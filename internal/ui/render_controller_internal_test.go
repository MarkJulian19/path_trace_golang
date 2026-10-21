package ui

import (
	"bytes"
	"image"
	"testing"
	"time"

	"fyne.io/fyne/v2/test"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	"github.com/MarkJulian19/path_trace_golang/internal/transport/gpu"
	engine "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

func TestRenderControllerCompletionAndModeSwitches(t *testing.T) {
	t.Parallel()
	app := test.NewApp()
	t.Cleanup(app.Quit)
	queue := make(chan func(), 128)
	uiDo := func(fn func()) { queue <- fn }
	sc := &scene.Scene{
		Camera:     scene.Camera{Position: scene.Vec3{Z: 4}, Up: scene.Vec3{Y: 1}, FOV: 45},
		Materials:  []scene.Material{{ID: "red", Type: scene.MaterialLambert, Albedo: scene.Color{R: 0.8}}},
		Objects:    []scene.Object{{Type: scene.ObjectSphere, MaterialID: "red", Size: scene.Vec3{X: 1}}},
		Settings:   scene.RenderSettings{Width: 65, Height: 49, SamplesPerPx: 1, MaxDepth: 2},
		Background: scene.Color{R: 0.1, G: 0.2, B: 0.8},
	}
	runtime := newUIRuntime("preview", sc, uiDo)
	session := newUISession(sc, nil, "", engine.New(nil, gpu.Adapter{}), runtime, uiDo)
	c := session.controller
	t.Cleanup(c.stop)
	for _, live := range []bool{true, false} {
		session.liveUpdate.SetChecked(live)
		c.startRender(true)
		job := controllerJob(c)
		select {
		case fn := <-queue:
			fn()
		case <-time.After(5 * time.Second):
			t.Fatal("render did not request a scene snapshot")
		}
		awaitRenderJob(t, job)
		drainUI(queue)
		if runtime.status.Text != "Done" || runtime.progressBar.Value != 1 {
			t.Fatalf(
				"live=%v: completed job lost its UI update: %s %.2f",
				live,
				runtime.status.Text,
				runtime.progressBar.Value,
			)
		}
		final, ok := session.lastFinalImage.(*image.RGBA)
		if !ok {
			t.Fatalf("live=%v: final image not saved", live)
		}
		displayed, ok := runtime.imgCanvas.Image.(*image.RGBA)
		if !ok || !bytes.Equal(displayed.Pix, final.Pix) {
			t.Fatalf("live=%v: final image was not displayed", live)
		}
	}
	modes := newRenderModeSelect(session.service, c.startRender)
	modes.SetSelected("Wireframe")
	job := controllerJob(c)
	c.launchJob(job)
	previous := c.displayedPreview
	pumpUIUntil(t, queue, func() bool {
		if c.displayedPreview == nil || c.displayedPreview == previous {
			return false
		}
		pixel := c.displayedPreview.RGBAAt(32, 24)
		return pixel.R > pixel.G && pixel.R > pixel.B
	})
	if session.service.GetCPURenderMode() != 1 || session.service.GetRenderMode() != 1 {
		t.Fatal("wireframe mode was not synchronized between CPU and GPU")
	}
	modes.SetSelected("Normal")
	c.startRender(true)
	pumpRenderJob(t, queue, job)
	next := controllerJob(c)
	if next == nil || next == job {
		t.Fatal("normal render was not started after cancelling wireframe")
	}
	pumpRenderJob(t, queue, next)
	drainUI(queue)
	if runtime.status.Text != "Done" || session.service.GetCPURenderMode() != 0 {
		t.Fatalf("normal mode did not complete: %s", runtime.status.Text)
	}
	session.service.SetBackend(engine.BackendGPU)
	c.startRender(true)
	job = controllerJob(c)
	pumpRenderJob(t, queue, job)
	drainUI(queue)
	if runtime.status.Text != "Done (CPU fallback)" {
		t.Fatalf("GPU fallback is hidden: %s", runtime.status.Text)
	}
}

func controllerJob(c *renderController) *renderJob {
	c.jobMu.Lock()
	defer c.jobMu.Unlock()
	return c.currentJob
}

func awaitRenderJob(t *testing.T, job *renderJob) {
	t.Helper()
	select {
	case <-job.done:
	case <-time.After(5 * time.Second):
		t.Fatal("render did not finish")
	}
}

func pumpRenderJob(t *testing.T, queue <-chan func(), job *renderJob) {
	t.Helper()
	timer := time.NewTimer(5 * time.Second)
	defer timer.Stop()
	for {
		select {
		case fn := <-queue:
			fn()
		case <-job.done:
			return
		case <-timer.C:
			t.Fatal("render did not finish while processing UI events")
		}
	}
}

func pumpUIUntil(t *testing.T, queue <-chan func(), done func() bool) {
	t.Helper()
	timer := time.NewTimer(5 * time.Second)
	defer timer.Stop()
	for !done() {
		select {
		case fn := <-queue:
			fn()
		case <-timer.C:
			t.Fatal("render or UI queue stalled")
		}
	}
}

func drainUI(queue <-chan func()) {
	for {
		select {
		case fn := <-queue:
			fn()
		default:
			return
		}
	}
}

func TestCloneSceneOwnsEditableData(t *testing.T) {
	t.Parallel()
	source := &scene.Scene{
		Materials: []scene.Material{{ID: "material"}}, Objects: []scene.Object{{ID: "object"}},
		Sky: &scene.Sky{Type: "solid"}, Fog: &scene.Fog{Density: 1},
	}
	snapshot := cloneScene(source)
	source.Materials[0].ID = "changed"
	source.Objects[0].ID = "changed"
	source.Sky.Type = "gradient"
	source.Fog.Density = 2
	if snapshot.Materials[0].ID != "material" || snapshot.Objects[0].ID != "object" ||
		snapshot.Sky.Type != "solid" || snapshot.Fog.Density != 1 {
		t.Fatal("render snapshot shares editable data with the scene")
	}
}

func TestProgressRefreshWithoutLivePreview(t *testing.T) {
	t.Parallel()
	c := &renderController{}
	state := &renderProgressState{
		cfg: engine.Config{Width: 2560, Height: 1440}, last: time.Now().Add(-time.Second),
	}
	if !c.shouldRefresh(state, 1, 100) {
		t.Fatal("2K render with live preview disabled stopped updating progress and elapsed time")
	}
	if c.shouldRefresh(state, 2, 100) {
		t.Fatal("progress updates are not throttled")
	}
	if !c.shouldRefresh(state, 100, 100) {
		t.Fatal("final progress update was suppressed")
	}
}

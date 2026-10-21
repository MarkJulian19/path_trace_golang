package gpu

import (
	"bytes"
	"image"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/go-gl/gl/v3.3-core/gl"

	renderentity "github.com/MarkJulian19/path_trace_golang/internal/entities/render"
	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	"github.com/MarkJulian19/path_trace_golang/internal/repository/scenejson"
)

var gpuIntegrationMu sync.Mutex

func TestGPUIntegration(t *testing.T) {
	t.Parallel()
	if os.Getenv("PATHTRACER_GPU_INTEGRATION") != "1" {
		t.Skip("set PATHTRACER_GPU_INTEGRATION=1 to run with an OpenGL 4.3 display")
	}
	gpuIntegrationMu.Lock()
	defer gpuIntegrationMu.Unlock()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	SetLivePreview(true)
	r := &gpuRenderer{}
	err := r.initGL()
	if err != nil {
		t.Fatalf("initialize actual OpenGL renderer: %v", err)
	}
	defer r.window.Destroy()
	SetDenoiseConfigFromUI(false, 1, 0.15)
	SetSmoothConfigFromUI(false, 1, 0)
	sc := &scene.Scene{
		Camera: scene.Camera{
			Position: scene.Vec3{Z: 4}, Up: scene.Vec3{Y: 1}, FOV: 45,
		},
		Materials: []scene.Material{{ID: "red", Type: scene.MaterialLambert, Albedo: scene.Color{R: 0.8}}},
		Objects: []scene.Object{{
			ID: "sphere", Type: scene.ObjectSphere, MaterialID: "red", Size: scene.Vec3{X: 1},
		}},
		Sky: &scene.Sky{Type: "solid", Color: scene.Color{R: 0.1, G: 0.25, B: 0.8}},
	}
	cases := []struct {
		name             string
		spp, mode, width int
		live, temporal   bool
	}{
		{name: "normal first frame", spp: 1, width: 65, live: true},
		{name: "wireframe after normal", spp: 1, mode: 1, width: 65, live: true},
		{name: "wireframe ignores path sample budget", spp: 17, mode: 1, width: 65, live: true},
		{name: "normal after wireframe", spp: 5, width: 65, live: true},
		{name: "temporal final readback", spp: 5, width: 65, temporal: true},
		{name: "temporal live readback", spp: 5, width: 65, live: true, temporal: true},
		{name: "resize and final readback", spp: 2, width: 33},
	}
	for _, tc := range cases {
		SetRenderModeFromUI(tc.mode)
		SetLivePreview(tc.live)
		temporalOn.Store(tc.temporal)
		SetRenderCancel(false)
		cfg := renderentity.Config{Width: tc.width, Height: 49, SamplesPerPx: tc.spp, MaxDepth: 3}
		img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
		completed := false
		progress := func(current, total int) { completed = current == total }
		err = r.renderOnce(sc, cfg, img, progress, func() bool { return false })
		if err != nil {
			t.Fatalf("%s: render: %v", tc.name, err)
		}
		if code := gl.GetError(); code != gl.NO_ERROR {
			t.Fatalf("%s: OpenGL error 0x%x", tc.name, code)
		}
		if !completed {
			t.Fatalf("%s: missing completion", tc.name)
		}
		center := img.RGBAAt(cfg.Width/2, cfg.Height/2)
		if center.R <= center.G || center.R <= center.B {
			t.Fatalf("%s: red sphere is missing: %v", tc.name, center)
		}
		if tc.mode == 0 {
			corner := img.RGBAAt(cfg.Width-1, 0)
			if corner.B <= corner.R || corner.G <= corner.R || corner.A != 255 {
				t.Fatalf("%s: sky layout or final texture is wrong: %v", tc.name, corner)
			}
		}
	}
	checkGPUCancellation(t, r, sc)
	checkGPUCancellationWithinPass(t, r, sc)
	checkGPUAdaptiveTiling(t, r, sc)
	checkGPUMaterials(t, r, sc)
	checkGPUEmptyScene(t, r)
	checkGPUBundledScenes(t, r)
}

func checkGPUAdaptiveTiling(t *testing.T, r *gpuRenderer, sc *scene.Scene) {
	t.Helper()
	previousLimit := gpuTileSize.Load()
	defer gpuTileSize.Store(previousLimit)
	cfg := renderentity.Config{Width: 257, Height: 193, SamplesPerPx: 9, MaxDepth: 4, Seed: 987}
	var reference []byte
	for _, limit := range []int64{32, 512} {
		gpuTileSize.Store(limit)
		img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
		err := r.renderOnce(sc, cfg, img, nil, func() bool { return false })
		if err != nil {
			t.Fatal(err)
		}
		if reference == nil {
			reference = bytes.Clone(img.Pix)
		} else if !bytes.Equal(reference, img.Pix) {
			t.Fatal("adaptive tiles overlap, leave gaps, or change per-pixel sample streams")
		}
	}
}

func checkGPUCancellationWithinPass(t *testing.T, r *gpuRenderer, sc *scene.Scene) {
	t.Helper()
	SetRenderModeFromUI(0)
	SetRenderCancel(false)
	warmConfig := renderentity.Config{Width: 32, Height: 32, SamplesPerPx: 1, MaxDepth: 16}
	warmImage := image.NewRGBA(image.Rect(0, 0, warmConfig.Width, warmConfig.Height))
	err := r.renderOnce(sc, warmConfig, warmImage, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	cfg := renderentity.Config{Width: 512, Height: 512, SamplesPerPx: 4096, MaxDepth: 16, Seed: 1}
	img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
	state, ok := r.prepareGPUFrame(sc, cfg, img, false, nil)
	if !ok {
		t.Fatal("failed to prepare cancellation probe")
	}
	started := time.Now()
	request := newGPUSampleRequest(state, cfg, img, nil, func() bool {
		return time.Since(started) >= 50*time.Millisecond
	})
	stats := r.renderSamples(request)
	if !stats.cancelled || time.Since(started) > 2*time.Second {
		t.Fatalf(
			"GPU cancellation stalled inside a pass: cancelled=%v elapsed=%s",
			stats.cancelled,
			time.Since(started),
		)
	}
	drainStarted := time.Now()
	gl.Finish()
	if elapsed := time.Since(drainStarted); elapsed > 250*time.Millisecond {
		t.Fatalf("cancelled GPU request left queued work: drain=%s", elapsed)
	}
	t.Logf("GPU cancellation including dispatch drain: %s", time.Since(started))
}

func checkGPUMaterials(t *testing.T, r *gpuRenderer, base *scene.Scene) {
	t.Helper()
	SetRenderModeFromUI(0)
	SetRenderCancel(false)
	sc := *base
	for _, typ := range []scene.MaterialType{
		scene.MaterialLambert, scene.MaterialMetal, scene.MaterialMirror, scene.MaterialDielectric, scene.MaterialEmissive,
	} {
		sc.Materials = []scene.Material{{
			ID: "red", Type: typ, Albedo: scene.Color{R: 1, G: 1, B: 1}, Rough: 0.3, IOR: 1.5,
			Emit: scene.Color{B: 1}, Power: 1,
		}}
		cfg := renderentity.Config{Width: 33, Height: 33, SamplesPerPx: 8, MaxDepth: 8}
		img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
		err := r.renderOnce(&sc, cfg, img, nil, func() bool { return false })
		if err != nil {
			t.Fatalf("material=%s: %v", typ, err)
		}
		pixel := img.RGBAAt(16, 16)
		if pixel.B == 0 || pixel.B <= pixel.R {
			t.Fatalf("material=%s: surface lost blue sky or emission: %v", typ, pixel)
		}
	}
}

func checkGPUCancellation(t *testing.T, r *gpuRenderer, sc *scene.Scene) {
	t.Helper()
	SetRenderModeFromUI(0)
	SetRenderCancel(false)
	cfg := renderentity.Config{Width: 32, Height: 18, SamplesPerPx: 20, MaxDepth: 2}
	img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
	cancelled := false
	lastSample := 0
	err := r.renderOnce(sc, cfg, img, func(current, _ int) {
		lastSample = current
		cancelled = true
	}, func() bool { return cancelled })
	if err != nil || !cancelled || lastSample >= cfg.SamplesPerPx {
		t.Fatalf("cancelled GPU job continued or reported completion: last=%d err=%v", lastSample, err)
	}
	SetRenderCancel(true)
	completed := false
	cfg.SamplesPerPx = 1
	err = r.renderOnce(
		sc,
		cfg,
		img,
		func(current, total int) { completed = current == total },
		func() bool { return false },
	)
	if err != nil || !completed {
		t.Fatalf("new GPU job inherited stale cancellation: completed=%v err=%v", completed, err)
	}
}

func checkGPUEmptyScene(t *testing.T, r *gpuRenderer) {
	t.Helper()
	SetRenderModeFromUI(0)
	SetRenderCancel(false)
	sc := &scene.Scene{
		Camera: scene.Camera{Position: scene.Vec3{Z: 4}, Up: scene.Vec3{Y: 1}, FOV: 45},
		Sky:    &scene.Sky{Type: "solid", Color: scene.Color{B: 0.8}},
	}
	cfg := renderentity.Config{Width: 1, Height: 1, SamplesPerPx: 1, MaxDepth: 2}
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	err := r.renderOnce(sc, cfg, img, nil, func() bool { return false })
	if err != nil || img.RGBAAt(0, 0).B == 0 {
		t.Fatalf("empty scene or one-pixel sky is black: pixel=%v err=%v", img.RGBAAt(0, 0), err)
	}
}

func checkGPUBundledScenes(t *testing.T, r *gpuRenderer) {
	t.Helper()
	paths, err := filepath.Glob("../../../scenes/*.json")
	validationPaths, validationErr := filepath.Glob("../../../scenes/validation/*.json")
	if validationErr != nil {
		t.Fatal(validationErr)
	}
	paths = append(paths, validationPaths...)
	galleryPaths, galleryErr := filepath.Glob("../../../scenes/gallery/*.json")
	if galleryErr != nil {
		t.Fatal(galleryErr)
	}
	paths = append(paths, galleryPaths...)
	if err != nil || len(paths) == 0 {
		t.Fatalf("find bundled scenes: paths=%v err=%v", paths, err)
	}
	for _, path := range paths {
		sc, loadErr := scenejson.New().Load(path)
		if loadErr != nil {
			t.Fatal(loadErr)
		}
		cfg := renderentity.Config{Width: 32, Height: 18, SamplesPerPx: 2, MaxDepth: 4, MaxRayDist: 1000}
		for _, mode := range []int{0, 1} {
			SetRenderModeFromUI(mode)
			img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
			renderErr := r.renderOnce(sc, cfg, img, nil, func() bool { return false })
			if renderErr != nil {
				t.Fatalf("%s mode=%d: %v", filepath.Base(path), mode, renderErr)
			}
			visible := false
			for i := 0; i < len(img.Pix); i += 4 {
				visible = visible || img.Pix[i] != 0 || img.Pix[i+1] != 0 || img.Pix[i+2] != 0
				if img.Pix[i+3] != 255 {
					t.Fatalf("%s mode=%d: missing pixel %d", filepath.Base(path), mode, i/4)
				}
			}
			if !visible {
				t.Fatalf("%s mode=%d: frame is entirely black", filepath.Base(path), mode)
			}
		}
		t.Logf("rendered %s on OpenGL in Normal and Wireframe", filepath.Base(path))
	}
}

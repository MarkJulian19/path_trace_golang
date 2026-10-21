package ui

import (
	"fmt"
	"image"
	"strconv"
	"sync"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/widget"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	engine "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

type videoUI struct {
	sc *scene.Scene

	settings    *scene.RenderSettings
	service     *engine.Service
	uiDo        func(func())
	status      *widget.Label
	progressBar *widget.ProgressBar
	timeLabel   *widget.Label
	mu          *sync.Mutex
	img         **image.RGBA
	canvas      *canvas.Image
	cam         scene.Camera
}

func newVideoBox(
	sc *scene.Scene,
	cam scene.Camera,
	settings *scene.RenderSettings,
	service *engine.Service,
	uiDo func(func()),
	status *widget.Label,
	progressBar *widget.ProgressBar,
	timeLabel *widget.Label,
	mu *sync.Mutex,
	img **image.RGBA,
	previewCanvas *canvas.Image,
) fyne.CanvasObject {
	controls := newVideoUI(sc, cam, settings, service, uiDo, status, progressBar, timeLabel, mu, img, previewCanvas)
	form, renderButton := newVideoForm(controls, cam)
	return newVideoLayout(form, renderButton)
}

func newVideoUI(
	sc *scene.Scene,
	cam scene.Camera,
	settings *scene.RenderSettings,
	service *engine.Service,
	uiDo func(func()),
	status *widget.Label,
	progressBar *widget.ProgressBar,
	timeLabel *widget.Label,
	mu *sync.Mutex,
	img **image.RGBA,
	previewCanvas *canvas.Image,
) *videoUI {
	return &videoUI{
		sc: sc, cam: cam, settings: settings, service: service, uiDo: uiDo, status: status,
		progressBar: progressBar, timeLabel: timeLabel, mu: mu, img: img, canvas: previewCanvas,
	}
}

type videoForm struct {
	startPosition, endPosition videoEntries
	startTarget, endTarget     videoEntries
	duration, fps, outputPath  *widget.Entry
}

func newVideoForm(controls *videoUI, cam scene.Camera) (videoForm, *widget.Button) {
	form := videoForm{
		startPosition: newVideoEntries(cam.Position), endPosition: newVideoEntries(cam.Position),
		startTarget: newVideoEntries(cam.Target), endTarget: newVideoEntries(cam.Target),
		duration: widget.NewEntry(), fps: widget.NewEntry(), outputPath: widget.NewEntry(),
	}
	form.duration.SetText("5.0")
	form.fps.SetText("30")
	form.outputPath.SetText("output_video.avi")
	renderButton := widget.NewButton("Render Video", func() {
		startCam := controls.cameraFromEntries(form.startPosition, form.startTarget)
		endCam := controls.cameraFromEntries(form.endPosition, form.endTarget)
		duration := parseVideoFloat(form.duration, uiDefaultVideoDuration)
		fps := parseVideoInt(form.fps, uiDefaultVideoFPS)
		path := form.outputPath.Text
		if path == "" {
			path = "output_video.avi"
		}
		controls.render(startCam, endCam, duration, fps, path)
	})
	return form, renderButton
}

func newVideoLayout(form videoForm, renderButton *widget.Button) fyne.CanvasObject {
	return container.NewVBox(
		widget.NewLabel("Video Generation"),
		widget.NewLabel("Start Camera"),
		videoCoordinateGrid("Start Pos", form.startPosition),
		videoCoordinateGrid("Start Target", form.startTarget),
		widget.NewLabel("End Camera"),
		videoCoordinateGrid("End Pos", form.endPosition),
		videoCoordinateGrid("End Target", form.endTarget),
		container.NewGridWithColumns(2,
			widget.NewLabel("Duration (s)"), form.duration,
			widget.NewLabel("FPS"), form.fps,
			widget.NewLabel("Output Path"), form.outputPath,
		),
		renderButton,
	)
}

type videoEntries struct{ x, y, z *widget.Entry }

func newVideoEntries(value scene.Vec3) videoEntries {
	entries := videoEntries{widget.NewEntry(), widget.NewEntry(), widget.NewEntry()}
	entries.x.SetText(fmt.Sprintf("%.2f", value.X))
	entries.y.SetText(fmt.Sprintf("%.2f", value.Y))
	entries.z.SetText(fmt.Sprintf("%.2f", value.Z))
	return entries
}

func videoCoordinateGrid(label string, entries videoEntries) fyne.CanvasObject {
	return container.NewGridWithColumns(2,
		widget.NewLabel(label+" X"), entries.x,
		widget.NewLabel(label+" Y"), entries.y,
		widget.NewLabel(label+" Z"), entries.z,
	)
}

func parseVideoFloat(entry *widget.Entry, fallback float64) float64 {
	value, err := strconv.ParseFloat(entry.Text, 64)
	if err != nil {
		return fallback
	}
	return value
}

func parseVideoInt(entry *widget.Entry, fallback int) int {
	value, err := strconv.ParseInt(entry.Text, 10, 64)
	if err != nil {
		return fallback
	}
	return int(value)
}

func (v *videoUI) cameraFromEntries(position, target videoEntries) scene.Camera {
	return scene.Camera{
		Position: scene.Vec3{
			X: parseVideoFloat(position.x, v.cam.Position.X),
			Y: parseVideoFloat(position.y, v.cam.Position.Y),
			Z: parseVideoFloat(position.z, v.cam.Position.Z),
		},
		Target: scene.Vec3{
			X: parseVideoFloat(target.x, v.cam.Target.X),
			Y: parseVideoFloat(target.y, v.cam.Target.Y),
			Z: parseVideoFloat(target.z, v.cam.Target.Z),
		},
		Up:          v.cam.Up,
		FOV:         v.cam.FOV,
		Aperture:    v.cam.Aperture,
		FocusDist:   v.cam.FocusDist,
		AspectRatio: v.cam.AspectRatio,
	}
}

func (v *videoUI) render(startCam, endCam scene.Camera, duration float64, fps int, outputPath string) {
	cfg := engine.Config{
		Width:        v.settings.Width,
		Height:       v.settings.Height,
		SamplesPerPx: v.settings.SamplesPerPx,
		MaxDepth:     v.settings.MaxDepth,
		MaxRayDist:   float32(v.settings.MaxRayDist),
	}
	if v.service.GetBackend() == engine.BackendGPU {
		v.service.PrepareGPU()
	}
	snapshot := cloneScene(v.sc)
	go v.renderJob(&snapshot, startCam, endCam, duration, fps, outputPath, cfg)
}

func (v *videoUI) renderJob(
	sc *scene.Scene,
	startCam, endCam scene.Camera,
	duration float64,
	fps int,
	outputPath string,
	cfg engine.Config,
) {
	previewCanvas, previewImg := v.videoPreviewState()
	v.showVideoProgress()
	lastPreviewUpdate := time.Now()
	progress := func(currentFrame, totalFrames int, frameImg image.Image) {
		lastPreviewUpdate = v.updateVideoPreview(
			previewCanvas,
			&previewImg,
			lastPreviewUpdate,
			currentFrame,
			totalFrames,
			frameImg,
		)
	}
	frames, err := v.service.VideoSequence(sc, startCam, endCam, duration, fps, cfg, progress)
	if err != nil {
		v.uiDo(func() { v.status.SetText(fmt.Sprintf("Video render error: %v", err)) })
		return
	}
	v.encodeVideo(frames, outputPath, fps)
}

func (v *videoUI) videoPreviewState() (*canvas.Image, *image.RGBA) {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.canvas, *v.img
}

func (v *videoUI) showVideoProgress() {
	v.uiDo(func() {
		v.status.SetText("Rendering video frames...")
		v.progressBar.SetValue(0)
		v.progressBar.Show()
	})
}

func (v *videoUI) encodeVideo(frames []image.Image, outputPath string, fps int) {
	v.uiDo(func() { v.status.SetText("Creating video file...") })
	videoProgress := func(currentFrame, totalFrames int) {
		value := float64(currentFrame) / float64(totalFrames)
		text := fmt.Sprintf("Encoding frame %d/%d", currentFrame, totalFrames)
		v.uiDo(func() { v.progressBar.SetValue(value); v.timeLabel.SetText(text) })
	}
	err := engine.CreateVideoFromFrames(frames, outputPath, float64(fps), videoProgress)
	if err != nil {
		v.uiDo(func() { v.status.SetText(fmt.Sprintf("Video creation error: %v", err)) })
		return
	}
	v.uiDo(func() { v.progressBar.SetValue(1); v.status.SetText("Video saved to " + outputPath) })
}

func (v *videoUI) updateVideoPreview(
	previewCanvas *canvas.Image,
	previewImg **image.RGBA,
	lastUpdate time.Time,
	currentFrame, totalFrames int,
	frameImg image.Image,
) time.Time {
	value := float64(currentFrame) / float64(totalFrames)
	text := fmt.Sprintf("Frame %d/%d", currentFrame, totalFrames)
	now := time.Now()
	if now.Sub(lastUpdate) < uiProgressUpdateInterval && currentFrame != totalFrames {
		v.uiDo(func() { v.progressBar.SetValue(value); v.timeLabel.SetText(text) })
		return lastUpdate
	}
	v.mu.Lock()
	bounds := frameImg.Bounds()
	*previewImg = image.NewRGBA(bounds)
	copyImage((*previewImg), frameImg)
	snapshot := *previewImg
	v.mu.Unlock()
	v.uiDo(func() {
		v.progressBar.SetValue(value)
		v.timeLabel.SetText(text)
		previewCanvas.Image = snapshot
		previewCanvas.Refresh()
	})
	return now
}

func copyImage(dst *image.RGBA, src image.Image) {
	bounds := src.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			dst.Set(x, y, src.At(x, y))
		}
	}
}

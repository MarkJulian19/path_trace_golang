package ui

import (
	"image"
	"image/color"
	"sync"

	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/widget"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

type uiRuntime struct {
	img       *image.RGBA
	imgCanvas *canvas.Image

	publishPreview func()
	setCanvasSize  func()
	status         *widget.Label
	progressBar    *widget.ProgressBar
	timeLabel      *widget.Label
	preview        scene.RenderSettings
	final          scene.RenderSettings

	mu          sync.Mutex
	maxDisplayW float32
	maxDisplayH float32
}

func newUIRuntime(mode string, sc *scene.Scene, uiDo func(func())) *uiRuntime {
	baseSettings := sceneRenderSettings(mode, sc)
	runtime := &uiRuntime{
		preview:     baseSettings,
		final:       baseSettings,
		maxDisplayW: float32(uiMaxDisplayWidth),
		maxDisplayH: float32(uiMaxDisplayHeight),
		img:         image.NewRGBA(image.Rect(0, 0, baseSettings.Width, baseSettings.Height)),
		status:      widget.NewLabel("Idle"),
		progressBar: widget.NewProgressBar(),
		timeLabel:   widget.NewLabel("Render time: -"),
	}
	runtime.final.SamplesPerPx *= 4
	runtime.final.MaxDepth *= 2
	fillImage(runtime.img, color.RGBA{0, 0, 0, 255})
	runtime.imgCanvas = canvas.NewImageFromImage(runtime.img)
	runtime.imgCanvas.FillMode = canvas.ImageFillContain
	var displayImg *image.RGBA
	runtime.publishPreview = newPreviewPublisher(&runtime.mu, &runtime.img, &displayImg, runtime.imgCanvas, uiDo)
	runtime.setCanvasSize = func() {
		setPreviewCanvasSize(runtime.imgCanvas, &runtime.preview, &runtime.maxDisplayW, &runtime.maxDisplayH)
	}
	runtime.setCanvasSize()
	runtime.progressBar.Hide()
	return runtime
}

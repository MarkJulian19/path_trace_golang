package ui

import (
	"image"
	"sync"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/canvas"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

func newPreviewPublisher(
	mu *sync.Mutex,
	img **image.RGBA,
	displayImg **image.RGBA,
	imgCanvas *canvas.Image,
	uiDo func(func()),
) func() {
	return func() {
		mu.Lock()
		source := *img
		if source == nil {
			mu.Unlock()
			return
		}
		bounds := source.Bounds()
		if *displayImg == nil || (*displayImg).Bounds() != bounds {
			*displayImg = image.NewRGBA(bounds)
		}
		copy((*displayImg).Pix, source.Pix)
		snapshot := *displayImg
		mu.Unlock()
		uiDo(func() {
			imgCanvas.Image = snapshot
			imgCanvas.Refresh()
		})
	}
}

func setPreviewCanvasSize(imgCanvas *canvas.Image, preview *scene.RenderSettings, maxWidth, maxHeight *float32) {
	aspect := float32(preview.Width) / float32(preview.Height)
	displayWidth := *maxWidth
	displayHeight := displayWidth / aspect
	if displayHeight > *maxHeight {
		displayHeight = *maxHeight
		displayWidth = displayHeight * aspect
	}
	imgCanvas.SetMinSize(fyne.NewSize(displayWidth, displayHeight))
}

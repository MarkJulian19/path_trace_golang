package ui

import (
	"fmt"
	"image"
	"sync"

	"fyne.io/fyne/v2/widget"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	engine "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
	scenecase "github.com/MarkJulian19/path_trace_golang/internal/usecases/scene"
)

func newSaveButtons(
	scenes *scenecase.Scene,
	sc *scene.Scene,
	scenePath string,
	status *widget.Label,
	outputPath *widget.Entry,
	mu *sync.Mutex,
	lastImage *image.Image,
	lastConfig *engine.Config,
	uiDo func(func()),
) (*widget.Button, *widget.Button) {
	saveScene := widget.NewButton("Save scene", func() {
		err := scenes.Save(scenePath, sc)
		if err != nil {
			status.SetText(fmt.Sprintf("Save error: %v", err))
			return
		}
		status.SetText("Scene saved")
	})
	saveImage := widget.NewButton("Save image (PNG)", func() {
		path := outputPath.Text
		if path == "" {
			path = "ui_render.png"
		}
		mu.Lock()
		savedImage := *lastImage
		savedConfig := *lastConfig
		mu.Unlock()
		if savedImage == nil {
			status.SetText("No final render available. Please render final image first.")
			return
		}
		uiDo(func() { status.SetText("Saving image...") })
		go savePNG(path, savedImage, savedConfig, status, uiDo)
	})
	return saveScene, saveImage
}

func savePNG(path string, imageValue image.Image, config engine.Config, status *widget.Label, uiDo func(func())) {
	err := engine.SavePNG(path, imageValue)
	if err != nil {
		message := fmt.Sprintf("Save image error: %v", err)
		uiDo(func() { status.SetText(message) })
		return
	}
	message := fmt.Sprintf(
		"Image saved to %s (%dx%d, %d samples)",
		path,
		config.Width,
		config.Height,
		config.SamplesPerPx,
	)
	uiDo(func() { status.SetText(message) })
}

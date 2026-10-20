package ui

import (
	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/widget"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	engine "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

func newScenePicker(
	sc *scene.Scene,
	preview *scene.RenderSettings,
	imgCanvas *canvas.Image,
	service *engine.Service,
	status *widget.Label,
	startRender *func(bool),
) fyne.CanvasObject {
	selected := -1
	dragging := false
	dragStartX, dragStartY := 0, 0
	var dragStartPosition scene.Vec3
	return newPickableCanvas(imgCanvas, func(x, y int) {
		cfg := engine.Config{
			Width: preview.Width, Height: preview.Height, SamplesPerPx: preview.SamplesPerPx,
			MaxDepth: preview.MaxDepth, MaxRayDist: float32(preview.MaxRayDist),
		}
		objIndex := pickObject(sc, cfg, x, y)
		handleObjectSelection(
			objIndex,
			x,
			y,
			sc,
			service,
			status,
			&selected,
			&dragging,
			&dragStartX,
			&dragStartY,
			&dragStartPosition,
			*startRender,
		)
	}, func(_, _, x2, y2 int) {
		if !dragging {
			return
		}
		dragObject(sc, engine.Config{
			Width: preview.Width, Height: preview.Height, SamplesPerPx: preview.SamplesPerPx,
			MaxDepth: preview.MaxDepth,
		}, selected, dragStartX, dragStartY, x2, y2, dragStartPosition, *startRender)
	})
}

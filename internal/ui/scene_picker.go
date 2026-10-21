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
	startInteractiveRender func(),
	selectedObject *int,
	transformMode *editorTransformMode,
	objectMoveEnabled *bool,
	onSelect func(int),
	onObjectUpdated func(),
) (fyne.CanvasObject, *sceneGizmo) {
	state := &scenePickerState{
		selected: selectedObject, transformMode: transformMode, objectMoveEnabled: objectMoveEnabled,
	}
	picker := newPickableCanvas(
		imgCanvas,
		func(x, y int) { state.tap(x, y, sc, preview, service, status, startRender, onSelect) },
		func(dx, dy int) {
			state.drag(dx, dy, sc, preview, startInteractiveRender, onObjectUpdated)
		},
	)
	gizmo := newSceneGizmo(
		sc, preview, state.selected, transformMode, objectMoveEnabled, startInteractiveRender, onObjectUpdated,
	)
	return newSceneViewport(picker, gizmo), gizmo
}

type scenePickerState struct {
	selected          *int
	transformMode     *editorTransformMode
	objectMoveEnabled *bool
	dragging          bool
}

func (p *scenePickerState) tap(
	x, y int,
	sc *scene.Scene,
	preview *scene.RenderSettings,
	service *engine.Service,
	status *widget.Label,
	startRender *func(bool),
	onSelect func(int),
) {
	cfg := engine.Config{
		Width: preview.Width, Height: preview.Height, SamplesPerPx: preview.SamplesPerPx,
		MaxDepth: preview.MaxDepth, MaxRayDist: float32(preview.MaxRayDist),
	}
	objIndex := pickObject(sc, cfg, x, y)
	handleObjectSelection(
		objIndex, sc, service, status, p.selected, &p.dragging, *startRender,
	)
	if onSelect != nil {
		onSelect(objIndex)
	}
}

func (p *scenePickerState) drag(
	deltaX, deltaY int,
	sc *scene.Scene,
	preview *scene.RenderSettings,
	startInteractiveRender func(),
	onObjectUpdated func(),
) {
	if !p.dragging || p.transformMode == nil || *p.transformMode != transformModeMove ||
		p.objectMoveEnabled == nil || !*p.objectMoveEnabled {
		return
	}
	dragObjectDelta(
		sc,
		engine.Config{
			Width: preview.Width, Height: preview.Height,
			SamplesPerPx: preview.SamplesPerPx, MaxDepth: preview.MaxDepth,
		},
		*p.selected, deltaX, deltaY, startInteractiveRender,
	)
	if onObjectUpdated != nil {
		onObjectUpdated()
	}
}

package ui

import (
	"fmt"

	"fyne.io/fyne/v2/widget"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	engine "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

func handleObjectSelection(
	objIndex, x, y int,
	sc *scene.Scene,
	service *engine.Service,
	status *widget.Label,
	selected *int,
	dragging *bool,
	dragStartX, dragStartY *int,
	dragStartPosition *scene.Vec3,
	startRender func(bool),
) {
	*selected = objIndex
	if objIndex < 0 {
		status.SetText("No object selected")
		setSelectedGPUObject(service, -1)
		*dragging = false
		startRender(false)
		return
	}
	status.SetText(fmt.Sprintf("Selected object: %d", objIndex))
	setSelectedGPUObject(service, objIndex)
	if objIndex >= len(sc.Objects) {
		startRender(false)
		return
	}
	*dragStartPosition = sc.Objects[objIndex].Position
	*dragStartX = x
	*dragStartY = y
	*dragging = true
	startRender(false)
}

func setSelectedGPUObject(service *engine.Service, index int) {
	if service.GetBackend() == engine.BackendGPU {
		service.SetSelectedObjectFromUI(index)
	}
}

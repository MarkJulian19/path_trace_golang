package ui

import (
	"fmt"

	"fyne.io/fyne/v2/widget"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	engine "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

func handleObjectSelection(
	objIndex int,
	sc *scene.Scene,
	service *engine.Service,
	status *widget.Label,
	selected *int,
	dragging *bool,
	startRender func(bool),
) {
	*selected = objIndex
	if objIndex < 0 {
		status.SetText("No object selected")
		setSelectedGPUObject(service, -1)
		*dragging = false
		startSelectionPreview(service, startRender)
		return
	}
	status.SetText(fmt.Sprintf("Selected object: %d", objIndex))
	setSelectedGPUObject(service, objIndex)
	if objIndex >= len(sc.Objects) {
		startSelectionPreview(service, startRender)
		return
	}
	*dragging = true
	startSelectionPreview(service, startRender)
}

func startSelectionPreview(service *engine.Service, startRender func(bool)) {
	if isWireframeMode(service) || startRender == nil {
		return
	}
	startRender(false)
}

func setSelectedGPUObject(service *engine.Service, index int) {
	if service.GetBackend() == engine.BackendGPU {
		service.SetSelectedObjectFromUI(index)
	}
}

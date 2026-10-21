package ui

import (
	"fyne.io/fyne/v2/widget"

	engine "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

func newBackendControls(
	service *engine.Service,
	startRender func(bool),
	denoiseEnabled *bool,
	denoiseSigmaS, denoiseSigmaR *float64,
	smoothEnabled *bool,
	smoothRadius *int,
	smoothStrength *float64,
) (*widget.RadioGroup, *widget.Label) {
	label := widget.NewLabel("Backend: CPU")
	backendSelect := widget.NewRadioGroup([]string{"CPU", "GPU"}, func(value string) {
		if value == "GPU" {
			label.SetText("Backend: GPU")
			service.SetBackend(engine.BackendGPU)
			service.SetDenoiseConfigFromUI(*denoiseEnabled, *denoiseSigmaS, *denoiseSigmaR)
			service.SetSmoothConfigFromUI(*smoothEnabled, *smoothRadius, *smoothStrength)
		} else {
			label.SetText("Backend: CPU")
			service.SetBackend(engine.BackendCPU)
		}
		startRender(false)
	})
	backendSelect.Horizontal = true
	backendSelect.Required = true
	if service.GetBackend() == engine.BackendGPU {
		backendSelect.SetSelected("GPU")
		label.SetText("Backend: GPU")
	} else {
		backendSelect.SetSelected("CPU")
	}
	return backendSelect, label
}

func newRenderModeSelect(
	service *engine.Service,
	startRender func(bool),
) *widget.RadioGroup {
	modeSelect := widget.NewRadioGroup([]string{"Normal", "Wireframe"}, func(selected string) {
		mode := 0
		if selected == "Wireframe" {
			mode = 1
		}
		service.SetRenderModeFromUI(mode)
		service.SetCPURenderMode(mode)
		startRender(false)
	})
	modeSelect.SetSelected("Normal")
	return modeSelect
}

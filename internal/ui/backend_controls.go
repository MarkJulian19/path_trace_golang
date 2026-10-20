package ui

import (
	"sync"

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
) (*widget.Slider, *widget.Label) {
	slider := widget.NewSlider(0, 1)
	slider.Step = 1
	slider.Value = 0
	label := widget.NewLabel("Backend: CPU")
	slider.OnChanged = func(value float64) {
		if value >= uiHalf {
			label.SetText("Backend: GPU")
			service.SetBackend(engine.BackendGPU)
			service.SetDenoiseConfigFromUI(*denoiseEnabled, *denoiseSigmaS, *denoiseSigmaR)
			service.SetSmoothConfigFromUI(*smoothEnabled, *smoothRadius, *smoothStrength)
		} else {
			label.SetText("Backend: CPU")
			service.SetBackend(engine.BackendCPU)
		}
		startRender(false)
	}
	return slider, label
}

func newRenderModeSelect(
	service *engine.Service,
	mu *sync.Mutex,
	stopCh *chan struct{},
	startRender func(bool),
) *widget.RadioGroup {
	modeSelect := widget.NewRadioGroup([]string{"Normal", "Wireframe"}, func(selected string) {
		mu.Lock()
		if *stopCh != nil {
			close(*stopCh)
			*stopCh = nil
		}
		mu.Unlock()
		mode := 0
		if selected == "Wireframe" {
			mode = 1
		}
		if service.GetBackend() == engine.BackendGPU {
			service.SetRenderModeFromUI(mode)
		} else {
			service.SetCPURenderMode(mode)
		}
		startRender(false)
	})
	modeSelect.SetSelected("Normal")
	return modeSelect
}

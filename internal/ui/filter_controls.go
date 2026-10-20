package ui

import (
	"fmt"
	"strconv"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/widget"

	engine "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

func newFilterBoxes(
	service *engine.Service,
	startRender func(bool),
	denoiseEnabled *bool,
	denoiseSigmaS, denoiseSigmaR *float64,
	smoothEnabled *bool,
	smoothRadius *int,
	smoothStrength *float64,
) (fyne.CanvasObject, fyne.CanvasObject) {
	return newDenoiseBox(service, startRender, denoiseEnabled, denoiseSigmaS, denoiseSigmaR),
		newSmoothBox(service, startRender, smoothEnabled, smoothRadius, smoothStrength)
}

func newDenoiseBox(
	service *engine.Service,
	startRender func(bool),
	enabled *bool,
	sigmaS, sigmaR *float64,
) fyne.CanvasObject {
	check := widget.NewCheck("GPU denoise (bilateral 3x3)", func(value bool) {
		*enabled = value
		applyDenoise(service, startRender, *enabled, *sigmaS, *sigmaR)
	})
	check.SetChecked(*enabled)
	sigmaSEntry := widget.NewEntry()
	sigmaSEntry.SetText(fmt.Sprintf("%.2f", *sigmaS))
	sigmaREntry := widget.NewEntry()
	sigmaREntry.SetText(fmt.Sprintf("%.2f", *sigmaR))
	apply := widget.NewButton("Apply GPU denoise", func() {
		*sigmaS = parsePositiveSettingsFloat(sigmaSEntry, *sigmaS)
		*sigmaR = parsePositiveSettingsFloat(sigmaREntry, *sigmaR)
		applyDenoise(service, startRender, *enabled, *sigmaS, *sigmaR)
	})
	return container.NewVBox(
		widget.NewLabel("GPU denoise"), check,
		container.NewGridWithColumns(2,
			widget.NewLabel("Sigma S (space)"), sigmaSEntry,
			widget.NewLabel("Sigma R (color)"), sigmaREntry,
		),
		apply,
	)
}

func applyDenoise(service *engine.Service, startRender func(bool), enabled bool, sigmaS, sigmaR float64) {
	if service.GetBackend() != engine.BackendGPU {
		return
	}
	service.SetDenoiseConfigFromUI(enabled, sigmaS, sigmaR)
	startRender(false)
}

func newSmoothBox(
	service *engine.Service,
	startRender func(bool),
	enabled *bool,
	radius *int,
	strength *float64,
) fyne.CanvasObject {
	check := widget.NewCheck("GPU extra smoothing (strong blur)", func(value bool) {
		*enabled = value
		applySmooth(service, startRender, *enabled, *radius, *strength)
	})
	check.SetChecked(*enabled)
	radiusEntry := widget.NewEntry()
	radiusEntry.SetText(strconv.Itoa(*radius))
	strengthEntry := widget.NewEntry()
	strengthEntry.SetText(fmt.Sprintf("%.2f", *strength))
	apply := widget.NewButton("Apply smoothing", func() {
		*radius = min(max(parseSettingsInt(radiusEntry, *radius), 1), uiMaxSmoothRadius)
		*strength = min(max(parseCameraFloat(strengthEntry, *strength), 0), 1)
		applySmooth(service, startRender, *enabled, *radius, *strength)
	})
	return container.NewVBox(
		widget.NewLabel("GPU extra smoothing"), check,
		container.NewGridWithColumns(2,
			widget.NewLabel("Radius (1-5)"), radiusEntry,
			widget.NewLabel("Strength (0-1)"), strengthEntry,
		),
		apply,
	)
}

func applySmooth(service *engine.Service, startRender func(bool), enabled bool, radius int, strength float64) {
	if service.GetBackend() != engine.BackendGPU {
		return
	}
	service.SetSmoothConfigFromUI(enabled, radius, strength)
	startRender(false)
}

func parsePositiveSettingsFloat(entry *widget.Entry, fallback float64) float64 {
	value, err := strconv.ParseFloat(entry.Text, 64)
	if err != nil || value <= 0 {
		return fallback
	}
	return value
}

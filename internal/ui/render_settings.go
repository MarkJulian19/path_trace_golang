package ui

import (
	"fmt"
	"image"
	"image/color"
	"strconv"
	"sync"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/widget"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

type renderSettingsUI struct {
	sc             *scene.Scene
	preview        *scene.RenderSettings
	final          *scene.RenderSettings
	maxDisplayW    *float32
	maxDisplayH    *float32
	status         *widget.Label
	mu             *sync.Mutex
	img            **image.RGBA
	imgCanvas      *canvas.Image
	stopCh         *chan struct{}
	lastFinalImage *image.Image
	setCanvasSize  func()
	startRender    func(bool)

	prevW, prevH, prevSpp, prevDepth, prevMaxRayDist *widget.Entry
	dispW, dispH                                     *widget.Entry
	finalW, finalH, finalSpp, finalDepth             *widget.Entry
	finalMaxRayDist                                  *widget.Entry

	fogDensity, fogColorR, fogColorG, fogColorB       *widget.Entry
	fogScatter, fogSigmaS, fogSigmaA, fogG            *widget.Entry
	fogHeteroStrength, fogNoiseScale, fogNoiseOctaves *widget.Entry
	fogAffectSky, fogEnabled, fogGPUVolumetric        *widget.Check
}

func newRenderSettingsUI(
	sc *scene.Scene,
	preview, final *scene.RenderSettings,
	maxDisplayW, maxDisplayH *float32,
	status *widget.Label,
	mu *sync.Mutex,
	img **image.RGBA,
	imgCanvas *canvas.Image,
	stopCh *chan struct{},
	lastFinalImage *image.Image,
	setCanvasSize func(),
	startRender func(bool),
) *renderSettingsUI {
	s := &renderSettingsUI{
		sc:             sc,
		preview:        preview,
		final:          final,
		maxDisplayW:    maxDisplayW,
		maxDisplayH:    maxDisplayH,
		status:         status,
		mu:             mu,
		img:            img,
		imgCanvas:      imgCanvas,
		stopCh:         stopCh,
		lastFinalImage: lastFinalImage,
		setCanvasSize:  setCanvasSize,
		startRender:    startRender,
	}
	s.newEntries()
	s.newFogControls()
	s.loadFog()
	s.loadSettings()
	return s
}

func (s *renderSettingsUI) newEntries() {
	s.prevW, s.prevH, s.prevSpp, s.prevDepth = widget.NewEntry(), widget.NewEntry(), widget.NewEntry(), widget.NewEntry()
	s.dispW, s.dispH = widget.NewEntry(), widget.NewEntry()
	s.finalW, s.finalH, s.finalSpp, s.finalDepth = widget.NewEntry(), widget.NewEntry(), widget.NewEntry(), widget.NewEntry()
	s.prevMaxRayDist, s.finalMaxRayDist = widget.NewEntry(), widget.NewEntry()
}

func (s *renderSettingsUI) newFogControls() {
	s.fogDensity, s.fogColorR, s.fogColorG, s.fogColorB = widget.NewEntry(), widget.NewEntry(), widget.NewEntry(), widget.NewEntry()
	s.fogScatter, s.fogSigmaS, s.fogSigmaA, s.fogG = widget.NewEntry(), widget.NewEntry(), widget.NewEntry(), widget.NewEntry()
	s.fogHeteroStrength, s.fogNoiseScale, s.fogNoiseOctaves = widget.NewEntry(), widget.NewEntry(), widget.NewEntry()
	s.fogAffectSky = widget.NewCheck("Affect sky", func(bool) {})
	s.fogEnabled = widget.NewCheck("Enable fog (GPU only)", func(bool) {})
	s.fogGPUVolumetric = widget.NewCheck("Volumetric scattering (GPU only)", func(bool) {})
}

func (s *renderSettingsUI) loadSettings() {
	s.prevW.SetText(strconv.Itoa(s.preview.Width))
	s.prevH.SetText(strconv.Itoa(s.preview.Height))
	s.prevSpp.SetText(strconv.Itoa(s.preview.SamplesPerPx))
	s.prevDepth.SetText(strconv.Itoa(s.preview.MaxDepth))
	s.prevMaxRayDist.SetText(strconv.FormatFloat(s.preview.MaxRayDist, 'f', 1, 64))
	s.dispW.SetText(strconv.Itoa(int(*s.maxDisplayW)))
	s.dispH.SetText(strconv.Itoa(int(*s.maxDisplayH)))
	s.finalW.SetText(strconv.Itoa(s.final.Width))
	s.finalH.SetText(strconv.Itoa(s.final.Height))
	s.finalSpp.SetText(strconv.Itoa(s.final.SamplesPerPx))
	s.finalDepth.SetText(strconv.Itoa(s.final.MaxDepth))
	s.finalMaxRayDist.SetText(strconv.FormatFloat(s.final.MaxRayDist, 'f', 1, 64))
}

func (s *renderSettingsUI) loadFog() {
	if s.sc.Fog == nil {
		s.loadDefaultFog()
		return
	}
	fog := s.sc.Fog
	s.fogEnabled.SetChecked(fog.Density > 0 || fog.SigmaS > 0 || fog.SigmaA > 0)
	s.fogDensity.SetText(fmt.Sprintf("%.3f", fog.Density))
	s.fogColorR.SetText(fmt.Sprintf("%.2f", fog.Color.R))
	s.fogColorG.SetText(fmt.Sprintf("%.2f", fog.Color.G))
	s.fogColorB.SetText(fmt.Sprintf("%.2f", fog.Color.B))
	s.setFogEntry(s.fogScatter, fog.Scatter, "1.0", "%.2f")
	s.setFogEntry(s.fogSigmaS, fog.SigmaS, "0.5", "%.3f")
	s.setFogEntry(s.fogSigmaA, fog.SigmaA, "0.1", "%.3f")
	s.fogG.SetText(fmt.Sprintf("%.2f", fog.G))
	s.setFogEntry(s.fogHeteroStrength, fog.HeteroStrength, "0.0", "%.2f")
	s.setFogEntry(s.fogNoiseScale, fog.NoiseScale, "3.0", "%.2f")
	if fog.NoiseOctaves > 0 {
		s.fogNoiseOctaves.SetText(strconv.Itoa(fog.NoiseOctaves))
	} else {
		s.fogNoiseOctaves.SetText("3")
	}
	s.fogAffectSky.SetChecked(fog.AffectSky)
	s.fogGPUVolumetric.SetChecked(fog.GPUVolumetric)
}

func (s *renderSettingsUI) loadDefaultFog() {
	s.fogEnabled.SetChecked(false)
	s.fogDensity.SetText("0.0")
	s.fogColorR.SetText("0.8")
	s.fogColorG.SetText("0.8")
	s.fogColorB.SetText("0.8")
	s.fogScatter.SetText("1.0")
	s.fogSigmaS.SetText("0.5")
	s.fogSigmaA.SetText("0.1")
	s.fogG.SetText("0.6")
	s.fogHeteroStrength.SetText("0.0")
	s.fogNoiseScale.SetText("3.0")
	s.fogNoiseOctaves.SetText("3")
	s.fogAffectSky.SetChecked(false)
	s.fogGPUVolumetric.SetChecked(true)
}

func (s *renderSettingsUI) setFogEntry(entry *widget.Entry, value float64, fallback, format string) {
	if value > 0 {
		entry.SetText(fmt.Sprintf(format, value))
		return
	}
	entry.SetText(fallback)
}

func (s *renderSettingsUI) apply() {
	s.applyPreviewSettings()
	s.applyFinalSettings()
	s.preview.MaxRayDist = parseSettingsFloat(s.prevMaxRayDist, s.preview.MaxRayDist)
	s.final.MaxRayDist = parseSettingsFloat(s.finalMaxRayDist, s.final.MaxRayDist)
	s.applyFog()
	*s.maxDisplayW = float32(parseSettingsInt(s.dispW, int(*s.maxDisplayW)))
	*s.maxDisplayH = float32(parseSettingsInt(s.dispH, int(*s.maxDisplayH)))
	s.resetPreviewImage()
	s.setCanvasSize()
	s.status.SetText("Render settings updated")
	s.startRender(false)
}

func (s *renderSettingsUI) applyPreviewSettings() {
	s.preview.Width = parseSettingsInt(s.prevW, s.preview.Width)
	s.preview.Height = parseSettingsInt(s.prevH, s.preview.Height)
	s.preview.SamplesPerPx = parseSettingsInt(s.prevSpp, s.preview.SamplesPerPx)
	s.preview.MaxDepth = parseSettingsInt(s.prevDepth, s.preview.MaxDepth)
}

func (s *renderSettingsUI) applyFinalSettings() {
	old := *s.final
	s.final.Width = parseSettingsInt(s.finalW, s.final.Width)
	s.final.Height = parseSettingsInt(s.finalH, s.final.Height)
	s.final.SamplesPerPx = parseSettingsInt(s.finalSpp, s.final.SamplesPerPx)
	s.final.MaxDepth = parseSettingsInt(s.finalDepth, s.final.MaxDepth)
	s.final.MaxRayDist = parseSettingsFloat(s.finalMaxRayDist, s.final.MaxRayDist)
	if old == *s.final {
		return
	}
	s.mu.Lock()
	*s.lastFinalImage = nil
	s.mu.Unlock()
}

func (s *renderSettingsUI) applyFog() {
	if !s.fogEnabled.Checked {
		s.sc.Fog = nil
		return
	}
	if s.sc.Fog == nil {
		s.sc.Fog = &scene.Fog{}
	}
	fog := s.sc.Fog
	fog.Density = parseSettingsFloat(s.fogDensity, 0)
	fog.Color = scene.Color{
		R: parseSettingsFloat(s.fogColorR, uiFogColorDefault),
		G: parseSettingsFloat(s.fogColorG, uiFogColorDefault),
		B: parseSettingsFloat(s.fogColorB, uiFogColorDefault),
	}
	fog.Scatter = clampSettings(parseSettingsFloat(s.fogScatter, uiUnitValue), 0, 1)
	fog.AffectSky = s.fogAffectSky.Checked
	fog.SigmaS = max(parseSettingsFloat(s.fogSigmaS, 0), 0)
	fog.SigmaA = max(parseSettingsFloat(s.fogSigmaA, 0), 0)
	fog.G = clampSettings(parseSettingsFloat(s.fogG, 0), -uiNineTenths, uiNineTenths)
	fog.HeteroStrength = clampSettings(parseSettingsFloat(s.fogHeteroStrength, 0), 0, 1)
	fog.NoiseScale = max(parseSettingsFloat(s.fogNoiseScale, uiFogNoiseScaleDefault), uiFloatEpsilon)
	fog.NoiseOctaves = min(max(parseSettingsInt(s.fogNoiseOctaves, uiFogNoiseOctavesDefault), 1), uiMaxSmoothRadius)
	fog.GPUVolumetric = s.fogGPUVolumetric.Checked
}

func parseSettingsInt(entry *widget.Entry, fallback int) int {
	value, err := strconv.Atoi(entry.Text)
	if err != nil || value <= 0 {
		return fallback
	}
	return value
}

func parseSettingsFloat(entry *widget.Entry, fallback float64) float64 {
	value, err := strconv.ParseFloat(entry.Text, 64)
	if err != nil || value <= 0 {
		return fallback
	}
	return value
}

func clampSettings(value, minValue, maxValue float64) float64 {
	return min(max(value, minValue), maxValue)
}

func (s *renderSettingsUI) resetPreviewImage() {
	s.mu.Lock()
	if *s.stopCh != nil {
		close(*s.stopCh)
		*s.stopCh = nil
	}
	*s.img = image.NewRGBA(image.Rect(0, 0, s.preview.Width, s.preview.Height))
	fillImage(*s.img, color.RGBA{0, 0, 0, 255})
	s.imgCanvas.Image = *s.img
	s.mu.Unlock()
}

func (s *renderSettingsUI) box() fyne.CanvasObject {
	apply := widget.NewButton("Apply render settings", s.apply)
	return container.NewVBox(
		widget.NewLabel("Render settings"),
		widget.NewLabel("Preview"),
		container.NewGridWithColumns(2,
			widget.NewLabel("Width"), s.prevW, widget.NewLabel("Height"), s.prevH,
			widget.NewLabel("Samples"), s.prevSpp, widget.NewLabel("Max reflections"), s.prevDepth,
			widget.NewLabel("Max ray distance"), s.prevMaxRayDist,
		),
		widget.NewLabel("Preview display (on screen)"),
		container.NewGridWithColumns(2, widget.NewLabel("Disp W"), s.dispW, widget.NewLabel("Disp H"), s.dispH),
		widget.NewLabel("Final"),
		container.NewGridWithColumns(2,
			widget.NewLabel("Width"), s.finalW, widget.NewLabel("Height"), s.finalH,
			widget.NewLabel("Samples"), s.finalSpp, widget.NewLabel("Max reflections"), s.finalDepth,
			widget.NewLabel("Max ray distance"), s.finalMaxRayDist,
		),
		widget.NewLabel("Fog (GPU)"), s.fogEnabled,
		container.NewGridWithColumns(2,
			widget.NewLabel("Density"), s.fogDensity, widget.NewLabel("Scatter"), s.fogScatter,
			widget.NewLabel("Color R"), s.fogColorR, widget.NewLabel("Color G"), s.fogColorG,
			widget.NewLabel("Color B"), s.fogColorB,
		),
		container.NewGridWithColumns(2,
			widget.NewLabel("Sigma S"), s.fogSigmaS, widget.NewLabel("Sigma A"), s.fogSigmaA,
			widget.NewLabel("g (anisotropy)"), s.fogG, widget.NewLabel("Hetero strength"), s.fogHeteroStrength,
			widget.NewLabel("Noise scale"), s.fogNoiseScale, widget.NewLabel("Noise octaves"), s.fogNoiseOctaves,
		),
		s.fogAffectSky, s.fogGPUVolumetric, apply,
	)
}

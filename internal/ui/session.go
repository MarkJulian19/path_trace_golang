package ui

import (
	"fmt"
	"image"
	"sync"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/widget"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	engine "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
	scenecase "github.com/MarkJulian19/path_trace_golang/internal/usecases/scene"
)

type uiSession struct {
	sc      *scene.Scene
	scenes  *scenecase.Scene
	service *engine.Service
	runtime *uiRuntime
	uiDo    func(func())
	mu      *sync.Mutex

	stopCh          chan struct{}
	renderTimer     *time.Timer
	lastFinalImage  image.Image
	lastFinalConfig engine.Config
	wireframeFPS    float64

	liveUpdate       *widget.Check
	camControlCheck  *widget.Check
	camControlActive bool
	denoiseEnabled   bool
	denoiseSigmaS    float64
	denoiseSigmaR    float64
	smoothEnabled    bool
	smoothRadius     int
	smoothStrength   float64

	controller      *renderController
	startRender     func(bool)
	clearFinalImage func()

	pickableCanvas   fyne.CanvasObject
	backendSlider    *widget.Slider
	backendLabel     *widget.Label
	cam              scene.Camera
	cameraEntries    [6]*widget.Entry
	cameraBox        fyne.CanvasObject
	videoBox         fyne.CanvasObject
	objectsBox       fyne.CanvasObject
	objectForm       fyne.CanvasObject
	materialsBox     fyne.CanvasObject
	materialForm     fyne.CanvasObject
	settingsBox      fyne.CanvasObject
	previewBtn       *widget.Button
	finalBtn         *widget.Button
	outputPath       *widget.Entry
	saveBtn          *widget.Button
	saveImageBtn     *widget.Button
	denoiseBox       fyne.CanvasObject
	smoothBox        fyne.CanvasObject
	wireframeLabel   *widget.Label
	wireframeSlider  *widget.Slider
	renderModeSelect *widget.RadioGroup
}

func newUISession(
	sc *scene.Scene,
	scenes *scenecase.Scene,
	scenePath string,
	service *engine.Service,
	runtime *uiRuntime,
	uiDo func(func()),
) *uiSession {
	s := &uiSession{
		sc:             sc,
		scenes:         scenes,
		service:        service,
		runtime:        runtime,
		uiDo:           uiDo,
		mu:             &runtime.mu,
		wireframeFPS:   uiWireframeFPSDefault,
		denoiseEnabled: true,
		denoiseSigmaS:  uiUnitValue,
		denoiseSigmaR:  uiDenoiseSigmaRDefault,
		smoothRadius:   2,
		smoothStrength: uiHalf,
	}
	s.initRender()
	s.initSceneControls(scenePath)
	return s
}

func (s *uiSession) initRender() {
	s.liveUpdate = widget.NewCheck("Live update while rendering", func(value bool) { s.service.SetLivePreview(value) })
	s.liveUpdate.SetChecked(true)
	s.service.SetLivePreview(true)
	s.camControlCheck = widget.NewCheck(
		"WASDQE camera control (preview)",
		func(value bool) { s.camControlActive = value },
	)
	s.controller = &renderController{
		sc: s.sc, service: s.service, preview: &s.runtime.preview, final: &s.runtime.final,
		img: &s.runtime.img, imgCanvas: s.runtime.imgCanvas, status: s.runtime.status,
		progressBar: s.runtime.progressBar, timeLabel: s.runtime.timeLabel, uiDo: s.uiDo,
		publish: s.runtime.publishPreview, mu: s.mu, stopCh: &s.stopCh, renderTimer: &s.renderTimer,
		lastImage: &s.lastFinalImage, lastConfig: &s.lastFinalConfig, wireframeFPS: &s.wireframeFPS,
		liveUpdate: s.liveUpdate, denoiseEnabled: &s.denoiseEnabled, denoiseSigmaS: &s.denoiseSigmaS,
		denoiseSigmaR: &s.denoiseSigmaR, smoothEnabled: &s.smoothEnabled, smoothRadius: &s.smoothRadius,
		smoothStrength: &s.smoothStrength,
	}
	s.startRender = s.controller.startRender
	s.clearFinalImage = s.controller.clearFinalImage
	s.pickableCanvas = newScenePicker(
		s.sc,
		&s.runtime.preview,
		s.runtime.imgCanvas,
		s.service,
		s.runtime.status,
		&s.startRender,
	)
	s.backendSlider, s.backendLabel = newBackendControls(
		s.service,
		s.startRender,
		&s.denoiseEnabled,
		&s.denoiseSigmaS,
		&s.denoiseSigmaR,
		&s.smoothEnabled,
		&s.smoothRadius,
		&s.smoothStrength,
	)
}

func (s *uiSession) initSceneControls(scenePath string) {
	s.initCameraAndVideo()
	s.initSceneObjects()
	s.initRenderSettings()
	s.initActions(scenePath)
	s.initDisplayControls()
}

func (s *uiSession) initCameraAndVideo() {
	s.cam = s.sc.Camera
	s.cameraEntries, s.cameraBox = newCameraUI(&s.cam, s.sc, s.runtime.status, s.clearFinalImage, s.startRender)
	s.videoBox = newVideoBox(
		s.sc, s.cam, &s.runtime.final, s.service, s.uiDo, s.runtime.status,
		s.runtime.progressBar, s.runtime.timeLabel, s.mu, &s.runtime.img, s.runtime.imgCanvas,
	)
}

func (s *uiSession) initSceneObjects() {
	materialIDs := make([]string, 0, len(s.sc.Materials))
	for _, material := range s.sc.Materials {
		materialIDs = append(materialIDs, material.ID)
	}
	materialUI := newMaterialUI(s.sc, s.runtime.status, s.clearFinalImage, s.startRender)
	s.materialsBox, s.materialForm = materialUI.boxes()
	objectUI := newObjectUI(s.sc, materialIDs, s.runtime.status, s.clearFinalImage, s.startRender)
	s.objectsBox, s.objectForm = objectUI.box()
}

func (s *uiSession) initRenderSettings() {
	settingsUI := newRenderSettingsUI(
		s.sc, &s.runtime.preview, &s.runtime.final, &s.runtime.maxDisplayW, &s.runtime.maxDisplayH,
		s.runtime.status, s.mu, &s.runtime.img, s.runtime.imgCanvas, &s.stopCh, &s.lastFinalImage,
		s.runtime.setCanvasSize, s.startRender,
	)
	s.settingsBox = settingsUI.box()
}

func (s *uiSession) initActions(scenePath string) {
	s.previewBtn = widget.NewButton("Preview render", func() { s.startRender(false) })
	s.finalBtn = widget.NewButton("Final render", func() { s.startRender(true) })
	s.outputPath = widget.NewEntry()
	s.outputPath.SetText("ui_render.png")
	s.saveBtn, s.saveImageBtn = newSaveButtons(
		s.scenes, s.sc, scenePath, s.runtime.status, s.outputPath, s.mu,
		&s.lastFinalImage, &s.lastFinalConfig, s.uiDo,
	)
	s.denoiseBox, s.smoothBox = newFilterBoxes(
		s.service, s.startRender, &s.denoiseEnabled, &s.denoiseSigmaS, &s.denoiseSigmaR,
		&s.smoothEnabled, &s.smoothRadius, &s.smoothStrength,
	)
}

func (s *uiSession) initDisplayControls() {
	s.wireframeLabel = widget.NewLabel("Wireframe FPS: 60")
	s.wireframeSlider = widget.NewSlider(10, uiWireframeFPSMax)
	s.wireframeSlider.Value = s.wireframeFPS
	s.wireframeSlider.Step = 1
	s.wireframeSlider.OnChanged = func(value float64) {
		s.mu.Lock()
		s.wireframeFPS = value
		s.mu.Unlock()
		s.wireframeLabel.SetText(fmtWireframeFPS(value))
	}
	s.renderModeSelect = newRenderModeSelect(s.service, s.mu, &s.stopCh, s.startRender)
}

func fmtWireframeFPS(value float64) string {
	return fmt.Sprintf("Wireframe FPS: %.0f", value)
}

func (s *uiSession) content() fyne.CanvasObject {
	left := container.NewVScroll(
		container.NewVBox(widget.NewLabel("Scene"), s.objectsBox, s.objectForm, s.materialsBox, s.materialForm),
	)
	left.SetMinSize(fyne.NewSize(uiPanelWidth, 0))
	right := container.NewVScroll(s.rightContent())
	right.SetMinSize(fyne.NewSize(uiValue300, 0))
	bottom := container.NewVBox(
		s.runtime.status,
		s.runtime.progressBar,
		s.runtime.timeLabel,
		widget.NewLabel("Timeline (coming soon)"),
	)
	center := container.NewHSplit(container.NewStack(s.pickableCanvas), right)
	center.SetOffset(uiRightPanelOffset)
	main := container.NewHSplit(left, center)
	main.SetOffset(uiLeftPanelOffset)
	content := container.NewVSplit(main, bottom)
	content.SetOffset(uiNineTenths)
	return content
}

func (s *uiSession) rightContent() *fyne.Container {
	return container.NewVBox(
		widget.NewLabel("Render Settings"),
		container.NewVBox(widget.NewLabel("Compute backend"), s.backendLabel, s.backendSlider),
		widget.NewLabel("Display Mode"), s.renderModeSelect, s.wireframeLabel, s.wireframeSlider,
		s.liveUpdate, s.camControlCheck, container.NewHBox(s.previewBtn, s.finalBtn), s.settingsBox,
		s.cameraBox, s.videoBox, s.denoiseBox, s.smoothBox,
		container.NewGridWithColumns(2, widget.NewLabel("Scene / Image path"), s.outputPath),
		container.NewHBox(s.saveBtn, s.saveImageBtn),
	)
}

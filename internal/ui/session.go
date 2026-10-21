package ui

import (
	"fmt"
	"image"
	"sync"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/widget"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	engine "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
	scenecase "github.com/MarkJulian19/path_trace_golang/internal/usecases/scene"
)

type uiSession struct {
	cameraEntries  [6]*widget.Entry
	lastFinalImage image.Image

	pickableCanvas fyne.CanvasObject

	cameraBox    fyne.CanvasObject
	videoBox     fyne.CanvasObject
	objectsBox   fyne.CanvasObject
	objectForm   fyne.CanvasObject
	materialsBox fyne.CanvasObject
	materialForm fyne.CanvasObject
	settingsBox  fyne.CanvasObject

	denoiseBox fyne.CanvasObject
	smoothBox  fyne.CanvasObject
	sc         *scene.Scene
	scenes     *scenecase.Scene
	service    *engine.Service
	runtime    *uiRuntime
	uiDo       func(func())
	mu         *sync.Mutex

	liveUpdate      *widget.Check
	camControlCheck *widget.Check

	controller             *renderController
	startRender            func(bool)
	startInteractiveRender func()
	clearFinalImage        func()
	gizmo                  *sceneGizmo
	leftPanel              fyne.CanvasObject
	rightPanel             fyne.CanvasObject
	bottomPanel            fyne.CanvasObject
	panelRestoreBar        *fyne.Container
	leftPanelRestore       fyne.CanvasObject
	rightPanelRestore      fyne.CanvasObject
	bottomPanelRestore     fyne.CanvasObject
	workspaceSplit         *container.Split
	centerSplit            *container.Split
	bodySplit              *container.Split

	backendSelect *widget.RadioGroup
	backendLabel  *widget.Label

	outputPath   *widget.Entry
	saveBtn      *widget.Button
	saveImageBtn *widget.Button

	wireframeLabel   *widget.Label
	wireframeSlider  *widget.Slider
	renderModeSelect *widget.RadioGroup
	objectUI         *objectUI
	inspectorTabs    *container.AppTabs
	transformSelect  *widget.RadioGroup
	objectMoveCheck  *widget.Check
	transformMode    editorTransformMode

	denoiseSigmaS float64
	denoiseSigmaR float64

	smoothRadius    int
	smoothStrength  float64
	cam             scene.Camera
	lastFinalConfig engine.Config
	wireframeFPS    float64
	selectedObject  int
	transformStep   float64

	camControlActive  bool
	denoiseEnabled    bool
	smoothEnabled     bool
	objectMoveEnabled bool
	controlsSyncing   bool
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
		selectedObject: -1,
		transformMode:  transformModeSelect,
		transformStep:  uiObjectMoveStep,
	}
	s.initRender()
	s.initSceneControls(scenePath)
	return s
}

func (s *uiSession) initRender() {
	s.initLiveUpdate()
	s.camControlCheck = widget.NewCheck(
		"Camera",
		s.setCameraControlEnabled,
	)
	s.initRenderController()
	s.initRenderWidgets()
}

func (s *uiSession) initRenderController() {
	s.controller = &renderController{
		sc: s.sc, service: s.service, preview: &s.runtime.preview, final: &s.runtime.final,
		img: &s.runtime.img, imgCanvas: s.runtime.imgCanvas, status: s.runtime.status,
		progressBar: s.runtime.progressBar, timeLabel: s.runtime.timeLabel, uiDo: s.uiDo,
		mu:        s.mu,
		lastImage: &s.lastFinalImage, lastConfig: &s.lastFinalConfig, wireframeFPS: &s.wireframeFPS,
		denoiseEnabled: &s.denoiseEnabled, denoiseSigmaS: &s.denoiseSigmaS,
		denoiseSigmaR: &s.denoiseSigmaR, smoothEnabled: &s.smoothEnabled, smoothRadius: &s.smoothRadius,
		smoothStrength: &s.smoothStrength,
	}
	s.controller.livePreview.Store(true)
	s.startRender = s.controller.startRender
	s.startInteractiveRender = s.controller.startInteractiveRender
	s.clearFinalImage = s.controller.clearFinalImage
}

func (s *uiSession) initRenderWidgets() {
	s.pickableCanvas, s.gizmo = newScenePicker(
		s.sc,
		&s.runtime.preview,
		s.runtime.imgCanvas,
		s.service,
		s.runtime.status,
		&s.startRender,
		s.startInteractiveRender,
		&s.selectedObject,
		&s.transformMode,
		&s.objectMoveEnabled,
		func(index int) { s.selectObject(index) },
		func() {
			s.objectUI.refreshSelected()
			s.clearFinalImage()
			s.gizmo.refresh()
		},
	)
	s.backendSelect, s.backendLabel = newBackendControls(
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

func (s *uiSession) initLiveUpdate() {
	s.liveUpdate = widget.NewCheck("Live update while rendering", func(value bool) {
		s.service.SetLivePreview(value)
		if s.controller != nil {
			s.controller.livePreview.Store(value)
		}
	})
	s.liveUpdate.SetChecked(true)
	s.service.SetLivePreview(true)
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
	s.objectUI = newObjectUI(
		s.sc,
		materialIDs,
		s.runtime.status,
		s.clearFinalImage,
		s.startRender,
		func(index int) { s.selectObject(index) },
	)
	s.objectsBox, s.objectForm = s.objectUI.box()
}

func (s *uiSession) selectObject(index int) {
	if index < 0 || index >= len(s.sc.Objects) {
		s.selectedObject = -1
		setSelectedGPUObject(s.service, -1)
		if s.objectUI != nil {
			s.objectUI.setSelected(-1)
		}
		s.refreshGizmo()
		s.runtime.status.SetText("No object selected")
		return
	}
	s.selectedObject = index
	setSelectedGPUObject(s.service, index)
	if s.objectUI != nil {
		s.objectUI.setSelected(index)
	}
	s.refreshGizmo()
	if s.inspectorTabs != nil {
		s.inspectorTabs.SelectIndex(0)
	}
	s.runtime.status.SetText("Selected object: " + s.sc.Objects[index].ID)
}

func (s *uiSession) refreshGizmo() {
	if s.gizmo != nil {
		s.gizmo.refresh()
	}
}

func (s *uiSession) initRenderSettings() {
	settingsUI := newRenderSettingsUI(
		s.sc, &s.runtime.preview, &s.runtime.final, &s.runtime.maxDisplayW, &s.runtime.maxDisplayH,
		s.runtime.status, s.mu, &s.runtime.img, s.runtime.imgCanvas, &s.lastFinalImage,
		s.runtime.setCanvasSize, s.refreshGizmo, s.startRender,
	)
	s.settingsBox = settingsUI.box()
}

func (s *uiSession) initActions(scenePath string) {
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
	s.renderModeSelect = newRenderModeSelect(s.service, s.startRender)
	s.transformMode = transformModeSelect
}

func fmtWireframeFPS(value float64) string {
	return fmt.Sprintf("Wireframe FPS: %.0f", value)
}

func (s *uiSession) content() fyne.CanvasObject {
	s.leftPanel = s.outliner()
	s.rightPanel = s.inspector()
	s.bottomPanel = s.statusBar()
	viewport := s.viewport()
	s.centerSplit = container.NewHSplit(viewport, s.rightPanel)
	s.centerSplit.SetOffset(uiRightPanelOffset)
	s.workspaceSplit = container.NewHSplit(s.leftPanel, s.centerSplit)
	s.workspaceSplit.SetOffset(uiLeftPanelOffset)
	s.bodySplit = container.NewVSplit(s.workspaceSplit, s.bottomPanel)
	s.bodySplit.SetOffset(uiEditorBodyOffset)
	return container.NewBorder(s.topBar(), nil, nil, nil, s.bodySplit)
}

package ui

import (
	"strconv"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/widget"
)

func (s *uiSession) topBar() fyne.CanvasObject {
	return container.NewHBox(s.transformToolbar(), s.panelRestoreToolbar())
}

func (s *uiSession) mainMenu(closeWindow func()) *fyne.MainMenu {
	return fyne.NewMainMenu(
		s.fileMenu(closeWindow),
		s.renderMenu(),
		s.viewMenu(),
		s.objectMenu(),
		s.panelsMenu(),
		s.helpMenu(),
	)
}

func (s *uiSession) fileMenu(closeWindow func()) *fyne.Menu {
	return fyne.NewMenu("File",
		fyne.NewMenuItem("Save scene", func() { s.runButton(s.saveBtn) }),
		fyne.NewMenuItem("Save image (PNG)", func() { s.runButton(s.saveImageBtn) }),
		fyne.NewMenuItemSeparator(),
		fyne.NewMenuItem("Close", closeWindow),
	)
}

func (s *uiSession) renderMenu() *fyne.Menu {
	return fyne.NewMenu("Render",
		fyne.NewMenuItem("Preview render", func() { s.startRender(false) }),
		fyne.NewMenuItem("Final render", func() { s.startRender(true) }),
		fyne.NewMenuItemSeparator(),
		fyne.NewMenuItem("Normal mode", func() { s.renderModeSelect.SetSelected("Normal") }),
		fyne.NewMenuItem("Wireframe mode", func() { s.renderModeSelect.SetSelected("Wireframe") }),
	)
}

func (s *uiSession) viewMenu() *fyne.Menu {
	return fyne.NewMenu("View",
		fyne.NewMenuItem("Camera movement", func() {
			s.camControlCheck.SetChecked(!s.camControlCheck.Checked)
		}),
		fyne.NewMenuItem("Reset layout", s.resetLayout),
	)
}

func (s *uiSession) objectMenu() *fyne.Menu {
	return fyne.NewMenu("Object",
		fyne.NewMenuItem("Select mode", func() { s.transformSelect.SetSelected(string(transformModeSelect)) }),
		fyne.NewMenuItem("Move mode", func() { s.transformSelect.SetSelected(string(transformModeMove)) }),
		fyne.NewMenuItem("Object movement", func() {
			s.objectMoveCheck.SetChecked(!s.objectMoveCheck.Checked)
		}),
		fyne.NewMenuItem("Clear selection", s.clearSelection),
	)
}

func (s *uiSession) panelsMenu() *fyne.Menu {
	return fyne.NewMenu("Panels",
		fyne.NewMenuItem("Show left panel", func() { s.setPanelVisible(s.leftPanel, true) }),
		fyne.NewMenuItem("Show right panel", func() { s.setPanelVisible(s.rightPanel, true) }),
		fyne.NewMenuItem("Show bottom panel", func() { s.setPanelVisible(s.bottomPanel, true) }),
	)
}

func (s *uiSession) helpMenu() *fyne.Menu {
	return fyne.NewMenu("Help",
		fyne.NewMenuItem("Controls", func() {
			s.runtime.status.SetText("Camera: WASDQE + arrows | Object: Move mode + arrows or gizmo")
		}),
		fyne.NewMenuItem("About", func() {
			s.runtime.status.SetText("Go Path Tracer scene editor")
		}),
	)
}

func (s *uiSession) runButton(button *widget.Button) {
	if button != nil && button.OnTapped != nil {
		button.OnTapped()
	}
}

func (s *uiSession) resetLayout() {
	s.centerSplit.SetOffset(uiRightPanelOffset)
	s.workspaceSplit.SetOffset(uiLeftPanelOffset)
	s.bodySplit.SetOffset(uiEditorBodyOffset)
	s.setPanelVisible(s.leftPanel, true)
	s.setPanelVisible(s.rightPanel, true)
	s.setPanelVisible(s.bottomPanel, true)
}

func (s *uiSession) panelRestoreToolbar() fyne.CanvasObject {
	s.leftPanelRestore = newCompactButton("›", func() {
		s.setPanelVisible(s.leftPanel, true)
	})
	s.rightPanelRestore = newCompactButton("‹", func() {
		s.setPanelVisible(s.rightPanel, true)
	})
	s.bottomPanelRestore = newCompactButton("⌃", func() {
		s.setPanelVisible(s.bottomPanel, true)
	})
	s.panelRestoreBar = container.NewHBox(
		s.leftPanelRestore,
		s.rightPanelRestore,
		s.bottomPanelRestore,
	)
	s.updatePanelRestoreButtons()
	return s.panelRestoreBar
}

func newCompactButton(symbol string, tapped func()) fyne.CanvasObject {
	button := widget.NewButton(symbol, tapped)
	button.Importance = widget.LowImportance
	return container.NewGridWrap(fyne.NewSize(uiCompactButtonSize, uiCompactButtonSize), button)
}

func (s *uiSession) setPanelVisible(panel fyne.CanvasObject, visible bool) {
	if visible {
		panel.Show()
	} else {
		panel.Hide()
	}
	s.refreshPanelLayout()
	s.updatePanelRestoreButtons()
}

func (s *uiSession) updatePanelRestoreButtons() {
	if s.panelRestoreBar == nil {
		return
	}
	if s.leftPanel.Visible() {
		s.leftPanelRestore.Hide()
	} else {
		s.leftPanelRestore.Show()
	}
	if s.rightPanel.Visible() {
		s.rightPanelRestore.Hide()
	} else {
		s.rightPanelRestore.Show()
	}
	if s.bottomPanel.Visible() {
		s.bottomPanelRestore.Hide()
	} else {
		s.bottomPanelRestore.Show()
	}
	s.panelRestoreBar.Refresh()
}

func (s *uiSession) refreshPanelLayout() {
	if s.workspaceSplit != nil {
		s.workspaceSplit.Refresh()
	}
	if s.centerSplit != nil {
		s.centerSplit.Refresh()
	}
	if s.bodySplit != nil {
		s.bodySplit.Refresh()
	}
	s.refreshGizmo()
}

func (s *uiSession) panelHeader(title string, hide func()) fyne.CanvasObject {
	return container.NewBorder(
		widget.NewLabel(title), nil, nil, newCompactButton("×", hide),
	)
}

func (s *uiSession) transformToolbar() fyne.CanvasObject {
	transformSelect := widget.NewRadioGroup(
		[]string{string(transformModeSelect), string(transformModeMove)},
		func(value string) {
			s.setTransformMode(editorTransformMode(value))
			if s.transformMode == transformModeMove {
				s.runtime.status.SetText("Move mode: arrows move the selected object")
				return
			}
			s.runtime.status.SetText("Select mode: click an object in the viewport")
		},
	)
	s.transformSelect = transformSelect
	transformSelect.Horizontal = true
	transformSelect.Required = true
	transformSelect.SetSelected(string(s.transformMode))

	stepEntry := widget.NewEntry()
	stepEntry.SetText(strconv.FormatFloat(s.transformStep, 'f', 2, 64))
	stepEntry.OnChanged = func(value string) {
		step, err := strconv.ParseFloat(value, 64)
		if err == nil && step > 0 {
			s.transformStep = step
		}
	}
	stepBox := container.NewGridWrap(fyne.NewSize(uiTransformStepWidth, uiTransformStepHeight), stepEntry)
	s.objectMoveCheck = widget.NewCheck("Object move", s.setObjectMoveEnabled)
	s.objectMoveCheck.SetChecked(s.objectMoveEnabled)
	clearSelection := newCompactButton("×", s.clearSelection)
	transform := container.NewHBox(
		widget.NewLabel("Transform"), transformSelect,
		widget.NewLabel("Step"), stepBox,
		s.objectMoveCheck, s.camControlCheck, clearSelection,
	)
	return transform
}

func (s *uiSession) setTransformMode(mode editorTransformMode) {
	s.transformMode = mode
	if s.controlsSyncing {
		return
	}
	s.setObjectMoveEnabled(mode == transformModeMove)
	if s.gizmo != nil {
		s.gizmo.refresh()
	}
}

func (s *uiSession) setObjectMoveEnabled(enabled bool) {
	s.objectMoveEnabled = enabled
	if s.controlsSyncing {
		return
	}
	if enabled && s.camControlActive {
		s.controlsSyncing = true
		s.camControlCheck.SetChecked(false)
		s.controlsSyncing = false
		s.camControlActive = false
	}
	if enabled && s.transformMode != transformModeMove && s.transformSelect != nil {
		s.controlsSyncing = true
		s.transformSelect.SetSelected(string(transformModeMove))
		s.controlsSyncing = false
		s.transformMode = transformModeMove
	}
	if s.objectMoveCheck != nil && s.objectMoveCheck.Checked != enabled {
		s.controlsSyncing = true
		s.objectMoveCheck.SetChecked(enabled)
		s.controlsSyncing = false
	}
	if s.gizmo != nil {
		s.gizmo.refresh()
	}
}

func (s *uiSession) setCameraControlEnabled(enabled bool) {
	s.camControlActive = enabled
	if s.controlsSyncing {
		return
	}
	if enabled && s.objectMoveEnabled {
		s.setObjectMoveEnabled(false)
	}
	if enabled {
		s.runtime.status.SetText("Camera movement enabled: WASDQE and arrows")
	} else {
		s.runtime.status.SetText("Camera movement disabled")
	}
}

func (s *uiSession) clearSelection() {
	s.selectObject(-1)
}

func (s *uiSession) outliner() fyne.CanvasObject {
	sections := widget.NewAccordion(
		widget.NewAccordionItem("Objects", s.objectsBox),
		widget.NewAccordionItem("Materials", s.materialsBox),
		widget.NewAccordionItem("Assets", futureAssetsPanel()),
	)
	sections.Open(0)
	view := container.NewVScroll(sections)
	view.SetMinSize(fyne.NewSize(uiPanelWidth, 0))
	content := container.NewBorder(s.panelHeader("OUTLINER", s.hideLeftPanel), nil, nil, nil, view)
	return newPanelHost(content, fyne.NewSize(uiPanelWidth, uiViewportMinHeight))
}

func (s *uiSession) viewport() fyne.CanvasObject {
	header := container.NewHBox(
		widget.NewLabel("VIEWPORT"),
		widget.NewLabel("  •  Click to select  •  Move mode: axis handles / drag / arrow keys"),
	)
	return container.NewBorder(header, nil, nil, nil, container.NewStack(s.pickableCanvas))
}

func (s *uiSession) inspector() fyne.CanvasObject {
	tabs := container.NewAppTabs(
		container.NewTabItem("Object", container.NewVScroll(s.objectForm)),
		container.NewTabItem("Material", container.NewVScroll(s.materialForm)),
		container.NewTabItem("Render", container.NewVScroll(s.renderInspector())),
		container.NewTabItem("Camera", container.NewVScroll(s.cameraInspector())),
		container.NewTabItem("Animation", container.NewVScroll(s.videoBox)),
		container.NewTabItem("Scene", container.NewVScroll(s.sceneInspector())),
	)
	tabs.SetTabLocation(container.TabLocationTop)
	s.inspectorTabs = tabs
	content := container.NewBorder(s.panelHeader("INSPECTOR", s.hideRightPanel), nil, nil, nil, tabs)
	return newPanelHost(content, fyne.NewSize(uiInspectorMinWidth, uiViewportMinHeight))
}

func (s *uiSession) renderInspector() fyne.CanvasObject {
	sections := widget.NewAccordion(
		widget.NewAccordionItem(
			"Backend",
			container.NewVBox(widget.NewLabel("Compute backend"), s.backendLabel, s.backendSelect),
		),
		widget.NewAccordionItem(
			"Display",
			container.NewVBox(
				widget.NewLabel("Render mode"), s.renderModeSelect,
				s.wireframeLabel, s.wireframeSlider, s.liveUpdate,
			),
		),
		widget.NewAccordionItem("Quality", s.settingsBox),
		widget.NewAccordionItem(
			"Post processing",
			container.NewVBox(s.denoiseBox, s.smoothBox),
		),
	)
	sections.Open(0)
	return sections
}

func (s *uiSession) cameraInspector() fyne.CanvasObject {
	sections := widget.NewAccordion(
		widget.NewAccordionItem("Camera", s.cameraBox),
		widget.NewAccordionItem(
			"Navigation",
			container.NewVBox(
				widget.NewLabel("WASDQE moves the camera; arrows rotate it"),
				widget.NewLabel("Use the Camera toggle in the top toolbar."),
			),
		),
	)
	sections.Open(0)
	return sections
}

func (s *uiSession) sceneInspector() fyne.CanvasObject {
	return container.NewVBox(
		widget.NewCard(
			"Scene file",
			"Save the editable JSON scene",
			container.NewVBox(
				widget.NewLabel("Scene / image path"), s.outputPath,
				container.NewHBox(s.saveBtn, s.saveImageBtn),
			),
		),
		widget.NewCard(
			"Workflow",
			"The editor is ready for collections, mesh assets and texture maps",
			futureWorkflowPanel(),
		),
	)
}

func (s *uiSession) statusBar() fyne.CanvasObject {
	content := container.NewBorder(
		s.panelHeader("STATUS", s.hideBottomPanel), nil, nil, nil,
		container.NewVBox(
			s.runtime.status,
			s.runtime.progressBar,
			s.runtime.timeLabel,
			widget.NewLabel("Timeline / keyframes: reserved"),
		),
	)
	return newPanelHost(content, fyne.NewSize(uiViewportMinWidth, uiStatusMinHeight))
}

// panelHost ограничивает минимальный размер панели, сохраняя прокрутку её содержимого.
type panelHost struct {
	widget.BaseWidget

	content fyne.CanvasObject
	minimum fyne.Size
}

func newPanelHost(content fyne.CanvasObject, minimum fyne.Size) *panelHost {
	host := &panelHost{content: content, minimum: minimum}
	host.ExtendBaseWidget(host)
	return host
}

// CreateRenderer размещает содержимое панели в доступной области.
func (p *panelHost) CreateRenderer() fyne.WidgetRenderer {
	return &panelHostRenderer{host: p}
}

type panelHostRenderer struct {
	host *panelHost
}

// Layout растягивает содержимое панели вместе с её контейнером.
func (r *panelHostRenderer) Layout(size fyne.Size) {
	r.host.content.Move(fyne.NewPos(0, 0))
	r.host.content.Resize(size)
}

// MinSize возвращает небольшой размер, достаточный для перетаскивания разделителя.
func (r *panelHostRenderer) MinSize() fyne.Size {
	return r.host.minimum
}

// Objects возвращает содержимое панели для отрисовки.
func (r *panelHostRenderer) Objects() []fyne.CanvasObject {
	return []fyne.CanvasObject{r.host.content}
}

// Refresh обновляет содержимое панели.
func (r *panelHostRenderer) Refresh() {
	r.host.content.Refresh()
}

// Destroy освобождает ресурсы отрисовщика панели.
func (r *panelHostRenderer) Destroy() {}

func (s *uiSession) hideLeftPanel() {
	if s.leftPanel.Visible() {
		s.setPanelVisible(s.leftPanel, false)
	}
}

func (s *uiSession) hideRightPanel() {
	if s.rightPanel.Visible() {
		s.setPanelVisible(s.rightPanel, false)
	}
}

func (s *uiSession) hideBottomPanel() {
	if s.bottomPanel.Visible() {
		s.setPanelVisible(s.bottomPanel, false)
	}
}

func futureAssetsPanel() fyne.CanvasObject {
	mesh := widget.NewButton("Mesh import", nil)
	mesh.Disable()
	textures := widget.NewButton("Texture slots", nil)
	textures.Disable()
	return container.NewVBox(
		widget.NewLabel("Asset boundaries are kept separate from primitive objects."),
		mesh,
		textures,
	)
}

func futureWorkflowPanel() fyne.CanvasObject {
	return container.NewVBox(
		widget.NewLabel("Planned editor layers:"),
		widget.NewLabel("• mesh geometry and normals"),
		widget.NewLabel("• UV coordinates and texture maps"),
		widget.NewLabel("• collections, gizmos and keyframes"),
	)
}

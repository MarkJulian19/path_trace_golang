package ui

import (
	"fmt"
	"strconv"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/widget"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

type materialUI struct {
	albedo [3]*widget.Entry
	emit   [3]*widget.Entry

	tint            [3]*widget.Entry
	smoothness      [2]*widget.Entry
	reflectivity    [2]*widget.Entry
	sc              *scene.Scene
	status          *widget.Label
	clearFinalImage func()
	startRender     func(bool)

	list       *widget.List
	typeSelect *widget.Select

	power *widget.Entry
	rough *widget.Entry
	ior   *widget.Entry

	absorption *widget.Entry
	selected   int
}

func newMaterialUI(sc *scene.Scene, status *widget.Label, clearFinalImage func(), startRender func(bool)) *materialUI {
	m := &materialUI{sc: sc, status: status, clearFinalImage: clearFinalImage, startRender: startRender, selected: -1}
	m.typeSelect = widget.NewSelect(
		[]string{
			string(scene.MaterialLambert),
			string(scene.MaterialMetal),
			string(scene.MaterialDielectric),
			string(scene.MaterialEmissive),
		},
		nil,
	)
	for i := range m.albedo {
		m.albedo[i] = widget.NewEntry()
		m.emit[i] = widget.NewEntry()
		m.tint[i] = widget.NewEntry()
	}
	m.power, m.rough, m.ior = widget.NewEntry(), widget.NewEntry(), widget.NewEntry()
	m.smoothness = [2]*widget.Entry{widget.NewEntry(), widget.NewEntry()}
	m.reflectivity = m.smoothness
	m.absorption = widget.NewEntry()
	m.list = widget.NewList(
		func() int { return len(m.sc.Materials) },
		func() fyne.CanvasObject { return widget.NewLabel("") },
		m.updateListItem,
	)
	m.list.OnSelected = func(id widget.ListItemID) { m.selected = id; m.load(id) }
	m.typeSelect.OnChanged = func(string) { m.updateVisibility() }
	m.setEnabled(false)
	return m
}

func (m *materialUI) updateListItem(id widget.ListItemID, object fyne.CanvasObject) {
	if id < 0 || id >= len(m.sc.Materials) {
		return
	}
	label, ok := object.(*widget.Label)
	if !ok {
		return
	}
	material := m.sc.Materials[id]
	label.SetText(fmt.Sprintf("%s (%s)", material.ID, material.Type))
}

func (m *materialUI) setEnabled(enabled bool) {
	entries := []*widget.Entry{
		m.albedo[0],
		m.albedo[1],
		m.albedo[2],
		m.emit[0],
		m.emit[1],
		m.emit[2],
		m.power,
		m.rough,
		m.ior,
		m.smoothness[0],
		m.smoothness[1],
		m.tint[0],
		m.tint[1],
		m.tint[2],
		m.absorption,
	}
	for _, entry := range entries {
		if enabled {
			entry.Enable()
		} else {
			entry.Disable()
		}
	}
	if enabled {
		m.typeSelect.Enable()
	} else {
		m.typeSelect.Disable()
	}
}

func (m *materialUI) updateVisibility() {
	if m.selected < 0 || m.selected >= len(m.sc.Materials) {
		return
	}
	material := m.sc.Materials[m.selected]
	metal := material.Type == scene.MaterialMetal || material.Type == scene.MaterialMirror
	dielectric := material.Type == scene.MaterialDielectric
	setEntryVisibility(m.smoothness[0], metal)
	setEntryVisibility(m.smoothness[1], metal)
	for _, entry := range m.tint {
		setEntryVisibility(entry, dielectric)
	}
	setEntryVisibility(m.absorption, dielectric)
}

func setEntryVisibility(entry *widget.Entry, visible bool) {
	if visible {
		entry.Show()
	} else {
		entry.Hide()
	}
}

func (m *materialUI) load(index int) {
	if index < 0 || index >= len(m.sc.Materials) {
		m.setEnabled(false)
		return
	}
	material := m.sc.Materials[index]
	m.setEnabled(true)
	m.typeSelect.SetSelected(string(material.Type))
	setMaterialEntries(m, material)
	m.updateVisibility()
}

func setMaterialEntries(m *materialUI, material scene.Material) {
	setColorEntries(m.albedo, material.Albedo)
	setColorEntries(m.emit, material.Emit)
	setMaterialBaseEntries(m, material)
	setMaterialSurfaceEntries(m, material)
	setMaterialTintEntries(m, material)
}

func setMaterialBaseEntries(m *materialUI, material scene.Material) {
	m.power.SetText(fmt.Sprintf("%.2f", material.Power))
	m.rough.SetText(fmt.Sprintf("%.2f", material.Rough))
	m.ior.SetText(fmt.Sprintf("%.2f", material.IOR))
}

func setMaterialSurfaceEntries(m *materialUI, material scene.Material) {
	smoothness := material.Smoothness
	if smoothness == 0 && (material.Type == scene.MaterialMetal || material.Type == scene.MaterialMirror) {
		smoothness = uiUnitValue - material.Rough
	}
	reflectivity := material.Reflectivity
	if reflectivity == 0 && (material.Type == scene.MaterialMetal || material.Type == scene.MaterialMirror) {
		reflectivity = uiUnitValue
	}
	m.smoothness[0].SetText(fmt.Sprintf("%.2f", smoothness))
	m.smoothness[1].SetText(fmt.Sprintf("%.2f", reflectivity))
}

func setMaterialTintEntries(m *materialUI, material scene.Material) {
	tintR, tintG, tintB := material.Tint.R, material.Tint.G, material.Tint.B
	if tintR == 0 && tintG == 0 && tintB == 0 && material.Type == scene.MaterialDielectric {
		tintR, tintG, tintB = uiUnitValue, uiUnitValue, uiUnitValue
	}
	setEntryValues(m.tint, tintR, tintG, tintB)
	absorption := material.AbsorptionScale
	if absorption == 0 && material.Type == scene.MaterialDielectric {
		absorption = uiAbsorptionScaleDefault
	}
	m.absorption.SetText(fmt.Sprintf("%.4f", absorption))
}

func setColorEntries(entries [3]*widget.Entry, value scene.Color) {
	setEntryValues(entries, value.R, value.G, value.B)
}

func setEntryValues(entries [3]*widget.Entry, values ...float64) {
	for i, value := range values {
		entries[i].SetText(fmt.Sprintf("%.2f", value))
	}
}

func (m *materialUI) apply() {
	if m.selected < 0 || m.selected >= len(m.sc.Materials) {
		m.status.SetText("No material selected")
		return
	}
	material := m.sc.Materials[m.selected]
	if m.typeSelect.Selected != "" {
		material.Type = scene.MaterialType(m.typeSelect.Selected)
	}
	material.Albedo = parseColor(m.albedo, material.Albedo)
	material.Emit = parseColor(m.emit, material.Emit)
	material.Power = parseEntry(m.power, material.Power)
	material.Rough = parseEntry(m.rough, material.Rough)
	material.IOR = parseEntry(m.ior, material.IOR)
	if material.Type == scene.MaterialMetal || material.Type == scene.MaterialMirror {
		applyMetal(&material, m.smoothness)
	}
	if material.Type == scene.MaterialDielectric {
		material.Tint = parseColor(m.tint, material.Tint)
	}
	m.sc.Materials[m.selected] = material
	m.list.Refresh()
	m.clearFinalImage()
	m.status.SetText("Material updated")
	m.startRender(false)
}

func applyMetal(material *scene.Material, entries [2]*widget.Entry) {
	material.Smoothness = clampUI(parseEntry(entries[0], material.Smoothness))
	material.Reflectivity = clampUI(parseEntry(entries[1], material.Reflectivity))
}

func clampUI(value float64) float64 { return min(max(value, 0), 1) }

func parseEntry(entry *widget.Entry, fallback float64) float64 {
	value, err := strconv.ParseFloat(entry.Text, 64)
	if err != nil {
		return fallback
	}
	return value
}

func parseColor(entries [3]*widget.Entry, fallback scene.Color) scene.Color {
	return scene.Color{
		R: parseEntry(entries[0], fallback.R),
		G: parseEntry(entries[1], fallback.G),
		B: parseEntry(entries[2], fallback.B),
	}
}

func (m *materialUI) boxes() (fyne.CanvasObject, fyne.CanvasObject) {
	listScroll := container.NewScroll(m.list)
	listScroll.SetMinSize(fyne.NewSize(0, uiPanelHeight))
	listBox := container.NewBorder(widget.NewLabel("Materials"), nil, nil, nil, listScroll)
	return listBox, m.form()
}

func (m *materialUI) form() fyne.CanvasObject {
	form := container.NewVBox(
		widget.NewLabel("Selected material"),
		m.fields(),
		widget.NewButton("Apply material", m.apply),
	)
	return form
}

func (m *materialUI) fields() fyne.CanvasObject {
	return container.NewGridWithColumns(2, m.fieldObjects()...)
}

func (m *materialUI) fieldObjects() []fyne.CanvasObject {
	return append(m.baseFieldObjects(), m.surfaceFieldObjects()...)
}

func (m *materialUI) baseFieldObjects() []fyne.CanvasObject {
	return []fyne.CanvasObject{
		widget.NewLabel("Type"),
		m.typeSelect,
		widget.NewLabel(
			"Albedo R",
		),
		m.albedo[0],
		widget.NewLabel("Albedo G"),
		m.albedo[1],
		widget.NewLabel("Albedo B"),
		m.albedo[2],
		widget.NewLabel(
			"Emit R",
		),
		m.emit[0],
		widget.NewLabel("Emit G"),
		m.emit[1],
		widget.NewLabel("Emit B"),
		m.emit[2],
		widget.NewLabel("Power"),
		m.power,
		widget.NewLabel("Rough"),
		m.rough,
		widget.NewLabel("IOR"),
		m.ior,
	}
}

func (m *materialUI) surfaceFieldObjects() []fyne.CanvasObject {
	return []fyne.CanvasObject{
		widget.NewLabel(
			"Smoothness (Metal)",
		),
		m.smoothness[0],
		widget.NewLabel("Reflectivity (Metal)"),
		m.smoothness[1],
		widget.NewLabel(
			"Tint R (Glass)",
		),
		m.tint[0],
		widget.NewLabel("Tint G (Glass)"),
		m.tint[1],
		widget.NewLabel("Tint B (Glass)"),
		m.tint[2],
		widget.NewLabel("Absorption Scale (Glass)"),
		m.absorption,
	}
}

type objectUI struct {
	position [3]*widget.Entry
	size     [3]*widget.Entry
	sc       *scene.Scene

	status          *widget.Label
	clearFinalImage func()
	startRender     func(bool)

	list           *widget.List
	typeSelect     *widget.Select
	materialSelect *widget.Select
	materialIDs    []string

	selected int
}

func newObjectUI(
	sc *scene.Scene,
	materialIDs []string,
	status *widget.Label,
	clearFinalImage func(),
	startRender func(bool),
) *objectUI {
	o := &objectUI{
		sc:              sc,
		materialIDs:     materialIDs,
		status:          status,
		clearFinalImage: clearFinalImage,
		startRender:     startRender,
		selected:        -1,
	}
	o.typeSelect = widget.NewSelect(
		[]string{string(scene.ObjectSphere), string(scene.ObjectPlane), string(scene.ObjectBox)},
		nil,
	)
	o.materialSelect = widget.NewSelect(materialIDs, nil)
	for i := range o.position {
		o.position[i] = widget.NewEntry()
		o.size[i] = widget.NewEntry()
	}
	o.list = widget.NewList(
		func() int { return len(o.sc.Objects) },
		func() fyne.CanvasObject { return widget.NewLabel("") },
		o.updateListItem,
	)
	o.list.OnSelected = func(id widget.ListItemID) { o.selected = id; o.load(id) }
	o.setEnabled(false)
	return o
}

func (o *objectUI) updateListItem(id widget.ListItemID, object fyne.CanvasObject) {
	if id < 0 || id >= len(o.sc.Objects) {
		return
	}
	label, ok := object.(*widget.Label)
	if !ok {
		return
	}
	objectValue := o.sc.Objects[id]
	label.SetText(fmt.Sprintf("%s (%s)", objectValue.ID, objectValue.Type))
}

func (o *objectUI) setEnabled(enabled bool) {
	entries := append(o.position[:], o.size[:]...)
	for _, entry := range entries {
		if enabled {
			entry.Enable()
		} else {
			entry.Disable()
		}
	}
	if enabled {
		o.typeSelect.Enable()
		o.materialSelect.Enable()
	} else {
		o.typeSelect.Disable()
		o.materialSelect.Disable()
	}
}

func (o *objectUI) load(index int) {
	if index < 0 || index >= len(o.sc.Objects) {
		o.setEnabled(false)
		return
	}
	value := o.sc.Objects[index]
	o.setEnabled(true)
	o.typeSelect.SetSelected(string(value.Type))
	o.materialSelect.SetSelected(value.MaterialID)
	setVecEntries(o.position, value.Position)
	setVecEntries(o.size, value.Size)
}

func setVecEntries(entries [3]*widget.Entry, value scene.Vec3) {
	setEntryValues(entries, value.X, value.Y, value.Z)
}

func (o *objectUI) apply() {
	if o.selected < 0 || o.selected >= len(o.sc.Objects) {
		o.status.SetText("No object selected")
		return
	}
	value := o.sc.Objects[o.selected]
	if o.typeSelect.Selected != "" {
		value.Type = scene.ObjectType(o.typeSelect.Selected)
	}
	if o.materialSelect.Selected != "" {
		value.MaterialID = o.materialSelect.Selected
	}
	value.Position = parseVec(o.position, value.Position)
	value.Size = parseVec(o.size, value.Size)
	o.sc.Objects[o.selected] = value
	o.list.Refresh()
	o.clearFinalImage()
	o.status.SetText("Object updated")
	o.startRender(false)
}

func parseVec(entries [3]*widget.Entry, fallback scene.Vec3) scene.Vec3 {
	return scene.Vec3{
		X: parseEntry(entries[0], fallback.X),
		Y: parseEntry(entries[1], fallback.Y),
		Z: parseEntry(entries[2], fallback.Z),
	}
}

func (o *objectUI) add(objectType scene.ObjectType) {
	value := scene.Object{
		ID:       fmt.Sprintf("%s-%d", objectType, len(o.sc.Objects)+1),
		Type:     objectType,
		Position: scene.Vec3{Y: uiHalf},
		Size:     scene.Vec3{X: 1, Y: 1, Z: 1},
	}
	if objectType == scene.ObjectSphere {
		value.Size = scene.Vec3{X: 1}
		value.Position.Y = 1
	}
	if len(o.materialIDs) > 0 {
		value.MaterialID = o.materialIDs[0]
	}
	o.sc.Objects = append(o.sc.Objects, value)
	o.list.Refresh()
	o.selected = len(o.sc.Objects) - 1
	o.list.Select(o.selected)
	o.clearFinalImage()
	o.status.SetText(fmt.Sprintf("%s added", objectType))
	o.startRender(false)
}

func (o *objectUI) remove() {
	if o.selected < 0 || o.selected >= len(o.sc.Objects) {
		o.status.SetText("No object selected")
		return
	}
	o.sc.Objects = append(o.sc.Objects[:o.selected], o.sc.Objects[o.selected+1:]...)
	o.selected = -1
	o.list.Refresh()
	o.clearFinalImage()
	o.status.SetText("Object removed")
	o.startRender(false)
}

func (o *objectUI) box() (fyne.CanvasObject, fyne.CanvasObject) {
	apply := widget.NewButton("Apply object", o.apply)
	addSphere := widget.NewButton("Add sphere", func() { o.add(scene.ObjectSphere) })
	addBox := widget.NewButton("Add box", func() { o.add(scene.ObjectBox) })
	remove := widget.NewButton("Remove selected", o.remove)
	listScroll := container.NewScroll(o.list)
	listScroll.SetMinSize(fyne.NewSize(0, uiPanelHeight))
	listBox := container.NewBorder(widget.NewLabel("Objects"), nil, nil, nil, listScroll)
	form := container.NewVBox(
		widget.NewLabel("Selected object"),
		container.NewGridWithColumns(
			2,
			widget.NewLabel("Type"),
			o.typeSelect,
			widget.NewLabel("Material"),
			o.materialSelect,
			widget.NewLabel("Pos X"),
			o.position[0],
			widget.NewLabel("Pos Y"),
			o.position[1],
			widget.NewLabel("Pos Z"),
			o.position[2],
			widget.NewLabel("Size X"),
			o.size[0],
			widget.NewLabel("Size Y"),
			o.size[1],
			widget.NewLabel("Size Z"),
			o.size[2],
		),
		apply,
		container.NewHBox(addSphere, addBox, remove),
	)
	return listBox, form
}

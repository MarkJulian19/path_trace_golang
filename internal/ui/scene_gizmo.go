package ui

import (
	"image/color"
	"math"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/driver/desktop"
	"fyne.io/fyne/v2/widget"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

type gizmoAxis int

const gizmoAxisCount = 3

const (
	gizmoAxisX gizmoAxis = iota
	gizmoAxisY
	gizmoAxisZ
)

var gizmoAxisColors = [...]color.NRGBA{
	{R: uiGizmoRedR, G: uiGizmoRedG, B: uiGizmoRedB, A: uiThemeAlpha},
	{R: uiGizmoGreenR, G: uiGizmoGreenG, B: uiThemeGreen, A: uiThemeAlpha},
	{R: uiGizmoBlueR, G: uiGizmoBlueG, B: uiGizmoRedR, A: uiThemeAlpha},
}

type sceneGizmo struct {
	sc                     *scene.Scene
	preview                *scene.RenderSettings
	selected               *int
	transformMode          *editorTransformMode
	objectMoveEnabled      *bool
	startInteractiveRender func()
	onObjectUpdated        func()

	lines   [gizmoAxisCount]*canvas.Line
	handles [gizmoAxisCount]*gizmoHandle
	labels  [gizmoAxisCount]*canvas.Text

	viewportSize fyne.Size
	center       fyne.Position
	worldLength  float64
	axes         [gizmoAxisCount]gizmoScreenAxis
}

type gizmoScreenAxis struct {
	endpoint fyne.Position
	vector   fyne.Position
	visible  bool
}

func newSceneGizmo(
	sc *scene.Scene,
	preview *scene.RenderSettings,
	selected *int,
	transformMode *editorTransformMode,
	objectMoveEnabled *bool,
	startInteractiveRender func(),
	onObjectUpdated func(),
) *sceneGizmo {
	gizmo := &sceneGizmo{
		sc: sc, preview: preview, selected: selected, transformMode: transformMode,
		objectMoveEnabled:      objectMoveEnabled,
		startInteractiveRender: startInteractiveRender, onObjectUpdated: onObjectUpdated,
	}
	for index := range gizmo.lines {
		gizmo.initAxis(index)
	}
	gizmo.hide()
	return gizmo
}

func (g *sceneGizmo) initAxis(index int) {
	axis := gizmoAxis(index)
	axisColor := gizmoAxisColors[index]
	g.lines[index] = canvas.NewLine(axisColor)
	g.lines[index].StrokeWidth = uiGizmoLineWidth
	g.handles[index] = newGizmoHandle(axisColor, func(delta fyne.Delta) {
		g.drag(axis, delta)
	})
	g.labels[index] = canvas.NewText(gizmoAxisName(axis), axisColor)
	g.labels[index].TextSize = uiGizmoLabelSize
	g.labels[index].TextStyle = fyne.TextStyle{Bold: true}
}

func gizmoAxisName(axis gizmoAxis) string {
	switch axis {
	case gizmoAxisX:
		return "X"
	case gizmoAxisY:
		return "Y"
	default:
		return "Z"
	}
}

func (g *sceneGizmo) objects() []fyne.CanvasObject {
	objects := make([]fyne.CanvasObject, 0, len(g.lines)*gizmoAxisCount)
	for _, line := range g.lines {
		objects = append(objects, line)
	}
	for _, handle := range g.handles {
		objects = append(objects, handle)
	}
	for _, label := range g.labels {
		objects = append(objects, label)
	}
	return objects
}

func (g *sceneGizmo) refresh() {
	g.update(g.viewportSize)
}

func (g *sceneGizmo) update(size fyne.Size) {
	g.viewportSize = size
	center, axes, worldLength, ok := g.projectAxes(size)
	if !ok {
		g.hide()
		return
	}
	g.center = center
	g.worldLength = worldLength
	g.axes = axes
	for index := range g.axes {
		g.updateAxis(index)
	}
}

func (g *sceneGizmo) updateAxis(index int) {
	axis := g.axes[index]
	if !axis.visible {
		g.lines[index].Hide()
		g.handles[index].Hide()
		g.labels[index].Hide()
		return
	}
	line := g.lines[index]
	line.Position1 = g.center
	line.Position2 = axis.endpoint
	line.Show()
	line.Refresh()
	handle := g.handles[index]
	handle.Resize(fyne.NewSize(uiGizmoHandleSize, uiGizmoHandleSize))
	handle.Move(axis.endpoint.Subtract(fyne.NewPos(uiGizmoHandleSize/2, uiGizmoHandleSize/2)))
	if g.canTransform() {
		handle.Show()
	} else {
		handle.Hide()
	}
	label := g.labels[index]
	label.Resize(fyne.NewSize(uiGizmoLabelSize, uiGizmoLabelSize))
	label.Move(axis.endpoint.AddXY(uiGizmoLabelOffset, -uiGizmoLabelSize/2))
	label.Show()
}

func (g *sceneGizmo) hide() {
	for index := range g.lines {
		g.lines[index].Hide()
		g.handles[index].Hide()
		g.labels[index].Hide()
	}
}

func (g *sceneGizmo) canTransform() bool {
	return g.transformMode != nil && *g.transformMode == transformModeMove &&
		g.objectMoveEnabled != nil && *g.objectMoveEnabled
}

func (g *sceneGizmo) drag(axis gizmoAxis, delta fyne.Delta) {
	if !g.canTransform() || !g.hasSelection() || int(axis) >= len(g.axes) {
		return
	}
	screenAxis := g.axes[axis]
	screenLength := math.Hypot(float64(screenAxis.vector.X), float64(screenAxis.vector.Y))
	if !screenAxis.visible || screenLength < 1 {
		return
	}
	pixelDelta := (float64(delta.DX)*float64(screenAxis.vector.X) +
		float64(delta.DY)*float64(screenAxis.vector.Y)) / screenLength
	worldDelta := pixelDelta * g.worldLength / screenLength
	if !moveObjectAlongAxis(g.sc, *g.selected, axis, worldDelta) {
		return
	}
	if g.onObjectUpdated != nil {
		g.onObjectUpdated()
	}
	g.refresh()
	if g.startInteractiveRender != nil {
		g.startInteractiveRender()
	}
}

func (g *sceneGizmo) projectAxes(size fyne.Size) (fyne.Position, [gizmoAxisCount]gizmoScreenAxis, float64, bool) {
	var axes [gizmoAxisCount]gizmoScreenAxis
	if !g.hasSelection() {
		return fyne.Position{}, axes, 0, false
	}
	offset, _, scale, ok := imageDisplayMetrics(size, g.sourceSize())
	if !ok {
		return fyne.Position{}, axes, 0, false
	}
	projection, ok := newGizmoProjection(g.sc, g.preview)
	if !ok {
		return fyne.Position{}, axes, 0, false
	}
	object := g.sc.Objects[*g.selected]
	centerSource, depth, ok := projection.project(object.Position)
	if !ok {
		return fyne.Position{}, axes, 0, false
	}
	center := imagePositionToViewport(centerSource, offset, scale)
	worldLength := max(depth*projection.viewportHeight*uiGizmoAxisLength, uiFloatEpsilon)
	for index := range axes {
		axisVector := gizmoAxisVector(gizmoAxis(index), worldLength)
		endpoint := scene.Vec3{
			X: object.Position.X + axisVector.X,
			Y: object.Position.Y + axisVector.Y,
			Z: object.Position.Z + axisVector.Z,
		}
		endpointSource, _, visible := projection.project(endpoint)
		axes[index] = gizmoScreenAxis{
			endpoint: imagePositionToViewport(endpointSource, offset, scale),
			visible:  visible,
		}
		axes[index].vector = axes[index].endpoint.Subtract(center)
	}
	return center, axes, worldLength, true
}

func (g *sceneGizmo) hasSelection() bool {
	return g.selected != nil && *g.selected >= 0 && *g.selected < len(g.sc.Objects)
}

func (g *sceneGizmo) sourceSize() fyne.Size {
	return fyne.NewSize(float32(g.preview.Width), float32(g.preview.Height))
}

func imagePositionToViewport(position fyne.Position, offset fyne.Position, scale float32) fyne.Position {
	return offset.AddXY(position.X*scale, position.Y*scale)
}

type gizmoProjection struct {
	cameraX, cameraY, cameraZ     float64
	forwardX, forwardY, forwardZ  float64
	uX, uY, uZ                    float64
	vX, vY, vZ                    float64
	viewportWidth, viewportHeight float64
	imageWidth, imageHeight       float64
}

func newGizmoProjection(sc *scene.Scene, preview *scene.RenderSettings) (gizmoProjection, bool) {
	forwardX := sc.Camera.Target.X - sc.Camera.Position.X
	forwardY := sc.Camera.Target.Y - sc.Camera.Position.Y
	forwardZ := sc.Camera.Target.Z - sc.Camera.Position.Z
	directionLength := math.Sqrt(forwardX*forwardX + forwardY*forwardY + forwardZ*forwardZ)
	if directionLength < uiFloatEpsilon || preview.Width <= 0 || preview.Height <= 0 {
		return gizmoProjection{}, false
	}
	forwardX /= directionLength
	forwardY /= directionLength
	forwardZ /= directionLength
	uX, uY, uZ, vX, vY, vZ := cameraViewportBasis(sc)
	viewportHeight := uiViewportDiameter * math.Tan(sc.Camera.FOV*math.Pi/uiDegreesPerHalfTurn/2)
	if viewportHeight <= uiFloatEpsilon {
		return gizmoProjection{}, false
	}
	aspect := sc.Camera.AspectRatio
	if aspect <= uiFloatEpsilon {
		aspect = float64(preview.Width) / float64(preview.Height)
	}
	return gizmoProjection{
		cameraX: sc.Camera.Position.X, cameraY: sc.Camera.Position.Y, cameraZ: sc.Camera.Position.Z,
		forwardX: forwardX, forwardY: forwardY, forwardZ: forwardZ,
		uX: uX, uY: uY, uZ: uZ, vX: vX, vY: vY, vZ: vZ,
		viewportWidth: aspect * viewportHeight, viewportHeight: viewportHeight,
		imageWidth: float64(preview.Width), imageHeight: float64(preview.Height),
	}, true
}

func (p gizmoProjection) project(point scene.Vec3) (fyne.Position, float64, bool) {
	fromCameraX := point.X - p.cameraX
	fromCameraY := point.Y - p.cameraY
	fromCameraZ := point.Z - p.cameraZ
	depth := fromCameraX*p.forwardX + fromCameraY*p.forwardY + fromCameraZ*p.forwardZ
	if depth <= uiFloatEpsilon {
		return fyne.Position{}, 0, false
	}
	px := p.uX*fromCameraX + p.uY*fromCameraY + p.uZ*fromCameraZ
	py := p.vX*fromCameraX + p.vY*fromCameraY + p.vZ*fromCameraZ
	return fyne.NewPos(
		float32((uiHalf+px/depth/p.viewportWidth)*p.imageWidth),
		float32((uiHalf-py/depth/p.viewportHeight)*p.imageHeight),
	), depth, true
}

func gizmoAxisVector(axis gizmoAxis, length float64) scene.Vec3 {
	switch axis {
	case gizmoAxisX:
		return scene.Vec3{X: length}
	case gizmoAxisY:
		return scene.Vec3{Y: length}
	default:
		return scene.Vec3{Z: length}
	}
}

type gizmoHandle struct {
	widget.BaseWidget

	color  color.Color
	onDrag func(fyne.Delta)
}

func newGizmoHandle(axisColor color.Color, onDrag func(fyne.Delta)) *gizmoHandle {
	handle := &gizmoHandle{color: axisColor, onDrag: onDrag}
	handle.ExtendBaseWidget(handle)
	return handle
}

// CreateRenderer создаёт отрисовщик круглой ручки оси.
func (h *gizmoHandle) CreateRenderer() fyne.WidgetRenderer {
	circle := canvas.NewCircle(h.color)
	circle.StrokeColor = color.White
	circle.StrokeWidth = 1
	return &gizmoHandleRenderer{circle: circle}
}

// MouseDown фиксирует нажатие основной кнопки на ручке оси.
func (h *gizmoHandle) MouseDown(ev *desktop.MouseEvent) {
	if ev.Button == desktop.MouseButtonPrimary {
		h.Refresh()
	}
}

// MouseUp завершает нажатие ручки оси.
func (h *gizmoHandle) MouseUp(*desktop.MouseEvent) {}

// Dragged передаёт перемещение мыши обработчику выбранной оси.
func (h *gizmoHandle) Dragged(ev *fyne.DragEvent) {
	if h.onDrag != nil {
		h.onDrag(ev.Dragged)
	}
}

// DragEnd завершает перетаскивание ручки оси.
func (h *gizmoHandle) DragEnd() {}

type gizmoHandleRenderer struct {
	circle *canvas.Circle
}

// Layout растягивает круг по размеру ручки.
func (r *gizmoHandleRenderer) Layout(size fyne.Size) {
	r.circle.Resize(size)
}

// MinSize возвращает размер области захвата ручки.
func (r *gizmoHandleRenderer) MinSize() fyne.Size {
	return fyne.NewSize(uiGizmoHandleSize, uiGizmoHandleSize)
}

// Objects возвращает объекты, отрисовываемые ручкой.
func (r *gizmoHandleRenderer) Objects() []fyne.CanvasObject {
	return []fyne.CanvasObject{r.circle}
}

// Refresh обновляет отображение ручки.
func (r *gizmoHandleRenderer) Refresh() {
	r.circle.Refresh()
}

// Destroy освобождает ресурсы отрисовщика.
func (r *gizmoHandleRenderer) Destroy() {}

type sceneViewportLayout struct {
	gizmo *sceneGizmo
}

// Layout растягивает viewport и перестраивает положение осей.
func (l *sceneViewportLayout) Layout(objects []fyne.CanvasObject, size fyne.Size) {
	if len(objects) == 0 {
		return
	}
	objects[0].Move(fyne.NewPos(0, 0))
	objects[0].Resize(size)
	l.gizmo.update(size)
}

// MinSize использует минимальный размер основного viewport.
func (l *sceneViewportLayout) MinSize(objects []fyne.CanvasObject) fyne.Size {
	if len(objects) == 0 {
		return fyne.Size{}
	}
	return objects[0].MinSize()
}

func newSceneViewport(picker fyne.CanvasObject, gizmo *sceneGizmo) fyne.CanvasObject {
	objects := []fyne.CanvasObject{picker}
	objects = append(objects, gizmo.objects()...)
	return container.New(&sceneViewportLayout{gizmo: gizmo}, objects...)
}

package ui

import (
	"math"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/driver/desktop"
	"fyne.io/fyne/v2/widget"
)

// pickableCanvas обертка для canvas.Image с поддержкой кликов.
type pickableCanvas struct {
	widget.BaseWidget

	img    *canvas.Image
	onTap  func(x, y int)
	onDrag func(dx, dy int)

	dragRemainderX float32
	dragRemainderY float32
	mouseDown      bool
	tapHandled     bool
}

func newPickableCanvas(img *canvas.Image, onTap func(x, y int), onDrag func(dx, dy int)) *pickableCanvas {
	p := &pickableCanvas{
		img:    img,
		onTap:  onTap,
		onDrag: onDrag,
	}
	p.ExtendBaseWidget(p)
	return p
}

// CreateRenderer создаёт отрисовщик изображения с обработкой ввода.
func (p *pickableCanvas) CreateRenderer() fyne.WidgetRenderer {
	return &pickableCanvasRenderer{
		pickable: p,
		img:      p.img,
	}
}

type pickableCanvasRenderer struct {
	pickable *pickableCanvas
	img      *canvas.Image
}

// Layout размещает изображение внутри виджета.
func (r *pickableCanvasRenderer) Layout(size fyne.Size) {
	r.img.Resize(size)
}

// MinSize возвращает минимальный размер изображения.
func (r *pickableCanvasRenderer) MinSize() fyne.Size {
	// Размер исходного изображения не должен блокировать перетаскивание
	// разделителей: оно масштабируется под фактический размер viewport.
	return fyne.NewSize(uiViewportMinWidth, uiViewportMinHeight)
}

// Objects возвращает объекты, отрисовываемые рендерером.
func (r *pickableCanvasRenderer) Objects() []fyne.CanvasObject {
	return []fyne.CanvasObject{r.img}
}

// Refresh обновляет изображение.
func (r *pickableCanvasRenderer) Refresh() {
	r.img.Refresh()
}

// Destroy освобождает ресурсы рендерера.
func (r *pickableCanvasRenderer) Destroy() {}

// Tapped обрабатывает клик на canvas.
func (p *pickableCanvas) Tapped(ev *fyne.PointEvent) {
	if p.mouseDown || p.tapHandled {
		return
	}
	if p.onTap == nil {
		return
	}
	x, y, ok := p.tapCoordinates(ev)
	if !ok {
		return
	}
	p.onTap(x, y)
}

// MouseDown выбирает объект в момент нажатия, чтобы тот же жест можно было сразу перетащить.
func (p *pickableCanvas) MouseDown(ev *desktop.MouseEvent) {
	if ev.Button != desktop.MouseButtonPrimary || p.onTap == nil {
		return
	}
	p.mouseDown = true
	p.tapHandled = true
	p.dragRemainderX = 0
	p.dragRemainderY = 0
	x, y, ok := p.tapCoordinates(&ev.PointEvent)
	if ok {
		p.onTap(x, y)
	}
}

// MouseUp завершает состояние нажатия основной кнопки мыши.
func (p *pickableCanvas) MouseUp(ev *desktop.MouseEvent) {
	if ev.Button == desktop.MouseButtonPrimary {
		p.mouseDown = false
	}
}

// Dragged обрабатывает перетаскивание.
func (p *pickableCanvas) Dragged(ev *fyne.DragEvent) {
	if p.onDrag == nil {
		return
	}
	delta := p.imageDelta(ev.Dragged)
	if delta.X == 0 && delta.Y == 0 {
		return
	}
	p.onDrag(int(delta.X), int(delta.Y))
}

// DragEnd завершает перетаскивание изображения.
func (p *pickableCanvas) DragEnd() {
	p.dragRemainderX = 0
	p.dragRemainderY = 0
}

func (p *pickableCanvas) tapCoordinates(ev *fyne.PointEvent) (int, int, bool) {
	size := p.Size()
	imgSize := p.sourceSize()
	offset, displaySize, scale, ok := imageDisplayMetrics(size, imgSize)
	if !ok {
		return 0, 0, false
	}
	imgPosition := ev.Position.Subtract(offset)
	if imgPosition.X < 0 || imgPosition.X > displaySize.Width ||
		imgPosition.Y < 0 || imgPosition.Y > displaySize.Height {
		return 0, 0, false
	}
	x := int(float64(imgPosition.X) / float64(scale))
	y := int(float64(imgPosition.Y) / float64(scale))
	return clampCoordinate(x, int(imgSize.Width)), clampCoordinate(y, int(imgSize.Height)), true
}

func (p *pickableCanvas) imageDelta(delta fyne.Delta) fyne.Position {
	_, _, scale, ok := imageDisplayMetrics(p.Size(), p.sourceSize())
	if !ok {
		return fyne.NewPos(0, 0)
	}
	p.dragRemainderX += delta.DX / scale
	p.dragRemainderY += delta.DY / scale
	dx := int(math.Trunc(float64(p.dragRemainderX)))
	dy := int(math.Trunc(float64(p.dragRemainderY)))
	p.dragRemainderX -= float32(dx)
	p.dragRemainderY -= float32(dy)
	return fyne.NewPos(float32(dx), float32(dy))
}

func imageDisplayMetrics(viewportSize, imageSize fyne.Size) (fyne.Position, fyne.Size, float32, bool) {
	if viewportSize.Width <= 0 || viewportSize.Height <= 0 || imageSize.Width <= 0 || imageSize.Height <= 0 {
		return fyne.Position{}, fyne.Size{}, 0, false
	}
	scale := min(viewportSize.Width/imageSize.Width, viewportSize.Height/imageSize.Height)
	displaySize := fyne.NewSize(imageSize.Width*scale, imageSize.Height*scale)
	offset := fyne.NewPos(
		(viewportSize.Width-displaySize.Width)/2,
		(viewportSize.Height-displaySize.Height)/2,
	)
	return offset, displaySize, scale, true
}

func (p *pickableCanvas) sourceSize() fyne.Size {
	if p.img.Image == nil {
		return fyne.NewSize(0, 0)
	}
	bounds := p.img.Image.Bounds()
	return fyne.NewSize(float32(bounds.Dx()), float32(bounds.Dy()))
}

func clampCoordinate(value, limit int) int {
	if value < 0 {
		return 0
	}
	if value >= limit {
		return limit - 1
	}
	return value
}

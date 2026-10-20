package ui

import (
	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/widget"
)

// pickableCanvas обертка для canvas.Image с поддержкой кликов.
type pickableCanvas struct {
	widget.BaseWidget

	img    *canvas.Image
	onTap  func(x, y int)
	onDrag func(x1, y1, x2, y2 int)
}

func newPickableCanvas(img *canvas.Image, onTap func(x, y int), onDrag func(x1, y1, x2, y2 int)) *pickableCanvas {
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
	return r.img.MinSize()
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
	if p.onTap == nil {
		return
	}
	x, y, ok := p.tapCoordinates(ev)
	if !ok {
		return
	}
	p.onTap(x, y)
}

// Dragged обрабатывает перетаскивание.
func (p *pickableCanvas) Dragged(ev *fyne.DragEvent) {
	if p.onDrag != nil {
		size := p.Size()
		imgSize := p.img.Size()

		scaleX := float32(imgSize.Width) / float32(size.Width)
		scaleY := float32(imgSize.Height) / float32(size.Height)

		x1 := int(float32(ev.Position.X) * scaleX)
		y1 := int(float32(ev.Position.Y) * scaleY)
		// Для drag используем текущую позицию как конечную
		x2 := x1
		y2 := y1

		p.onDrag(x1, y1, x2, y2)
	}
}

// DragEnd завершает перетаскивание изображения.
func (p *pickableCanvas) DragEnd() {}

func (p *pickableCanvas) tapCoordinates(ev *fyne.PointEvent) (int, int, bool) {
	size := p.Size()
	imgSize := p.img.Size()
	scaleX := float32(size.Width) / float32(imgSize.Width)
	scaleY := float32(size.Height) / float32(imgSize.Height)
	scale := min(scaleX, scaleY)
	displayW := float32(imgSize.Width) * scale
	displayH := float32(imgSize.Height) * scale
	offsetX := (float32(size.Width) - displayW) / 2
	offsetY := (float32(size.Height) - displayH) / 2
	imgX := ev.Position.X - offsetX
	imgY := ev.Position.Y - offsetY
	if imgX < 0 || imgX > displayW || imgY < 0 || imgY > displayH {
		return 0, 0, false
	}
	x := int(float64(imgX) / float64(scale))
	y := int(float64(imgY) / float64(scale))
	return clampCoordinate(x, int(imgSize.Width)), clampCoordinate(y, int(imgSize.Height)), true
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

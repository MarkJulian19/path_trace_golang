package ui

import (
	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/widget"
)

// pickableCanvas обертка для canvas.Image с поддержкой кликов
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

func (r *pickableCanvasRenderer) Layout(size fyne.Size) {
	r.img.Resize(size)
}

func (r *pickableCanvasRenderer) MinSize() fyne.Size {
	return r.img.MinSize()
}

func (r *pickableCanvasRenderer) Objects() []fyne.CanvasObject {
	return []fyne.CanvasObject{r.img}
}

func (r *pickableCanvasRenderer) Refresh() {
	r.img.Refresh()
}

func (r *pickableCanvasRenderer) Destroy() {}

// Tapped обрабатывает клик на canvas
func (p *pickableCanvas) Tapped(ev *fyne.PointEvent) {
	if p.onTap != nil {
		// Преобразуем координаты клика в координаты изображения
		// Учитываем, что изображение может быть масштабировано и центрировано
		size := p.Size()
		imgSize := p.img.Size()
		
		// Вычисляем масштаб с учетом FillMode = ImageFillContain
		// Изображение масштабируется так, чтобы полностью поместиться в виджет
		scaleX := float32(size.Width) / float32(imgSize.Width)
		scaleY := float32(size.Height) / float32(imgSize.Height)
		scale := scaleX
		if scaleY < scaleX {
			scale = scaleY
		}
		
		// Вычисляем размер отображаемого изображения
		displayW := float32(imgSize.Width) * scale
		displayH := float32(imgSize.Height) * scale
		
		// Вычисляем смещение для центрирования
		offsetX := (float32(size.Width) - displayW) / 2
		offsetY := (float32(size.Height) - displayH) / 2
		
		// Преобразуем координаты с учетом смещения
		imgX := ev.Position.X - offsetX
		imgY := ev.Position.Y - offsetY
		
		// Проверяем, что клик внутри изображения
		if imgX < 0 || imgX > displayW || imgY < 0 || imgY > displayH {
			return
		}
		
		// Преобразуем в координаты исходного изображения
		// Важно: используем точное деление для правильного преобразования
		x := int(float64(imgX) / float64(scale))
		y := int(float64(imgY) / float64(scale))
		
		// Ограничиваем координаты
		if x < 0 {
			x = 0
		}
		if x >= int(imgSize.Width) {
			x = int(imgSize.Width) - 1
		}
		if y < 0 {
			y = 0
		}
		if y >= int(imgSize.Height) {
			y = int(imgSize.Height) - 1
		}
		
		p.onTap(x, y)
	}
}

// Dragged обрабатывает перетаскивание
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

func (p *pickableCanvas) DragEnd() {}


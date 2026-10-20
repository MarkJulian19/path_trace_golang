package gpu

import (
	"image"

	renderentity "github.com/MarkJulian19/path_trace_golang/internal/entities/render"
	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

// Adapter предоставляет GPU-рендер и его настройки через порты сценария.
type Adapter struct{}

// Render строит изображение на GPU.
func (Adapter) Render(
	value *scene.Scene,
	config renderentity.Config,
	img *image.RGBA,
	progress renderentity.Progress,
) error {
	return Render(value, config, img, progress)
}

// SetSelectedObjectFromUI выбирает объект для подсветки.
func (Adapter) SetSelectedObjectFromUI(index int) {
	SetSelectedObjectFromUI(index)
}

// SetLivePreview включает или выключает промежуточное обновление изображения.
func (Adapter) SetLivePreview(enabled bool) {
	SetLivePreview(enabled)
}

// GetRenderMode возвращает текущий режим GPU-рендера.
func (Adapter) GetRenderMode() int {
	return GetRenderMode()
}

// SetRenderCancel устанавливает флаг отмены GPU-рендера.
func (Adapter) SetRenderCancel(cancel bool) {
	SetRenderCancel(cancel)
}

// SetDenoiseConfigFromUI задаёт параметры подавления шума.
func (Adapter) SetDenoiseConfigFromUI(enabled bool, sigmaS, sigmaR float64) {
	SetDenoiseConfigFromUI(enabled, sigmaS, sigmaR)
}

// SetSmoothConfigFromUI задаёт параметры дополнительного сглаживания.
func (Adapter) SetSmoothConfigFromUI(enabled bool, radius int, strength float64) {
	SetSmoothConfigFromUI(enabled, radius, strength)
}

// SetRenderModeFromUI задаёт режим GPU-рендера.
func (Adapter) SetRenderModeFromUI(mode int) {
	SetRenderModeFromUI(mode)
}

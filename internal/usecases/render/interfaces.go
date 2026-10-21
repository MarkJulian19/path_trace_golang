package render

import (
	"image"

	renderentity "github.com/MarkJulian19/path_trace_golang/internal/entities/render"
	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

//go:generate mockgen -source=interfaces.go -destination mocks/mock.go -package mocks

// GPUBackend выполняет построение изображения с использованием GPU.
type GPUBackend interface {
	Render(value *scene.Scene, config renderentity.Config, target *image.RGBA, progress renderentity.Progress) error
}

// GPUInitializer создаёт оконные ресурсы GPU в главном потоке приложения.
type GPUInitializer interface {
	Initialize() error
}

// GPUBackendCancelable is an optional extension used by interactive rendering.
// It lets a backend stop before a queued request starts and between dispatch tiles.
type GPUBackendCancelable interface {
	RenderWithCancel(
		value *scene.Scene,
		config renderentity.Config,
		target *image.RGBA,
		progress renderentity.Progress,
		cancelled func() bool,
	) error
}

// GPUControls изменяет параметры GPU-рендера из входного адаптера.
type GPUControls interface {
	SetSelectedObjectFromUI(index int)
	SetLivePreview(enabled bool)
	GetRenderMode() int
	SetRenderCancel(cancel bool)
	SetDenoiseConfigFromUI(enabled bool, sigmaS, sigmaR float64)
	SetSmoothConfigFromUI(enabled bool, radius int, strength float64)
	SetRenderModeFromUI(mode int)
}

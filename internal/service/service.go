package service

import (
	"github.com/MarkJulian19/path_trace_golang/internal/repository/scenejson"
	"github.com/MarkJulian19/path_trace_golang/internal/transport/gpu"
	"github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
	scenecase "github.com/MarkJulian19/path_trace_golang/internal/usecases/scene"
)

// Container содержит собранные сценарии приложения.
type Container struct {
	Scenes *scenecase.Scene
	Render *render.Service
}

// New собирает зависимости приложения и подключает адаптеры рендера.
func New() *Container {
	gpuAdapter := gpu.Adapter{}

	return &Container{
		Scenes: scenecase.New(scenejson.New()),
		Render: render.New(gpuAdapter, gpuAdapter),
	}
}

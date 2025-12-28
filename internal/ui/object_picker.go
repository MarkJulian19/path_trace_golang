package ui

import (
	"github.com/user/pathtracer/internal/engine"
	"github.com/user/pathtracer/internal/scene"
)

// pickObject выполняет raycast для определения объекта под курсором
// Возвращает индекс объекта или -1, если объект не найден
func pickObject(sc *scene.Scene, cfg engine.RenderConfig, x, y int) int {
	return engine.PickObject(sc, cfg, x, y)
}


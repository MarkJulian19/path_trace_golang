package ui

import (
	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	"github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

// pickObject выполняет raycast для определения объекта под курсором
// Возвращает индекс объекта или -1, если объект не найден.
func pickObject(sc *scene.Scene, cfg render.Config, x, y int) int {
	return render.PickObject(sc, cfg, x, y)
}

package scene

import domain "github.com/MarkJulian19/path_trace_golang/internal/entities/scene"

//go:generate mockgen -source=interfaces.go -destination mocks/mock.go -package mocks

// Repository загружает и сохраняет сцены.
type Repository interface {
	Load(path string) (*domain.Scene, error)
	Save(path string, value *domain.Scene) error
}

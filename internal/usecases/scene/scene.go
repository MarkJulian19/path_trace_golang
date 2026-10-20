package scene

import (
	"fmt"

	domain "github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

// Scene предоставляет сценарии работы со сценами через порт хранилища.
type Scene struct {
	repository Repository
}

// New создаёт сценарий работы со сценами.
func New(repository Repository) *Scene {
	return &Scene{repository: repository}
}

// Load загружает сцену по пути к JSON-файлу.
func (s *Scene) Load(path string) (*domain.Scene, error) {
	value, err := s.repository.Load(path)
	if err != nil {
		return nil, fmt.Errorf("load scene: %w", err)
	}
	return value, nil
}

// Save сохраняет сцену по указанному пути.
func (s *Scene) Save(path string, value *domain.Scene) error {
	err := s.repository.Save(path, value)
	if err != nil {
		return fmt.Errorf("save scene: %w", err)
	}
	return nil
}

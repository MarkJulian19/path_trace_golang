package scenejson

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

// Repository сохраняет сцены в JSON-файлах.
type Repository struct{}

// New создаёт хранилище сцен на основе JSON-файлов.
func New() Repository {
	return Repository{}
}

// Load читает сцену из JSON-файла.
func (Repository) Load(path string) (result *scene.Scene, err error) {
	f, err := os.Open(filepath.Clean(path))
	if err != nil {
		return nil, fmt.Errorf("open scene: %w", err)
	}
	defer func() {
		closeErr := f.Close()
		if closeErr != nil && err == nil {
			err = fmt.Errorf("close scene: %w", closeErr)
		}
	}()

	var fileScene sceneFile
	err = json.NewDecoder(f).Decode(&fileScene)
	if err != nil {
		return nil, fmt.Errorf("decode scene: %w", err)
	}

	return toEntity(fileScene), nil
}

// Save записывает сцену в JSON-файл.
func (Repository) Save(path string, sc *scene.Scene) (err error) {
	f, err := os.Create(filepath.Clean(path))
	if err != nil {
		return fmt.Errorf("create scene: %w", err)
	}
	defer func() {
		closeErr := f.Close()
		if closeErr != nil && err == nil {
			err = fmt.Errorf("close scene: %w", closeErr)
		}
	}()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	err = enc.Encode(fromEntity(sc))
	if err != nil {
		return fmt.Errorf("encode scene: %w", err)
	}

	return nil
}

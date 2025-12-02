package scene

import (
	"encoding/json"
	"fmt"
	"os"
)

// Load reads a Scene from a JSON file.
func Load(path string) (*Scene, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open scene: %w", err)
	}
	defer f.Close()

	var sc Scene
	if err := json.NewDecoder(f).Decode(&sc); err != nil {
		return nil, fmt.Errorf("decode scene: %w", err)
	}
	return &sc, nil
}

// Save writes a Scene to a JSON file.
func Save(path string, sc *Scene) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create scene: %w", err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(sc); err != nil {
		return fmt.Errorf("encode scene: %w", err)
	}
	return nil
}

// (Rendering helpers moved to engine package to avoid import cycle)


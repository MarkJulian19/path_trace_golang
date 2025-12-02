package engine

import (
	"fmt"
	"image"
	"image/png"
	"os"

	"github.com/user/pathtracer/internal/scene"
)

// RenderScene performs path tracing of the given scene using provided settings.
func RenderScene(sc *scene.Scene, settings scene.RenderSettings) (image.Image, error) {
	cfg := RenderConfig{
		Width:        settings.Width,
		Height:       settings.Height,
		SamplesPerPx: settings.SamplesPerPx,
		MaxDepth:     settings.MaxDepth,
	}
	img := Render(sc, cfg)
	return img, nil
}

// RenderSettingsForMode returns reasonable defaults for preview/final modes.
func RenderSettingsForMode(mode string) scene.RenderSettings {
	switch mode {
	case "final":
		return scene.RenderSettings{
			Width:        1920,
			Height:       1080,
			SamplesPerPx: 1000,
			MaxDepth:     80,
		}
	default:
		return scene.RenderSettings{
			Width:        400,
			Height:       225,
			SamplesPerPx: 20,
			MaxDepth:     20,
		}
	}
}

// SavePNG writes an image to a PNG file.
func SavePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create png: %w", err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		return fmt.Errorf("encode png: %w", err)
	}
	return nil
}

package render

import (
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

// Scene строит изображение сцены с указанными параметрами.
func Scene(sc *scene.Scene, settings scene.RenderSettings) (image.Image, error) {
	cfg := Config{
		Width:        settings.Width,
		Height:       settings.Height,
		SamplesPerPx: settings.SamplesPerPx,
		MaxDepth:     settings.MaxDepth,
		MaxRayDist:   float32(settings.MaxRayDist),
	}
	img := Render(sc, cfg)
	return img, nil
}

// SettingsForMode возвращает параметры для режима предпросмотра или финального вывода.
func SettingsForMode(mode string) scene.RenderSettings {
	switch mode {
	case "final":
		return scene.RenderSettings{
			Width:        defaultWidth,
			Height:       defaultHeight,
			SamplesPerPx: 1000,
			MaxDepth:     defaultMaxDepth,
			MaxRayDist:   defaultMaxRayDistance, // Максимальная длина луча по умолчанию
		}
	default:
		return scene.RenderSettings{
			Width:        previewWidth,
			Height:       previewHeight,
			SamplesPerPx: previewSamplesPerPixel,
			MaxDepth:     previewMaxDepth,
			MaxRayDist:   defaultMaxRayDistance, // Максимальная длина луча по умолчанию
		}
	}
}

// SavePNG writes an image to a PNG file.
func SavePNG(path string, img image.Image) (err error) {
	f, err := os.Create(filepath.Clean(path))
	if err != nil {
		return fmt.Errorf("create png: %w", err)
	}
	defer func() {
		closeErr := f.Close()
		if closeErr != nil && err == nil {
			err = fmt.Errorf("close png: %w", closeErr)
		}
	}()
	err = png.Encode(f, img)
	if err != nil {
		return fmt.Errorf("encode png: %w", err)
	}
	return nil
}

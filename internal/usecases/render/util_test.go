package render_test

import (
	"image"
	"image/color"
	"os"
	"path/filepath"
	"testing"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	render "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

func TestSceneUsesRenderSettings(t *testing.T) {
	t.Parallel()
	settings := scene.RenderSettings{
		Width:        testWidth,
		Height:       testHeight,
		SamplesPerPx: 1,
		MaxDepth:     1,
		MaxRayDist:   testMaxRayDist,
	}

	img, err := render.Scene(benchmarkScene(), settings)
	if err != nil {
		t.Fatalf("Scene() error = %v", err)
	}
	if got := img.Bounds().Size(); got.X != settings.Width || got.Y != settings.Height {
		t.Fatalf("scene image size = %v, want %dx%d", got, settings.Width, settings.Height)
	}
}

func TestSavePNGWritesImage(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "render.png")
	want := image.NewRGBA(image.Rect(0, 0, testWidth, testHeight))
	want.SetRGBA(0, 0, color.RGBA{R: 255, G: 128, B: 64, A: 255})

	err := render.SavePNG(path, want)
	if err != nil {
		t.Fatalf("SavePNG() error = %v", err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat PNG: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("SavePNG() created an empty file")
	}
}

func TestSavePNGReportsCreateError(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "missing", "render.png")
	err := render.SavePNG(path, testVideoImage())
	if err == nil {
		t.Fatal("SavePNG() accepted a path in a missing directory")
	}
}

func BenchmarkScene(b *testing.B) {
	settings := scene.RenderSettings{
		Width:        benchmarkWidth,
		Height:       benchmarkHeight,
		SamplesPerPx: benchmarkSamples,
		MaxDepth:     benchmarkDepth,
		MaxRayDist:   testMaxRayDist,
	}
	sc := benchmarkScene()
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		benchmarkSceneImage, errBenchmarkSceneRender = render.Scene(sc, settings)
	}
}

func BenchmarkSavePNG(b *testing.B) {
	path := filepath.Join(b.TempDir(), "render.png")
	img := testVideoImage()
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		errBenchmarkPNG = render.SavePNG(path, img)
	}
}

var (
	benchmarkSceneImage     image.Image
	errBenchmarkSceneRender error
	errBenchmarkPNG         error
)

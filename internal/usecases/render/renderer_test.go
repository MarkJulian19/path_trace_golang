package render_test

import (
	"image"
	"sync/atomic"
	"testing"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	render "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

const (
	previewWidth       = 400
	previewHeight      = 225
	previewSamples     = 20
	previewDepth       = 20
	finalWidth         = 1920
	finalHeight        = 1080
	finalSamples       = 1000
	finalDepth         = 80
	benchmarkWidth     = 64
	benchmarkHeight    = 36
	benchmarkSamples   = 4
	benchmarkDepth     = 5
	benchmarkLargeW    = 128
	benchmarkLargeH    = 72
	benchmarkLargeSpp  = 2
	interpolationStart = -1
	interpolationMid   = 0.5
	interpolationEnd   = 2
	maxColorChannel    = uint32(1<<16 - 1)
)

func TestSettingsForMode(t *testing.T) {
	t.Parallel()
	testCases := []struct {
		name     string
		mode     string
		expected scene.RenderSettings
	}{
		{
			name: "preview",
			mode: "preview",
			expected: scene.RenderSettings{
				Width:        previewWidth,
				Height:       previewHeight,
				SamplesPerPx: previewSamples,
				MaxDepth:     previewDepth,
				MaxRayDist:   testMaxRayDist,
			},
		},
		{
			name: "final",
			mode: "final",
			expected: scene.RenderSettings{
				Width:        finalWidth,
				Height:       finalHeight,
				SamplesPerPx: finalSamples,
				MaxDepth:     finalDepth,
				MaxRayDist:   testMaxRayDist,
			},
		},
		{
			name: "unknown uses preview",
			mode: "unknown",
			expected: scene.RenderSettings{
				Width:        previewWidth,
				Height:       previewHeight,
				SamplesPerPx: previewSamples,
				MaxDepth:     previewDepth,
				MaxRayDist:   testMaxRayDist,
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			if got := render.SettingsForMode(testCase.mode); got != testCase.expected {
				t.Fatalf("SettingsForMode(%q) = %#v, want %#v", testCase.mode, got, testCase.expected)
			}
		})
	}
}

func TestInterpolateCameraClampsParameter(t *testing.T) {
	t.Parallel()
	start := scene.Camera{
		Position: scene.Vec3{X: 1, Y: 2, Z: 3},
		Target:   scene.Vec3{X: 4, Y: 5, Z: 6},
		FOV:      testCameraFOV,
	}
	end := scene.Camera{
		Position: scene.Vec3{X: 11, Y: 12, Z: 13},
		Target:   scene.Vec3{X: 14, Y: 15, Z: 16},
	}

	testCases := []struct {
		name  string
		value float64
		want  scene.Vec3
	}{
		{name: "below range", value: interpolationStart, want: start.Position},
		{name: "middle", value: interpolationMid, want: scene.Vec3{X: 6, Y: 7, Z: 8}},
		{name: "above range", value: interpolationEnd, want: end.Position},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			got := render.InterpolateCamera(start, end, testCase.value)
			if got.Position != testCase.want {
				t.Fatalf("position = %#v, want %#v", got.Position, testCase.want)
			}
			if got.FOV != start.FOV {
				t.Fatalf("unchanged camera fields were modified: FOV=%v", got.FOV)
			}
		})
	}
}

func TestRenderProducesExpectedImage(t *testing.T) {
	t.Parallel()

	img := render.Render(benchmarkScene(), testConfig())
	if got, want := img.Bounds().Dx(), testWidth; got != want {
		t.Fatalf("image width = %d, want %d", got, want)
	}
	if got, want := img.Bounds().Dy(), testHeight; got != want {
		t.Fatalf("image height = %d, want %d", got, want)
	}

	pixel := img.At(testWidth/2, testHeight/2)
	_, _, _, alpha := pixel.RGBA()
	if alpha != maxColorChannel {
		t.Fatalf("center pixel alpha = %d, want %d", alpha, maxColorChannel)
	}
}

func TestIntoReportsProgressAndRejectsWrongBuffer(t *testing.T) {
	t.Parallel()
	cfg := testConfig()
	var progressCalls atomic.Int32
	img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))

	render.Into(benchmarkScene(), cfg, img, func(_, _ int) {
		progressCalls.Add(1)
	})
	if progressCalls.Load() == 0 {
		t.Fatal("render progress callback was not called")
	}

	wrongSize := image.NewRGBA(image.Rect(0, 0, 1, 1))
	render.Into(benchmarkScene(), cfg, wrongSize, nil)
	if got := wrongSize.Bounds().Dx(); got != 1 {
		t.Fatalf("wrong-size image was changed to width %d", got)
	}
}

func TestPickObjectReturnsNearestObjectAndHandlesEmptyScene(t *testing.T) {
	t.Parallel()
	config := render.Config{
		Width:        benchmarkWidth,
		Height:       benchmarkHeight,
		SamplesPerPx: benchmarkSamples,
		MaxDepth:     benchmarkDepth,
		MaxRayDist:   testMaxRayDist,
	}
	sc := benchmarkScene()

	if got := render.PickObject(sc, config, config.Width/2, config.Height/2); got != 2 {
		t.Fatalf("center object index = %d, want nearest box index 2", got)
	}
	if got := render.PickObject(&scene.Scene{}, config, 0, 0); got != -1 {
		t.Fatalf("empty scene object index = %d, want -1", got)
	}
}

func BenchmarkRender(b *testing.B) {
	benchmarkCases := []struct {
		name         string
		width        int
		height       int
		samplesPerPx int
		maxDepth     int
	}{
		{name: "small", width: testWidth, height: testHeight, samplesPerPx: testSamples, maxDepth: testDepth},
		{
			name:         "medium",
			width:        benchmarkWidth,
			height:       benchmarkHeight,
			samplesPerPx: benchmarkSamples,
			maxDepth:     benchmarkDepth,
		},
		{
			name:         "large",
			width:        benchmarkLargeW,
			height:       benchmarkLargeH,
			samplesPerPx: benchmarkLargeSpp,
			maxDepth:     benchmarkDepth,
		},
	}

	for _, benchmarkCase := range benchmarkCases {
		b.Run(benchmarkCase.name, func(b *testing.B) {
			config := render.Config{
				Width:        benchmarkCase.width,
				Height:       benchmarkCase.height,
				SamplesPerPx: benchmarkCase.samplesPerPx,
				MaxDepth:     benchmarkCase.maxDepth,
				MaxRayDist:   testMaxRayDist,
			}
			sc := benchmarkScene()
			b.SetBytes(int64(benchmarkCase.width * benchmarkCase.height))
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				benchmarkImage = render.Render(sc, config)
			}
		})
	}
}

func BenchmarkInto(b *testing.B) {
	sc := benchmarkScene()
	config := testConfig()
	img := image.NewRGBA(image.Rect(0, 0, config.Width, config.Height))
	b.SetBytes(int64(config.Width * config.Height))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		render.Into(sc, config, img, nil)
	}
}

func BenchmarkPickObject(b *testing.B) {
	sc := benchmarkScene()
	config := render.Config{
		Width:        benchmarkWidth,
		Height:       benchmarkHeight,
		SamplesPerPx: benchmarkSamples,
		MaxDepth:     benchmarkDepth,
		MaxRayDist:   testMaxRayDist,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		benchmarkObjectIndex = render.PickObject(sc, config, config.Width/2, config.Height/2)
	}
}

func BenchmarkSettingsForMode(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		benchmarkSettings = render.SettingsForMode("preview")
	}
}

var (
	benchmarkImage       image.Image
	benchmarkObjectIndex int
	benchmarkSettings    scene.RenderSettings
)

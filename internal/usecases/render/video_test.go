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

const (
	testVideoFPS      = 1
	testVideoDuration = 1.0
	testVideoQuality  = 75
	testVideoPixel    = 32
)

func TestDefaultVideoEncodingConfig(t *testing.T) {
	t.Parallel()
	config := render.DefaultVideoEncodingConfig()
	if config.Bitrate <= 0 || config.Quality <= 0 || config.PixelFormat == "" {
		t.Fatalf("invalid default video config: %#v", config)
	}
}

func TestVideoSequenceRestoresCamera(t *testing.T) {
	t.Parallel()
	sc := benchmarkScene()
	original := sc.Camera
	end := original
	end.Position.X += 1
	progressCalls := 0

	frames, err := render.VideoSequence(
		sc,
		original,
		end,
		testVideoDuration,
		testVideoFPS,
		testConfig(),
		func(_, _ int, _ image.Image) { progressCalls++ },
	)
	if err != nil {
		t.Fatalf("VideoSequence() error = %v", err)
	}
	if len(frames) != 1 {
		t.Fatalf("frame count = %d, want 1", len(frames))
	}
	if sc.Camera != original {
		t.Fatalf("camera was not restored: got %#v, want %#v", sc.Camera, original)
	}
	if progressCalls != 1 {
		t.Fatalf("progress calls = %d, want 1", progressCalls)
	}
}

func TestCreateVideoFromFramesValidatesAndWritesAVI(t *testing.T) {
	t.Parallel()
	err := render.CreateVideoFromFrames(
		nil,
		filepath.Join(t.TempDir(), "empty"),
		testVideoFPS,
		nil,
	)
	if err == nil {
		t.Fatal("empty video was accepted")
	}

	outputPath := filepath.Join(t.TempDir(), "movie")
	progressCalls := 0
	err = render.CreateVideoFromFramesWithConfig(
		[]image.Image{testVideoImage()},
		outputPath,
		testVideoFPS,
		render.VideoEncodingConfig{Bitrate: testVideoQuality, Quality: testVideoQuality, PixelFormat: "yuv420p"},
		func(_, _ int) { progressCalls++ },
	)
	if err != nil {
		t.Fatalf("CreateVideoFromFramesWithConfig() error = %v", err)
	}
	info, err := os.Stat(outputPath + ".avi")
	if err != nil {
		t.Fatalf("stat AVI: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("created AVI is empty")
	}
	if progressCalls != 1 {
		t.Fatalf("progress calls = %d, want 1", progressCalls)
	}
}

func testVideoImage() image.Image {
	img := image.NewRGBA(image.Rect(0, 0, testVideoPixel, testVideoPixel))
	img.SetRGBA(0, 0, color.RGBA{R: 255, G: 128, B: 64, A: 255})
	return img
}

func BenchmarkInterpolateCamera(b *testing.B) {
	start := benchmarkScene().Camera
	end := start
	end.Position.X += 1
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		benchmarkVideoCamera = render.InterpolateCamera(start, end, interpolationMid)
	}
}

func BenchmarkDefaultVideoEncodingConfig(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		benchmarkVideoConfig = render.DefaultVideoEncodingConfig()
	}
}

func BenchmarkVideoSequence(b *testing.B) {
	start := benchmarkScene().Camera
	end := start
	end.Position.X += 1
	sc := benchmarkScene()
	config := testConfig()
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		benchmarkVideoFrames, errBenchmarkVideo = render.VideoSequence(
			sc, start, end, testVideoDuration, testVideoFPS, config, nil,
		)
	}
}

func BenchmarkServiceVideoSequence(b *testing.B) {
	service := render.New(nil, nil)
	start := benchmarkScene().Camera
	end := start
	end.Position.X += 1
	sc := benchmarkScene()
	config := testConfig()
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		benchmarkVideoFrames, errBenchmarkVideo = service.VideoSequence(
			sc, start, end, testVideoDuration, testVideoFPS, config, nil,
		)
	}
}

func BenchmarkCreateVideoFromFrames(b *testing.B) {
	frame := testVideoImage()
	outputPath := filepath.Join(b.TempDir(), "benchmark.avi")
	frames := []image.Image{frame}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		errBenchmarkVideo = render.CreateVideoFromFrames(frames, outputPath, testVideoFPS, nil)
	}
}

var (
	benchmarkVideoCamera scene.Camera
	benchmarkVideoFrames []image.Image
	benchmarkVideoConfig render.VideoEncodingConfig
	errBenchmarkVideo    error
)

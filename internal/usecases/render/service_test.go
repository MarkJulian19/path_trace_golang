package render_test

import (
	"errors"
	"image"
	"image/color"
	"testing"

	"github.com/golang/mock/gomock"

	renderentity "github.com/MarkJulian19/path_trace_golang/internal/entities/render"
	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	render "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
	renderMocks "github.com/MarkJulian19/path_trace_golang/internal/usecases/render/mocks"
)

const (
	testObjectIndex    = 1
	testRenderMode     = 1
	testSigmaS         = 1.0
	testSigmaR         = 0.15
	testSmoothRadius   = 2
	testSmoothStrength = 0.5
	testColorRed       = 255
	testColorGreen     = 128
	testColorBlue      = 64
	testColorAlpha     = 255
)

var errGPUUnavailable = errors.New("GPU unavailable")

func TestServiceDefaultsToCPU(t *testing.T) {
	t.Parallel()
	service := render.New(nil, nil)
	if got := service.GetBackend(); got != render.BackendCPU {
		t.Fatalf("default backend = %v, want CPU", got)
	}
	if got := service.GetRenderMode(); got != 0 {
		t.Fatalf("nil GPU render mode = %d, want 0", got)
	}
}

func TestServiceNormalizesBackendAndCPUMode(t *testing.T) {
	t.Parallel()
	service := render.New(nil, nil)
	service.SetBackend(render.BackendGPU)
	if got := service.GetBackend(); got != render.BackendGPU {
		t.Fatalf("GPU backend = %v, want GPU", got)
	}

	service.SetBackend(render.Backend(99))
	if got := service.GetBackend(); got != render.BackendCPU {
		t.Fatalf("invalid backend = %v, want CPU", got)
	}

	service.SetCPURenderMode(-1)
	if got := service.GetCPURenderMode(); got != 0 {
		t.Fatalf("negative CPU mode = %d, want 0", got)
	}
	service.SetCPURenderMode(99)
	if got := service.GetCPURenderMode(); got != 1 {
		t.Fatalf("large CPU mode = %d, want 1", got)
	}
}

func TestServiceGPUBackendDelegatesRender(t *testing.T) {
	t.Parallel()
	controller := gomock.NewController(t)
	backend := renderMocks.NewMockGPUBackend(controller)
	service := render.New(backend, nil)
	service.SetBackend(render.BackendGPU)
	sc := benchmarkScene()
	config := testConfig()

	backend.EXPECT().Render(sc, config, gomock.Any(), gomock.Any()).DoAndReturn(
		func(_ *scene.Scene, _ renderentity.Config, target *image.RGBA, _ renderentity.Progress) error {
			target.SetRGBA(0, 0, colorForTest())
			return nil
		},
	)

	img := service.Render(sc, config)
	if got := img.Bounds().Dx(); got != config.Width {
		t.Fatalf("rendered image width = %d, want %d", got, config.Width)
	}
	if got := img.Bounds().Dy(); got != config.Height {
		t.Fatalf("rendered image height = %d, want %d", got, config.Height)
	}
}

func TestServiceSceneUsesSettings(t *testing.T) {
	t.Parallel()
	service := render.New(nil, nil)
	settings := scene.RenderSettings{
		Width:        testWidth,
		Height:       testHeight,
		SamplesPerPx: 1,
		MaxDepth:     1,
		MaxRayDist:   testMaxRayDist,
	}

	img, err := service.Scene(benchmarkScene(), settings)
	if err != nil {
		t.Fatalf("Scene() error = %v", err)
	}
	if got := img.Bounds().Size(); got.X != settings.Width || got.Y != settings.Height {
		t.Fatalf("service image size = %v, want %dx%d", got, settings.Width, settings.Height)
	}
}

func TestServiceFallsBackToCPUWhenGPUFails(t *testing.T) {
	t.Parallel()
	controller := gomock.NewController(t)
	backend := renderMocks.NewMockGPUBackend(controller)
	service := render.New(backend, nil)
	service.SetBackend(render.BackendGPU)
	sc := benchmarkScene()
	config := render.Config{
		Width:        testWidth,
		Height:       testHeight,
		SamplesPerPx: 1,
		MaxDepth:     1,
		MaxRayDist:   testMaxRayDist,
	}
	backend.EXPECT().Render(sc, config, gomock.Any(), gomock.Any()).Return(errGPUUnavailable)

	img := service.Render(sc, config)
	if got := img.Bounds().Size(); got.X != config.Width || got.Y != config.Height {
		t.Fatalf("fallback image size = %v, want %dx%d", got, config.Width, config.Height)
	}
}

func TestServiceForwardsGPUControls(t *testing.T) {
	t.Parallel()
	controller := gomock.NewController(t)
	controls := renderMocks.NewMockGPUControls(controller)
	service := render.New(nil, controls)

	controls.EXPECT().SetSelectedObjectFromUI(testObjectIndex)
	controls.EXPECT().SetLivePreview(true)
	controls.EXPECT().GetRenderMode().Return(testRenderMode)
	controls.EXPECT().SetRenderCancel(true)
	controls.EXPECT().SetDenoiseConfigFromUI(true, testSigmaS, testSigmaR)
	controls.EXPECT().SetSmoothConfigFromUI(true, testSmoothRadius, testSmoothStrength)
	controls.EXPECT().SetRenderModeFromUI(testRenderMode)

	service.SetSelectedObjectFromUI(testObjectIndex)
	service.SetLivePreview(true)
	if got := service.GetRenderMode(); got != testRenderMode {
		t.Fatalf("GPU render mode = %d, want %d", got, testRenderMode)
	}
	service.SetRenderCancel(true)
	service.SetDenoiseConfigFromUI(true, testSigmaS, testSigmaR)
	service.SetSmoothConfigFromUI(true, testSmoothRadius, testSmoothStrength)
	service.SetRenderModeFromUI(testRenderMode)
}

func colorForTest() color.RGBA {
	return color.RGBA{R: testColorRed, G: testColorGreen, B: testColorBlue, A: testColorAlpha}
}

type benchmarkGPUControls struct{}

func (benchmarkGPUControls) SetSelectedObjectFromUI(index int) {
	benchmarkControlState.selectedObject = index
}

func (benchmarkGPUControls) SetLivePreview(enabled bool) {
	benchmarkControlState.livePreview = enabled
}

func (benchmarkGPUControls) GetRenderMode() int { return benchmarkControlState.renderMode }

func (benchmarkGPUControls) SetRenderCancel(cancel bool) {
	benchmarkControlState.renderCancel = cancel
}

func (benchmarkGPUControls) SetDenoiseConfigFromUI(enabled bool, sigmaS, sigmaR float64) {
	benchmarkControlState.denoiseEnabled = enabled
	benchmarkControlState.sigmaS = sigmaS
	benchmarkControlState.sigmaR = sigmaR
}

func (benchmarkGPUControls) SetSmoothConfigFromUI(enabled bool, radius int, strength float64) {
	benchmarkControlState.smoothEnabled = enabled
	benchmarkControlState.smoothRadius = radius
	benchmarkControlState.smoothStrength = strength
}

func (benchmarkGPUControls) SetRenderModeFromUI(mode int) {
	benchmarkControlState.renderMode = mode
}

func BenchmarkServiceAPI(b *testing.B) {
	sc := benchmarkScene()
	config := testConfig()
	settings := scene.RenderSettings{
		Width:        config.Width,
		Height:       config.Height,
		SamplesPerPx: config.SamplesPerPx,
		MaxDepth:     config.MaxDepth,
		MaxRayDist:   float64(config.MaxRayDist),
	}

	b.Run("New", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkService = render.New(nil, nil)
		}
	})
	b.Run("SetBackend", func(b *testing.B) {
		service := render.New(nil, nil)
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			service.SetBackend(render.BackendGPU)
		}
	})
	b.Run("GetBackend", func(b *testing.B) {
		service := render.New(nil, nil)
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkBackend = service.GetBackend()
		}
	})
	b.Run("SetCPURenderMode", func(b *testing.B) {
		service := render.New(nil, nil)
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			service.SetCPURenderMode(testRenderMode)
		}
	})
	b.Run("GetCPURenderMode", func(b *testing.B) {
		service := render.New(nil, nil)
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkMode = service.GetCPURenderMode()
		}
	})
	b.Run("Render", func(b *testing.B) {
		service := render.New(nil, nil)
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkServiceImage = service.Render(sc, config)
		}
	})
	b.Run("Into", func(b *testing.B) {
		service := render.New(nil, nil)
		img := image.NewRGBA(image.Rect(0, 0, config.Width, config.Height))
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			service.Into(sc, config, img, nil)
		}
	})
	b.Run("Scene", func(b *testing.B) {
		service := render.New(nil, nil)
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkServiceImage, errBenchmarkService = service.Scene(sc, settings)
		}
	})
}

func BenchmarkServiceGPUControls(b *testing.B) {
	service := render.New(nil, benchmarkGPUControls{})
	b.Run("SetSelectedObjectFromUI", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			service.SetSelectedObjectFromUI(testObjectIndex)
		}
	})
	b.Run("SetLivePreview", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			service.SetLivePreview(true)
		}
	})
	b.Run("GetRenderMode", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkMode = service.GetRenderMode()
		}
	})
	b.Run("SetRenderCancel", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			service.SetRenderCancel(true)
		}
	})
	b.Run("SetDenoiseConfigFromUI", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			service.SetDenoiseConfigFromUI(true, testSigmaS, testSigmaR)
		}
	})
	b.Run("SetSmoothConfigFromUI", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			service.SetSmoothConfigFromUI(true, testSmoothRadius, testSmoothStrength)
		}
	})
	b.Run("SetRenderModeFromUI", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			service.SetRenderModeFromUI(testRenderMode)
		}
	})
}

var (
	benchmarkService      *render.Service
	benchmarkServiceImage image.Image
	errBenchmarkService   error
	benchmarkBackend      render.Backend
	benchmarkMode         int
	benchmarkControlState struct {
		sigmaS, sigmaR, smoothStrength float64
		selectedObject, renderMode     int
		smoothRadius                   int
		livePreview, renderCancel      bool
		denoiseEnabled, smoothEnabled  bool
	}
)

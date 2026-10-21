package render

import (
	"fmt"
	"image"
	"os"
	"sync"

	renderentity "github.com/MarkJulian19/path_trace_golang/internal/entities/render"
	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	"github.com/MarkJulian19/path_trace_golang/internal/renderfilter"
)

// Service управляет выбором средства рендера и связывает сценарий с адаптером GPU.
type Service struct {
	gpuBackend  GPUBackend
	gpuControls GPUControls

	backendMu sync.RWMutex
	backend   Backend

	cpuModeMu       sync.RWMutex
	cpuMode         int
	fallbackWarning sync.Once
	denoiseMu       sync.RWMutex
	denoise         renderfilter.Config
}

// New создаёт сценарий рендера с необязательным GPU-адаптером.
func New(gpuBackend GPUBackend, gpuControls GPUControls) *Service {
	return &Service{
		gpuBackend:  gpuBackend,
		gpuControls: gpuControls,
		backend:     BackendCPU,
		denoise:     renderfilter.Normalize(renderfilter.Config{Enabled: true}),
	}
}

// SetBackend выбирает средство построения изображения.
func (s *Service) SetBackend(backend Backend) {
	s.backendMu.Lock()
	defer s.backendMu.Unlock()

	if backend != BackendGPU {
		backend = BackendCPU
	}
	s.backend = backend
}

// GetBackend возвращает выбранное средство построения изображения.
func (s *Service) GetBackend() Backend {
	s.backendMu.RLock()
	defer s.backendMu.RUnlock()

	return s.backend
}

// PrepareGPU создаёт оконные ресурсы в главном потоке до фонового рендера.
// Ошибку повторно возвращает сам GPU-адаптер при построении кадра.
func (s *Service) PrepareGPU() {
	initializer, ok := s.gpuBackend.(GPUInitializer)
	if !ok {
		return
	}
	err := initializer.Initialize()
	if err != nil {
		s.warnGPUFallback(err)
	}
}

// Render строит изображение сцены и возвращает новый буфер изображения.
func (s *Service) Render(sc *scene.Scene, cfg renderentity.Config) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
	s.Into(sc, cfg, img, nil)
	return img
}

// Into строит сцену в переданном изображении и сообщает о ходе работы.
func (s *Service) Into(
	sc *scene.Scene,
	cfg renderentity.Config,
	img *image.RGBA,
	progress renderentity.Progress,
) {
	s.IntoWithCancel(sc, cfg, img, progress, nil)
}

// IntoWithCancel строит сцену и позволяет интерактивному клиенту отменить
// текущий CPU/GPU job до запуска следующего.
func (s *Service) IntoWithCancel(
	sc *scene.Scene,
	cfg renderentity.Config,
	img *image.RGBA,
	progress renderentity.Progress,
	cancelled func() bool,
) Backend {
	if s.GetBackend() == BackendGPU {
		return s.renderIntoGPU(sc, cfg, img, progress, cancelled)
	}

	s.renderCPUFiltered(sc, cfg, img, progress, cancelled)
	return BackendCPU
}

// Scene строит изображение сцены по настройкам качества.
func (s *Service) Scene(sc *scene.Scene, settings scene.RenderSettings) (image.Image, error) {
	cfg := renderentity.Config{
		Width:        settings.Width,
		Height:       settings.Height,
		SamplesPerPx: settings.SamplesPerPx,
		MaxDepth:     settings.MaxDepth,
		MaxRayDist:   float32(settings.MaxRayDist),
	}

	return s.Render(sc, cfg), nil
}

// SetCPURenderMode устанавливает режим рендеринга CPU.
func (s *Service) SetCPURenderMode(mode int) {
	s.cpuModeMu.Lock()
	defer s.cpuModeMu.Unlock()

	s.cpuMode = clampRenderMode(mode)
}

// GetCPURenderMode возвращает текущий режим рендеринга CPU.
func (s *Service) GetCPURenderMode() int {
	s.cpuModeMu.RLock()
	defer s.cpuModeMu.RUnlock()

	return s.cpuMode
}

// SetSelectedObjectFromUI передаёт GPU индекс выбранного объекта.
func (s *Service) SetSelectedObjectFromUI(index int) {
	if s.gpuControls != nil {
		s.gpuControls.SetSelectedObjectFromUI(index)
	}
}

// SetLivePreview включает или выключает промежуточное обновление изображения.
func (s *Service) SetLivePreview(enabled bool) {
	if s.gpuControls != nil {
		s.gpuControls.SetLivePreview(enabled)
	}
}

// GetRenderMode возвращает текущий режим GPU-рендера.
func (s *Service) GetRenderMode() int {
	if s.gpuControls == nil {
		return 0
	}
	return s.gpuControls.GetRenderMode()
}

// SetRenderCancel устанавливает флаг отмены GPU-рендера.
func (s *Service) SetRenderCancel(cancel bool) {
	if s.gpuControls != nil {
		s.gpuControls.SetRenderCancel(cancel)
	}
}

// SetDenoiseConfigFromUI задаёт параметры подавления шума.
func (s *Service) SetDenoiseConfigFromUI(enabled bool, sigmaS, sigmaR float64) {
	s.denoiseMu.Lock()
	s.denoise = renderfilter.Normalize(renderfilter.Config{Enabled: enabled, SigmaS: sigmaS, SigmaR: sigmaR})
	s.denoiseMu.Unlock()
	if s.gpuControls != nil {
		s.gpuControls.SetDenoiseConfigFromUI(enabled, sigmaS, sigmaR)
	}
}

// SetSmoothConfigFromUI задаёт параметры дополнительного сглаживания.
func (s *Service) SetSmoothConfigFromUI(enabled bool, radius int, strength float64) {
	if s.gpuControls != nil {
		s.gpuControls.SetSmoothConfigFromUI(enabled, radius, strength)
	}
}

// SetRenderModeFromUI устанавливает режим GPU-рендера.
func (s *Service) SetRenderModeFromUI(mode int) {
	if s.gpuControls != nil {
		s.gpuControls.SetRenderModeFromUI(mode)
	}
}

func (s *Service) renderIntoGPU(
	sc *scene.Scene,
	cfg renderentity.Config,
	img *image.RGBA,
	progress renderentity.Progress,
	cancelled func() bool,
) Backend {
	if s.gpuBackend == nil {
		s.renderCPUFiltered(sc, cfg, img, progress, cancelled)
		return BackendCPU
	}

	var err error
	if backend, ok := s.gpuBackend.(GPUBackendCancelable); ok {
		err = backend.RenderWithCancel(sc, cfg, img, progress, cancelled)
	} else {
		err = s.gpuBackend.Render(sc, cfg, img, progress)
	}
	if err != nil {
		s.warnGPUFallback(err)
		s.renderCPUFiltered(sc, cfg, img, progress, cancelled)
		return BackendCPU
	}
	return BackendGPU
}

func (s *Service) warnGPUFallback(err error) {
	s.fallbackWarning.Do(func() {
		fmt.Fprintf(os.Stderr, "GPU render error: %v\nFalling back to CPU renderer.\n", err)
	})
}

func clampRenderMode(mode int) int {
	if mode < 0 {
		return 0
	}
	if mode > 1 {
		return 1
	}
	return mode
}

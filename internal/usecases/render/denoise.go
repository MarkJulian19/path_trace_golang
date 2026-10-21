package render

import (
	"image"

	renderentity "github.com/MarkJulian19/path_trace_golang/internal/entities/render"
	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	"github.com/MarkJulian19/path_trace_golang/internal/renderfilter"
)

func (s *Service) renderCPUFiltered(sc *scene.Scene, cfg Config, img *image.RGBA,
	progress renderentity.Progress, cancelled func() bool,
) {
	if img.Bounds().Dx() != cfg.Width || img.Bounds().Dy() != cfg.Height {
		return
	}
	mode := s.GetCPURenderMode()
	s.denoiseMu.RLock()
	filter := s.denoise
	s.denoiseMu.RUnlock()
	if mode != 0 || !filter.Enabled || cfg.SamplesPerPx < renderRGBAComponents {
		renderIntoCPUWithCancel(sc, cfg, img, progress, mode, cancelled)
		return
	}
	intermediate := incompleteCPUProgress(progress)
	renderIntoCPUWithCancel(sc, cfg, img, intermediate, mode, cancelled)
	if isCancelled(cancelled) {
		return
	}
	complete := renderfilter.ApplyRGBA(img, filter, cancelled)
	if complete && progress != nil && !isCancelled(cancelled) {
		progress(cfg.SamplesPerPx, cfg.SamplesPerPx)
	}
}

func incompleteCPUProgress(progress renderentity.Progress) func(int, int) {
	if progress == nil {
		return nil
	}
	return func(current, total int) {
		if current < total {
			progress(current, total)
		}
	}
}

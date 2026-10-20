package ui

import (
	"fmt"
	"image"
	"image/color"
	"log"
	"sync"
	"time"

	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/widget"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	engine "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

type renderController struct {
	sc      *scene.Scene
	service *engine.Service
	preview *scene.RenderSettings
	final   *scene.RenderSettings

	img          **image.RGBA
	imgCanvas    *canvas.Image
	status       *widget.Label
	progressBar  *widget.ProgressBar
	timeLabel    *widget.Label
	uiDo         func(func())
	publish      func()
	mu           *sync.Mutex
	stopCh       *chan struct{}
	renderTimer  **time.Timer
	lastImage    *image.Image
	lastConfig   *engine.Config
	wireframeFPS *float64
	liveUpdate   *widget.Check

	denoiseEnabled *bool
	denoiseSigmaS  *float64
	denoiseSigmaR  *float64
	smoothEnabled  *bool
	smoothRadius   *int
	smoothStrength *float64
}

type renderProgressState struct {
	cfg     engine.Config
	started time.Time
	last    time.Time
	live    bool
}

func (c *renderController) doRender(final bool) {
	go func() {
		log.Println("render goroutine started, final =", final)
		if isWireframeMode(c.service) && !final {
			c.streamWireframe()
			return
		}
		c.renderFrame(final)
	}()
}

func (c *renderController) streamWireframe() {
	c.uiDo(func() { c.status.SetText("Streaming wireframe...") })
	cfg := engine.Config{
		Width:        c.preview.Width,
		Height:       c.preview.Height,
		SamplesPerPx: 1,
		MaxDepth:     1,
		MaxRayDist:   float32(c.preview.MaxRayDist),
	}
	c.prepareImage(cfg, true)
	frameDuration := c.wireframeFrameDuration()
	ticker := time.NewTicker(frameDuration)
	defer func() { ticker.Stop() }()
	for {
		select {
		case <-*c.stopCh:
			c.uiDo(func() { c.status.SetText("Streaming stopped") })
			return
		case <-ticker.C:
			if c.wireframeStopped() {
				return
			}
			c.renderWireframeFrame(cfg)
			newDuration := c.wireframeFrameDuration()
			if newDuration != frameDuration {
				frameDuration = newDuration
				ticker.Stop()
				ticker = time.NewTicker(frameDuration)
			}
		}
	}
}

func (c *renderController) wireframeFrameDuration() time.Duration {
	c.mu.Lock()
	fps := *c.wireframeFPS
	c.mu.Unlock()
	return time.Duration(float64(time.Second) / fps)
}

func (c *renderController) wireframeStopped() bool {
	select {
	case <-*c.stopCh:
		c.uiDo(func() { c.status.SetText("Streaming stopped") })
		return true
	default:
	}
	c.mu.Lock()
	active := isWireframeMode(c.service)
	c.mu.Unlock()
	if !active {
		c.uiDo(func() { c.status.SetText("Mode changed, stopping stream") })
	}
	return !active
}

func (c *renderController) renderWireframeFrame(cfg engine.Config) {
	c.mu.Lock()
	renderImg := *c.img
	c.mu.Unlock()
	c.service.Into(c.sc, cfg, renderImg, func(int, int) {})
	c.publish()
}

func (c *renderController) renderFrame(final bool) {
	c.uiDo(func() { c.status.SetText("Rendering...") })
	started := time.Now()
	cfg := c.config(final)
	c.prepareImage(cfg, false)
	c.uiDo(func() {
		c.progressBar.SetValue(0)
		c.progressBar.Show()
		c.timeLabel.SetText("Render time: 0.00s")
	})
	live := c.liveUpdate.Checked
	progressState := &renderProgressState{cfg: cfg, started: started, last: started, live: live}
	progress := func(currentSample, totalSamples int) {
		c.updateProgress(progressState, currentSample, totalSamples)
	}
	c.configureGPU(live)
	c.service.Into(c.sc, cfg, *c.img, progress)
	select {
	case <-*c.stopCh:
		return
	default:
	}
	if !live {
		c.publish()
	}
	if final {
		c.saveFinal(cfg)
	}
	c.finishRender(started)
}

func (c *renderController) config(final bool) engine.Config {
	settings := c.preview
	if final {
		settings = c.final
	}
	return engine.Config{
		Width:        settings.Width,
		Height:       settings.Height,
		SamplesPerPx: settings.SamplesPerPx,
		MaxDepth:     settings.MaxDepth,
		MaxRayDist:   float32(settings.MaxRayDist),
	}
}

func (c *renderController) prepareImage(cfg engine.Config, wireframe bool) {
	c.mu.Lock()
	needCanvasBind := false
	current := *c.img
	if current.Bounds().Dx() != cfg.Width || current.Bounds().Dy() != cfg.Height {
		current = image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
		*c.img = current
		needCanvasBind = true
	}
	if wireframe {
		fillImage(current, color.RGBA{25, 25, 25, 255})
	} else if !isWireframeMode(c.service) {
		fillImage(current, color.RGBA{0, 0, 0, 255})
	}
	c.mu.Unlock()
	if needCanvasBind {
		c.uiDo(func() { c.imgCanvas.Image = current })
	}
}

func fillImage(img *image.RGBA, value color.RGBA) {
	for y := range img.Bounds().Dy() {
		for x := range img.Bounds().Dx() {
			img.Set(x, y, value)
		}
	}
}

func (c *renderController) updateProgress(state *renderProgressState, currentSample, totalSamples int) {
	select {
	case <-*c.stopCh:
		if c.service.GetBackend() == engine.BackendGPU {
			c.service.SetRenderCancel(true)
		}
		return
	default:
	}
	totalSamples = max(totalSamples, 1)
	currentSample = min(currentSample, totalSamples)
	progressValue := float64(currentSample) / float64(totalSamples)
	timeText := fmt.Sprintf("Render time: %.2fs", time.Since(state.started).Seconds())
	refresh := c.shouldRefresh(state, currentSample, totalSamples)
	c.uiDo(func() {
		c.progressBar.SetValue(progressValue)
		c.timeLabel.SetText(timeText)
	})
	if refresh {
		c.publish()
	}
}

func (c *renderController) shouldRefresh(state *renderProgressState, currentSample, totalSamples int) bool {
	pixelCount := state.cfg.Width * state.cfg.Height
	if state.live {
		interval := progressInterval(pixelCount)
		refresh := time.Since(state.last) >= interval || currentSample == totalSamples
		if refresh {
			state.last = time.Now()
		}
		return refresh
	}
	if pixelCount <= 2000*2000 {
		return false
	}
	refresh := time.Since(state.last) >= uiFrameMillis*time.Millisecond || currentSample == totalSamples
	if refresh {
		state.last = time.Now()
	}
	return refresh
}

func progressInterval(pixelCount int) time.Duration {
	switch {
	case pixelCount > 2000*2000:
		return uiPreviewIntervalSlow * time.Millisecond
	case pixelCount > 1000*1000:
		return uiPreviewIntervalNormal * time.Millisecond
	default:
		return uiProgressUpdateInterval
	}
}

func (c *renderController) configureGPU(live bool) {
	if c.service.GetBackend() != engine.BackendGPU {
		return
	}
	c.service.SetDenoiseConfigFromUI(*c.denoiseEnabled, *c.denoiseSigmaS, *c.denoiseSigmaR)
	c.service.SetSmoothConfigFromUI(*c.smoothEnabled, *c.smoothRadius, *c.smoothStrength)
	c.service.SetLivePreview(live)
	c.service.SetRenderCancel(false)
}

func (c *renderController) saveFinal(cfg engine.Config) {
	c.mu.Lock()
	cloned := image.NewRGBA((*c.img).Bounds())
	copy(cloned.Pix, (*c.img).Pix)
	*c.lastImage = cloned
	*c.lastConfig = cfg
	c.mu.Unlock()
}

func (c *renderController) finishRender(started time.Time) {
	doneText := fmt.Sprintf("Render time: %.2fs", time.Since(started).Seconds())
	c.uiDo(func() {
		c.progressBar.SetValue(1)
		c.timeLabel.SetText(doneText)
		c.status.SetText("Done")
	})
	log.Println("render finished")
}

func (c *renderController) clearFinalImage() {
	c.mu.Lock()
	*c.lastImage = nil
	c.mu.Unlock()
}

func (c *renderController) startRender(final bool) {
	c.mu.Lock()
	if *c.renderTimer != nil {
		(*c.renderTimer).Stop()
		*c.renderTimer = nil
	}
	if *c.stopCh != nil {
		close(*c.stopCh)
	}
	*c.stopCh = make(chan struct{})
	c.mu.Unlock()
	if final {
		c.doRender(true)
		return
	}
	c.mu.Lock()
	*c.renderTimer = time.AfterFunc(uiValue300*time.Millisecond, func() {
		c.mu.Lock()
		*c.renderTimer = nil
		c.mu.Unlock()
		c.doRender(false)
	})
	c.mu.Unlock()
}

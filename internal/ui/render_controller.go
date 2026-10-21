package ui

import (
	"fmt"
	"image"
	"image/color"
	"log"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/widget"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	engine "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

type renderController struct {
	lastInteractive time.Time

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
	mu           *sync.Mutex
	lastImage    *image.Image
	lastConfig   *engine.Config
	wireframeFPS *float64

	denoiseEnabled *bool
	denoiseSigmaS  *float64
	denoiseSigmaR  *float64
	smoothEnabled  *bool
	smoothRadius   *int
	smoothStrength *float64

	currentJob            *renderJob
	latestJob             *renderJob
	pendingJob            *pendingRender
	renderTimer           *time.Timer
	interactiveTimer      *time.Timer
	pendingPreview        *previewUpdate
	displayedPreview      *image.RGBA
	previewPool           []*image.RGBA
	interactiveGeneration uint64

	jobMu sync.Mutex

	previewMu   sync.Mutex
	livePreview atomic.Bool

	previewScheduled bool
	closed           bool
}

// renderJob хранит отмену задания; следующее задание ждёт освобождения рендера.
type renderJob struct {
	cancel    chan struct{}
	done      chan struct{}
	delay     time.Duration
	final     bool
	started   bool
	cancelled bool
}

type pendingRender struct {
	final bool
	delay time.Duration
}

type previewUpdate struct {
	job *renderJob
	img *image.RGBA
}

type renderProgressState struct {
	started time.Time
	last    time.Time
	job     *renderJob
	img     *image.RGBA
	cfg     engine.Config

	backend engine.Backend
	mu      sync.Mutex
	live    bool
}

func newRenderJob(final bool, delay time.Duration) *renderJob {
	return &renderJob{cancel: make(chan struct{}), done: make(chan struct{}), final: final, delay: delay}
}

func (c *renderController) doRender(job *renderJob) {
	defer c.finishJob(job)
	log.Println("render goroutine started, final =", job.final)
	if c.jobCancelled(job) {
		return
	}
	snapshot, ok := c.captureSnapshot(job)
	if !ok {
		return
	}
	if snapshot.wireframe && !job.final {
		c.streamWireframe(job, snapshot)
		return
	}
	c.renderFrame(job, snapshot)
}

func (c *renderController) streamWireframe(job *renderJob, snapshot renderSnapshot) {
	c.updateUI(job, func() { c.status.SetText("Streaming wireframe...") })
	cfg := snapshot.cfg
	cfg.SamplesPerPx, cfg.MaxDepth = 1, 1
	renderImg := c.prepareImage(job, cfg, true)
	frameDuration := snapshot.interval
	ticker := time.NewTicker(frameDuration)
	defer func() { ticker.Stop() }()
	c.renderWireframeFrame(job, &snapshot.scene, cfg, renderImg)
	for {
		select {
		case <-job.cancel:
			return
		case <-ticker.C:
			next, ok := c.captureSnapshot(job)
			if !ok || !next.wireframe {
				return
			}
			c.renderWireframeFrame(job, &next.scene, cfg, renderImg)
			newDuration := next.interval
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
	fps = max(fps, 1)
	return time.Duration(float64(time.Second) / fps)
}

func (c *renderController) renderWireframeFrame(
	job *renderJob, sc *scene.Scene, cfg engine.Config, renderImg *image.RGBA,
) {
	// Предыдущий интерактивный рендер мог оставить флаг отмены GPU. Обычный
	// рендер сбрасывает его в configureRenderer, а поток wireframe обходит этот путь.
	if c.service.GetBackend() == engine.BackendGPU {
		c.service.SetRenderCancel(false)
	}
	progress := func(currentSample, totalSamples int) {
		if currentSample < totalSamples {
			return
		}
		c.updateUI(job, func() {
			c.progressBar.SetValue(1)
			c.progressBar.Show()
		})
	}
	backend := c.service.IntoWithCancel(sc, cfg, renderImg, progress, func() bool { return c.jobCancelled(job) })
	if !c.jobCancelled(job) {
		if backend == engine.BackendCPU && c.service.GetBackend() == engine.BackendGPU {
			c.updateUI(job, func() { c.status.SetText("Streaming wireframe (CPU fallback)...") })
		}
		c.publishJob(job, renderImg)
	}
}

func (c *renderController) renderFrame(job *renderJob, snapshot renderSnapshot) {
	c.updateUI(job, func() { c.status.SetText("Rendering...") })
	started := time.Now()
	cfg := snapshot.cfg
	renderImg := c.prepareImage(job, cfg, false)
	c.updateUI(job, func() {
		c.progressBar.SetValue(0)
		c.progressBar.Show()
		c.timeLabel.SetText("Render time: 0.00s")
	})
	live := snapshot.live
	progressState := &renderProgressState{
		job: job, img: renderImg, cfg: cfg, started: started, last: started,
		live: live, backend: snapshot.backend,
	}
	progress := func(currentSample, totalSamples int) {
		c.updateProgress(progressState, currentSample, totalSamples)
	}
	backend := c.service.IntoWithCancel(
		&snapshot.scene, cfg, renderImg, progress, func() bool { return c.jobCancelled(job) },
	)
	if c.jobCancelled(job) {
		return
	}
	c.publishJob(job, renderImg)
	if job.final {
		c.saveFinal(job, cfg, renderImg)
	}
	c.finishRender(job, started, snapshot.backend == engine.BackendGPU && backend == engine.BackendCPU)
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

func (c *renderController) prepareImage(job *renderJob, cfg engine.Config, wireframe bool) *image.RGBA {
	current := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
	value := color.RGBA{0, 0, 0, 255}
	if wireframe {
		value = color.RGBA{25, 25, 25, 255}
	}
	fillImage(current, value)
	c.mu.Lock()
	*c.img = current
	c.mu.Unlock()
	c.publishJob(job, current)
	return current
}

func fillImage(img *image.RGBA, value color.RGBA) {
	for y := range img.Bounds().Dy() {
		for x := range img.Bounds().Dx() {
			img.Set(x, y, value)
		}
	}
}

func (c *renderController) updateProgress(state *renderProgressState, currentSample, totalSamples int) {
	if c.jobCancelled(state.job) {
		if state.backend == engine.BackendGPU {
			c.service.SetRenderCancel(true)
		}
		return
	}
	totalSamples = max(totalSamples, 1)
	currentSample = min(currentSample, totalSamples)
	progressValue := float64(currentSample) / float64(totalSamples)
	timeText := fmt.Sprintf("Render time: %.2fs", time.Since(state.started).Seconds())
	refresh := c.shouldRefresh(state, currentSample, totalSamples)
	if !refresh {
		return
	}
	c.updateUI(state.job, func() {
		c.progressBar.SetValue(progressValue)
		c.timeLabel.SetText(timeText)
	})
	// CPU публикует законченные плитки последовательно под блокировкой.
	if state.live {
		c.publishJob(state.job, state.img)
	}
}

func (c *renderController) shouldRefresh(state *renderProgressState, currentSample, totalSamples int) bool {
	state.mu.Lock()
	defer state.mu.Unlock()
	pixelCount := state.cfg.Width * state.cfg.Height
	if state.live {
		interval := progressInterval(pixelCount)
		refresh := time.Since(state.last) >= interval || currentSample == totalSamples
		if refresh {
			state.last = time.Now()
		}
		return refresh
	}
	refresh := time.Since(state.last) >= uiProgressUpdateInterval || currentSample == totalSamples
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

func (c *renderController) configureRenderer(live bool) {
	c.service.SetDenoiseConfigFromUI(*c.denoiseEnabled, *c.denoiseSigmaS, *c.denoiseSigmaR)
	if c.service.GetBackend() != engine.BackendGPU {
		return
	}
	c.service.PrepareGPU()
	c.service.SetSmoothConfigFromUI(*c.smoothEnabled, *c.smoothRadius, *c.smoothStrength)
	c.service.SetLivePreview(live)
	c.service.SetRenderCancel(false)
}

func (c *renderController) saveFinal(job *renderJob, cfg engine.Config, renderImg *image.RGBA) {
	if !c.jobActive(job) {
		return
	}
	cloned := image.NewRGBA(renderImg.Bounds())
	copy(cloned.Pix, renderImg.Pix)
	c.jobMu.Lock()
	defer c.jobMu.Unlock()
	if c.currentJob != job || job.cancelled {
		return
	}
	c.mu.Lock()
	*c.lastImage = cloned
	*c.lastConfig = cfg
	c.mu.Unlock()
}

func (c *renderController) finishRender(job *renderJob, started time.Time, fallback bool) {
	if !c.jobActive(job) {
		return
	}
	doneText := fmt.Sprintf("Render time: %.2fs", time.Since(started).Seconds())
	c.updateUI(job, func() {
		c.progressBar.SetValue(1)
		c.timeLabel.SetText(doneText)
		c.status.SetText("Done")
		if fallback {
			c.status.SetText("Done (CPU fallback)")
		}
	})
	log.Println("render finished")
}

func (c *renderController) clearFinalImage() {
	c.mu.Lock()
	*c.lastImage = nil
	c.mu.Unlock()
}

func (c *renderController) jobCancelled(job *renderJob) bool {
	select {
	case <-job.cancel:
		return true
	default:
		return false
	}
}

func (c *renderController) jobActive(job *renderJob) bool {
	c.jobMu.Lock()
	active := c.currentJob == job && !job.cancelled
	c.jobMu.Unlock()
	return active
}

func (c *renderController) previewJobValid(job *renderJob) bool {
	c.jobMu.Lock()
	valid := !job.cancelled && c.latestJob == job
	c.jobMu.Unlock()
	return valid
}

func (c *renderController) updateUI(job *renderJob, fn func()) {
	if !c.jobActive(job) {
		return
	}
	c.uiDo(func() {
		if c.previewJobValid(job) {
			fn()
		}
	})
}

func (c *renderController) publishJob(job *renderJob, source *image.RGBA) {
	if !c.jobActive(job) {
		return
	}
	c.previewMu.Lock()
	if c.pendingPreview != nil {
		c.previewPool = append(c.previewPool, c.pendingPreview.img)
	}
	frame := c.acquirePreviewFrameLocked(source.Bounds())
	copy(frame.Pix, source.Pix)
	c.pendingPreview = &previewUpdate{job: job, img: frame}
	schedule := !c.previewScheduled
	c.previewScheduled = true
	c.previewMu.Unlock()
	if schedule {
		c.uiDo(c.flushPreview)
	}
}

func (c *renderController) acquirePreviewFrameLocked(bounds image.Rectangle) *image.RGBA {
	for index, frame := range slices.Backward(c.previewPool) {
		if frame.Bounds() == bounds {
			c.previewPool = append(c.previewPool[:index], c.previewPool[index+1:]...)
			return frame
		}
	}
	c.previewPool = nil
	return image.NewRGBA(bounds)
}

func (c *renderController) flushPreview() {
	c.previewMu.Lock()
	update := c.pendingPreview
	c.pendingPreview = nil
	c.previewScheduled = false
	if update == nil || !c.previewJobValid(update.job) {
		if update != nil {
			c.previewPool = append(c.previewPool, update.img)
		}
		c.previewMu.Unlock()
		return
	}
	oldDisplayed := c.displayedPreview
	c.displayedPreview = update.img
	if oldDisplayed != nil {
		c.previewPool = append(c.previewPool, oldDisplayed)
	}
	c.previewMu.Unlock()
	c.imgCanvas.Image = update.img
	c.imgCanvas.Refresh()
}

func (c *renderController) cancelJobLocked(job *renderJob) {
	if job == nil || job.cancelled {
		return
	}
	job.cancelled = true
	close(job.cancel)
	if !job.started {
		close(job.done)
	}
}

func (c *renderController) stopTimerLocked() {
	if c.renderTimer != nil {
		c.renderTimer.Stop()
		c.renderTimer = nil
	}
}

func (c *renderController) stopInteractiveTimerLocked() {
	c.interactiveGeneration++
	if c.interactiveTimer != nil {
		c.interactiveTimer.Stop()
		c.interactiveTimer = nil
	}
}

func (c *renderController) scheduleJobLocked(job *renderJob) bool {
	if job.final || job.delay <= 0 {
		return true
	}
	c.renderTimer = time.AfterFunc(job.delay, func() {
		c.launchJob(job)
	})
	return false
}

func (c *renderController) launchJob(job *renderJob) {
	c.jobMu.Lock()
	if c.currentJob != job || job.cancelled || job.started {
		c.jobMu.Unlock()
		return
	}
	job.started = true
	c.stopTimerLocked()
	c.jobMu.Unlock()
	go c.doRender(job)
}

func (c *renderController) finishJob(job *renderJob) {
	c.jobMu.Lock()
	if c.currentJob != job {
		c.jobMu.Unlock()
		return
	}
	c.currentJob = nil
	pending := c.pendingJob
	c.pendingJob = nil
	var next *renderJob
	if pending != nil {
		next = newRenderJob(pending.final, pending.delay)
		c.currentJob = next
		c.latestJob = next
		if !c.scheduleJobLocked(next) {
			next = nil
		}
	}
	c.jobMu.Unlock()
	close(job.done)
	if next != nil {
		c.launchJob(next)
	}
}

func (c *renderController) startRender(final bool) {
	delay := uiRenderDebounce
	if final {
		delay = 0
	}
	c.startRenderWithDelay(final, delay)
}

func (c *renderController) startRenderWithDelay(final bool, delay time.Duration) {
	c.jobMu.Lock()
	if c.closed {
		c.jobMu.Unlock()
		return
	}
	c.stopInteractiveTimerLocked()
	c.stopTimerLocked()
	if c.currentJob != nil {
		if !c.currentJob.started {
			c.cancelJobLocked(c.currentJob)
			c.currentJob = newRenderJob(final, delay)
			c.latestJob = c.currentJob
			next := c.currentJob
			immediate := c.scheduleJobLocked(next)
			c.jobMu.Unlock()
			if immediate {
				c.launchJob(next)
			}
			return
		}
		c.cancelJobLocked(c.currentJob)
		c.pendingJob = &pendingRender{final: final, delay: delay}
		c.service.SetRenderCancel(true)
		c.jobMu.Unlock()
		return
	}
	job := newRenderJob(final, delay)
	c.currentJob = job
	c.latestJob = job
	immediate := c.scheduleJobLocked(job)
	c.jobMu.Unlock()
	if immediate {
		c.launchJob(job)
	}
}

// startInteractiveRender запускает превью сразу, но не чаще заданного UI-интервала.
// Это сохраняет отзывчивость перетаскивания и не создаёт рендер на каждое событие мыши.
func (c *renderController) startInteractiveRender() {
	// Wireframe уже работает непрерывным потоком. Перезапуск на каждое
	// движение объекта очищает кадр и создаёт лишнюю очередь рендеров.
	if isWireframeMode(c.service) {
		return
	}
	c.jobMu.Lock()
	if c.closed {
		c.jobMu.Unlock()
		return
	}
	wait := time.Duration(0)
	if !c.lastInteractive.IsZero() {
		wait = max(uiInteractiveRenderRate-time.Since(c.lastInteractive), 0)
	}
	if wait <= 0 {
		c.lastInteractive = time.Now()
		c.jobMu.Unlock()
		c.startRenderWithDelay(false, 0)
		return
	}
	if c.interactiveTimer == nil {
		c.scheduleInteractiveTimerLocked(wait)
	}
	c.jobMu.Unlock()
}

func (c *renderController) scheduleInteractiveTimerLocked(wait time.Duration) {
	generation := c.interactiveGeneration
	c.interactiveTimer = time.AfterFunc(wait, func() {
		c.runInteractiveTimer(generation)
	})
}

func (c *renderController) runInteractiveTimer(generation uint64) {
	c.jobMu.Lock()
	if generation != c.interactiveGeneration || c.closed {
		c.jobMu.Unlock()
		return
	}
	c.interactiveTimer = nil
	c.lastInteractive = time.Now()
	c.jobMu.Unlock()
	c.startRenderWithDelay(false, 0)
}

func (c *renderController) stop() {
	c.jobMu.Lock()
	defer c.jobMu.Unlock()
	c.closed = true
	c.stopTimerLocked()
	c.stopInteractiveTimerLocked()
	c.pendingJob = nil
	c.cancelJobLocked(c.currentJob)
	c.service.SetRenderCancel(true)
}

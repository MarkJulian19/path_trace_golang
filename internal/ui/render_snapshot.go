package ui

import (
	"slices"
	"time"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	engine "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

type renderSnapshot struct {
	scene     scene.Scene
	cfg       engine.Config
	interval  time.Duration
	backend   engine.Backend
	wireframe bool
	live      bool
}

// captureSnapshot читает редактируемую сцену в очереди интерфейса, чтобы
// рендер не обращался к полям, которые пользователь меняет во время работы.
func (c *renderController) captureSnapshot(job *renderJob) (renderSnapshot, bool) {
	ready := make(chan renderSnapshot, 1)
	c.uiDo(func() {
		if !c.jobActive(job) {
			return
		}
		live := c.livePreview.Load()
		c.configureRenderer(live)
		ready <- renderSnapshot{
			scene: cloneScene(c.sc), cfg: c.config(job.final), interval: c.wireframeFrameDuration(),
			backend: c.service.GetBackend(), wireframe: isWireframeMode(c.service), live: live,
		}
	})
	select {
	case snapshot := <-ready:
		return snapshot, !c.jobCancelled(job)
	case <-job.cancel:
		return renderSnapshot{}, false
	}
}

func cloneScene(source *scene.Scene) scene.Scene {
	cloned := *source
	cloned.Objects = slices.Clone(source.Objects)
	cloned.Materials = slices.Clone(source.Materials)
	if source.Sky != nil {
		cloned.Sky = new(*source.Sky)
	}
	if source.Fog != nil {
		cloned.Fog = new(*source.Fog)
	}
	return cloned
}

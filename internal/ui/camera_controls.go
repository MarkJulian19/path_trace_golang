package ui

import (
	"fmt"
	"math"
	"sync"
	"time"

	"fyne.io/fyne/v2"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	engine "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

type cameraController struct {
	posX             cameraEntry
	posY             cameraEntry
	posZ             cameraEntry
	lookX            cameraEntry
	lookY            cameraEntry
	lookZ            cameraEntry
	active           *bool
	camera           *scene.Camera
	scene            *scene.Scene
	service          *engine.Service
	mu               *sync.Mutex
	start            func(bool)
	interactiveStart func()
	viewportUpdated  func()

	selectedObject    *int
	transformMode     *editorTransformMode
	objectMoveEnabled *bool
	objectStep        *float64
	objectUpdated     func()

	keys       map[fyne.KeyName]bool
	lastKeys   map[fyne.KeyName]time.Time
	done       chan struct{}
	moveSpeed  float64
	rotSpeed   float64
	keyTimeout time.Duration

	keysMu sync.RWMutex
}

func startSessionCamera(w fyne.Window, s *uiSession) func() {
	return startCameraControlWithTransform(w.Canvas(), cameraControllerOptions{
		active:           &s.camControlActive,
		selectedObject:   &s.selectedObject,
		camera:           &s.cam,
		scene:            s.sc,
		service:          s.service,
		mu:               s.mu,
		start:            s.startRender,
		interactiveStart: s.startInteractiveRender,
		viewportUpdated:  s.refreshGizmo,
		entries: [6]cameraEntry{
			s.cameraEntries[0], s.cameraEntries[1], s.cameraEntries[2],
			s.cameraEntries[3], s.cameraEntries[4], s.cameraEntries[5],
		},
		transformMode:     &s.transformMode,
		objectMoveEnabled: &s.objectMoveEnabled,
		objectStep:        &s.transformStep,
		objectUpdated: func() {
			s.objectUI.refreshSelected()
			s.refreshGizmo()
			s.clearFinalImage()
			s.runtime.status.SetText("Moved object: " + s.sc.Objects[s.selectedObject].ID)
		},
	})
}

type cameraEntry interface {
	SetText(string)
}

type cameraKeys struct {
	w, s, a, d, q, e      bool
	left, right, up, down bool
	pageUp, pageDown      bool
}

type cameraBasis struct {
	dirX, dirY, dirZ float64
	dirLen           float64
	uX, uY, uZ       float64
	yaw, pitch       float64
}

type cameraControllerOptions struct {
	active            *bool
	selectedObject    *int
	camera            *scene.Camera
	scene             *scene.Scene
	service           *engine.Service
	mu                *sync.Mutex
	start             func(bool)
	interactiveStart  func()
	entries           [6]cameraEntry
	transformMode     *editorTransformMode
	objectMoveEnabled *bool
	objectStep        *float64
	objectUpdated     func()
	viewportUpdated   func()
}

func startCameraControlWithTransform(canvas fyne.Canvas, options cameraControllerOptions) func() {
	c := newCameraController(options)
	canvas.SetOnTypedKey(c.keyPressed)
	ticker := time.NewTicker(16 * time.Millisecond)
	go c.run(ticker)
	var once sync.Once
	return func() { once.Do(func() { close(c.done) }) }
}

func newCameraController(options cameraControllerOptions) *cameraController {
	return &cameraController{
		active:           options.active,
		camera:           options.camera,
		scene:            options.scene,
		service:          options.service,
		mu:               options.mu,
		start:            options.start,
		interactiveStart: options.interactiveStart,
		viewportUpdated:  options.viewportUpdated,
		posX:             options.entries[0],
		posY:             options.entries[1],
		posZ:             options.entries[2],
		lookX:            options.entries[3],
		lookY:            options.entries[4],
		lookZ:            options.entries[5],
		selectedObject:   options.selectedObject,
		transformMode:    options.transformMode,
		objectStep:       options.objectStep,
		objectUpdated:    options.objectUpdated,
		moveSpeed:        2,
		rotSpeed:         2,
		keyTimeout:       uiKeyTimeoutMillis * time.Millisecond,
		keys:             make(map[fyne.KeyName]bool),
		lastKeys:         make(map[fyne.KeyName]time.Time),
		done:             make(chan struct{}),
	}
}

func (c *cameraController) keyPressed(event *fyne.KeyEvent) {
	objectMode := c.objectTransformActive()
	if !*c.active && (!objectMode || !isObjectTransformKey(event.Name)) {
		return
	}
	c.keysMu.Lock()
	c.keys[event.Name] = true
	c.lastKeys[event.Name] = time.Now()
	c.keysMu.Unlock()
}

func (c *cameraController) run(ticker *time.Ticker) {
	defer ticker.Stop()
	for {
		select {
		case <-c.done:
			return
		case <-ticker.C:
			runOnUI(func() {
				select {
				case <-c.done:
					return
				default:
					c.tick()
				}
			})
		}
	}
}

func (c *cameraController) tick() {
	if c.objectTransformActive() {
		c.tickObject()
		return
	}
	if !*c.active {
		return
	}
	c.expireKeys()
	c.mu.Lock()
	basis, ok := newCameraBasis(*c.camera)
	if !ok {
		c.mu.Unlock()
		return
	}
	keys := c.currentKeys()
	dt := uiFrameMillis / uiMillisPerSecond
	changed := applyCameraMovement(c.camera, basis, keys, c.moveSpeed*dt)
	rotated := applyCameraRotation(c.camera, &basis, keys, c.rotSpeed*dt)
	c.mu.Unlock()
	if !changed && !rotated {
		return
	}
	c.updateSceneAndEntries()
	if c.viewportUpdated != nil {
		c.viewportUpdated()
	}
	if !isWireframeMode(c.service) {
		if c.interactiveStart != nil {
			c.interactiveStart()
			return
		}
		c.start(false)
	}
}

func (c *cameraController) objectTransformActive() bool {
	return c.transformMode != nil && *c.transformMode == transformModeMove &&
		c.objectMoveEnabled != nil && *c.objectMoveEnabled &&
		c.selectedObject != nil && *c.selectedObject >= 0 && *c.selectedObject < len(c.scene.Objects)
}

func (c *cameraController) tickObject() {
	c.expireKeys()
	keys := c.currentKeys()
	step := uiObjectMoveStep
	if c.objectStep != nil {
		step = max(*c.objectStep, uiFloatEpsilon)
	}
	index := *c.selectedObject
	c.mu.Lock()
	changed := applyObjectTranslation(&c.scene.Objects[index], keys, step)
	c.mu.Unlock()
	if !changed {
		return
	}
	if c.objectUpdated != nil {
		c.objectUpdated()
	}
	if !isWireframeMode(c.service) {
		if c.interactiveStart != nil {
			c.interactiveStart()
			return
		}
		c.start(false)
	}
}

func (c *cameraController) expireKeys() {
	now := time.Now()
	c.keysMu.Lock()
	for key, last := range c.lastKeys {
		if now.Sub(last) > c.keyTimeout {
			c.keys[key] = false
		}
	}
	c.keysMu.Unlock()
}

func (c *cameraController) currentKeys() cameraKeys {
	c.keysMu.RLock()
	defer c.keysMu.RUnlock()
	return cameraKeys{
		w: c.keys[fyne.KeyW], s: c.keys[fyne.KeyS], a: c.keys[fyne.KeyA], d: c.keys[fyne.KeyD],
		q: c.keys[fyne.KeyQ], e: c.keys[fyne.KeyE], left: c.keys[fyne.KeyLeft], right: c.keys[fyne.KeyRight],
		up: c.keys[fyne.KeyUp], down: c.keys[fyne.KeyDown], pageUp: c.keys[fyne.KeyPageUp],
		pageDown: c.keys[fyne.KeyPageDown],
	}
}

func newCameraBasis(camera scene.Camera) (cameraBasis, bool) {
	dirX := camera.Target.X - camera.Position.X
	dirY := camera.Target.Y - camera.Position.Y
	dirZ := camera.Target.Z - camera.Position.Z
	dirLen := math.Sqrt(dirX*dirX + dirY*dirY + dirZ*dirZ)
	if dirLen < uiFloatEpsilon {
		return cameraBasis{}, false
	}
	dirX /= dirLen
	dirY /= dirLen
	dirZ /= dirLen
	uX := camera.Up.Y*dirZ - camera.Up.Z*dirY
	uY := camera.Up.Z*dirX - camera.Up.X*dirZ
	uZ := camera.Up.X*dirY - camera.Up.Y*dirX
	uLen := math.Sqrt(uX*uX + uY*uY + uZ*uZ)
	if uLen > uiFloatEpsilon {
		uX /= uLen
		uY /= uLen
		uZ /= uLen
	}
	return cameraBasis{
		dirX: dirX, dirY: dirY, dirZ: dirZ, dirLen: dirLen,
		uX: uX, uY: uY, uZ: uZ,
		yaw: math.Atan2(dirZ, dirX), pitch: math.Atan2(dirY, math.Hypot(dirX, dirZ)),
	}, true
}

func applyCameraMovement(camera *scene.Camera, basis cameraBasis, keys cameraKeys, step float64) bool {
	changed := false
	if keys.w {
		translateCamera(camera, basis.dirX*step, basis.dirY*step, basis.dirZ*step)
		changed = true
	}
	if keys.s {
		translateCamera(camera, -basis.dirX*step, -basis.dirY*step, -basis.dirZ*step)
		changed = true
	}
	if keys.a {
		translateCamera(camera, -basis.uX*step, -basis.uY*step, -basis.uZ*step)
		changed = true
	}
	if keys.d {
		translateCamera(camera, basis.uX*step, basis.uY*step, basis.uZ*step)
		changed = true
	}
	if keys.q {
		translateCamera(camera, 0, -step, 0)
		changed = true
	}
	if keys.e {
		translateCamera(camera, 0, step, 0)
		changed = true
	}
	return changed
}

func translateCamera(camera *scene.Camera, x, y, z float64) {
	camera.Position.X += x
	camera.Position.Y += y
	camera.Position.Z += z
	camera.Target.X += x
	camera.Target.Y += y
	camera.Target.Z += z
}

func applyCameraRotation(camera *scene.Camera, basis *cameraBasis, keys cameraKeys, step float64) bool {
	rotated := false
	if keys.left {
		basis.yaw -= step
		rotated = true
	}
	if keys.right {
		basis.yaw += step
		rotated = true
	}
	if keys.up {
		basis.pitch = max(basis.pitch-step, -math.Pi/2+uiPitchMargin)
		rotated = true
	}
	if keys.down {
		basis.pitch = min(basis.pitch+step, math.Pi/2-uiPitchMargin)
		rotated = true
	}
	if !rotated {
		return false
	}
	camera.Target.X = camera.Position.X + basis.dirLen*math.Cos(basis.pitch)*math.Cos(basis.yaw)
	camera.Target.Y = camera.Position.Y + basis.dirLen*math.Sin(basis.pitch)
	camera.Target.Z = camera.Position.Z + basis.dirLen*math.Cos(basis.pitch)*math.Sin(basis.yaw)
	return true
}

func (c *cameraController) updateSceneAndEntries() {
	c.mu.Lock()
	c.scene.Camera = *c.camera
	c.posX.SetText(fmt.Sprintf("%.2f", c.camera.Position.X))
	c.posY.SetText(fmt.Sprintf("%.2f", c.camera.Position.Y))
	c.posZ.SetText(fmt.Sprintf("%.2f", c.camera.Position.Z))
	c.lookX.SetText(fmt.Sprintf("%.2f", c.camera.Target.X))
	c.lookY.SetText(fmt.Sprintf("%.2f", c.camera.Target.Y))
	c.lookZ.SetText(fmt.Sprintf("%.2f", c.camera.Target.Z))
	c.mu.Unlock()
}

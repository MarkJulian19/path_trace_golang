package ui

import (
	"math"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	engine "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

func dragObjectDelta(sc *scene.Scene, cfg engine.Config, selected, deltaX, deltaY int, startRender func()) {
	if selected < 0 || selected >= len(sc.Objects) {
		return
	}
	aspect := cameraAspect(sc, cfg)
	theta := sc.Camera.FOV * math.Pi / uiDegreesPerHalfTurn
	viewportHeight := uiViewportDiameter * math.Tan(theta/2)
	viewportWidth := aspect * viewportHeight
	focusDist := cameraFocusDistance(sc)
	scaleX := viewportWidth * focusDist / float64(cfg.Width)
	scaleY := viewportHeight * focusDist / float64(cfg.Height)
	worldDx := float64(deltaX) * scaleX
	worldDy := -float64(deltaY) * scaleY
	uX, uY, uZ, vX, vY, vZ := cameraViewportBasis(sc)
	obj := &sc.Objects[selected]
	obj.Position.X += uX*worldDx + vX*worldDy
	obj.Position.Y += uY*worldDx + vY*worldDy
	obj.Position.Z += uZ*worldDx + vZ*worldDy
	if startRender != nil {
		startRender()
	}
}

func moveObjectAlongAxis(sc *scene.Scene, selected int, axis gizmoAxis, delta float64) bool {
	if selected < 0 || selected >= len(sc.Objects) {
		return false
	}
	object := &sc.Objects[selected]
	switch axis {
	case gizmoAxisX:
		object.Position.X += delta
	case gizmoAxisY:
		object.Position.Y += delta
	case gizmoAxisZ:
		object.Position.Z += delta
	default:
		return false
	}
	return true
}

func cameraAspect(sc *scene.Scene, cfg engine.Config) float64 {
	if sc.Camera.AspectRatio != 0 {
		return sc.Camera.AspectRatio
	}
	return float64(cfg.Width) / float64(cfg.Height)
}

func cameraFocusDistance(sc *scene.Scene) float64 {
	if sc.Camera.FocusDist != 0 {
		return sc.Camera.FocusDist
	}
	dx := sc.Camera.Position.X - sc.Camera.Target.X
	dy := sc.Camera.Position.Y - sc.Camera.Target.Y
	dz := sc.Camera.Position.Z - sc.Camera.Target.Z
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

func cameraViewportBasis(sc *scene.Scene) (uX, uY, uZ, vX, vY, vZ float64) {
	wX := sc.Camera.Position.X - sc.Camera.Target.X
	wY := sc.Camera.Position.Y - sc.Camera.Target.Y
	wZ := sc.Camera.Position.Z - sc.Camera.Target.Z
	wLen := math.Sqrt(wX*wX + wY*wY + wZ*wZ)
	if wLen > 0 {
		wX /= wLen
		wY /= wLen
		wZ /= wLen
	}
	uX = sc.Camera.Up.Y*wZ - sc.Camera.Up.Z*wY
	uY = sc.Camera.Up.Z*wX - sc.Camera.Up.X*wZ
	uZ = sc.Camera.Up.X*wY - sc.Camera.Up.Y*wX
	uLen := math.Sqrt(uX*uX + uY*uY + uZ*uZ)
	if uLen > 0 {
		uX /= uLen
		uY /= uLen
		uZ /= uLen
	}
	vX = wY*uZ - wZ*uY
	vY = wZ*uX - wX*uZ
	vZ = wX*uY - wY*uX
	vLen := math.Sqrt(vX*vX + vY*vY + vZ*vZ)
	if vLen > 0 {
		vX /= vLen
		vY /= vLen
		vZ /= vLen
	}
	return
}

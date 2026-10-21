package ui

import (
	"fyne.io/fyne/v2"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

type editorTransformMode string

const (
	transformModeSelect editorTransformMode = "Select"
	transformModeMove   editorTransformMode = "Move"
)

func applyObjectTranslation(object *scene.Object, keys cameraKeys, step float64) bool {
	if object == nil || step <= 0 {
		return false
	}

	changed := false
	if keys.left {
		object.Position.X -= step
		changed = true
	}
	if keys.right {
		object.Position.X += step
		changed = true
	}
	if keys.up {
		object.Position.Z -= step
		changed = true
	}
	if keys.down {
		object.Position.Z += step
		changed = true
	}
	if keys.pageUp {
		object.Position.Y += step
		changed = true
	}
	if keys.pageDown {
		object.Position.Y -= step
		changed = true
	}
	return changed
}

func isObjectTransformKey(key fyne.KeyName) bool {
	switch key {
	case fyne.KeyLeft, fyne.KeyRight, fyne.KeyUp, fyne.KeyDown, fyne.KeyPageUp, fyne.KeyPageDown:
		return true
	default:
		return false
	}
}

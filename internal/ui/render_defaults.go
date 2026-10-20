package ui

import (
	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	engine "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

func sceneRenderSettings(mode string, sc *scene.Scene) scene.RenderSettings {
	settings := engine.SettingsForMode(mode)
	if sc.Settings.Width <= 0 || sc.Settings.Height <= 0 {
		return settings
	}
	settings.Width = sc.Settings.Width
	settings.Height = sc.Settings.Height
	if sc.Settings.SamplesPerPx > 0 {
		settings.SamplesPerPx = sc.Settings.SamplesPerPx
	}
	if sc.Settings.MaxDepth > 0 {
		settings.MaxDepth = sc.Settings.MaxDepth
	}
	if sc.Settings.MaxRayDist > 0 {
		settings.MaxRayDist = sc.Settings.MaxRayDist
	}
	return settings
}

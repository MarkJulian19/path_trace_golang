package scenejson

import (
	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

type sceneFile struct {
	Name       string             `json:"name"`
	Camera     cameraFile         `json:"camera"`
	Objects    []objectFile       `json:"objects"`
	Materials  []materialFile     `json:"materials"`
	Settings   renderSettingsFile `json:"settings"`
	Background colorFile          `json:"background"`
	Sky        *skyFile           `json:"sky"`
	Fog        *fogFile           `json:"fog,omitempty"`
}

type vec3File struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	Z float64 `json:"z"`
}

type colorFile struct {
	R float64 `json:"r"`
	G float64 `json:"g"`
	B float64 `json:"b"`
}

type cameraFile struct {
	Position    vec3File `json:"position"`
	Target      vec3File `json:"target"`
	Up          vec3File `json:"up"`
	FOV         float64  `json:"fov"`
	Aperture    float64  `json:"aperture"`
	FocusDist   float64  `json:"focus_dist"`
	AspectRatio float64  `json:"aspect_ratio"`
}

type materialFile struct {
	ID              string    `json:"id"`
	Type            string    `json:"type"`
	Albedo          colorFile `json:"albedo"`
	Rough           float64   `json:"rough"`
	IOR             float64   `json:"ior"`
	Emit            colorFile `json:"emit"`
	Power           float64   `json:"power"`
	Absorption      colorFile `json:"absorption"`
	Smoothness      float64   `json:"smoothness"`
	Reflectivity    float64   `json:"reflectivity"`
	Tint            colorFile `json:"tint"`
	AbsorptionScale float64   `json:"absorption_scale"`
}

type objectFile struct {
	ID         string   `json:"id"`
	Type       string   `json:"type"`
	Position   vec3File `json:"position"`
	Size       vec3File `json:"size"`
	MaterialID string   `json:"material_id"`
}

type renderSettingsFile struct {
	Width        int     `json:"width"`
	Height       int     `json:"height"`
	SamplesPerPx int     `json:"samples_per_px"`
	MaxDepth     int     `json:"max_depth"`
	MaxRayDist   float64 `json:"max_ray_dist"`
}

type fogFile struct {
	Density        float64   `json:"density"`
	Color          colorFile `json:"color"`
	Scatter        float64   `json:"scatter"`
	SigmaS         float64   `json:"sigma_s"`
	SigmaA         float64   `json:"sigma_a"`
	G              float64   `json:"g"`
	HeteroStrength float64   `json:"hetero_strength"`
	NoiseScale     float64   `json:"noise_scale"`
	NoiseOctaves   int       `json:"noise_octaves"`
	AffectSky      bool      `json:"affect_sky"`
	GPUVolumetric  bool      `json:"gpu_volumetric"`
}

type skyFile struct {
	Type    string    `json:"type"`
	Color   colorFile `json:"color"`
	Horizon colorFile `json:"horizon"`
	Zenith  colorFile `json:"zenith"`
}

func toEntity(value sceneFile) *scene.Scene {
	result := &scene.Scene{
		Name:       value.Name,
		Camera:     cameraToEntity(value.Camera),
		Objects:    objectsToEntity(value.Objects),
		Materials:  materialsToEntity(value.Materials),
		Settings:   renderSettingsToEntity(value.Settings),
		Background: colorToEntity(value.Background),
		Sky:        skyToEntity(value.Sky),
		Fog:        fogToEntity(value.Fog),
	}

	return result
}

func fromEntity(value *scene.Scene) sceneFile {
	result := sceneFile{
		Name:       value.Name,
		Camera:     cameraFromEntity(value.Camera),
		Objects:    objectsFromEntity(value.Objects),
		Materials:  materialsFromEntity(value.Materials),
		Settings:   renderSettingsFromEntity(value.Settings),
		Background: colorFromEntity(value.Background),
		Sky:        skyFromEntity(value.Sky),
		Fog:        fogFromEntity(value.Fog),
	}

	return result
}

func objectsToEntity(values []objectFile) []scene.Object {
	objects := make([]scene.Object, 0, len(values))
	for _, value := range values {
		objects = append(objects, scene.Object{
			ID:         value.ID,
			Type:       scene.ObjectType(value.Type),
			Position:   vec3ToEntity(value.Position),
			Size:       vec3ToEntity(value.Size),
			MaterialID: value.MaterialID,
		})
	}
	return objects
}

func materialsToEntity(values []materialFile) []scene.Material {
	materials := make([]scene.Material, 0, len(values))
	for _, value := range values {
		materials = append(materials, scene.Material{
			ID:              value.ID,
			Type:            scene.MaterialType(value.Type),
			Albedo:          colorToEntity(value.Albedo),
			Rough:           value.Rough,
			IOR:             value.IOR,
			Emit:            colorToEntity(value.Emit),
			Power:           value.Power,
			Absorption:      colorToEntity(value.Absorption),
			Smoothness:      value.Smoothness,
			Reflectivity:    value.Reflectivity,
			Tint:            colorToEntity(value.Tint),
			AbsorptionScale: value.AbsorptionScale,
		})
	}
	return materials
}

func objectsFromEntity(values []scene.Object) []objectFile {
	objects := make([]objectFile, 0, len(values))
	for _, value := range values {
		objects = append(objects, objectFile{
			ID:         value.ID,
			Type:       string(value.Type),
			Position:   vec3FromEntity(value.Position),
			Size:       vec3FromEntity(value.Size),
			MaterialID: value.MaterialID,
		})
	}
	return objects
}

func materialsFromEntity(values []scene.Material) []materialFile {
	materials := make([]materialFile, 0, len(values))
	for _, value := range values {
		materials = append(materials, materialFile{
			ID:              value.ID,
			Type:            string(value.Type),
			Albedo:          colorFromEntity(value.Albedo),
			Rough:           value.Rough,
			IOR:             value.IOR,
			Emit:            colorFromEntity(value.Emit),
			Power:           value.Power,
			Absorption:      colorFromEntity(value.Absorption),
			Smoothness:      value.Smoothness,
			Reflectivity:    value.Reflectivity,
			Tint:            colorFromEntity(value.Tint),
			AbsorptionScale: value.AbsorptionScale,
		})
	}
	return materials
}

func skyToEntity(value *skyFile) *scene.Sky {
	if value == nil {
		return nil
	}
	return &scene.Sky{
		Type:    value.Type,
		Color:   colorToEntity(value.Color),
		Horizon: colorToEntity(value.Horizon),
		Zenith:  colorToEntity(value.Zenith),
	}
}

func fogToEntity(value *fogFile) *scene.Fog {
	if value == nil {
		return nil
	}
	return &scene.Fog{
		Density:        value.Density,
		Color:          colorToEntity(value.Color),
		Scatter:        value.Scatter,
		SigmaS:         value.SigmaS,
		SigmaA:         value.SigmaA,
		G:              value.G,
		HeteroStrength: value.HeteroStrength,
		NoiseScale:     value.NoiseScale,
		NoiseOctaves:   value.NoiseOctaves,
		AffectSky:      value.AffectSky,
		GPUVolumetric:  value.GPUVolumetric,
	}
}

func skyFromEntity(value *scene.Sky) *skyFile {
	if value == nil {
		return nil
	}
	return &skyFile{
		Type:    value.Type,
		Color:   colorFromEntity(value.Color),
		Horizon: colorFromEntity(value.Horizon),
		Zenith:  colorFromEntity(value.Zenith),
	}
}

func fogFromEntity(value *scene.Fog) *fogFile {
	if value == nil {
		return nil
	}
	return &fogFile{
		Density:        value.Density,
		Color:          colorFromEntity(value.Color),
		Scatter:        value.Scatter,
		SigmaS:         value.SigmaS,
		SigmaA:         value.SigmaA,
		G:              value.G,
		HeteroStrength: value.HeteroStrength,
		NoiseScale:     value.NoiseScale,
		NoiseOctaves:   value.NoiseOctaves,
		AffectSky:      value.AffectSky,
		GPUVolumetric:  value.GPUVolumetric,
	}
}

func vec3ToEntity(value vec3File) scene.Vec3 {
	return scene.Vec3{X: value.X, Y: value.Y, Z: value.Z}
}

func vec3FromEntity(value scene.Vec3) vec3File {
	return vec3File{X: value.X, Y: value.Y, Z: value.Z}
}

func colorToEntity(value colorFile) scene.Color {
	return scene.Color{R: value.R, G: value.G, B: value.B}
}

func colorFromEntity(value scene.Color) colorFile {
	return colorFile{R: value.R, G: value.G, B: value.B}
}

func cameraToEntity(value cameraFile) scene.Camera {
	return scene.Camera{
		Position:    vec3ToEntity(value.Position),
		Target:      vec3ToEntity(value.Target),
		Up:          vec3ToEntity(value.Up),
		FOV:         value.FOV,
		Aperture:    value.Aperture,
		FocusDist:   value.FocusDist,
		AspectRatio: value.AspectRatio,
	}
}

func cameraFromEntity(value scene.Camera) cameraFile {
	return cameraFile{
		Position:    vec3FromEntity(value.Position),
		Target:      vec3FromEntity(value.Target),
		Up:          vec3FromEntity(value.Up),
		FOV:         value.FOV,
		Aperture:    value.Aperture,
		FocusDist:   value.FocusDist,
		AspectRatio: value.AspectRatio,
	}
}

func renderSettingsToEntity(value renderSettingsFile) scene.RenderSettings {
	return scene.RenderSettings{
		Width:        value.Width,
		Height:       value.Height,
		SamplesPerPx: value.SamplesPerPx,
		MaxDepth:     value.MaxDepth,
		MaxRayDist:   value.MaxRayDist,
	}
}

func renderSettingsFromEntity(value scene.RenderSettings) renderSettingsFile {
	return renderSettingsFile{
		Width:        value.Width,
		Height:       value.Height,
		SamplesPerPx: value.SamplesPerPx,
		MaxDepth:     value.MaxDepth,
		MaxRayDist:   value.MaxRayDist,
	}
}

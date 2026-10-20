package scenejson

import (
	"encoding/json"

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
	FocusDist   float64  `json:"focusDist"`
	AspectRatio float64  `json:"aspectRatio"`
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
	AbsorptionScale float64   `json:"absorptionScale"`
}

type objectFile struct {
	ID         string   `json:"id"`
	Type       string   `json:"type"`
	Position   vec3File `json:"position"`
	Size       vec3File `json:"size"`
	MaterialID string   `json:"materialId"`
}

type renderSettingsFile struct {
	Width        int     `json:"width"`
	Height       int     `json:"height"`
	SamplesPerPx int     `json:"samplesPerPx"`
	MaxDepth     int     `json:"maxDepth"`
	MaxRayDist   float64 `json:"maxRayDist"`
}

type fogFile struct {
	Density        float64   `json:"density"`
	Color          colorFile `json:"color"`
	Scatter        float64   `json:"scatter"`
	SigmaS         float64   `json:"sigmaS"`
	SigmaA         float64   `json:"sigmaA"`
	G              float64   `json:"g"`
	HeteroStrength float64   `json:"heteroStrength"`
	NoiseScale     float64   `json:"noiseScale"`
	NoiseOctaves   int       `json:"noiseOctaves"`
	AffectSky      bool      `json:"affectSky"`
	GPUVolumetric  bool      `json:"gpuVolumetric"`
}

type skyFile struct {
	Type    string    `json:"type"`
	Color   colorFile `json:"color"`
	Horizon colorFile `json:"horizon"`
	Zenith  colorFile `json:"zenith"`
}

func unmarshalLegacyJSON(data []byte, target any, replacements map[string]string) error {
	var fields map[string]json.RawMessage
	err := json.Unmarshal(data, &fields)
	if err != nil {
		return err
	}
	for currentKey, legacyKey := range replacements {
		if raw, ok := fields[legacyKey]; ok {
			fields[currentKey] = raw
			delete(fields, legacyKey)
		}
	}
	normalized, err := json.Marshal(fields)
	if err != nil {
		return err
	}
	return json.Unmarshal(normalized, target)
}

func marshalLegacyJSON(value any, replacements map[string]string) ([]byte, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var fields map[string]any
	err = json.Unmarshal(data, &fields)
	if err != nil {
		return nil, err
	}
	for currentKey, legacyKey := range replacements {
		if value, ok := fields[currentKey]; ok {
			fields[legacyKey] = value
			delete(fields, currentKey)
		}
	}
	return json.Marshal(fields)
}

// UnmarshalJSON decodes camera fields while accepting legacy snake_case names.
func (value *cameraFile) UnmarshalJSON(data []byte) error {
	type alias cameraFile
	var decoded alias
	err := unmarshalLegacyJSON(data, &decoded, map[string]string{
		"focusDist":   "focus_dist",
		"aspectRatio": "aspect_ratio",
	})
	if err != nil {
		return err
	}
	*value = cameraFile(decoded)
	return nil
}

// MarshalJSON encodes camera fields using legacy snake_case names.
func (value cameraFile) MarshalJSON() ([]byte, error) {
	type alias cameraFile
	return marshalLegacyJSON(alias(value), map[string]string{
		"focusDist":   "focus_dist",
		"aspectRatio": "aspect_ratio",
	})
}

// UnmarshalJSON decodes material fields while accepting legacy snake_case names.
func (value *materialFile) UnmarshalJSON(data []byte) error {
	type alias materialFile
	var decoded alias
	err := unmarshalLegacyJSON(data, &decoded, map[string]string{
		"absorptionScale": "absorption_scale",
	})
	if err != nil {
		return err
	}
	*value = materialFile(decoded)
	return nil
}

// MarshalJSON encodes material fields using legacy snake_case names.
func (value materialFile) MarshalJSON() ([]byte, error) {
	type alias materialFile
	return marshalLegacyJSON(alias(value), map[string]string{
		"absorptionScale": "absorption_scale",
	})
}

// UnmarshalJSON decodes object fields while accepting legacy snake_case names.
func (value *objectFile) UnmarshalJSON(data []byte) error {
	type alias objectFile
	var decoded alias
	err := unmarshalLegacyJSON(data, &decoded, map[string]string{
		"materialId": "material_id",
	})
	if err != nil {
		return err
	}
	*value = objectFile(decoded)
	return nil
}

// MarshalJSON encodes object fields using legacy snake_case names.
func (value objectFile) MarshalJSON() ([]byte, error) {
	type alias objectFile
	return marshalLegacyJSON(alias(value), map[string]string{
		"materialId": "material_id",
	})
}

// UnmarshalJSON decodes render settings while accepting legacy snake_case names.
func (value *renderSettingsFile) UnmarshalJSON(data []byte) error {
	type alias renderSettingsFile
	var decoded alias
	err := unmarshalLegacyJSON(data, &decoded, map[string]string{
		"samplesPerPx": "samples_per_px",
		"maxDepth":     "max_depth",
		"maxRayDist":   "max_ray_dist",
	})
	if err != nil {
		return err
	}
	*value = renderSettingsFile(decoded)
	return nil
}

// MarshalJSON encodes render settings using legacy snake_case names.
func (value renderSettingsFile) MarshalJSON() ([]byte, error) {
	type alias renderSettingsFile
	return marshalLegacyJSON(alias(value), map[string]string{
		"samplesPerPx": "samples_per_px",
		"maxDepth":     "max_depth",
		"maxRayDist":   "max_ray_dist",
	})
}

// UnmarshalJSON decodes fog fields while accepting legacy snake_case names.
func (value *fogFile) UnmarshalJSON(data []byte) error {
	type alias fogFile
	var decoded alias
	err := unmarshalLegacyJSON(data, &decoded, map[string]string{
		"sigmaS":         "sigma_s",
		"sigmaA":         "sigma_a",
		"heteroStrength": "hetero_strength",
		"noiseScale":     "noise_scale",
		"noiseOctaves":   "noise_octaves",
		"affectSky":      "affect_sky",
		"gpuVolumetric":  "gpu_volumetric",
	})
	if err != nil {
		return err
	}
	*value = fogFile(decoded)
	return nil
}

// MarshalJSON encodes fog fields using legacy snake_case names.
func (value fogFile) MarshalJSON() ([]byte, error) {
	type alias fogFile
	return marshalLegacyJSON(alias(value), map[string]string{
		"sigmaS":         "sigma_s",
		"sigmaA":         "sigma_a",
		"heteroStrength": "hetero_strength",
		"noiseScale":     "noise_scale",
		"noiseOctaves":   "noise_octaves",
		"affectSky":      "affect_sky",
		"gpuVolumetric":  "gpu_volumetric",
	})
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

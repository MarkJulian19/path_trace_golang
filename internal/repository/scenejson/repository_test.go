package scenejson_test

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	scenejson "github.com/MarkJulian19/path_trace_golang/internal/repository/scenejson"
)

const (
	repositoryWidth        = 24
	repositoryHeight       = 16
	repositorySamples      = 3
	repositoryDepth        = 5
	repositoryRayDist      = 80
	repositoryFOV          = 50
	repositoryAperture     = 0.1
	repositoryFocusDist    = 6
	repositoryAspect       = 1.5
	repositoryDensity      = 0.02
	repositoryScatter      = 0.7
	repositoryNoiseScale   = 3
	repositoryNoiseOctaves = 2
	legacyFocusDistance    = 5
)

func repositoryTestScene() *scene.Scene {
	return &scene.Scene{
		Name: "repository-scene",
		Camera: scene.Camera{
			Position:    scene.Vec3{X: 1, Y: 2, Z: 6},
			Target:      scene.Vec3{X: 0, Y: 1, Z: 0},
			Up:          scene.Vec3{X: 0, Y: 1, Z: 0},
			FOV:         repositoryFOV,
			Aperture:    repositoryAperture,
			FocusDist:   repositoryFocusDist,
			AspectRatio: repositoryAspect,
		},
		Objects: []scene.Object{
			{
				ID:         "sphere",
				Type:       scene.ObjectSphere,
				Position:   scene.Vec3{X: 1, Y: 1, Z: 0},
				Size:       scene.Vec3{X: 2},
				MaterialID: "glass",
			},
		},
		Materials: []scene.Material{
			{
				ID:              "glass",
				Type:            scene.MaterialDielectric,
				Albedo:          scene.Color{R: 0.8, G: 0.9, B: 1},
				Rough:           0.1,
				IOR:             1.5,
				Emit:            scene.Color{R: 0.1, G: 0.2, B: 0.3},
				Power:           2,
				Absorption:      scene.Color{R: 0.01, G: 0.02, B: 0.03},
				Smoothness:      0.4,
				Reflectivity:    0.6,
				Tint:            scene.Color{R: 0.9, G: 0.95, B: 1},
				AbsorptionScale: 0.02,
			},
		},
		Settings: scene.RenderSettings{
			Width:        repositoryWidth,
			Height:       repositoryHeight,
			SamplesPerPx: repositorySamples,
			MaxDepth:     repositoryDepth,
			MaxRayDist:   repositoryRayDist,
		},
		Background: scene.Color{R: 0.1, G: 0.2, B: 0.3},
		Sky: &scene.Sky{
			Type:    "gradient",
			Color:   scene.Color{R: 0.1, G: 0.1, B: 0.1},
			Horizon: scene.Color{R: 0.2, G: 0.3, B: 0.4},
			Zenith:  scene.Color{R: 0.5, G: 0.6, B: 0.7},
		},
		Fog: &scene.Fog{
			Density:        repositoryDensity,
			Color:          scene.Color{R: 0.3, G: 0.4, B: 0.5},
			Scatter:        repositoryScatter,
			SigmaS:         0.01,
			SigmaA:         0.02,
			G:              0.1,
			HeteroStrength: 0.4,
			NoiseScale:     repositoryNoiseScale,
			NoiseOctaves:   repositoryNoiseOctaves,
			AffectSky:      true,
			GPUVolumetric:  true,
		},
	}
}

func TestRepositoryRoundTripPreservesScene(t *testing.T) {
	t.Parallel()
	repository := scenejson.New()
	path := filepath.Join(t.TempDir(), "scene.json")
	want := repositoryTestScene()

	err := repository.Save(path, want)
	if err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	got, err := repository.Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("round trip changed scene:\n got: %#v\nwant: %#v", got, want)
	}
}

func TestRepositoryLoadsLegacySnakeCaseJSON(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "legacy.json")
	legacyJSON := []byte(`{
  "name": "legacy",
  "camera": {
    "position": {"x": 1, "y": 2, "z": 3},
    "target": {"x": 0, "y": 0, "z": 0},
    "up": {"x": 0, "y": 1, "z": 0},
    "fov": 45,
    "aperture": 0,
    "focus_dist": 5,
    "aspect_ratio": 1.5
  },
  "objects": [{
    "id": "sphere",
    "type": "sphere",
    "position": {"x": 0, "y": 0, "z": 0},
    "size": {"x": 1, "y": 0, "z": 0},
    "material_id": "mat"
  }],
  "materials": [{
    "id": "mat",
    "type": "lambert",
    "albedo": {"r": 1, "g": 0, "b": 0},
    "rough": 0.5,
    "ior": 1.5,
    "emit": {"r": 0, "g": 0, "b": 0},
    "power": 1,
    "absorption": {"r": 0, "g": 0, "b": 0},
    "smoothness": 0,
    "reflectivity": 0,
    "tint": {"r": 1, "g": 1, "b": 1},
    "absorption_scale": 0
  }],
  "settings": {
    "width": 24,
    "height": 16,
    "samples_per_px": 3,
    "max_depth": 5,
    "max_ray_dist": 80
  },
  "background": {"r": 0.1, "g": 0.2, "b": 0.3}
}`)
	err := os.WriteFile(path, legacyJSON, 0o600)
	if err != nil {
		t.Fatalf("write legacy scene: %v", err)
	}

	got, err := scenejson.New().Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if got.Name != "legacy" || got.Camera.FocusDist != legacyFocusDistance {
		t.Fatalf("legacy scene was not decoded: %#v", got)
	}
	if len(got.Objects) != 1 || got.Objects[0].MaterialID != "mat" {
		t.Fatalf("legacy objects were not decoded: %#v", got.Objects)
	}
}

func TestRepositoryReturnsContextualErrors(t *testing.T) {
	t.Parallel()
	missingPath := filepath.Join(t.TempDir(), "missing.json")
	_, err := scenejson.New().Load(missingPath)
	if err == nil || !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("Load() error = %v, want wrapped os.ErrNotExist", err)
	}

	invalidPath := filepath.Join(t.TempDir(), "invalid.json")
	err = os.WriteFile(invalidPath, []byte("{"), 0o600)
	if err != nil {
		t.Fatalf("write invalid scene: %v", err)
	}
	_, err = scenejson.New().Load(invalidPath)
	if err == nil {
		t.Fatal("Load() accepted invalid JSON")
	}
}

func BenchmarkRepositoryLoad(b *testing.B) {
	repository := scenejson.New()
	path := filepath.Join(b.TempDir(), "scene.json")
	err := repository.Save(path, repositoryTestScene())
	if err != nil {
		b.Fatalf("prepare scene: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		benchmarkRepositoryScene, errBenchmarkRepository = repository.Load(path)
	}
}

func BenchmarkRepositoryNew(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		benchmarkRepositoryValue = scenejson.New()
	}
}

func BenchmarkRepositorySave(b *testing.B) {
	repository := scenejson.New()
	path := filepath.Join(b.TempDir(), "scene.json")
	value := repositoryTestScene()
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		errBenchmarkRepository = repository.Save(path, value)
	}
}

var (
	benchmarkRepositoryScene *scene.Scene
	benchmarkRepositoryValue scenejson.Repository
	errBenchmarkRepository   error
)

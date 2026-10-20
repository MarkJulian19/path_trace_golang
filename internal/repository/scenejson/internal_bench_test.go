//go:build benchinternal

package scenejson

import (
	"testing"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

func BenchmarkInternalModelConversion(b *testing.B) {
	value := internalRepositoryScene()
	fileValue := fromEntity(value)

	b.Run("toEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalScene = toEntity(fileValue)
		}
	})
	b.Run("fromEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalFile = fromEntity(value)
		}
	})
	b.Run("objectsToEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalObjects = objectsToEntity(fileValue.Objects)
		}
	})
	b.Run("objectsFromEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalFiles = objectsFromEntity(value.Objects)
		}
	})
	b.Run("materialsToEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalMaterials = materialsToEntity(fileValue.Materials)
		}
	})
	b.Run("materialsFromEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalMaterialFiles = materialsFromEntity(value.Materials)
		}
	})
	b.Run("skyToEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalSky = skyToEntity(fileValue.Sky)
		}
	})
	b.Run("skyFromEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalSkyFile = skyFromEntity(value.Sky)
		}
	})
	b.Run("fogToEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalFog = fogToEntity(fileValue.Fog)
		}
	})
	b.Run("fogFromEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalFogFile = fogFromEntity(value.Fog)
		}
	})
	b.Run("vec3ToEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalVec3 = vec3ToEntity(fileValue.Camera.Position)
		}
	})
	b.Run("vec3FromEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalVec3File = vec3FromEntity(value.Camera.Position)
		}
	})
	b.Run("colorToEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalColor = colorToEntity(fileValue.Background)
		}
	})
	b.Run("colorFromEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalColorFile = colorFromEntity(value.Background)
		}
	})
	b.Run("cameraToEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalCamera = cameraToEntity(fileValue.Camera)
		}
	})
	b.Run("cameraFromEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalCameraFile = cameraFromEntity(value.Camera)
		}
	})
	b.Run("renderSettingsToEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalSettings = renderSettingsToEntity(fileValue.Settings)
		}
	})
	b.Run("renderSettingsFromEntity", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalSettingsFile = renderSettingsFromEntity(value.Settings)
		}
	})
}

func internalRepositoryScene() *scene.Scene {
	return &scene.Scene{
		Name: "internal-benchmark",
		Camera: scene.Camera{
			Position:    scene.Vec3{X: 1, Y: 2, Z: 6},
			Target:      scene.Vec3{Y: 1},
			Up:          scene.Vec3{Y: 1},
			FOV:         50,
			Aperture:    0.1,
			FocusDist:   6,
			AspectRatio: 1.5,
		},
		Objects: []scene.Object{
			{
				ID:         "sphere",
				Type:       scene.ObjectSphere,
				Position:   scene.Vec3{Y: 1},
				Size:       scene.Vec3{X: 2},
				MaterialID: "glass",
			},
		},
		Materials: []scene.Material{
			{
				ID:     "glass",
				Type:   scene.MaterialDielectric,
				Albedo: scene.Color{R: 0.8, G: 0.9, B: 1},
				IOR:    1.5,
				Emit:   scene.Color{R: 0.1, G: 0.2, B: 0.3},
				Power:  2,
			},
		},
		Settings: scene.RenderSettings{
			Width:        24,
			Height:       16,
			SamplesPerPx: 3,
			MaxDepth:     5,
			MaxRayDist:   80,
		},
		Background: scene.Color{R: 0.1, G: 0.2, B: 0.3},
		Sky: &scene.Sky{
			Type:    "gradient",
			Color:   scene.Color{R: 0.1, G: 0.1, B: 0.1},
			Horizon: scene.Color{R: 0.2, G: 0.3, B: 0.4},
			Zenith:  scene.Color{R: 0.5, G: 0.6, B: 0.7},
		},
		Fog: &scene.Fog{
			Density:        0.02,
			Color:          scene.Color{R: 0.3, G: 0.4, B: 0.5},
			Scatter:        0.7,
			SigmaS:         0.01,
			SigmaA:         0.02,
			G:              0.1,
			HeteroStrength: 0.4,
			NoiseScale:     3,
			NoiseOctaves:   2,
			AffectSky:      true,
			GPUVolumetric:  true,
		},
	}
}

var (
	benchmarkInternalScene         *scene.Scene
	benchmarkInternalFile          sceneFile
	benchmarkInternalObjects       []scene.Object
	benchmarkInternalFiles         []objectFile
	benchmarkInternalMaterials     []scene.Material
	benchmarkInternalMaterialFiles []materialFile
	benchmarkInternalSky           *scene.Sky
	benchmarkInternalSkyFile       *skyFile
	benchmarkInternalFog           *scene.Fog
	benchmarkInternalFogFile       *fogFile
	benchmarkInternalVec3          scene.Vec3
	benchmarkInternalVec3File      vec3File
	benchmarkInternalColor         scene.Color
	benchmarkInternalColorFile     colorFile
	benchmarkInternalCamera        scene.Camera
	benchmarkInternalCameraFile    cameraFile
	benchmarkInternalSettings      scene.RenderSettings
	benchmarkInternalSettingsFile  renderSettingsFile
)

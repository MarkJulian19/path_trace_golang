//go:build benchinternal

package render

import (
	"bytes"
	"image"
	"image/color"
	"os"
	"path/filepath"
	"testing"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

const (
	internalBenchmarkWidth  = 32
	internalBenchmarkHeight = 18
	internalBenchmarkSpp    = 2
	internalBenchmarkDepth  = 4
	internalBenchmarkRayMax = 1000
)

func BenchmarkInternalMath(b *testing.B) {
	left := v(1, 2, 3)
	right := v(4, 5, 6)
	unit := v(0, 1, 0)
	rng := newRandSource()

	b.Run("v", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalVec = v(1, 2, 3)
		}
	})
	b.Run("add", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalVec = left.add(right)
		}
	})
	b.Run("sub", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalVec = left.sub(right)
		}
	})
	b.Run("mul", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalVec = left.mul(2)
		}
	})
	b.Run("div", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalVec = left.div(2)
		}
	})
	b.Run("dot", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalFloat = left.dot(right)
		}
	})
	b.Run("cross", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalVec = left.cross(right)
		}
	})
	b.Run("length", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalFloat = left.length()
		}
	})
	b.Run("unit", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalVec = left.unit()
		}
	})
	b.Run("reflectVec", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalVec = reflectVec(left, unit)
		}
	})
	b.Run("refractVec", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalVec = refractVec(left.unit(), unit, 0.67)
		}
	})
	b.Run("ray.at", func(b *testing.B) {
		r := ray{orig: left, dir: right}
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalVec = r.at(0.5)
		}
	})
	b.Run("randomInUnitSphere", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalVec = randomInUnitSphere(rng)
		}
	})
	b.Run("randomCosineDirection", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalVec = randomCosineDirection(unit, rng)
		}
	})
}

func BenchmarkInternalRandom(b *testing.B) {
	b.Run("newRandSource", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalRNG = newRandSource()
		}
	})
	b.Run("Float64", func(b *testing.B) {
		rng := newRandSource()
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalFloat = rng.Float64()
		}
	})
}

func BenchmarkInternalCamera(b *testing.B) {
	sc := internalBenchmarkScene()
	cfg := internalBenchmarkConfig()
	rng := newRandSource()
	b.Run("newCamera", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalCamera = newCamera(sc.Camera, cfg, rng)
		}
	})
	camera := newCamera(sc.Camera, cfg, rng)
	b.Run("getRay", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalRay = camera.getRay(0.5, 0.5)
		}
	})
}

func BenchmarkInternalMaterials(b *testing.B) {
	sc := internalBenchmarkScene()
	rng := newRandSource()
	record := hitRecord{p: v(0, 0, 0), normal: v(0, 1, 0), frontFace: true}
	incoming := ray{orig: v(0, 1, 2), dir: v(0, -0.2, -1).unit()}

	b.Run("convertMaterial", func(b *testing.B) {
		materialValue := sc.Materials[0]
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalMaterial = convertMaterial(materialValue)
		}
	})
	b.Run("clamp01", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalFloat = clamp01(0.5)
		}
	})
	b.Run("emitted", func(b *testing.B) {
		materialValue := convertMaterial(sc.Materials[3])
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalVec = materialValue.emitted()
		}
	})
	b.Run("scatter-lambert", func(b *testing.B) {
		materialValue := convertMaterial(sc.Materials[0])
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalScatterOK, benchmarkInternalVec, benchmarkInternalRay = materialValue.scatter(
				rng,
				incoming,
				&record,
			)
		}
	})
	b.Run("scatter-metal", func(b *testing.B) {
		materialValue := convertMaterial(sc.Materials[1])
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalScatterOK, benchmarkInternalVec, benchmarkInternalRay = materialValue.scatter(
				rng,
				incoming,
				&record,
			)
		}
	})
	b.Run("scatter-dielectric", func(b *testing.B) {
		materialValue := convertMaterial(sc.Materials[2])
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalScatterOK, benchmarkInternalVec, benchmarkInternalRay = materialValue.scatter(
				rng,
				incoming,
				&record,
			)
		}
	})
	b.Run("scatter-emissive", func(b *testing.B) {
		materialValue := convertMaterial(sc.Materials[3])
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalScatterOK, benchmarkInternalVec, benchmarkInternalRay = materialValue.scatter(
				rng,
				incoming,
				&record,
			)
		}
	})
	b.Run("scatter-mirror", func(b *testing.B) {
		materialValue := convertMaterial(sc.Materials[4])
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalScatterOK, benchmarkInternalVec, benchmarkInternalRay = materialValue.scatter(
				rng,
				incoming,
				&record,
			)
		}
	})
	b.Run("reflectance", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalFloat = reflectance(0.5, 1.5)
		}
	})
}

func BenchmarkInternalObjects(b *testing.B) {
	sc := internalBenchmarkScene()
	r := ray{orig: v(0, 1, 5), dir: v(0, -0.1, -1).unit()}
	world := sceneToWorld(sc)

	b.Run("sceneToWorld", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalWorld = sceneToWorld(sc)
		}
	})
	b.Run("setFaceNormal", func(b *testing.B) {
		record := hitRecord{}
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			record.setFaceNormal(r, v(0, 1, 0))
		}
	})
	b.Run("sphere.hit", func(b *testing.B) {
		object := world[1].(sphere)
		record := hitRecord{}
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalHit = object.hit(r, 0.001, 1000, &record)
		}
	})
	b.Run("plane.hit", func(b *testing.B) {
		object := world[0].(plane)
		record := hitRecord{}
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalHit = object.hit(r, 0.001, 1000, &record)
		}
	})
	b.Run("box.hit", func(b *testing.B) {
		object := world[2].(box)
		record := hitRecord{}
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalHit = object.hit(r, 0.001, 1000, &record)
		}
	})
}

func BenchmarkInternalRayColor(b *testing.B) {
	sc := internalBenchmarkScene()
	world := sceneToWorld(sc)
	rng := newRandSource()
	record := hitRecord{}
	incoming := ray{orig: v(0, 1, 5), dir: v(0, -0.1, -1).unit()}
	background := func(_ ray) vec3 { return v(0.05, 0.05, 0.05) }
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		benchmarkInternalColor = rayColorOpt(incoming, world, background, internalBenchmarkDepth, rng, &record, 0)
	}
}

func BenchmarkInternalVideo(b *testing.B) {
	frame := image.NewRGBA(image.Rect(0, 0, 32, 32))
	frame.SetRGBA(0, 0, color.RGBA{R: 255, G: 128, B: 64, A: 255})
	b.Run("encodeFrameToJPEG", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalJPEG, benchmarkInternalError = encodeFrameToJPEG(frame, 75)
		}
	})

	data, err := encodeFrameToJPEG(frame, 75)
	if err != nil {
		b.Fatalf("prepare JPEG: %v", err)
	}
	path := filepath.Join(b.TempDir(), "internal.avi")
	b.Run("createAVIWithMotionJPEG", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			file, createErr := os.Create(path)
			if createErr != nil {
				b.Fatal(createErr)
			}
			benchmarkInternalError = createAVIWithMotionJPEG(file, [][]byte{data}, 32, 32, 1)
			closeErr := file.Close()
			if benchmarkInternalError == nil {
				benchmarkInternalError = closeErr
			}
		}
	})
}

func BenchmarkInternalBinaryWriters(b *testing.B) {
	var buffer bytes.Buffer
	b.Run("writeUint16LE", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			buffer.Reset()
			benchmarkInternalError = writeUint16LE(&buffer, 1)
		}
	})
	b.Run("writeUint32LE", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			buffer.Reset()
			benchmarkInternalError = writeUint32LE(&buffer, 1)
		}
	})
	b.Run("writeInt16LE", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			buffer.Reset()
			benchmarkInternalError = writeInt16LE(&buffer, 1)
		}
	})
	b.Run("writeInt32LE", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			buffer.Reset()
			benchmarkInternalError = writeInt32LE(&buffer, 1)
		}
	})
	b.Run("writeUint8LE", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			buffer.Reset()
			benchmarkInternalError = writeUint8LE(&buffer, 1)
		}
	})
	b.Run("writeString", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			buffer.Reset()
			benchmarkInternalError = writeString(&buffer, "RIFF")
		}
	})
}

func internalBenchmarkScene() *scene.Scene {
	return &scene.Scene{
		Camera: scene.Camera{
			Position:    scene.Vec3{Z: 5},
			Target:      scene.Vec3{},
			Up:          scene.Vec3{Y: 1},
			FOV:         45,
			FocusDist:   5,
			AspectRatio: 16.0 / 9.0,
		},
		Objects: []scene.Object{
			{Type: scene.ObjectPlane, Position: scene.Vec3{Y: -1}, MaterialID: "ground"},
			{Type: scene.ObjectSphere, Position: scene.Vec3{X: -1}, Size: scene.Vec3{X: 1}, MaterialID: "lambert"},
			{
				Type:       scene.ObjectBox,
				Position:   scene.Vec3{X: 1},
				Size:       scene.Vec3{X: 2, Y: 2, Z: 2},
				MaterialID: "metal",
			},
		},
		Materials: []scene.Material{
			{ID: "ground", Type: scene.MaterialLambert, Albedo: scene.Color{R: 0.4, G: 0.4, B: 0.4}, Rough: 0.7},
			{ID: "lambert", Type: scene.MaterialLambert, Albedo: scene.Color{R: 0.8, G: 0.2, B: 0.1}, Rough: 0.4},
			{ID: "metal", Type: scene.MaterialMetal, Albedo: scene.Color{R: 0.7, G: 0.7, B: 0.75}, Rough: 0.2},
			{ID: "dielectric", Type: scene.MaterialDielectric, IOR: 1.5},
			{ID: "emissive", Type: scene.MaterialEmissive, Emit: scene.Color{R: 1, G: 0.9, B: 0.7}, Power: 4},
			{ID: "mirror", Type: scene.MaterialMirror, Albedo: scene.Color{R: 0.9, G: 0.9, B: 0.9}},
		},
		Settings: scene.RenderSettings{
			Width:        internalBenchmarkWidth,
			Height:       internalBenchmarkHeight,
			SamplesPerPx: internalBenchmarkSpp,
			MaxDepth:     internalBenchmarkDepth,
			MaxRayDist:   internalBenchmarkRayMax,
		},
		Background: scene.Color{R: 0.05, G: 0.05, B: 0.05},
	}
}

func internalBenchmarkConfig() Config {
	return Config{
		Width:        internalBenchmarkWidth,
		Height:       internalBenchmarkHeight,
		SamplesPerPx: internalBenchmarkSpp,
		MaxDepth:     internalBenchmarkDepth,
		MaxRayDist:   internalBenchmarkRayMax,
	}
}

var (
	benchmarkInternalVec       vec3
	benchmarkInternalFloat     float64
	benchmarkInternalRNG       *randSource
	benchmarkInternalCamera    camera
	benchmarkInternalRay       ray
	benchmarkInternalMaterial  material
	benchmarkInternalScatterOK bool
	benchmarkInternalWorld     []hittable
	benchmarkInternalHit       bool
	benchmarkInternalColor     vec3
	benchmarkInternalJPEG      []byte
	benchmarkInternalError     error
)

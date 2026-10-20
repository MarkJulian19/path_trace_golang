package render_test

import (
	"flag"
	"os"
	"testing"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	"github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

const (
	testWidth       = 32
	testHeight      = 18
	testSamples     = 2
	testDepth       = 4
	testMaxRayDist  = 1000
	testCameraFOV   = 45
	testFocusDist   = 5
	testSphereSize  = 1
	testPlaneY      = -1
	testBoxSize     = 2
	testMetalRough  = 0.2
	testMetalSmooth = 0.8
	testMetalIOR    = 1.5
	testLightPower  = 4
	testAspectRatio = 16.0 / 9.0
	backgroundValue = 0.05
)

func TestMain(m *testing.M) {
	benchmarking := false
	if benchmarkFlag := flag.Lookup("test.bench"); benchmarkFlag != nil {
		benchmarking = benchmarkFlag.Value.String() != ""
	}
	workersSet := false
	if !benchmarking {
		err := os.Setenv("PATHTRACER_WORKERS", "1")
		if err != nil {
			os.Exit(1)
		}
		workersSet = true
	}
	code := m.Run()
	if workersSet {
		err := os.Unsetenv("PATHTRACER_WORKERS")
		if err != nil && code == 0 {
			code = 1
		}
	}
	os.Exit(code)
}

func benchmarkScene() *scene.Scene {
	return &scene.Scene{
		Name: "benchmark-scene",
		Camera: scene.Camera{
			Position:    scene.Vec3{X: 0, Y: 1, Z: 5},
			Target:      scene.Vec3{X: 0, Y: 0, Z: 0},
			Up:          scene.Vec3{X: 0, Y: 1, Z: 0},
			FOV:         testCameraFOV,
			FocusDist:   testFocusDist,
			AspectRatio: testAspectRatio,
		},
		Objects: []scene.Object{
			{
				ID:         "ground",
				Type:       scene.ObjectPlane,
				Position:   scene.Vec3{Y: testPlaneY},
				MaterialID: "ground-material",
			},
			{
				ID:         "sphere",
				Type:       scene.ObjectSphere,
				Position:   scene.Vec3{X: -1, Y: 0, Z: 0},
				Size:       scene.Vec3{X: testSphereSize},
				MaterialID: "lambert-material",
			},
			{
				ID:         "box",
				Type:       scene.ObjectBox,
				Position:   scene.Vec3{X: 1, Y: 0, Z: 0},
				Size:       scene.Vec3{X: testBoxSize, Y: testBoxSize, Z: testBoxSize},
				MaterialID: "metal-material",
			},
			{
				ID:         "light",
				Type:       scene.ObjectSphereLight,
				Position:   scene.Vec3{X: 0, Y: 3, Z: 1},
				Size:       scene.Vec3{X: testSphereSize},
				MaterialID: "light-material",
			},
		},
		Materials: []scene.Material{
			{
				ID:         "ground-material",
				Type:       scene.MaterialLambert,
				Albedo:     scene.Color{R: 0.4, G: 0.4, B: 0.4},
				Rough:      0.7,
				Power:      1,
				Absorption: scene.Color{},
			},
			{
				ID:         "lambert-material",
				Type:       scene.MaterialLambert,
				Albedo:     scene.Color{R: 0.8, G: 0.2, B: 0.1},
				Rough:      0.4,
				Power:      1,
				Absorption: scene.Color{},
			},
			{
				ID:         "metal-material",
				Type:       scene.MaterialMetal,
				Albedo:     scene.Color{R: 0.7, G: 0.7, B: 0.75},
				Rough:      testMetalRough,
				Smoothness: testMetalSmooth,
				IOR:        testMetalIOR,
				Power:      1,
			},
			{
				ID:         "light-material",
				Type:       scene.MaterialEmissive,
				Emit:       scene.Color{R: 1, G: 0.9, B: 0.7},
				Power:      testLightPower,
				Albedo:     scene.Color{},
				Rough:      0,
				Absorption: scene.Color{},
			},
		},
		Settings: scene.RenderSettings{
			Width:        testWidth,
			Height:       testHeight,
			SamplesPerPx: testSamples,
			MaxDepth:     testDepth,
			MaxRayDist:   testMaxRayDist,
		},
		Background: scene.Color{R: backgroundValue, G: backgroundValue, B: backgroundValue},
	}
}

func testConfig() render.Config {
	return render.Config{
		Width:        testWidth,
		Height:       testHeight,
		SamplesPerPx: testSamples,
		MaxDepth:     testDepth,
		MaxRayDist:   testMaxRayDist,
	}
}

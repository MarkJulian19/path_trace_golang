package render

import (
	"image"
	"math"
	"testing"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

func TestCPUFogUsesGPUDefaults(t *testing.T) {
	t.Parallel()
	fog := newCPUFog(scene.Fog{Density: 0.5, Scatter: 0.4})
	if fog.sigmaS != 0.2 || fog.sigmaA != 0.3 {
		t.Fatalf("fog coefficients = (%v, %v), want (0.2, 0.3)", fog.sigmaS, fog.sigmaA)
	}
	if fog.noiseScale != 4 || fog.noiseOctaves != 3 {
		t.Fatalf("fog noise defaults = (%v, %d), want (4, 3)", fog.noiseScale, fog.noiseOctaves)
	}
}

func TestCPUTonemapMatchesGPUFormula(t *testing.T) {
	t.Parallel()
	if got := cpuACES(0); got != 0 {
		t.Fatalf("ACES(0) = %v, want 0", got)
	}
	wantACES := 2.54 / 3.16
	if got := cpuACES(1); math.Abs(got-wantACES) > 1e-12 {
		t.Fatalf("ACES(1) = %v, want %v", got, wantACES)
	}
	if got := cpuTonemapGamma(1); math.Abs(got-math.Sqrt(wantACES)) > 1e-12 {
		t.Fatalf("gamma(1) = %v, want %v", got, math.Sqrt(wantACES))
	}
}

func TestCPUVolumetricParityRenderIsFinite(t *testing.T) {
	t.Parallel()
	sc := &scene.Scene{
		Camera: scene.Camera{
			Position: scene.Vec3{Z: 3},
			Target:   scene.Vec3{},
			Up:       scene.Vec3{Y: 1},
			FOV:      45,
		},
		Sky: &scene.Sky{Type: "solid", Color: scene.Color{R: 0.2, G: 0.3, B: 0.5}},
		Fog: &scene.Fog{
			Density:       0.08,
			Color:         scene.Color{R: 0.6, G: 0.7, B: 0.8},
			Scatter:       0.5,
			AffectSky:     true,
			GPUVolumetric: true,
		},
	}
	cfg := Config{Width: 8, Height: 8, SamplesPerPx: 1, MaxDepth: 3, MaxRayDist: 100}
	img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
	Into(sc, cfg, img, nil)
	for i := 3; i < len(img.Pix); i += 4 {
		if img.Pix[i] != 255 {
			t.Fatalf("pixel alpha at %d = %d, want 255", i/4, img.Pix[i])
		}
	}
}

func TestCPURenderWorldMatchesGPULightSourceSlots(t *testing.T) {
	t.Parallel()

	sc := &scene.Scene{
		Objects: []scene.Object{
			{
				ID:         "fallback-material-object",
				Type:       scene.ObjectSphere,
				Position:   scene.Vec3{X: -3, Y: 1},
				Size:       scene.Vec3{X: 1},
				MaterialID: "missing-material-id",
			},
			{
				ID:         "zero-power-light",
				Type:       scene.ObjectSphereLight,
				Position:   scene.Vec3{Y: 3},
				Size:       scene.Vec3{X: 0.5},
				MaterialID: "zero-power",
			},
			{
				ID:         "box-light",
				Type:       scene.ObjectBox,
				Position:   scene.Vec3{Y: 5},
				Size:       scene.Vec3{X: 2, Y: 0.2, Z: 2},
				MaterialID: "box-light",
			},
			{
				ID:         "non-light",
				Type:       scene.ObjectSphere,
				Position:   scene.Vec3{X: 3, Y: 1},
				Size:       scene.Vec3{X: 1},
				MaterialID: "diffuse",
			},
		},
		Materials: []scene.Material{
			{
				ID:    "fallback-light",
				Type:  scene.MaterialEmissive,
				Emit:  scene.Color{R: 1, G: 0.5, B: 0.25},
				Power: 2,
			},
			{
				ID:    "zero-power",
				Type:  scene.MaterialEmissive,
				Emit:  scene.Color{G: 1},
				Power: 0,
			},
			{
				ID:    "box-light",
				Type:  scene.MaterialEmissive,
				Emit:  scene.Color{R: 0.5, G: 0.5, B: 0.5},
				Power: 4,
			},
			{
				ID:   "diffuse",
				Type: scene.MaterialLambert,
			},
		},
	}

	world := newCPURenderWorld(sc, Config{MaxRayDist: 100})
	if len(world.objects) != len(sc.Objects) {
		t.Fatalf("world object count = %d, want %d", len(world.objects), len(sc.Objects))
	}
	if len(world.lights) != 3 {
		t.Fatalf("light slot count = %d, want 3", len(world.lights))
	}

	for i, object := range world.objects {
		if object.sceneIdx != i {
			t.Errorf("world object %d has scene index %d, want %d", i, object.sceneIdx, i)
		}
	}
	if world.lights[0].sceneIdx != 0 || !world.lights[0].valid {
		t.Fatalf("fallback-material light = %+v, want valid scene slot 0", world.lights[0])
	}
	if world.lights[1].sceneIdx != 1 || !world.lights[1].valid {
		t.Fatalf("zero-power light = %+v, want valid scene slot 1", world.lights[1])
	}
	if world.lights[1].emit != (vec3{}) {
		t.Fatalf("zero-power light emission = %+v, want zero", world.lights[1].emit)
	}
	if world.lights[2].sceneIdx != 2 || world.lights[2].valid {
		t.Fatalf("box light = %+v, want invalid geometry slot 2", world.lights[2])
	}
	if world.lights[0].emit != v(2, 1, 0.5) {
		t.Fatalf("fallback-material emission = %+v, want (2, 1, 0.5)", world.lights[0].emit)
	}
}

func TestCPUDirectLightFromSphereSourceIsNonZero(t *testing.T) {
	t.Parallel()

	sc := &scene.Scene{
		Objects: []scene.Object{
			{
				ID:         "ground",
				Type:       scene.ObjectPlane,
				MaterialID: "ground",
			},
			{
				ID:         "light",
				Type:       scene.ObjectSphereLight,
				Position:   scene.Vec3{Y: 3},
				Size:       scene.Vec3{X: 1},
				MaterialID: "light",
			},
		},
		Materials: []scene.Material{
			{ID: "ground", Type: scene.MaterialLambert},
			{ID: "light", Type: scene.MaterialEmissive, Emit: scene.Color{R: 1, G: 0.8, B: 0.6}, Power: 4},
		},
	}
	world := newCPURenderWorld(sc, Config{MaxRayDist: 100})
	if len(world.lights) != 1 || !world.lights[0].valid {
		t.Fatalf("sphere light slots = %+v, want one valid slot", world.lights)
	}

	surfHit := hitRecord{p: v(0, 0, 0), normal: v(0, 1, 0)}
	total := vec3{}
	for seed := uint64(1); seed <= 128; seed++ {
		rng := &randSource{state: 0x9e3779b97f4a7c15 ^ seed}
		contrib := estimateDirectLightSingleCPU(
			world,
			&world.lights[0],
			surfHit,
			v(1, 1, 1),
			rng,
		)
		total = total.add(contrib)
	}
	if total.x <= 0 || total.y <= 0 || total.z <= 0 {
		t.Fatalf("direct light sum = %+v, want positive contribution in every channel", total)
	}
}

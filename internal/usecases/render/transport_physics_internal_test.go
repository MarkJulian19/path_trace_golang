package render

import (
	"math"
	"testing"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

func TestDielectricFresnelAndSnell(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct{ cosine, eta, want float64 }{
		{1, 1 / 1.5, 0.04},
		{0.5, 1 / 1.5, 0.08918671280221276},
		{0.5, 1.5, 1},
		{0, 1, 0},
		{0, 1 / 1.5, 1},
	} {
		if got := reflectance(tc.cosine, tc.eta); math.Abs(got-tc.want) > 1e-12 {
			t.Errorf("Fresnel(%v,%v)=%v, want %v", tc.cosine, tc.eta, got, tc.want)
		}
	}
	normal := v(0, 1, 0)
	incident := v(0.6, -0.8, 0)
	transmitted := refractVec(incident, normal, 1/1.5)
	reverse := refractVec(transmitted.mul(-1), normal.mul(-1), 1.5)
	if math.Abs(transmitted.x-0.4) > 1e-12 || math.Abs(transmitted.length()-1) > 1e-12 ||
		reverse.add(incident).length() > 1e-12 {
		t.Fatalf("Snell refraction or reciprocity failed: transmitted=%+v reverse=%+v", transmitted, reverse)
	}
}

func TestPickingRayUsesPixelCenter(t *testing.T) {
	t.Parallel()
	sc := &scene.Scene{Camera: scene.Camera{
		Position: scene.Vec3{Z: 3}, Up: scene.Vec3{Y: 1},
		FOV: 90, AspectRatio: 1, Aperture: 1,
	}}
	center := pickRay(sc, Config{Width: 1, Height: 1}, 0, 0)
	corner := pickRay(sc, Config{Width: 2, Height: 2}, 0, 0)
	if center.dir.sub(v(0, 0, -1)).length() > 1e-12 ||
		corner.dir.sub(v(-0.5, 0.5, -1).unit()).length() > 1e-12 {
		t.Fatalf("picking projection: single-pixel ray=%+v, corner ray=%+v", center, corner)
	}
}

func TestNestedDielectricInterfaces(t *testing.T) {
	t.Parallel()
	world := &cpuRenderWorld{objects: []hittable{
		{kind: hittableSphere, sphere: sphere{radius: 3, mat: material{typ: matDielectric, ior: 1.5}}},
		{kind: hittableSphere, sphere: sphere{radius: 1, mat: material{typ: matDielectric, ior: 1.33}}},
	}}
	r := ray{dir: v(1, 0, 0)}
	if got := mediumAtCPU(world, r, -1); got != 1 {
		t.Fatalf("camera inside nested glass: medium=%v, want 1", got)
	}
	rec := hitRecord{p: v(1, 0, 0), normal: v(-1, 0, 0), objectIdx: 1}
	eta, next := dielectricInterfaceCPU(world, r, rec, 1)
	if math.Abs(eta-1.33/1.5) > 1e-12 || next != 0 {
		t.Fatalf("inner-to-outer glass: eta=%v medium=%v", eta, next)
	}
}

func TestGlassSlabTransport(t *testing.T) {
	t.Parallel()
	sc := &scene.Scene{
		Materials: []scene.Material{{
			ID: "glass", Type: scene.MaterialDielectric,
			IOR: 1, Absorption: scene.Color{R: 1, G: 1, B: 1}, AbsorptionScale: 1,
		}},
		Objects: []scene.Object{{Type: scene.ObjectBox, MaterialID: "glass", Size: scene.Vec3{X: 2, Y: 2, Z: 2}}},
	}
	world := newCPURenderWorld(sc, Config{MaxRayDist: 100})
	rng := &randSource{state: 1234567}
	got := rayColorGPUParity(ray{orig: v(0, 0, 3), dir: v(0, 0, -1)}, world,
		func(ray) vec3 { return v(1, 1, 1) }, 3, rng)
	if math.Abs(got.x-math.Exp(-2)) > 0.00001 {
		t.Fatalf("glass slab transmission=%v, want exp(-2)", got.x)
	}
}

func TestSphereLightsAnalyticIrradiance(t *testing.T) {
	t.Parallel()
	for _, count := range []int{1, 2, 9} {
		sc, want := analyticLightScene(count)
		world := newCPURenderWorld(sc, Config{MaxRayDist: 100})
		for _, depth := range []int{1, 2} {
			rng := &randSource{state: 123456789}
			sum := 0.0
			const trials = 60000
			for range trials {
				value := rayColorGPUParity(ray{orig: v(0, 0.5, 0), dir: v(0, -1, 0)}, world,
					func(ray) vec3 { return vec3{} }, depth, rng)
				sum += value.x
			}
			got := sum / trials
			t.Logf("lights=%d depth=%d radiance=%v analytic=%v", count, depth, got, want)
			if math.Abs(got-want) > want*0.025 {
				t.Errorf("lights=%d depth=%d: radiance=%v, want %v", count, depth, got, want)
			}
		}
	}
}

func analyticLightScene(count int) (*scene.Scene, float64) {
	sc := &scene.Scene{Materials: []scene.Material{
		{ID: "white", Type: scene.MaterialLambert, Albedo: scene.Color{R: 1, G: 1, B: 1}},
		{ID: "lamp", Type: scene.MaterialEmissive, Emit: scene.Color{R: 4, G: 4, B: 4}, Power: 1},
	}, Objects: []scene.Object{{Type: scene.ObjectPlane, MaterialID: "white"}}}
	for i := range count {
		phi := 2 * math.Pi * float64(i) / float64(count)
		sc.Objects = append(sc.Objects, scene.Object{
			Type: scene.ObjectSphere, MaterialID: "lamp",
			Position: scene.Vec3{X: 4 * math.Cos(phi), Y: 3, Z: 4 * math.Sin(phi)}, Size: scene.Vec3{X: 0.2},
		})
	}
	// Irradiance of a sphere: Le*pi*(radius/distance)^2*cos(axis,normal).
	return sc, float64(count) * 4 * 0.2 * 0.2 / 25 * 3 / 5
}

func TestGGXReciprocityAndWhiteFurnace(t *testing.T) {
	t.Parallel()
	normal := v(0, 1, 0)
	for _, roughness := range []float64{0.01, 0.2, 0.6, 1} {
		m := material{typ: matMetal, albedo: v(1, 1, 1), reflectivity: 1, smoothness: 1 - roughness}
		for _, cosine := range []float64{1, 0.2} {
			incident := v(math.Sqrt(1-cosine*cosine), -cosine, 0)
			rng := &randSource{state: 123456789}
			energy := 0.0
			const trials = 60000
			for range trials {
				wi := sampleGGXCPU(incident.mul(-1), normal, roughness, rng)
				f, pdf := surfaceBSDFCPU(&m, incident, wi, normal)
				if pdf <= 0 {
					continue
				}
				reciprocal, _ := surfaceBSDFCPU(&m, wi.mul(-1), incident.mul(-1), normal)
				if math.Abs(f.x-reciprocal.x) > max(1, f.x)*1e-8 {
					t.Fatal("GGX violates reciprocity")
				}
				energy += f.x * wi.dot(normal) / pdf
			}
			energy /= trials
			t.Logf("GGX roughness=%v cosine=%v reflected energy=%v", roughness, cosine, energy)
			if math.IsNaN(energy) || energy < 0 || energy > 1.02 {
				t.Errorf("GGX violates energy conservation: %v", energy)
			}
		}
	}
}

func TestFogSamplingAndLensDistribution(t *testing.T) {
	t.Parallel()
	rng := &randSource{state: 123456789}
	fog := cpuFog{sigmaA: 0.7, volumetric: true}
	const trials = 100000
	survival, meanCosine, radiusSquared := 0.0, 0.0, 0.0
	cameraValue := camera{rng: rng, lensRadius: 1, u: v(1, 0, 0), v: v(0, 1, 0), lowerLeftCorner: v(0, 0, -1)}
	for range trials {
		_, _, scattered := sampleFogDistanceCPU(ray{dir: v(1, 0, 0)}, 2, fog, rng)
		if !scattered {
			survival++
		}
		meanCosine += samplePhaseHGCPU(v(0, 0, 1), 0.7, rng).z
		lens := cameraValue.getRay(0, 0).orig
		radiusSquared += lens.dot(lens)
	}
	if math.Abs(survival/trials-math.Exp(-1.4)) > 0.005 || math.Abs(meanCosine/trials-0.7) > 0.005 ||
		math.Abs(radiusSquared/trials-0.5) > 0.005 {
		t.Fatalf("incorrect distributions: fog=%v phase=%v lens r²=%v", survival/trials,
			meanCosine/trials, radiusSquared/trials)
	}
}

func TestHeterogeneousFogTransmittance(t *testing.T) {
	t.Parallel()
	fog := cpuFog{
		sigmaS: 0.4, sigmaA: 0.6, volumetric: true,
		heteroStrength: 0.7, noiseScale: 0.4, noiseOctaves: 3,
	}
	r := ray{orig: v(0.1, 0.2, 0.3), dir: v(1, 0, 0)}
	const steps = 100000
	opticalDepth := 0.0
	for i := range steps {
		_, extinction := fog.mediumCoefficients(r.at((float64(i) + 0.5) * 2 / steps))
		opticalDepth += extinction * 2 / steps
	}
	rng := &randSource{state: 123456789}
	sum := 0.0
	for range steps {
		sum += fogTransmittanceCPU(r, 2, fog, rng)
	}
	want := math.Exp(-opticalDepth)
	if got := sum / steps; math.Abs(got-want) > 0.005 {
		t.Fatalf("heterogeneous transmittance=%v, quadrature=%v", got, want)
	}
}

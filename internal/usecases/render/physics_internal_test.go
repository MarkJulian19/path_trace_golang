package render

import (
	"math"
	"testing"
)

func TestGlassBeerLambert(t *testing.T) {
	t.Parallel()
	m := &material{absorption: v(2, 1, 0), absorptionScale: 1, tint: v(1, 1, 1)}
	for _, distance := range []float64{0, 1, 2, 20} {
		got := dielectricAttenuation(m, distance)
		want := v(math.Exp(-2*distance), math.Exp(-distance), 1)
		if got.sub(want).length() > 1e-12 {
			t.Errorf("distance=%v: transmission=%+v, want %+v", distance, got, want)
		}
	}
	a := dielectricAttenuation(m, 1)
	b := dielectricAttenuation(m, 2)
	if mulVec3(a, b).sub(dielectricAttenuation(m, 3)).length() > 1e-12 {
		t.Fatal("transmittance is not multiplicative along the ray")
	}
}

func TestBoxRayStartingInsideFindsExit(t *testing.T) {
	t.Parallel()
	shape := box{min: v(-1, -1, -1), max: v(1, 1, 1)}
	var rec hitRecord
	if !shape.hit(ray{dir: v(1, 0, 0)}, renderMinRayDistance, 100, &rec) {
		t.Fatal("ray inside box did not find its exit")
	}
	if rec.t != 1 || rec.frontFace || rec.normal != v(-1, 0, 0) {
		t.Fatalf("exit intersection is wrong: %+v", rec)
	}
}

func TestHenyeyGreensteinIntegratesToOne(t *testing.T) {
	t.Parallel()
	const steps = 10000
	for _, g := range []float64{-0.8, 0, 0.8} {
		sum := 0.0
		for i := range steps {
			cosine := -1 + (float64(i)+0.5)*2/steps
			sum += phaseHGCPU(cosine, g) * 4 * math.Pi / steps
		}
		if math.Abs(sum-1) > 1e-5 {
			t.Errorf("g=%v: phase integral=%v, want 1", g, sum)
		}
	}
}

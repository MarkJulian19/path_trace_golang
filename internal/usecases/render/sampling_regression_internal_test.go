package render

import (
	"math"
	"runtime"
	"testing"
)

func TestWorkerCountUsesAllAvailableCPUs(t *testing.T) {
	t.Setenv("PATHTRACER_WORKERS", "")
	if got := configuredWorkerCount(); got != runtime.NumCPU() {
		t.Fatalf("default renderer reserves CPUs: workers=%d available=%d", got, runtime.NumCPU())
	}
	t.Setenv("PATHTRACER_WORKERS", "14")
	if got := configuredWorkerCount(); got != 14 {
		t.Fatalf("explicit worker limit was ignored: got %d", got)
	}
}

func TestGGXNearNormalRotatesWithView(t *testing.T) {
	t.Parallel()
	normal := v(0, 0, 1)
	view := v(0.01, 0, 1).unit()
	rotatedView := v(0, view.x, view.z)
	first, second := &randSource{state: 1234567}, &randSource{state: 1234567}
	for range 1000 {
		a := sampleGGXCPU(view, normal, 0.2, first)
		b := sampleGGXCPU(rotatedView, normal, 0.2, second)
		if b.sub(v(-a.y, a.x, a.z)).length() > 1e-10 {
			t.Fatalf("GGX basis loses rotational symmetry near normal: first=%+v rotated=%+v", a, b)
		}
	}
}

func TestPathRandomDistribution(t *testing.T) {
	t.Parallel()
	const count = 200000
	rng := &randSource{state: 123456789}
	var mean, secondMoment, covariance float64
	var bins [16]int
	for range count {
		x, y := rng.Float64(), rng.Float64()
		if x < 0 || x >= 1 || y < 0 || y >= 1 {
			t.Fatal("random value is outside [0, 1)")
		}
		mean += x
		secondMoment += x * x
		covariance += (x - 0.5) * (y - 0.5)
		bins[int(x*4)*4+int(y*4)]++
	}
	if math.Abs(mean/count-0.5) > 0.003 || math.Abs(secondMoment/count-1.0/3) > 0.003 ||
		math.Abs(covariance/count) > 0.001 {
		t.Fatalf("biased or correlated random pairs: mean=%v moment=%v covariance=%v",
			mean/count, secondMoment/count, covariance/count)
	}
	for _, bin := range bins {
		if math.Abs(float64(bin)-count/16) > count/16*0.05 {
			t.Fatalf("non-uniform 2D random samples: bins=%v", bins)
		}
	}
}

func TestFloorLightGradientMatchesAnalyticFalloff(t *testing.T) {
	t.Parallel()
	sc, _ := analyticLightScene(1)
	world := newCPURenderWorld(sc, Config{MaxRayDist: 100})
	const trials = 12000
	for _, radius := range []float64{0, 0.5, 1, 2, 4, 8, 16} {
		point := v(4+radius, 0.5, 0)
		distanceSquared := radius*radius + 9
		want := 4 * 0.2 * 0.2 * 3 / math.Pow(distanceSquared, 1.5)
		for _, seed := range []uint64{1, 123456789} {
			rng := &randSource{state: seed}
			var sum float64
			for range trials {
				sum += rayColorGPUParity(ray{orig: point, dir: v(0, -1, 0)}, world,
					func(ray) vec3 { return vec3{} }, 1, rng).x
			}
			if got := sum / trials; math.Abs(got-want) > want*0.01 {
				t.Errorf("floor radius=%v seed=%d: got=%v analytic=%v", radius, seed, got, want)
			}
		}
	}
}

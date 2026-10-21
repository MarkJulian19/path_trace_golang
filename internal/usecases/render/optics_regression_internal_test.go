package render

import (
	"bytes"
	"image"
	"math"
	"testing"

	"github.com/MarkJulian19/path_trace_golang/internal/repository/scenejson"
)

func TestReflectedRaysLeaveConvexSurfaces(t *testing.T) {
	t.Parallel()
	origin := v(6, 2.8, 8)
	for _, center := range []vec3{v(-2.2, 1, 0), v(0, 1, 0), v(2.2, 1, 0)} {
		objects := []hittable{
			{kind: hittableSphere, sphere: sphere{center: center, radius: 1}},
			{kind: hittableBox, box: box{min: center.sub(v(1, 1, 1)), max: center.add(v(1, 1, 1))}},
		}
		for _, object := range objects {
			rng := &randSource{state: 123456789}
			for range 8192 {
				jitter := v(2*rng.Float64()-1, 2*rng.Float64()-1, 2*rng.Float64()-1).mul(0.1)
				r := ray{orig: origin, dir: center.sub(origin).unit().add(jitter).unit()}
				var hit, next hitRecord
				if !object.hit(r, 0, 100, &hit) {
					continue
				}
				reflected := offsetRayCPU(hit.p, hit.normal, reflectVec(r.dir, hit.normal))
				if object.hit(reflected, 0, 100, &next) {
					t.Fatalf(
						"reflected ray hits its own surface: kind=%v center=%+v distance=%v",
						object.kind,
						center,
						next.t,
					)
				}
			}
		}
	}
}

func TestThinGlassKeepsBothInterfaces(t *testing.T) {
	t.Parallel()
	const thickness = 0.00002
	glass := material{typ: matDielectric, ior: 1.5, tint: v(1, 1, 1)}
	world := &cpuRenderWorld{objects: []hittable{{
		kind: hittableBox,
		box:  box{min: v(-1, -1, 3), max: v(1, 1, 3+thickness), mat: glass},
	}}, maxRayDist: 100}
	r := ray{orig: v(0, 0, 4), dir: v(0.5, 0, -math.Sqrt(0.75))}
	var entry, exit hitRecord
	if !hitClosestCPU(world, r, 100, &entry) {
		t.Fatal("missing glass entry")
	}
	inside := refractVec(r.dir, entry.normal, 1/1.5)
	r = offsetRayCPU(entry.p, entry.normal, inside)
	if !hitClosestCPU(world, r, 100, &exit) || exit.frontFace {
		t.Fatalf("offset skipped the exit of thin glass: origin=%+v exit=%+v", r.orig, exit)
	}
	outgoing := refractVec(r.dir, exit.normal, 1.5)
	if outgoing.sub(v(0.5, 0, -math.Sqrt(0.75))).length() > 1e-12 {
		t.Fatalf("parallel glass slab changed the emergent direction: %+v", outgoing)
	}
}

func TestCPUFilterAndSeedAreEffective(t *testing.T) {
	t.Parallel()
	sc, err := scenejson.New().Load("../../../scenes/validation/lighting_noise.json")
	if err != nil {
		t.Fatal(err)
	}
	service := New(nil, nil)
	cfg := Config{Width: 64, Height: 36, SamplesPerPx: 16, MaxDepth: 12, MaxRayDist: 100, Seed: 42}
	service.SetDenoiseConfigFromUI(false, 1, 0.15)
	raw := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
	repeated := image.NewRGBA(raw.Bounds())
	service.Into(sc, cfg, raw, nil)
	service.Into(sc, cfg, repeated, nil)
	if !bytes.Equal(raw.Pix, repeated.Pix) {
		t.Fatal("fixed seed is not reproducible across worker scheduling")
	}
	service.SetDenoiseConfigFromUI(true, 1, 0.15)
	filtered := image.NewRGBA(raw.Bounds())
	service.Into(sc, cfg, filtered, nil)
	if bytes.Equal(raw.Pix, filtered.Pix) {
		t.Fatal("CPU denoise control did not change the image")
	}
}

func TestSlabLateralDisplacement(t *testing.T) {
	t.Parallel()
	const thickness = 0.5
	const incidentAngle = math.Pi / 4
	transmittedAngle := math.Asin(math.Sin(incidentAngle) / 1.5)
	incident := v(math.Sin(incidentAngle), 0, -math.Cos(incidentAngle))
	normal := v(0, 0, 1)
	inside := refractVec(incident, normal, 1/1.5)
	displacement := thickness * inside.x / -inside.z
	want := thickness * math.Tan(transmittedAngle)
	if math.Abs(displacement-want) > 1e-12 {
		t.Fatalf("slab displacement=%v, want %v", displacement, want)
	}
}

func TestNestedHalfspacesChooseInnermostMedium(t *testing.T) {
	t.Parallel()
	world := &cpuRenderWorld{objects: []hittable{
		{kind: hittablePlane, pln: plane{point: v(0, 2, 0), mat: material{typ: matDielectric, ior: 1.5}}},
		{kind: hittablePlane, pln: plane{point: v(0, 1, 0), mat: material{typ: matDielectric, ior: 1.33}}},
	}}
	for _, direction := range []vec3{v(0, 1, 0), v(0, -1, 0), v(1, 0, 0)} {
		if got := mediumAtCPU(world, ray{dir: direction}, -1); got != 1 {
			t.Errorf("medium depends on ray direction %+v: got %d, want 1", direction, got)
		}
	}
}

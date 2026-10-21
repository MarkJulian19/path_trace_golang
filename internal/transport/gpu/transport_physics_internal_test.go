package gpu

import (
	"image"
	"math"
	"testing"
	"time"

	"github.com/go-gl/gl/v3.3-core/gl"

	renderentity "github.com/MarkJulian19/path_trace_golang/internal/entities/render"
	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

func readLinearGPU(r *gpuRenderer, width, height int) []float32 {
	values := make([]float32, width*height*gpuRGBAComponents)
	gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)
	gl.BindTexture(gl.TEXTURE_2D, r.imgTexture)
	gl.GetTexImage(gl.TEXTURE_2D, 0, gl.RGBA, gl.FLOAT, gl.Ptr(values))
	return values
}

func checkGPUAnalyticTransport(t *testing.T, r *gpuRenderer, prefix string) {
	t.Helper()
	program, err := createComputeProgram(prefix+analyticProbeMain, "analytic transport")
	if err != nil {
		t.Fatal(err)
	}
	defer gl.DeleteProgram(program)
	r.program = program
	for _, count := range []int{1, 2, 9} {
		sc := gpuAnalyticLightScene(count)
		want := float64(count) * 4 * 0.2 * 0.2 / 25 * 3 / 5
		checkGPUProbeMean(t, r, sc, want)
	}
	sc := gpuAnalyticLightScene(1)
	sc.Objects[1] = scene.Object{
		Type: scene.ObjectBox, MaterialID: "lamp",
		Position: scene.Vec3{Y: 3}, Size: scene.Vec3{X: 2, Y: 0.01, Z: 2},
	}
	checkGPUProbeMean(t, r, sc, boxLightReference())
	sc = &scene.Scene{
		Materials: []scene.Material{{
			ID: "mirror", Type: scene.MaterialMirror,
			Albedo: scene.Color{R: 0.6, G: 0.6, B: 0.6}, Reflectivity: 0.75,
		}},
		Objects: []scene.Object{{Type: scene.ObjectPlane, MaterialID: "mirror"}},
		Sky:     &scene.Sky{Type: "solid", Color: scene.Color{R: 0.8, G: 0.8, B: 0.8}},
	}
	checkGPUProbeMean(t, r, sc, 0.36)
	sc.Materials = []scene.Material{{ID: "glass", Type: scene.MaterialDielectric, IOR: 1.5}}
	sc.Objects = []scene.Object{{Type: scene.ObjectBox, MaterialID: "glass", Size: scene.Vec3{X: 2, Y: 2, Z: 2}}}
	sc.Sky.Color = scene.Color{R: 1, G: 1, B: 1}
	checkGPUProbeMean(t, r, sc, 2.25)
	sc.Materials = []scene.Material{
		{ID: "lamp", Type: scene.MaterialEmissive, Emit: scene.Color{R: 1, G: 1, B: 1}, Power: 1},
	}
	sc.Objects = []scene.Object{{Type: scene.ObjectPlane, MaterialID: "lamp"}}
	sc.Fog = &scene.Fog{GPUVolumetric: true, SigmaA: 0.7, Color: scene.Color{R: 1, G: 1, B: 1}}
	checkGPUProbeMean(t, r, sc, math.Exp(-0.35))
}

func checkGPUProbeMean(t *testing.T, r *gpuRenderer, sc *scene.Scene, want float64) {
	t.Helper()
	SetRenderCancel(false)
	cfg := renderentity.Config{Width: 32, Height: 32, SamplesPerPx: 1, MaxDepth: 12, MaxRayDist: 100}
	img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
	err := r.renderOnce(sc, cfg, img, nil, func() bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	values := readLinearGPU(r, cfg.Width, cfg.Height)
	mean := 0.0
	for i := 0; i < len(values); i += gpuRGBAComponents {
		mean += float64(values[i]) / float64(cfg.Width*cfg.Height)
	}
	t.Logf("GPU analytic objects=%d radiance=%v reference=%v", len(sc.Objects), mean, want)
	if math.IsNaN(mean) || math.Abs(mean-want) > want*0.035 {
		t.Fatalf("GPU radiance=%v, analytic=%v", mean, want)
	}
}

func gpuAnalyticLightScene(count int) *scene.Scene {
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
	return sc
}

func boxLightReference() float64 {
	const steps = 400
	const height = 2.995
	sum := 0.0
	for i := range steps {
		for j := range steps {
			x, z := -1+(float64(i)+0.5)*2/steps, -1+(float64(j)+0.5)*2/steps
			d2 := x*x + z*z + height*height
			sum += 4 / math.Pi * height * height / (d2 * d2) * 4 / (steps * steps)
		}
	}
	return sum
}

func checkGPUBatching(t *testing.T, r *gpuRenderer) {
	t.Helper()
	program, err := createComputeProgram(computeShader, "physical accumulation")
	if err != nil {
		t.Fatal(err)
	}
	defer gl.DeleteProgram(program)
	r.program = program
	sc := gpuAnalyticLightScene(2)
	sc.Camera = scene.Camera{Position: scene.Vec3{Y: 2, Z: 5}, Up: scene.Vec3{Y: 1}, FOV: 45}
	cfg := renderentity.Config{Width: 65, Height: 33, SamplesPerPx: 17, MaxDepth: 5, MaxRayDist: 100}
	deterministicGPUFrame(t, r, sc, cfg, 1, false)
	deterministicGPUFrame(t, r, sc, cfg, 4, false)
	reference, referenceElapsed := deterministicGPUFrame(t, r, sc, cfg, 1, false)
	t.Logf("GPU batch=1 elapsed=%v", referenceElapsed)
	for _, batch := range []int{4, 16} {
		for _, temporal := range []bool{false, true} {
			values, elapsed := deterministicGPUFrame(t, r, sc, cfg, batch, temporal)
			maxDifference := 0.0
			for i := range values {
				maxDifference = max(maxDifference, math.Abs(float64(values[i]-reference[i])))
			}
			t.Logf("GPU batch=%d temporal=%v max HDR difference=%v elapsed=%v", batch, temporal, maxDifference, elapsed)
			if maxDifference > 0.00002 {
				t.Fatalf("batching or temporal targets changed the estimator: delta=%v", maxDifference)
			}
		}
	}
}

func deterministicGPUFrame(t *testing.T, r *gpuRenderer, sc *scene.Scene, cfg renderentity.Config,
	batch int, temporal bool,
) ([]float32, time.Duration) {
	t.Helper()
	previousBatch, previousTemporal := gpuSamplesPerPass.Load(), temporalOn.Load()
	gpuSamplesPerPass.Store(int64(batch))
	temporalOn.Store(temporal)
	defer gpuSamplesPerPass.Store(previousBatch)
	defer temporalOn.Store(previousTemporal)
	img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
	frame, ok := r.prepareGPUFrame(sc, cfg, img, false, func() bool { return false })
	if !ok {
		t.Fatal("prepare deterministic GPU frame")
	}
	completed := 0
	request := newGPUSampleRequest(
		frame,
		cfg,
		img,
		func(current, _ int) { completed = current },
		func() bool { return false },
	)
	request.frameSeed = 123456789
	start := time.Now()
	r.renderSamples(request)
	gl.Finish()
	elapsed := time.Since(start)
	if completed != cfg.SamplesPerPx {
		t.Fatalf("sample budget: completed=%d, requested=%d", completed, cfg.SamplesPerPx)
	}
	return readLinearGPU(r, cfg.Width, cfg.Height), elapsed
}

const analyticProbeMain = `
void main() {
 ivec2 p=ivec2(gl_GlobalInvocationID.xy)+uDispatchOffset;
 if(p.x>=uWidth || p.y>=uHeight) return;
 uint state=hash_u(uint(p.x)*1973u ^ uint(p.y)*9277u ^ 123456789u);
 vec3 sum=vec3(0);
 for(int i=0;i<64;++i) sum+=rayColor(makeRay(vec3(0,0.5,0),vec3(0,-1,0)),state);
 imageStore(destTex,p,vec4(sum/64.0,1));
}
`

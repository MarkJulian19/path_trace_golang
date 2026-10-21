package gpu

import (
	"image"
	"math"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/go-gl/gl/v3.3-core/gl"

	renderentity "github.com/MarkJulian19/path_trace_golang/internal/entities/render"
	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

// TestGPUPhysicsAudit проверяет функции действующего шейдера через линейный HDR-буфер.
func TestGPUPhysicsAudit(t *testing.T) {
	t.Parallel()
	if os.Getenv("PATHTRACER_GPU_INTEGRATION") != "1" {
		t.Skip("requires an OpenGL 4.3 display; set PATHTRACER_GPU_INTEGRATION=1")
	}
	gpuIntegrationMu.Lock()
	defer gpuIntegrationMu.Unlock()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r := &gpuRenderer{}
	err := r.initGL()
	if err != nil {
		t.Fatal(err)
	}
	defer r.window.Destroy()
	prefix := computeShader[:strings.LastIndex(computeShader, "void main()")]
	program, err := createComputeProgram(prefix+physicsProbeMain, "physics audit")
	if err != nil {
		t.Fatal(err)
	}
	defer gl.DeleteProgram(program)
	r.program = program
	SetRenderModeFromUI(0)
	SetRenderCancel(false)
	SetLivePreview(false)
	SetDenoiseConfigFromUI(false, 1, 0.15)
	SetSmoothConfigFromUI(false, 1, 0)
	sc := &scene.Scene{
		Materials: []scene.Material{{
			ID: "glass", Type: scene.MaterialDielectric, IOR: 1,
			Absorption: scene.Color{R: 1, G: 1, B: 1}, AbsorptionScale: 1,
		}},
		Objects: []scene.Object{{Type: scene.ObjectBox, MaterialID: "glass", Size: scene.Vec3{X: 2, Y: 2, Z: 2}}},
		Sky:     &scene.Sky{Type: "solid", Color: scene.Color{R: 1, G: 1, B: 1}},
	}
	cfg := renderentity.Config{Width: 5, Height: 1, SamplesPerPx: 1, MaxDepth: 8, MaxRayDist: 100}
	img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
	err = r.renderOnce(sc, cfg, img, nil, func() bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	values := make([]float32, cfg.Width*4)
	gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)
	gl.BindTexture(gl.TEXTURE_2D, r.imgTexture)
	gl.GetTexImage(gl.TEXTURE_2D, 0, gl.RGBA, gl.FLOAT, gl.Ptr(values))
	cases := []struct {
		name            string
		index           int
		want, tolerance float64
	}{
		{name: "exact Fresnel at 60 degrees", index: 0, want: 0.08918671280221276, tolerance: 0.0001},
		{name: "ray exiting box", index: 4, want: 1, tolerance: 0.00001},
		{name: "Beer-Lambert through two units of glass", index: 8, want: math.Exp(-2), tolerance: 0.001},
		{name: "HDR radiance 100000 stays finite", index: 12, want: 100000, tolerance: 100},
		{name: "thin oblique slab preserves exit refraction", index: 16, want: 0.5, tolerance: 0.00001},
	}
	for _, tc := range cases {
		got := float64(values[tc.index])
		if math.Abs(got-tc.want) > tc.tolerance || math.IsInf(got, 0) || math.IsNaN(got) {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
		}
	}
	checkGPUNestedMedia(t, r, prefix)
	checkGPUSamplingRegression(t, r, prefix)
	checkGPUAnalyticTransport(t, r, prefix)
	checkGPUBatching(t, r)
}

const physicsProbeMain = `
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy)+uDispatchOffset;
    if(p.x>=uWidth || p.y>=uHeight) return;
    float result=0.0;
    if(p.x==0) result=reflectance(0.5,2.0/3.0);
    if(p.x==1) {
        Hit h;
        if(hitBox(vec3(-1),vec3(1),makeRay(vec3(0),vec3(1,0,0)),0.001,100.0,h)) result=h.t;
    }
    if(p.x==2) {
        uint state=1234567u;
        result=rayColor(makeRay(vec3(0,0,3),vec3(0,0,-1)),state).r;
    }
    if(p.x==3) result=100000.0;
    if(p.x==4) {
        result=-999.0;
        Ray ray=makeRay(vec3(0,0,4),vec3(0.5,0,-sqrt(0.75)));
        vec3 lo=vec3(-1,-1,3),hi=vec3(1,1,3.00002);
        Hit entry,exit;
        if(hitBox(lo,hi,ray,0.0,100.0,entry)) {
            vec3 inside=refractVec(ray.dir,entry.normal,1.0/1.5);
            ray=offsetRay(entry.p,entry.normal,inside);
            if(hitBox(lo,hi,ray,0.0,100.0,exit) && !exit.frontFace)
                result=refractVec(ray.dir,exit.normal,1.5).x;
        }
    }
    imageStore(destTex,p,vec4(result,result,result,1));
}
`

package gpu

import (
	"image"
	"math"
	"testing"

	"github.com/go-gl/gl/v3.3-core/gl"

	renderentity "github.com/MarkJulian19/path_trace_golang/internal/entities/render"
	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

func checkGPUSamplingRegression(t *testing.T, r *gpuRenderer, prefix string) {
	t.Helper()
	program, err := createComputeProgram(prefix+samplingRegressionProbe, "sampling regression")
	if err != nil {
		t.Fatal(err)
	}
	defer gl.DeleteProgram(program)
	r.program = program
	cfg := renderentity.Config{Width: 2560, Height: 1, SamplesPerPx: 1, MaxDepth: 1}
	img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
	err = r.renderOnce(&scene.Scene{}, cfg, img, nil, func() bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	values := readLinearGPU(r, cfg.Width, cfg.Height)
	if values[0] > 0.00001 {
		t.Errorf("GPU GGX basis loses rotational symmetry: error=%v", values[0])
	}
	if values[4] != 1 {
		t.Error("distinct pixel coordinates share the same GPU random stream")
	}
	if values[16] != 0 {
		t.Errorf("GPU plane intersection leaves the exact surface: error=%v", values[16])
	}
	if values[20] != 0 || values[24] != 0 {
		t.Errorf("reflected rays hit their own convex surface: spheres=%v boxes=%v", values[20], values[24])
	}
	for i := 8; i < 16; i++ {
		if values[i*gpuRGBAComponents] != 1 {
			t.Errorf("GPU grazing plane case %d failed", i-8)
		}
	}
	if values[28] != 1 {
		t.Error("GPU box intersection selected an adjacent face normal near an edge")
	}
	if math.Abs(float64(values[8])-0.5) > 0.005 || math.Abs(float64(values[12])) > 0.002 {
		t.Errorf("biased or correlated GPU random samples: mean=%v covariance=%v", values[8], values[12])
	}
}

const samplingRegressionProbe = `
void main() {
 ivec2 p=ivec2(gl_GlobalInvocationID.xy)+uDispatchOffset;
 if(p.x>=uWidth || p.y>=uHeight) return;
 float result=0.0;
 if(p.x==0) {
  vec3 view=normalize(vec3(0.01,0,1));
  vec3 rotated=vec3(0,view.x,view.z);
  uint first=1234567u,second=first;
  for(int i=0;i<1000;++i) {
   vec3 a=sampleGGX(view,vec3(0,0,1),0.2,first);
   vec3 b=sampleGGX(rotated,vec3(0,0,1),0.2,second);
   result=max(result,length(b-vec3(-a.y,a.x,a.z)));
  }
 }
 if(p.x==1) result=pathSampleSeed(ivec2(226,1),0u)!=pathSampleSeed(ivec2(215,4),0u) ? 1.0 : 0.0;
 if(p.x==4) {
  uint state=123456789u;
  for(int i=0;i<4096;++i) {
   Ray ray=makeRay(vec3(6,2.8,8),vec3(2.0*rng(state)-1.0,-0.1-rng(state),2.0*rng(state)-1.0));
   Hit hit;
   if(hitPlane(vec3(0),vec3(0,1,0),ray,0.0,100.0,hit)) result=max(result,abs(hit.p.y));
  }
 }
 if(p.x>=2 && p.x<4) {
  uint state=123456789u;
  for(int i=0;i<65536;++i) {
   float x=rng(state),y=rng(state);
   result+=p.x==2 ? x : (x-0.5)*(y-0.5);
  }
  result/=65536.0;
 }
 if(p.x==5 || p.x==6) {
  uint state=123456789u;
  vec3 origin=vec3(6,2.8,8);
  for(int i=0;i<8192;++i) {
   vec3 center=vec3(2.2*float(i%3-1),1,0);
   Ray ray=makeRay(origin,normalize(center-origin)+0.1*vec3(2.0*rng(state)-1.0,2.0*rng(state)-1.0,2.0*rng(state)-1.0));
   Hit hit,next;
   bool found=p.x==5 ? hitSphere(center,1.0,ray,0.0,100.0,hit) : hitBox(center-vec3(1),center+vec3(1),ray,0.0,100.0,hit,false);
   if(!found) continue;
   Ray reflected=offsetRay(hit.p,hit.normal,reflectVec(ray.dir,hit.normal));
   bool selfHit=p.x==5 ? hitSphere(center,1.0,reflected,0.0,100.0,next) : hitBox(center-vec3(1),center+vec3(1),reflected,0.0,100.0,next,false);
   if(selfHit) result+=1.0;
  }
 }
 if(p.x==7) {
  Hit hit;
  Ray ray=makeRay(vec3(0.99995,3,0),vec3(0,-1,0));
  if(hitBox(vec3(-1),vec3(1),ray,0.0,100.0,hit,false)) result=dot(hit.normal,vec3(0,1,0));
 }
 if(p.x>=8 && p.x<16) {
  int index=p.x-8;
  const vec3 origins[8]=vec3[8](vec3(0,1e-6,0),vec3(0,-1e-6,0),vec3(0,1e-4,0),vec3(0,1e-6,0),vec3(0),vec3(0,1e-6,0),vec3(0),vec3(0,1e-20,0));
  const vec3 directions[8]=vec3[8](vec3(1,-5e-7,0),vec3(1,5e-7,0),vec3(1,-5e-7,0),vec3(1,0,0),vec3(1,0,0),vec3(1,5e-7,0),vec3(1,-5e-7,0),vec3(1,-5e-21,0));
  Hit hit;
  bool found=hitPlane(vec3(0),vec3(0,1,0),makeRay(origins[index],directions[index]),1e-9,100.0,hit);
  bool expected=index==0 || index==1 || index==7;
  bool correct=found==expected;
  if(found) correct=correct && abs(hit.t-2.0)<1e-4 && hit.p.y==0.0 && hit.frontFace==(origins[index].y>0.0);
  result=correct ? 1.0 : 0.0;
 }
 imageStore(destTex,p,vec4(result,result,result,1));
}
`

func checkGPUNestedMedia(t *testing.T, r *gpuRenderer, prefix string) {
	t.Helper()
	program, err := createComputeProgram(prefix+nestedMediaProbe, "nested media")
	if err != nil {
		t.Fatal(err)
	}
	defer gl.DeleteProgram(program)
	r.program = program
	sc := &scene.Scene{Materials: []scene.Material{
		{ID: "outer", Type: scene.MaterialDielectric, IOR: 1.5},
		{ID: "inner", Type: scene.MaterialDielectric, IOR: 1.33},
	}, Objects: []scene.Object{
		{Type: scene.ObjectPlane, MaterialID: "outer", Position: scene.Vec3{Y: 2}},
		{Type: scene.ObjectPlane, MaterialID: "inner", Position: scene.Vec3{Y: 1}},
	}}
	cfg := renderentity.Config{Width: 3, Height: 1, SamplesPerPx: 1, MaxDepth: 1}
	img := image.NewRGBA(image.Rect(0, 0, 3, 1))
	err = r.renderOnce(sc, cfg, img, nil, func() bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	values := readLinearGPU(r, 3, 1)
	for i := range 3 {
		if values[i*gpuRGBAComponents] != 1 {
			t.Fatalf("nested medium changes with view direction: pixel=%d value=%v", i, values[i*gpuRGBAComponents])
		}
	}
}

const nestedMediaProbe = `
void main() {
 ivec2 p=ivec2(gl_GlobalInvocationID.xy)+uDispatchOffset;
 if(p.x>=uWidth || p.y>=uHeight) return;
 vec3 direction=p.x==0 ? vec3(0,1,0) : (p.x==1 ? vec3(0,-1,0) : vec3(1,0,0));
 float medium=float(mediumAt(makeRay(vec3(0),direction),-1));
 imageStore(destTex,p,vec4(medium,medium,medium,1));
}
`

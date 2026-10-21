package gpu

// pathTransportShader uses the same BSDFs, medium interfaces and MIS as the CPU integrator.
const pathTransportShader = `
vec3 rayColor(Ray r, inout uint state) {
    vec3 throughput=vec3(1), radiance=vec3(0);
    int medium=mediumAt(r,-1);
    float previousPDF=0.0, etaScale=1.0;
    vec3 previousPoint=r.orig;
    float horizon=uMaxRayDist>0.0 ? uMaxRayDist : 1000.0;
    for (int bounce=0; bounce<uMaxDepth; ++bounce) {
        if (max(throughput.r,max(throughput.g,throughput.b))<=0.0) break;
        Hit h;
        bool found=hitWorld(r,0.0,horizon,h);
        float distance=found ? h.t : horizon;
        if (medium<0 && fogGpuVolumetric>0.5 && (found || fogAffectSky>0.5)) {
            float scatterDistance;
            vec3 scatterWeight;
            if (sampleFogDistance(r,distance,state,scatterDistance,scatterWeight)) {
                vec3 point=rayAt(r,scatterDistance);
                throughput*=scatterWeight;
                radiance+=throughput*estimateFogDirect(point,r.dir,state,bounce+1<uMaxDepth);
                vec3 direction=samplePhaseHG(r.dir,fogG,state);
                previousPDF=phaseHG(dot(r.dir,direction),fogG);
                previousPoint=point;
                r=makeRay(point,direction);
                if (bounce>=3) {
                    float probability=min(1.0,etaScale*max(throughput.r,max(throughput.g,throughput.b)));
                    if (probability<=0.0 || rng(state)>=probability) break;
                    throughput/=probability;
                }
                continue;
            }
        }
        throughput*=mediumTransmission(medium,distance);
        if (!found) {
            vec3 bg=backgroundColor(r);
            if (fogGpuVolumetric<0.5 && fogAffectSky>0.5) bg=applyFog(bg,50.0);
            radiance+=throughput*bg;
            break;
        }
        int typ; float rough,ior,smoothness,reflectivity,scale;
        vec3 albedo,emit,absorption,tint;
        readMaterial(h.matIndex,typ,rough,ior,albedo,emit,absorption,smoothness,reflectivity,tint,scale);
        if (typ==MAT_EMISSIVE) {
            if (h.frontFace) {
                float pdf=lightPDFAtHit(h,previousPoint);
                float weight=previousPDF>0.0 ? previousPDF/(previousPDF+pdf) : 1.0;
                radiance+=throughput*emit*weight;
            }
            break;
        }
        vec3 direction,weight;
        float sampledPDF=0.0;
        if (typ==MAT_LAMBERT) {
            radiance+=throughput*estimateDirectLight(r,h,state,bounce+1<uMaxDepth);
            direction=randomCosineDirection(h.normal,state);
            sampledPDF=max(0.0,dot(direction,h.normal))*INV_PI;
            weight=albedo;
        } else if (typ==MAT_METAL || typ==MAT_MIRROR) {
            rough=1.0-smoothness;
            if (typ==MAT_METAL && rough>1e-4) {
                radiance+=throughput*estimateDirectLight(r,h,state,bounce+1<uMaxDepth);
                direction=sampleGGX(-r.dir,h.normal,rough,state);
                float pdf;
                vec3 f=surfaceBSDF(r,h,direction,pdf);
                sampledPDF=pdf;
                if (pdf<=0.0) break;
                weight=f*(max(0.0,dot(direction,h.normal))/pdf);
            } else {
                direction=reflectVec(r.dir,h.normal);
                weight=albedo*reflectivity;
                if (typ==MAT_METAL) weight=fresnelMetal(-dot(r.dir,h.normal),weight);
            }
        } else if (typ==MAT_DIELECTRIC) {
            int next=h.frontFace ? h.objIndex : mediumAt(offsetRay(h.p,h.normal,r.dir),h.objIndex);
            float eta=mediumIOR(medium)/mediumIOR(next);
            float cosine=clamp(-dot(r.dir,h.normal),0.0,1.0);
            if (rng(state)<reflectance(cosine,eta)) {
                direction=reflectVec(r.dir,h.normal);
                weight=vec3(1);
            } else {
                direction=refractVec(r.dir,h.normal,eta);
                weight=vec3(eta*eta);
                etaScale/=eta*eta;
                medium=next;
            }
        } else break;
        previousPDF=sampledPDF;
        previousPoint=h.p;
        throughput*=weight;
        if (bounce>=3) {
            float probability=min(1.0,etaScale*max(throughput.r,max(throughput.g,throughput.b)));
            if (probability<=0.0 || rng(state)>=probability) break;
            throughput/=probability;
        }
        r=offsetRay(h.p,h.normal,direction);
    }
    return radiance;
}

uint pathSampleSeed(ivec2 pixel, uint sampleIndex) {
    uint rasterIndex=uint(pixel.y)*uint(uWidth)+uint(pixel.x);
    return hash_u(rasterIndex ^ hash_u(uFrameSeed+sampleIndex));
}

void main() {
    ivec2 pix=ivec2(gl_GlobalInvocationID.xy)+uDispatchOffset;
    if (pix.x>=uWidth || pix.y>=uHeight) return;
    vec3 sum=vec3(0);
    for (int i=0; i<uSamplesPerPx; ++i) {
        uint index=uint(uSampleCount-uSamplesPerPx+i);
        uint state=pathSampleSeed(pix,index);
        vec2 uv=(vec2(pix.x,uHeight-1-pix.y)+vec2(rng(state),rng(state)))/vec2(uWidth,uHeight);
        Ray ray;
        buildCamera(uv,ray,state);
        sum+=rayColor(ray,state);
    }
    int base=(pix.y*uWidth+pix.x)*3;
    accumData[base]+=sum.r;
    accumData[base+1]+=sum.g;
    accumData[base+2]+=sum.b;
    vec3 mean=vec3(accumData[base],accumData[base+1],accumData[base+2])/float(uSampleCount);
    imageStore(destTex,pix,vec4(mean,1));
}
`

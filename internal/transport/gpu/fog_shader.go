package gpu

// Delta tracking and ratio tracking preserve the heterogeneous medium estimator.
const fogShader = `
vec3 shadowTransmittance(Ray ray, float distance, inout uint state) {
 int medium=mediumAt(ray,-1);
 return medium>=0 ? mediumTransmission(medium,distance) : vec3(fogTransmittance(ray,distance,state));
}

float fogMajorant() {
    float extinction=max(0.0,fogSigmaS)+max(0.0,fogSigmaA);
    if (extinction<=0.0) extinction=max(0.0,fogDensity);
    return extinction*(1.0+clamp(fogHeteroStrength,0.0,1.0));
}

float fogTransmittance(Ray ray, float distance, inout uint state) {
    if (fogGpuVolumetric<0.5) return 1.0;
    float majorant=fogMajorant();
    if (majorant<=0.0) return 1.0;
    if (fogHeteroStrength<=0.0) return exp(-majorant*distance);
    float t=0.0, weight=1.0;
    while (true) {
        t+=-log(1.0-rng(state))/majorant;
        if (t>=distance) return weight;
        float s,a,e; mediumCoeffs(rayAt(ray,t),s,a,e);
        weight*=max(0.0,1.0-e/majorant);
        if (weight<=0.0) return 0.0;
    }
}

bool sampleFogDistance(Ray ray, float distance, inout uint state, out float t, out vec3 weight) {
    float majorant=fogMajorant();
    if (majorant<=0.0) return false;
    t=0.0;
    while (true) {
        t+=-log(1.0-rng(state))/majorant;
        if (t>=distance) return false;
        float s,a,e; mediumCoeffs(rayAt(ray,t),s,a,e);
        if (rng(state)*majorant<e) {
            weight=clamp(fogColor.rgb,vec3(0),vec3(1))*(s/e);
            return true;
        }
    }
}

vec3 samplePhaseHG(vec3 axis, float g, inout uint state) {
    float u=2.0*rng(state)-1.0, denominator=1.0+g*u;
    float cosine=clamp((u+0.5*g*(u*u+3.0)+g*g*u+0.5*g*g*g*(u*u-1.0))/(denominator*denominator),-1.0,1.0);
    float phi=TWO_PI*rng(state), sine=sqrt(max(0.0,1.0-cosine*cosine));
    vec3 tangent=normalize(cross(abs(axis.x)>0.9 ? vec3(0,1,0) : vec3(1,0,0),axis));
    return normalize(tangent*(sine*cos(phi))+cross(axis,tangent)*(sine*sin(phi))+axis*cosine);
}

vec3 estimateFogDirect(vec3 position, vec3 incident, inout uint state, bool continuation) {
    vec3 sum=vec3(0);
    int count=getLightCount();
    if (count==0) return sum;
    int samples=min(count,8), start=count>8 ? int(rng(state)*float(count)) : 0;
    for (int i=0; i<samples; ++i) {
        int object=getLightObjectIndex((start+i)%count);
        vec3 point,normal; float areaPDF;
        if (!sampleLightGeometry(object,position,point,normal,areaPDF,state)) continue;
        vec3 delta=point-position;
        float distance=length(delta);
        if (distance<=0.0) continue;
        vec3 wi=delta/distance;
        float cosine=max(0.0,dot(normal,-wi));
        if (cosine<=0.0) continue;
        Ray shadow=makeRay(position,wi); 
        if (occluded(shadow,distance-0.00002,object)) continue;
        float lightPDF=areaPDF*float(samples)/float(count);
        float phase=phaseHG(dot(incident,wi),fogG);
        int mat=int(objData[object*OBJ_STRIDE+1]+0.5)*MAT_STRIDE;
        vec3 emit=vec3(matData[mat+8],matData[mat+9],matData[mat+10]);
        sum+=emit*(phase/(lightPDF+(continuation ? phase : 0.0)))*fogTransmittance(shadow,distance,state);
    }
    return sum;
}
`

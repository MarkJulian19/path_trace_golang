package gpu

const physicsShader = `
// Смещение на несколько ULP сохраняет тонкие границы в пределах точности float32.
Ray offsetRay(vec3 point, vec3 normal, vec3 direction) {
    vec3 side=dot(direction,normal)<0.0 ? -normal : normal;
    // Малые компоненты нормали тоже должны сдвигать координату наружу.
    ivec3 offset=ivec3(ceil(8.0*abs(side)))*ivec3(sign(side));
    ivec3 bits=floatBitsToInt(point);
    vec3 shifted=intBitsToFloat(bits+ivec3(point.x<0.0 ? -offset.x : offset.x,
        point.y<0.0 ? -offset.y : offset.y,point.z<0.0 ? -offset.z : offset.z));
    for(int i=0;i<3;++i) if(abs(point[i])<0.03125) shifted[i]=point[i]+1e-7*side[i];
    return makeRay(shifted,normalize(direction));
}

int mediumAt(Ray r, int exclude) {
    int medium=-1;
    float nearest=1e30;
    for(int i=0;i<objData.length()/OBJ_STRIDE;++i) {
        if(i==exclude) continue;
        int typ,mat; vec3 pos,size;
        readObject(i,typ,mat,pos,size);
        if(int(matData[mat*MAT_STRIDE]+0.5)!=MAT_DIELECTRIC) continue;
        vec3 delta=r.orig-pos;
        float distance;
        if(typ==OBJ_SPHERE) distance=size.x-length(delta);
        else if(typ==OBJ_BOX) {
            vec3 clearance=0.5*size-abs(delta);
            distance=min(clearance.x,min(clearance.y,clearance.z));
        } else distance=pos.y-r.orig.y;
        if(distance>0.0 && distance<nearest) { medium=i; nearest=distance; }
    }
    return medium;
}

float mediumIOR(int object) {
    if (object < 0) return 1.0;
    int mat = int(objData[object*OBJ_STRIDE+1]+0.5);
    return matData[mat*MAT_STRIDE+2];
}

vec3 mediumTransmission(int object, float distance) {
    if (object < 0 || distance <= 0.0) return vec3(1);
    int mat = int(objData[object*OBJ_STRIDE+1]+0.5)*MAT_STRIDE;
    vec3 sigma = max(vec3(matData[mat+12],matData[mat+13],matData[mat+14]),vec3(0));
    float scale = max(0.0,matData[mat+15]);
    vec3 tint = clamp(vec3(matData[mat+16],matData[mat+17],matData[mat+18]),vec3(0),vec3(1));
    return exp(-sigma*scale*distance)*pow(tint,vec3(distance));
}

float ggxG1(float cosine, float alpha) {
    if (cosine <= 0.0) return 0.0;
    return 2.0*cosine/(cosine+sqrt(alpha*alpha+(1.0-alpha*alpha)*cosine*cosine));
}

vec3 fresnelMetal(float cosine, vec3 f0) {
    float x = 1.0-clamp(cosine,0.0,1.0);
    return f0+(vec3(1)-f0)*(x*x*x*x*x);
}

vec3 surfaceBSDF(Ray r, Hit h, vec3 wi, out float pdf) {
    pdf = 0.0;
    float nv = -dot(r.dir,h.normal), nl = dot(wi,h.normal);
    if (nv <= 0.0 || nl <= 0.0) return vec3(0);
    int typ; float rough, ior, smoothness, reflectivity, scale;
    vec3 albedo, emit, absorption, tint;
    readMaterial(h.matIndex,typ,rough,ior,albedo,emit,absorption,smoothness,reflectivity,tint,scale);
    if (typ == MAT_LAMBERT) { pdf=nl*INV_PI; return albedo*INV_PI; }
    rough=1.0-smoothness;
    if (typ != MAT_METAL || rough <= 1e-4) return vec3(0);
    vec3 halfVector = normalize(wi-r.dir);
    float nh=clamp(dot(h.normal,halfVector),0.0,1.0), vh=abs(dot(r.dir,halfVector));
    if (nh <= 0.0 || vh <= 1e-12) return vec3(0);
    float alpha=rough*rough;
    float denominator=(1.0-nh*nh)+nh*nh*alpha*alpha;
    float D=alpha*alpha/(PI*denominator*denominator);
    float g1View=ggxG1(nv,alpha);
    float G=g1View*ggxG1(nl,alpha);
    pdf=D*g1View/(4.0*nv);
    return fresnelMetal(vh,albedo*reflectivity)*(D*G/(4.0*nv*nl));
}

vec3 visibleBoxAreas(vec3 center, vec3 size, vec3 reference) {
 vec3 areas=vec3(size.y*size.z,size.x*size.z,size.x*size.y);
 vec3 delta=abs(reference-center);
 for(int i=0;i<3;++i) if(delta[i]<=size[i]*0.5) areas[i]=0.0;
 return areas;
}

float sphereLightPDF(vec3 center, float radius, vec3 reference) {
 vec3 delta=center-reference;
 float distanceSquared=dot(delta,delta);
 if(distanceSquared<=radius*radius) return 0.0;
 float sineSquared=radius*radius/distanceSquared;
 float oneMinusCosine=sineSquared/(1.0+sqrt(max(0.0,1.0-sineSquared)));
 return 1.0/(TWO_PI*oneMinusCosine);
}

float lightPDFAtHit(Hit h, vec3 previous) {
 if(lightIndices.length()==0) return 0.0;
 int typ,mat; vec3 position,size;
 readObject(h.objIndex,typ,mat,position,size);
 float pdf=0.0;
 if(typ==OBJ_SPHERE) pdf=sphereLightPDF(position,size.x,previous);
 else if(typ==OBJ_BOX) {
  vec3 areas=visibleBoxAreas(position,size,previous);
  float area=areas.x+areas.y+areas.z;
  vec3 delta=h.p-previous;
  float cosine=abs(dot(normalize(delta),h.normal));
  if(area>0.0 && cosine>0.0) pdf=dot(delta,delta)/(area*cosine);
 }
 return pdf*min(1.0,8.0/float(lightIndices.length()));
}
`

package gpu

const lightingShader = `
int getLightCount() { return lightIndices.length(); }
int getLightObjectIndex(int index) { return lightIndices[index]; }

bool sampleLightGeometry(int object, vec3 reference, out vec3 point, out vec3 normal, out float pdf, inout uint state) {
 int typ,mat; vec3 position,size;
 readObject(object,typ,mat,position,size);
 if(typ==OBJ_SPHERE) {
  pdf=sphereLightPDF(position,size.x,reference);
  if(pdf<=0.0) return false;
  vec3 delta=position-reference;
  float distance=length(delta);
  vec3 axis=delta/distance;
  float deltaCosine=rng(state)/(TWO_PI*pdf),cosine=1.0-deltaCosine;
  float sine=sqrt(max(0.0,deltaCosine*(2.0-deltaCosine))),phi=TWO_PI*rng(state);
  vec3 tangent=normalize(cross(abs(axis.z)<0.999 ? vec3(0,0,1) : vec3(1,0,0),axis));
  vec3 wi=normalize(axis*cosine+tangent*(sine*cos(phi))+cross(axis,tangent)*(sine*sin(phi)));
  float projection=distance*cosine;
  float halfChord=sqrt(max(0.0,size.x*size.x-distance*distance*sine*sine));
  point=reference+wi*((distance*distance-size.x*size.x)/(projection+halfChord));
  normal=normalize(point-position);
 } else if(typ==OBJ_BOX) {
  vec3 areas=visibleBoxAreas(position,size,reference);
  float area=areas.x+areas.y+areas.z;
  if(area<=0.0) return false;
  float choice=rng(state)*area;
  int axis=0;
  while(axis<2 && choice>=areas[axis]) { choice-=areas[axis]; ++axis; }
  float side=reference[axis]<position[axis] ? -1.0 : 1.0;
  point=position;
  for(int i=0;i<3;++i) point[i]+=(i==axis ? side*0.5 : rng(state)-0.5)*size[i];
  normal=vec3(0); normal[axis]=side;
  vec3 delta=point-reference;
  float cosine=max(0.0,dot(normal,-normalize(delta)));
  if(cosine<=0.0) return false;
  pdf=dot(delta,delta)/(area*cosine);
 } else return false;
 return true;
}

// Defined below alongside the medium coefficients.
float fogTransmittance(Ray ray, float distance, inout uint state);
vec3 shadowTransmittance(Ray ray, float distance, inout uint state);

vec3 estimateDirectLight(Ray ray, Hit hit, inout uint state, bool continuation) {
    int count=getLightCount();
    if (count==0) return vec3(0);
    int samples=min(count,8);
    int start=count>8 ? int(rng(state)*float(count)) : 0;
    vec3 sum=vec3(0);
    float inclusion=float(samples)/float(count);
    for (int i=0; i<samples; ++i) {
        int object=getLightObjectIndex((start+i)%count);
        vec3 point,normal; float areaPDF;
        if (!sampleLightGeometry(object,hit.p,point,normal,areaPDF,state)) continue;
        vec3 delta=point-hit.p;
        float distance=length(delta);
        if (distance<=0.0) continue;
        vec3 wi=delta/distance;
        float nl=max(0.0,dot(hit.normal,wi)), cosine=max(0.0,dot(normal,-wi));
        if (nl<=0.0 || cosine<=0.0) continue;
        Ray shadow=offsetRay(hit.p,hit.normal,wi);
        
        float endpoint=length(point-shadow.orig);
        shadow.dir=normalize(point-shadow.orig);
        if (occluded(shadow,endpoint-0.00002,object)) continue;
        float bsdfPDF;
        vec3 f=surfaceBSDF(ray,hit,wi,bsdfPDF);
        float lightPDF=areaPDF*inclusion;
        int mat=int(objData[object*OBJ_STRIDE+1]+0.5)*MAT_STRIDE;
        vec3 emit=vec3(matData[mat+8],matData[mat+9],matData[mat+10]);
        vec3 transmittance=shadowTransmittance(shadow,endpoint,state);
        sum+=f*emit*(nl/(lightPDF+(continuation ? bsdfPDF : 0.0)))*transmittance;
    }
    return sum;
}
`

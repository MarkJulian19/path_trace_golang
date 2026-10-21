package gpu

const computeShader = `
#version 430
layout(local_size_x = 32, local_size_y = 32) in;

// Используем формат rgba32f, чтобы соответствовать текстуре RGBA32F на стороне Go.
layout(binding = 0, rgba32f) uniform writeonly image2D destTex;

// Текстура предыдущего кадра для Temporal Accumulation (уменьшение шума между кадрами)
layout(binding = 9) uniform sampler2D prevFrameTex;

uniform int uWidth;
uniform int uHeight;
uniform int uSamplesPerPx;
uniform int uMaxDepth;
uniform float uMaxRayDist; // Максимальная длина луча (для предотвращения бесконечной трассировки)
uniform uint uFrameSeed;
uniform int uSampleCount; // Текущее количество накопленных сэмплов
uniform int uSelectedObject; // Индекс выбранного объекта (-1 = нет выбора)
uniform ivec2 uDispatchOffset;
uniform bool uTemporalEnabled;

// Камера (совпадает с scene.Camera)
layout(std140, binding = 1) uniform CameraBlock {
    vec4 camPos;     // xyz = position
    vec4 camTarget;  // xyz = target
    vec4 camUp;      // xyz = up
    float camFov;
    float camAperture;
    float camFocusDist;
    float camAspect;
};

// Sky / background
layout(std140, binding = 2) uniform SkyBlock {
    // Используем float для skyType, чтобы соответствовать передаче данных из Go (float32).
    // 0.0 = background color, 1.0 = solid, 2.0 = gradient
    float skyType;
    float _padSky0;
    float _padSky1;
    float _padSky2;
    vec4 skyColor;    // background or solid color
    vec4 skyHorizon;  // gradient horizon
    vec4 skyZenith;   // gradient zenith
};

// Однородный / объёмный туман (fog)
layout(std140, binding = 5) uniform FogBlock {
    float fogDensity;       // исторический параметр плотности (для совместимости)
    float fogScatter;       // 0..1 — сила рассеяния (множитель для sigma_s)
    float fogAffectSky;     // 0 или 1
    float fogGpuVolumetric; // >0: включён физически‑реалистичный объёмный туман
    vec4  fogColor;         // базовый цвет тумана
    // Физические коэффициенты объёмной среды:
    // sigma_t = sigma_a + sigma_s
    float fogSigmaS;        // коэффициент рассеяния
    float fogSigmaA;        // коэффициент поглощения
    float fogG;             // анизотропия фазовой функции Хеньи–Грина
    float fogHeteroStrength;// 0..1, сила неоднородности (шум)
    float fogNoiseScale;    // масштаб шума
    float fogNoiseOctaves;  // количество октав шума (1..5)
    float _padFog1;
    float _padFog2;
};

// Потоковые массивы материалов и объектов в виде плоских float массивов.
// Форматы должны совпадать с тем, как мы пакуем данные на Go-стороне.
// Материал: [typ, rough, ior, smoothness,
//            albedo.r, albedo.g, albedo.b, reflectivity,
//            emit.r, emit.g, emit.b, pad0,
//            absorption.r, absorption.g, absorption.b, absorption_scale,
//            tint.r, tint.g, tint.b, pad2] (20 float)
layout(std430, binding = 3) buffer Materials {
    float matData[];
};

// Объект: [type, materialIndex, pad0, pad1,
//          pos.x, pos.y, pos.z, pad2,
//          size.x, size.y, size.z, pad3] (12 float)
layout(std430, binding = 4) buffer Objects {
    float objData[];
};

// Список индексов эмиссивных объектов (оптимизация для быстрого доступа к источникам света)
layout(std430, binding = 6) buffer LightIndicesBlock {
    int lightIndices[];
};

// Накопительный буфер для сэмплов (width * height * 3 * float32)
layout(std430, binding = 7) buffer AccumBuffer {
    float accumData[];
};

// Константы: должны совпадать с engine/materials.go и engine/objects.go
const int MAT_LAMBERT   = 0;
const int MAT_METAL     = 1;
const int MAT_DIELECTRIC= 2;
const int MAT_EMISSIVE  = 3;
const int MAT_MIRROR    = 4;

const int OBJ_SPHERE    = 0;
const int OBJ_PLANE     = 1;
const int OBJ_BOX       = 2;

const int MAT_STRIDE    = 20;
const int OBJ_STRIDE    = 12;
const float PI = 3.14159265359;
const float TWO_PI = 6.28318530718;
const float INV_PI = 0.31830988618;
const float INV_TWO_PI = 0.15915494309;

uint hash_u(uint x) {
    x ^= x >> 17;
    x *= 0xed5ad4bbU;
    x ^= x >> 11;
    x *= 0xac4c1b51U;
    x ^= x >> 15;
    x *= 0x31848babU;
    x ^= x >> 14;
    return x;
}

float rng(inout uint state) {
    state = hash_u(state + 0x9e3779b9u);
    return float(state >> 8u) * (1.0 / 16777216.0);
}

struct Ray {
    vec3 orig;
    vec3 dir;
};

struct Hit {
    vec3 p;
    vec3 normal;
    float t;
    int matIndex;
    int objIndex;  // Индекс объекта для правильного поиска выхода из стеклянных объектов
    bool frontFace;
};

vec3 vec3_from4(vec4 v) { return v.xyz; }

Ray makeRay(vec3 o, vec3 d) {
    return Ray(o, normalize(d));
}

vec3 rayAt(Ray r, float t) {
    return r.orig + t * r.dir;
}

void setFaceNormal(Ray r, inout Hit h, vec3 outwardNormal) {
    h.frontFace = dot(r.dir, outwardNormal) < 0.0;
    h.normal = h.frontFace ? outwardNormal : -outwardNormal;
}

// Чтение материала
void readMaterial(int idx, out int typ, out float rough, out float ior,
                  out vec3 albedo, out vec3 emit, out vec3 absorption,
                  out float smoothness, out float reflectivity, out vec3 tint,
                  out float absorptionScale) {
    int base = idx * MAT_STRIDE;
    float ftyp   = matData[base + 0];
    rough        = matData[base + 1];
    ior          = matData[base + 2];
    smoothness   = matData[base + 3];
    typ          = int(ftyp + 0.5);

    albedo = vec3(matData[base + 4], matData[base + 5], matData[base + 6]);
    reflectivity = matData[base + 7];
    emit   = vec3(matData[base + 8], matData[base + 9], matData[base + 10]);
    absorption = vec3(matData[base + 12], matData[base + 13], matData[base + 14]);
    absorptionScale = matData[base + 15];
    tint = vec3(matData[base + 16], matData[base + 17], matData[base + 18]);
}

// Чтение объекта
void readObject(int idx, out int typ, out int matIndex, out vec3 pos, out vec3 size) {
    int base = idx * OBJ_STRIDE;
    float ftyp = objData[base + 0];
    float fmat = objData[base + 1];
    typ      = int(ftyp + 0.5);
    matIndex = int(fmat + 0.5);

    pos  = vec3(objData[base + 4], objData[base + 5], objData[base + 6]);
    size = vec3(objData[base + 8], objData[base + 9], objData[base + 10]);
}

// Пересечения - оптимизировано: кэширование деления
bool hitSphere(vec3 center, float radius, Ray r, float tMin, float tMax, inout Hit h) {
    float a=dot(r.dir,r.dir);
    if(a<=0.0 || radius<=0.0) return false;
    vec3 oc=r.orig-center;
    float midpoint=-dot(oc,r.dir)/a;
    vec3 closest=oc+midpoint*r.dir;
    float discriminant=radius*radius-dot(closest,closest);
    if(discriminant<0.0) return false;
    float halfChord=sqrt(discriminant/a);
    float root=midpoint-halfChord;
    if(root<tMin || root>tMax) {
        root=midpoint+halfChord;
        if(root<tMin || root>tMax) return false;
    }
    h.t=root;
    h.p=rayAt(r,root);
    vec3 outward=normalize(h.p-center);
    // Восстанавливаем точку на поверхности до смещения вторичного луча.
    h.p=center+radius*outward;
    setFaceNormal(r,h,outward);
    return true;
}

bool hitPlane(vec3 point, vec3 normal, Ray r, float tMin, float tMax, inout Hit h) {
    float denom = dot(normal, r.dir);
    if (abs(denom) < 1e-6) return false;
    float t = dot(point - r.orig, normal) / denom;
    if (t < tMin || t > tMax) return false;
    h.t = t;
    h.p = rayAt(r, t);
    // Проецируем точку на плоскость: ошибка FP32 не должна вызывать самозатенение.
    h.p -= normal * dot(h.p - point, normal);
    setFaceNormal(r, h, normal);
    return true;
}

// Вычисляет нормаль для точки на поверхности куба
vec3 computeBoxNormal(vec3 localPoint, vec3 halfSize) {
    // Локальная точка уже относительно центра куба
    // Нормаль определяется по ближайшей грани
    
    // Находим расстояние до каждой из шести граней
    vec3 distToPositive = halfSize - localPoint;
    vec3 distToNegative = localPoint + halfSize;
    
    // Находим минимальное расстояние
    float minDist = min(distToPositive.x, min(distToPositive.y, min(distToPositive.z,
                      min(distToNegative.x, min(distToNegative.y, distToNegative.z)))));
    
    // Определяем, к какой грани ближе всего
    if (abs(minDist - distToPositive.x) < 1e-5) {
        return vec3(1.0, 0.0, 0.0);
    } else if (abs(minDist - distToNegative.x) < 1e-5) {
        return vec3(-1.0, 0.0, 0.0);
    } else if (abs(minDist - distToPositive.y) < 1e-5) {
        return vec3(0.0, 1.0, 0.0);
    } else if (abs(minDist - distToNegative.y) < 1e-5) {
        return vec3(0.0, -1.0, 0.0);
    } else if (abs(minDist - distToPositive.z) < 1e-5) {
        return vec3(0.0, 0.0, 1.0);
    } else {
        return vec3(0.0, 0.0, -1.0);
    }
}

// Основная функция пересечения с кубом
// findExit: false = ищем вход (t0), true = ищем выход (t1)
// УЛУЧШЕННАЯ ФУНКЦИЯ ПЕРЕСЕЧЕНИЯ С КУБОМ
bool hitBox(vec3 bmin, vec3 bmax, Ray r, float tMin, float tMax, inout Hit h, bool findExit) {
    float t0 = -1e30;
    float t1 = 1e30;
    vec3 entryNormal=vec3(0), exitNormal=vec3(0);
    
    for (int i = 0; i < 3; i++) {
        float direction = r.dir[i];
        float orig = (i == 0 ? r.orig.x : (i == 1 ? r.orig.y : r.orig.z));
        float minV = (i == 0 ? bmin.x : (i == 1 ? bmin.y : bmin.z));
        float maxV = (i == 0 ? bmax.x : (i == 1 ? bmax.y : bmax.z));
        if (direction == 0.0) {
            if (orig < minV || orig > maxV) return false;
            continue;
        }
        float invD = 1.0 / direction;
        
        float tNear = (minV - orig) * invD;
        float tFar  = (maxV - orig) * invD;
        
        if (invD < 0.0) {
            float tmp = tNear; tNear = tFar; tFar = tmp;
        }
        
        if(tNear>t0) {
            t0=tNear;
            entryNormal=vec3(0);
            entryNormal[i]=direction<0.0 ? 1.0 : -1.0;
        }
        if(tFar<t1) {
            t1=tFar;
            exitNormal=vec3(0);
            exitNormal[i]=direction<0.0 ? -1.0 : 1.0;
        }
        
        if (t1 < t0) return false;
    }
    
    // Выбираем правильное пересечение
    bool exiting=findExit || t0<tMin;
    h.t = exiting ? t1 : t0;
    
    if (h.t < tMin || h.t > tMax) {
        return false;
    }
    
    h.p = clamp(rayAt(r,h.t),bmin,bmax);
    vec3 outwardNormal=exiting ? exitNormal : entryNormal;
    for(int i=0;i<3;++i) if(outwardNormal[i]!=0.0) h.p[i]=outwardNormal[i]<0.0 ? bmin[i] : bmax[i];
    setFaceNormal(r, h, outwardNormal);
    return true;
}

// Перегрузка для обратной совместимости (по умолчанию ищем вход)
bool hitBox(vec3 bmin, vec3 bmax, Ray r, float tMin, float tMax, inout Hit h) {
    return hitBox(bmin, bmax, r, tMin, tMax, h, false);
}

// Общие пересечения для продолжения пути и проверки видимости.
bool hitObject(int index, Ray ray, float near, float far, inout Hit h) {
 int typ,mat; vec3 position,size;
 readObject(index,typ,mat,position,size);
 h.matIndex=mat; h.objIndex=index;
 if(typ==OBJ_SPHERE) return hitSphere(position,size.x,ray,near,far,h);
 if(typ==OBJ_PLANE) return hitPlane(position,vec3(0,1,0),ray,near,far,h);
 return hitBox(position-size*0.5,position+size*0.5,ray,near,far,h);
}

bool hitWorld(Ray ray, float near, float far, out Hit h) {
 bool found=false;
 for(int i=0;i<objData.length()/OBJ_STRIDE;++i) {
  Hit candidate;
  if(hitObject(i,ray,near,far,candidate)) { h=candidate; far=h.t; found=true; }
 }
 return found;
}

// Outward-facing points of these convex emitters cannot occlude themselves.
bool occluded(Ray ray, float distance, int emitter) {
 for(int i=0;i<objData.length()/OBJ_STRIDE;++i) {
  if(i==emitter) continue;
  Hit h;
  if(hitObject(i,ray,0.0,distance,h)) return true;
 }
 return false;
}

vec3 fastCosineSample(vec3 normal, vec2 rngPair) {
    float phi = 6.28318530718 * rngPair.x;
    float cosTheta = sqrt(rngPair.y);
    float sinTheta = sqrt(1.0 - rngPair.y);
    
    // Быстрая локальная система координат
    vec3 axis = abs(normal.z) < 0.999 ? vec3(0,0,1) : vec3(1,0,0);
    vec3 u = normalize(cross(axis, normal));
    vec3 v = cross(normal, u);
    
    return normalize(u * (sinTheta * cos(phi)) + 
                     v * (sinTheta * sin(phi)) + 
                     normal * cosTheta);
}

// Оптимизированный GGX сэмплинг
vec3 fastGGXSample(vec3 viewDir, vec3 normal, float roughness, vec2 randomPair) {
 vec3 up=abs(normal.z)<0.999 ? vec3(0,0,1) : vec3(1,0,0);
 vec3 tangent=normalize(cross(up,normal)), bitangent=cross(normal,tangent);
 float alpha=roughness*roughness;
 vec3 stretched=normalize(vec3(alpha*dot(viewDir,tangent),alpha*dot(viewDir,bitangent),dot(viewDir,normal)));
 float lengthSquared=dot(stretched.xy,stretched.xy);
 vec3 t1=lengthSquared>0.0 ? vec3(-stretched.y,stretched.x,0)/sqrt(lengthSquared) : vec3(1,0,0);
 vec3 t2=cross(stretched,t1);
 float radius=sqrt(randomPair.x),phi=TWO_PI*randomPair.y;
 vec2 p=radius*vec2(cos(phi),sin(phi));
 float blend=(1.0+stretched.z)*0.5;
 p.y=(1.0-blend)*sqrt(max(0.0,1.0-p.x*p.x))+blend*p.y;
 vec3 sampled=p.x*t1+p.y*t2+sqrt(max(0.0,1.0-dot(p,p)))*stretched;
 vec3 h=normalize(vec3(alpha*sampled.xy,max(0.0,sampled.z)));
 return reflect(-viewDir,normalize(tangent*h.x+bitangent*h.y+normal*h.z));
}
// ========== КОНЕЦ ОПТИМИЗИРОВАННОГО IMPORTANCE SAMPLING ==========

vec3 randomCosineDirection(vec3 normal, inout uint state) {
    // Используем оптимизированную версию с парой случайных чисел
    float r1 = rng(state);
    float r2 = rng(state);
    vec2 rngPair = vec2(r1, r2);
    return fastCosineSample(normal, rngPair);
}

// GGX/Trowbridge-Reitz importance sampling для металлических материалов
// Генерирует направление отражения на основе GGX distribution
vec3 sampleGGX(vec3 viewDir, vec3 normal, float roughness, inout uint state) {
    // Используем оптимизированную версию с парой случайных чисел
    float r1 = rng(state);
    float r2 = rng(state);
    vec2 rngPair = vec2(r1, r2);
    return fastGGXSample(viewDir, normal, roughness, rngPair);
}

// Оптимизированная функция отражения
vec3 reflectVec(vec3 v, vec3 n) {
    // Оптимизация: кэшируем dot product
    float vn = dot(v, n);
    return v - 2.0 * vn * n;
}

// ОБНОВЛЕННАЯ ФИЗИЧЕСКИ КОРРЕКТНАЯ ФУНКЦИЯ ПРЕЛОМЛЕНИЯ
// Используем правильную формулу Снеллиуса с учетом IOR и нормалей
vec3 refractVec(vec3 v, vec3 n, float eta) {
    // Оптимизация: кэшируем dot product
    float vn = dot(-v, n);
    float cosTheta = min(vn, 1.0);
    float cosTheta2 = cosTheta * cosTheta;
    float sinTheta2 = 1.0 - cosTheta2;
    
    // Оптимизация: проверяем полное внутреннее отражение без sqrt
    float eta2SinTheta2 = eta * eta * sinTheta2;
    if (eta2SinTheta2 > 1.0) {
        // Полное внутреннее отражение
        return reflectVec(v, n);
    }
    
    float sinTheta = sqrt(sinTheta2);
    vec3 rOutPerp = eta * (v + cosTheta * n);
    float rOutPerpLenSq = dot(rOutPerp, rOutPerp);
    float rOutParallelLen = sqrt(1.0 - min(rOutPerpLenSq, 1.0));
    vec3 rOutParallel = -rOutParallelLen * n;
    return rOutPerp + rOutParallel;
}

// Улучшенная функция отражения Френеля с учетом обоих случаев (вход/выход)
float reflectance(float cosine, float relIOR) {
    // relIOR = n1/n2; неполяризованные точные уравнения Френеля.
    if (relIOR == 1.0) return 0.0;
    cosine = clamp(cosine, 0.0, 1.0);
    float sinT2 = relIOR * relIOR * (1.0 - cosine*cosine);
    if (sinT2 >= 1.0) return 1.0;
    float cosT = sqrt(1.0-sinT2);
    float rs = (relIOR*cosine-cosT)/(relIOR*cosine+cosT);
    float rp = (cosine-relIOR*cosT)/(cosine+relIOR*cosT);
    return 0.5*(rs*rs+rp*rp);
}

` + physicsShader + `

// Простейший диффузный BRDF (Lambertian) - оптимизировано: умножение вместо деления
` + lightingShader + `

// Фон / небо
vec3 backgroundColor(Ray r) {
    // Оптимизация: быстрая нормализация через inversesqrt
    float dirLen2 = dot(r.dir, r.dir);
    vec3 dir = r.dir * inversesqrt(dirLen2);
    int st = int(round(skyType));
    if (st == 2) {
        // gradient
        float t = (dir.y + 1.0) * 0.5;
        t = clamp(t, 0.0, 1.0);
        return mix(skyHorizon.rgb, skyZenith.rgb, t);
    } else if (st == 1) {
        // solid
        return skyColor.rgb;
    } else {
        // deprecated background
        return skyColor.rgb;
    }
}

// Камера: строим луч аналогично CPU newCamera/getRay
void buildCamera(vec2 uv, inout Ray r, inout uint state) {
    float aspect = camAspect != 0.0 ? camAspect : (float(uWidth) / float(uHeight));
    float theta = camFov * 3.14159265359 / 180.0;
    float h = tan(theta * 0.5);
    float viewportHeight = 2.0 * h;
    float viewportWidth = aspect * viewportHeight;

    vec3 origin = camPos.xyz;
    vec3 target = camTarget.xyz;
    vec3 up = camUp.xyz;

    // Оптимизация: быстрая нормализация через inversesqrt
    vec3 wBase = origin - target;
    float wLen2 = dot(wBase, wBase);
    vec3 w = wBase * inversesqrt(wLen2);
    vec3 uBase = cross(up, w);
    float uLen2 = dot(uBase, uBase);
    vec3 u = uBase * inversesqrt(uLen2);
    vec3 v = cross(w, u);

    float focusDist = camFocusDist != 0.0 ? camFocusDist : length(origin - target);
    vec3 horizontal = viewportWidth * focusDist * u;
    vec3 vertical   = viewportHeight * focusDist * v;
    vec3 lowerLeftCorner = origin - 0.5 * horizontal - 0.5 * vertical - w * focusDist;

    float lensRadius = camAperture * 0.5;
    if (lensRadius > 0.0) {
        float radius = lensRadius * sqrt(rng(state));
        float phi = TWO_PI * rng(state);
        vec3 rd = vec3(radius*cos(phi), radius*sin(phi), 0);
        vec3 offset = u * rd.x + v * rd.y;
        vec3 dir = lowerLeftCorner + uv.x * horizontal + uv.y * vertical - origin - offset;
        r.orig = origin + offset;
        // Оптимизация: быстрая нормализация через inversesqrt
        float dirLen2 = dot(dir, dir);
        r.dir = dir * inversesqrt(dirLen2);
    } else {
        vec3 dir = lowerLeftCorner + uv.x * horizontal + uv.y * vertical - origin;
        r.orig = origin;
        // Оптимизация: быстрая нормализация через inversesqrt
        float dirLen2 = dot(dir, dir);
        r.dir = dir * inversesqrt(dirLen2);
    }
}

vec3 applyFog(vec3 radiance, float distance) {
    if (fogDensity <= 0.0 || distance <= 0.0) {
        return radiance;
    }
    float d = max(distance, 0.0);
    float att = exp(-fogDensity * d);
    vec3 fogCol = fogColor.rgb;
    return radiance * att + fogCol * (1.0 - att);
}

// Фазовая функция Хеньи–Грина (HG) с параметром анизотропии g.
// g = 0   -> изотропное рассеяние
// g > 0   -> вперёд направленное (god rays от источников)
// g < 0   -> обратное рассеяние
float phaseHG(float cosTheta, float g) {
    float gg = g * g;
    float denom = 1.0 + gg - 2.0 * g * cosTheta;
    float denomSqrt = sqrt(max(denom, 1e-6));
    // Оптимизация: используем INV_PI вместо деления на PI
    return (1.0 - gg) * INV_PI * 0.25 / (denom * denomSqrt);
}

// Простейший трёхмерный hash‑шум.
float hash31(vec3 p) {
    vec3 q = vec3(
        dot(p, vec3(127.1, 311.7, 74.7)),
        dot(p, vec3(269.5, 183.3, 246.1)),
        dot(p, vec3(113.5, 271.9, 124.6))
    );
    return fract(sin(q.x + q.y + q.z) * 43758.5453);
}

// Фрактальный 3D‑шум для неоднородного тумана - оптимизировано: упрощение для малого количества октав
float volumeNoise(vec3 p) {
    float octaves = clamp(fogNoiseOctaves, 1.0, 5.0);
    int oct = int(octaves);
    
    // Оптимизация: быстрая версия для 1-2 октав
    if (oct <= 1) {
        return hash31(p * fogNoiseScale);
    }
    
    // Оптимизация: быстрая версия для 2 октав
    if (oct == 2) {
        float n1 = hash31(p * fogNoiseScale);
        float n2 = hash31(p * fogNoiseScale * 2.0);
        return mix(n1, n2, 0.5);
    }
    
    // Полная версия для 3+ октав
    float amp = 1.0;
    float freq = fogNoiseScale;
    float sum = 0.0;
    float norm = 0.0;
    for (int i = 0; i < 5; i++) {
        if (i >= oct) break;
        sum += hash31(p * freq) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    if (norm <= 0.0) return 1.0;
    return sum / norm; // 0..1
}

// Локальная экстинкция в точке с учётом неоднородности.
void mediumCoeffs(vec3 pos, out float sigma_s, out float sigma_a, out float sigma_t) {
    sigma_s = max(fogSigmaS, 0.0);
    sigma_a = max(fogSigmaA, 0.0);

    if (sigma_s <= 0.0 && sigma_a <= 0.0 && fogDensity > 0.0) {
        // fallback: выводим из fogDensity/fogScatter.
        float baseSigmaT = fogDensity;
        float sMul = clamp(fogScatter, 0.0, 1.0);
        sigma_s = baseSigmaT * sMul;
        sigma_a = baseSigmaT - sigma_s;
        if (sigma_a < 0.0) sigma_a = 0.0;
    }

    sigma_t = sigma_s + sigma_a;
    if (sigma_t <= 0.0) {
        sigma_s = 0.0;
        sigma_a = 0.0;
        return;
    }

    if (fogHeteroStrength > 0.0) {
        float n = volumeNoise(pos);
        // 0..1 -> (1-k)..(1+k)
        float k = clamp(fogHeteroStrength, 0.0, 1.0);
        float scale = mix(1.0 - k, 1.0 + k, n);
        sigma_s *= scale;
        sigma_a *= scale;
        sigma_t = sigma_s + sigma_a;
    }
}

` + fogShader + pathTransportShader

const wireframeShaderSource = `
#version 430
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba32f) uniform writeonly image2D destTex;

uniform int uWidth;
uniform int uHeight;
uniform int uSamplesPerPx;
uniform int uMaxDepth;
uniform uint uFrameSeed;
uniform int uSampleCount;
uniform int uSelectedObject; // Индекс выбранного объекта (-1 = нет выбора)
uniform ivec2 uDispatchOffset;

layout(std140, binding = 1) uniform CameraBlock {
    vec4 camPos;
    vec4 camTarget;
    vec4 camUp;
    float camFov;
    float camAperture;
    float camFocusDist;
    float camAspect;
};

layout(std430, binding = 3) readonly buffer MaterialBuffer {
    float matData[];
};

layout(std430, binding = 4) readonly buffer ObjectBuffer {
    float objData[];
};

layout(std430, binding = 7) buffer AccumBuffer {
    float accumData[];
};

struct Ray {
    vec3 orig;
    vec3 dir;
};

struct Hit {
    vec3 p;
    vec3 normal;
    float t;
    int matIndex;
    int objIndex;
    bool frontFace;
};

uint hash_u(uint x) {
    x ^= x >> 16;
    x *= 0x85ebca6bU;
    x ^= x >> 13;
    x *= 0xc2b2ae35U;
    x ^= x >> 16;
    return x;
}

float rng(inout uint state) {
    state = hash_u(state + 0x9e3779b9u);
    return float(state >> 8u) * (1.0 / 16777216.0);
}

void setFaceNormal(Ray r, inout Hit h, vec3 outwardNormal) {
    h.frontFace = dot(r.dir, outwardNormal) < 0.0;
    h.normal = h.frontFace ? outwardNormal : -outwardNormal;
}

const int OBJ_SPHERE = 0;
const int OBJ_PLANE = 1;
const int OBJ_BOX = 2;

const int MAT_LAMBERT = 0;
const int MAT_METAL = 1;
const int MAT_DIELECTRIC = 2;
const int MAT_EMISSIVE = 3;
const int MAT_MIRROR = 4;

const int OBJ_STRIDE = 12; // Должно совпадать с основным шейдером!

void readObject(int idx, out int typ, out int matIdx, out vec3 pos, out vec3 size) {
    int base = idx * OBJ_STRIDE;
    float ftyp = objData[base + 0];
    float fmat = objData[base + 1];
    typ = int(ftyp + 0.5);
    matIdx = int(fmat + 0.5);
    pos = vec3(objData[base + 4], objData[base + 5], objData[base + 6]);
    size = vec3(objData[base + 8], objData[base + 9], objData[base + 10]);
}

// Читаем количество объектов из буфера
int getObjectCount() {
    return int(objData.length()) / OBJ_STRIDE;
}

bool hitSphere(vec3 center, float radius, Ray r, float tMin, float tMax, inout Hit rec) {
    vec3 oc = r.orig - center;
    float a = dot(r.dir, r.dir);
    float halfB = dot(oc, r.dir);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = halfB * halfB - a * c;
    
    if (discriminant < 0.0) return false;
    float sqrtD = sqrt(discriminant);
    float root = (-halfB - sqrtD) / a;
    if (root < tMin || root > tMax) {
        root = (-halfB + sqrtD) / a;
        if (root < tMin || root > tMax) return false;
    }
    
    rec.t = root;
    rec.p = r.orig + root * r.dir;
    vec3 outwardNormal = (rec.p - center) / radius;
    setFaceNormal(r, rec, outwardNormal);
    return true;
}

bool hitPlane(vec3 point, vec3 normal, Ray r, float tMin, float tMax, inout Hit rec) {
    float denom = dot(normal, r.dir);
    if (abs(denom) < 1e-6) return false;
    float t = dot(point - r.orig, normal) / denom;
    if (t < tMin || t > tMax) return false;
    rec.t = t;
    rec.p = r.orig + t * r.dir;
    rec.p -= normal * dot(rec.p - point, normal);
    setFaceNormal(r, rec, normal);
    return true;
}

bool hitBox(vec3 bmin, vec3 bmax, Ray r, float tMin, float tMax, inout Hit rec) {
    vec3 invDir = 1.0 / r.dir;
    vec3 t0 = (bmin - r.orig) * invDir;
    vec3 t1 = (bmax - r.orig) * invDir;
    vec3 tMinVec = min(t0, t1);
    vec3 tMaxVec = max(t0, t1);
    float tNear = max(max(tMinVec.x, tMinVec.y), tMinVec.z);
    float tFar = min(min(tMaxVec.x, tMaxVec.y), tMaxVec.z);
    
    if (tFar < 0.0 || tNear > tFar) return false;
    float t = tNear;
    if (t < tMin) {
        t = tFar;
        if (t < tMin || t > tMax) return false;
    }
    if (t > tMax) return false;
    
    rec.t = t;
    rec.p = r.orig + t * r.dir;
    
    // УЛУЧШЕННОЕ вычисление нормали для четких граней (Minecraft-стиль)
    // Определяем, какая грань была пересечена первой
    vec3 center = (bmin + bmax) * 0.5;
    vec3 halfSize = (bmax - bmin) * 0.5;
    vec3 localP = rec.p - center;
    
    // Вычисляем расстояния до каждой грани
    vec3 distToFace = halfSize - abs(localP);
    
    // Находим ближайшую грань (минимальное расстояние)
    float minDist = min(distToFace.x, min(distToFace.y, distToFace.z));
    
    // Определяем нормаль по ближайшей грани
    vec3 normal = vec3(0.0);
    const float epsilon = 1e-5;
    
    if (abs(distToFace.x - minDist) < epsilon) {
        // Грань по X
        normal = vec3(sign(localP.x), 0.0, 0.0);
    } else if (abs(distToFace.y - minDist) < epsilon) {
        // Грань по Y
        normal = vec3(0.0, sign(localP.y), 0.0);
    } else {
        // Грань по Z
        normal = vec3(0.0, 0.0, sign(localP.z));
    }
    
    setFaceNormal(r, rec, normal);
    return true;
}

// УДАЛЕНО: функции hitWorld и isInShadow больше не нужны для wireframe режима
// Они были слишком медленными и вызывали проблемы с производительностью

void buildCamera(vec2 uv, inout Ray r, inout uint state) {
    float aspect = camAspect != 0.0 ? camAspect : (float(uWidth) / float(uHeight));
    float theta = camFov * 3.14159265359 / 180.0;
    float h = tan(theta * 0.5);
    float viewportHeight = 2.0 * h;
    float viewportWidth = aspect * viewportHeight;
    
    vec3 origin = camPos.xyz;
    vec3 target = camTarget.xyz;
    vec3 up = camUp.xyz;
    
    vec3 wBase = origin - target;
    float wLen2 = dot(wBase, wBase);
    vec3 w = wBase * inversesqrt(wLen2);
    vec3 uBase = cross(up, w);
    float uLen2 = dot(uBase, uBase);
    vec3 u = uBase * inversesqrt(uLen2);
    vec3 v = cross(w, u);
    
    float focusDist = camFocusDist != 0.0 ? camFocusDist : length(origin - target);
    vec3 horizontal = viewportWidth * focusDist * u;
    vec3 vertical = viewportHeight * focusDist * v;
    vec3 lowerLeftCorner = origin - 0.5 * horizontal - 0.5 * vertical - w * focusDist;
    
    vec3 dir = lowerLeftCorner + uv.x * horizontal + uv.y * vertical - origin;
    r.orig = origin;
    float dirLen2 = dot(dir, dir);
    r.dir = dir * inversesqrt(dirLen2);
}

// Чтение материала для wireframe режима
void readMaterialSimple(int idx, out int typ, out vec3 albedo, out vec3 emit) {
    int base = idx * 20; // MAT_STRIDE = 20
    float ftyp = matData[base + 0];
    typ = int(ftyp + 0.5);
    albedo = vec3(matData[base + 4], matData[base + 5], matData[base + 6]);
    emit = vec3(matData[base + 8], matData[base + 9], matData[base + 10]);
}

vec3 wireframeColor(Ray r, inout uint state) {
    int objCount = getObjectCount();
    if (objCount <= 0) return vec3(0.1, 0.1, 0.1);
    
    Hit closestHit;
    bool hitFound = false;
    float closest = 1e30;
    
    // Оптимизация: кэшируем часто используемые значения
    vec3 rDir = r.dir;
    vec3 rOrig = r.orig;
    
    // Находим ближайший объект с ранним выходом
    for (int i = 0; i < objCount; i++) {
        int typ, matIdx;
        vec3 pos, size;
        readObject(i, typ, matIdx, pos, size);
        
        Hit temp;
        temp.matIndex = matIdx;
        temp.objIndex = i;
        bool hitObj = false;
        
        float tMin = 0.001;
        float tMax = closest;
        
        // Оптимизация: ранний выход для объектов, которые точно не пересекаются
        // Используем упрощенные проверки для быстрого отсечения
        
        if (typ == OBJ_SPHERE) {
            hitObj = hitSphere(pos, size.x, r, tMin, tMax, temp);
        } else if (typ == OBJ_PLANE) {
            const vec3 planeNormal = vec3(0.0, 1.0, 0.0);
            hitObj = hitPlane(pos, planeNormal, r, tMin, tMax, temp);
        } else if (typ == OBJ_BOX) {
            // Оптимизация: предвычисляем bmin/bmax один раз
            vec3 halfSize = 0.5 * size;
            vec3 bmin = pos - halfSize;
            vec3 bmax = pos + halfSize;
            hitObj = hitBox(bmin, bmax, r, tMin, tMax, temp);
        }
        
        if (hitObj && temp.t < closest) {
            hitFound = true;
            closest = temp.t;
            closestHit = temp;
            // Оптимизация: ранний выход не применяем, так как нужно найти ближайший объект
        }
    }
    
    if (!hitFound) {
        return vec3(0.1, 0.1, 0.1);
    }
    
    // Проверяем, является ли объект выбранным (кэшируем проверку)
    bool isSelected = (uSelectedObject >= 0 && closestHit.objIndex == uSelectedObject);
    
    // Читаем тип объекта и материал
    int objTyp, matIdx;
    vec3 objPos, objSize;
    readObject(closestHit.objIndex, objTyp, matIdx, objPos, objSize);
    
    int matTyp;
    vec3 albedo, emit;
    readMaterialSimple(closestHit.matIndex, matTyp, albedo, emit);
    
    // Оптимизация: кэшируем нормаль
    vec3 normal = closestHit.normal;
    float ny = normal.y; // Кэшируем для быстрого доступа
    
    // MINECRAFT-СТИЛЬ ОСВЕЩЕНИЕ: оптимизированная версия
    float brightness = 1.0;
    
    if (objTyp == OBJ_BOX || objTyp == OBJ_PLANE) {
        // Оптимизация: используем кэшированное значение ny
        if (ny > 0.9) {
            brightness = 1.0;
        } else if (ny < -0.9) {
            brightness = 0.4;
        } else {
            brightness = 0.6;
        }
    } else if (objTyp == OBJ_SPHERE) {
        // Оптимизация: предвычисляем lightDir один раз (константа)
        const vec3 lightDir = vec3(0.4472136, 0.8944272, 0.2683282); // normalize(vec3(0.5, 1.0, 0.3))
        float ndotl = max(0.0, dot(normal, lightDir));
        brightness = 0.5 + ndotl * 0.5;
    }
    
    // Базовый цвет с учетом яркости грани
    vec3 color = albedo * brightness;
    
    // Оптимизация: используем switch-like структуру для материалов
    if (matTyp == 3) { // MAT_EMISSIVE
        color = emit * 3.0;
    } else if (matTyp == 2) { // MAT_DIELECTRIC
        const vec3 tint = vec3(0.85, 0.9, 0.95);
        color = mix(color, tint * brightness, 0.4);
    } else if (matTyp == 1 || matTyp == 4) { // MAT_METAL или MAT_MIRROR
        color = mix(color, albedo * brightness * 1.3, 0.8);
    }
    
    // Подсветка выбранного объекта - оптимизированная версия
    if (isSelected) {
        const vec3 selectionColor = vec3(0.0, 1.0, 0.0);
        // Оптимизация: используем уже нормализованный r.dir (если он нормализован)
        // Для упрощения используем -r.dir напрямую
        vec3 viewDir = -rDir; // Оптимизация: не нормализуем, если r.dir уже нормализован
        float viewDirLen2 = dot(viewDir, viewDir);
        if (abs(viewDirLen2 - 1.0) > 0.01) {
            viewDir *= inversesqrt(viewDirLen2);
        }
        
        float viewDot = dot(viewDir, normal);
        float edgeFactor = abs(viewDot);
        
        // Оптимизация: упрощенные проверки с предвычисленными константами
        if (edgeFactor < 0.2) {
            float edgeStrength = 1.0 - (edgeFactor * 5.0); // 1.0 / 0.2 = 5.0
            color = mix(color, selectionColor, edgeStrength * 0.8);
        } else if (edgeFactor < 0.5) {
            float transition = (edgeFactor - 0.2) * 3.3333333; // 1.0 / 0.3
            color = mix(color, selectionColor, (1.0 - transition) * 0.4);
        } else {
            color = mix(color, selectionColor, 0.15);
        }
    }
    
    // Ограничиваем яркость
    color = min(color, vec3(1.0));
    return color;
}

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy) + uDispatchOffset;
    if (pix.x >= uWidth || pix.y >= uHeight) return;
    
    uint state = uint(pix.y * uWidth + pix.x) + uFrameSeed;
    
    // Для wireframe используем один сэмпл
    float u = (float(pix.x)+0.5) / float(max(uWidth, 1));
    float fy = float(uHeight - 1 - pix.y);
    float v = (fy+0.5) / float(max(uHeight, 1));
    
    Ray r;
    buildCamera(vec2(u, v), r, state);
    vec3 col = wireframeColor(r, state);
    
    // Пишем цвет напрямую
    imageStore(destTex, pix, vec4(col, 1.0));
}
`

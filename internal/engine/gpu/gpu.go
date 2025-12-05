package gpu

import (
	"fmt"
	"image"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-gl/gl/v3.3-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"

	"github.com/user/pathtracer/internal/scene"
)

// acesTonemap применяет простую аппроксимацию ACES-фильма к одному
// каналу линейного HDR-цвета (0..+inf) и возвращает значение в диапазоне [0,1].
func acesTonemap(x float32) float32 {
	if x <= 0 {
		return 0
	}
	// Классическая аппроксимация ACES:
	// x*(a*x + b) / (x*(c*x + d) + e)
	const a = 2.51
	const b = 0.03
	const c = 2.43
	const d = 0.59
	const e = 0.14

	y := float64(x)
	num := y * (a*y + b)
	den := y*(c*y+d) + e
	if den <= 0 {
		return 0
	}
	r := num / den
	if r < 0 {
		r = 0
	} else if r > 1 {
		r = 1
	}
	return float32(r)
}

// denoiseConfig описывает параметры пост-денойзинга для GPU-рендера.
type denoiseConfig struct {
	Enabled bool
	SigmaS  float64
	SigmaR  float64
}

var (
	denoiseCfgOnce  sync.Once
	denoiseCfg      denoiseConfig
	denoiseOverride bool
)

// getDenoiseConfig читает настройки денойзера из переменных окружения один раз.
// PATHTRACER_GPU_DENOISE: "0"/"false"/"off" — выкл, "1"/"true"/"on" — вкл (по умолчанию: вкл).
// PATHTRACER_GPU_DENOISE_SIGMA_S: пространственный радиус (по умолчанию 1.0).
// PATHTRACER_GPU_DENOISE_SIGMA_R: радиус по цвету в sRGB (0..1), по умолчанию 0.15.
func getDenoiseConfig() denoiseConfig {
	if denoiseOverride {
		return denoiseCfg
	}
	denoiseCfgOnce.Do(func() {
		cfg := denoiseConfig{
			Enabled: true,
			SigmaS:  1.0,
			SigmaR:  0.15,
		}

		if v := os.Getenv("PATHTRACER_GPU_DENOISE"); v != "" {
			switch vLower := strings.ToLower(v); vLower {
			case "0", "false", "off", "no":
				cfg.Enabled = false
			case "1", "true", "on", "yes":
				cfg.Enabled = true
			}
		}
		if v := os.Getenv("PATHTRACER_GPU_DENOISE_SIGMA_S"); v != "" {
			if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 {
				cfg.SigmaS = f
			}
		}
		if v := os.Getenv("PATHTRACER_GPU_DENOISE_SIGMA_R"); v != "" {
			if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 {
				cfg.SigmaR = f
			}
		}

		denoiseCfg = cfg
	})
	return denoiseCfg
}

// SetDenoiseConfigFromUI позволяет UI/CLI переопределить настройки денойзинга
// во время работы приложения, не полагаясь только на переменные окружения.
func SetDenoiseConfigFromUI(enabled bool, sigmaS, sigmaR float64) {
	if sigmaS <= 0 {
		sigmaS = 1.0
	}
	if sigmaR <= 0 {
		sigmaR = 0.15
	}
	denoiseCfg = denoiseConfig{
		Enabled: enabled,
		SigmaS:  sigmaS,
		SigmaR:  sigmaR,
	}
	denoiseOverride = true
}

// Дополнительный конфиг для более сильного сглаживания (blur) после денойза.
type smoothConfig struct {
	Enabled  bool
	Radius   int     // радиус бокса (в пикселях), 1..5
	Strength float64 // сила смешивания 0..1 (0 = нет эффекта, 1 = только blur)
}

var (
	smoothCfgOnce  sync.Once
	smoothCfg      smoothConfig
	smoothOverride bool
)

// getSmoothConfig читает настройки сглаживания из переменных окружения один раз.
// PATHTRACER_GPU_SMOOTH: "0"/"false"/"off" — выкл, "1"/"true"/"on" — вкл (по умолчанию: выкл).
// PATHTRACER_GPU_SMOOTH_RADIUS: целое 1..5 (по умолчанию 2).
// PATHTRACER_GPU_SMOOTH_STRENGTH: 0..1 (по умолчанию 0.5).
func getSmoothConfig() smoothConfig {
	if smoothOverride {
		return smoothCfg
	}
	smoothCfgOnce.Do(func() {
		cfg := smoothConfig{
			Enabled:  false,
			Radius:   2,
			Strength: 0.5,
		}

		if v := os.Getenv("PATHTRACER_GPU_SMOOTH"); v != "" {
			switch strings.ToLower(v) {
			case "1", "true", "on", "yes":
				cfg.Enabled = true
			case "0", "false", "off", "no":
				cfg.Enabled = false
			}
		}
		if v := os.Getenv("PATHTRACER_GPU_SMOOTH_RADIUS"); v != "" {
			if r, err := strconv.Atoi(v); err == nil {
				if r < 1 {
					r = 1
				}
				if r > 5 {
					r = 5
				}
				cfg.Radius = r
			}
		}
		if v := os.Getenv("PATHTRACER_GPU_SMOOTH_STRENGTH"); v != "" {
			if f, err := strconv.ParseFloat(v, 64); err == nil {
				if f < 0 {
					f = 0
				}
				if f > 1 {
					f = 1
				}
				cfg.Strength = f
			}
		}

		smoothCfg = cfg
	})
	return smoothCfg
}

// SetSmoothConfigFromUI переопределяет настройки сглаживания из UI.
func SetSmoothConfigFromUI(enabled bool, radius int, strength float64) {
	if radius < 1 {
		radius = 1
	}
	if radius > 5 {
		radius = 5
	}
	if strength < 0 {
		strength = 0
	}
	if strength > 1 {
		strength = 1
	}
	smoothCfg = smoothConfig{
		Enabled:  enabled,
		Radius:   radius,
		Strength: strength,
	}
	smoothOverride = true
}

// gpuRenderer owns a hidden GLFW window and GL resources used for compute rendering.
type gpuRenderer struct {
	initOnce   sync.Once
	initErr    error
	window     *glfw.Window
	program    uint32
	imgTexture uint32
	pbo        uint32
	matSSBO    uint32
	objSSBO    uint32
	lightSSBO  uint32 // SSBO для списка индексов эмиссивных объектов
	accumSSBO  uint32 // SSBO для накопительного буфера на GPU (width * height * 3 * float32)
	camUBO     uint32
	skyUBO     uint32
	fogUBO     uint32
	width      int
	height     int
	// accum содержит накопленные значения цвета в диапазоне [0,1] для каждого пикселя (R,G,B).
	// Используется только для чтения финального результата, накопление происходит на GPU.
	accum []float32
}

// RenderConfig is a minimal copy of engine.RenderConfig to avoid import cycles.
type RenderConfig struct {
	Width        int
	Height       int
	SamplesPerPx int
	MaxDepth     int
}

// Go-side copies of material / object type constants used in GLSL.
// Must stay in sync with values in the compute shader.
const (
	MAT_LAMBERT    = 0
	MAT_METAL      = 1
	MAT_DIELECTRIC = 2
	MAT_EMISSIVE   = 3
	MAT_MIRROR     = 4
)

const (
	OBJ_SPHERE = 0
	OBJ_PLANE  = 1
	OBJ_BOX    = 2
)

// renderRequest is sent from callers to the dedicated GL worker goroutine.
type renderRequest struct {
	sc       *scene.Scene
	cfg      RenderConfig
	img      *image.RGBA
	progress func()
	done     chan error
}

var (
	renderer   gpuRenderer
	renderCh   chan renderRequest
	workerOnce sync.Once
)

// ensureWorker starts the dedicated GL worker goroutine exactly once.
func ensureWorker() {
	workerOnce.Do(func() {
		renderCh = make(chan renderRequest)
		go renderWorker()
	})
}

// renderWorker owns the GL context and processes all GPU render requests.
// It always runs on a single locked OS thread, which is required by OpenGL.
func renderWorker() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := renderer.initGL(); err != nil {
		// Если не удалось инициализировать GL, логируем ошибку и отвечаем ошибкой на все запросы.
		fmt.Fprintf(os.Stderr, "GPU initialization failed: %v\n", err)
		for req := range renderCh {
			req.done <- err
		}
		return
	}

	fmt.Fprintf(os.Stderr, "GPU renderer initialized successfully\n")

	for req := range renderCh {
		err := renderer.renderOnce(req.sc, req.cfg, req.img, req.progress)
		if err != nil {
			fmt.Fprintf(os.Stderr, "GPU render error: %v\n", err)
		}
		req.done <- err
	}
}

// initGL must be called from the GL worker goroutine (locked OS thread).
func (r *gpuRenderer) initGL() error {
	r.initOnce.Do(func() {
		if err := glfw.Init(); err != nil {
			r.initErr = fmt.Errorf("glfw init: %w", err)
			return
		}

		glfw.WindowHint(glfw.Visible, glfw.False)
		glfw.WindowHint(glfw.ContextVersionMajor, 4)
		glfw.WindowHint(glfw.ContextVersionMinor, 3)
		glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)

		w, err := glfw.CreateWindow(1, 1, "pathtracer-gpu-hidden", nil, nil)
		if err != nil {
			r.initErr = fmt.Errorf("glfw create window: %w", err)
			return
		}
		r.window = w
		w.MakeContextCurrent()

		if err := gl.Init(); err != nil {
			r.initErr = fmt.Errorf("gl init: %w", err)
			return
		}

		// Create textures/buffers; resized on first use.
		gl.GenTextures(1, &r.imgTexture)
		gl.BindTexture(gl.TEXTURE_2D, r.imgTexture)
		// Храним результат в HDR-подобном формате (RGBA16F), чтобы минимизировать квантование
		// и аккуратно накапливать линейный цвет. Фильтры не важны для compute, но оставляем NEAREST.
		gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
		gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)

		gl.GenBuffers(1, &r.pbo)
		gl.GenBuffers(1, &r.matSSBO)
		gl.GenBuffers(1, &r.objSSBO)
		gl.GenBuffers(1, &r.lightSSBO)
		gl.GenBuffers(1, &r.accumSSBO)
		gl.GenBuffers(1, &r.camUBO)
		gl.GenBuffers(1, &r.skyUBO)
		gl.GenBuffers(1, &r.fogUBO)

		// Compute shader: полноценный (упрощённый) path tracing с поддержкой
		// нескольких материалов, диэлектриков, базового importance sampling источников света
		// и простого однородного тумана.
		const computeSrc = `
#version 430
layout(local_size_x = 16, local_size_y = 16) in;

// Используем формат rgba16f, чтобы соответствовать текстуре RGBA16F на стороне Go.
layout(binding = 0, rgba16f) uniform writeonly image2D destTex;

uniform int uWidth;
uniform int uHeight;
uniform int uSamplesPerPx;
uniform int uMaxDepth;
uniform uint uFrameSeed;
uniform int uSampleCount; // Текущее количество накопленных сэмплов

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
    vec3 _padSky0;    // выравнивание std140
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

// Быстрый хэш-рандом по пикселю и сэмплу
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
    state = hash_u(state);
    return float(state) / 4294967296.0;
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

// Пересечения
bool hitSphere(vec3 center, float radius, Ray r, float tMin, float tMax, inout Hit h) {
    vec3 oc = r.orig - center;
    float a = dot(r.dir, r.dir);
    float halfB = dot(oc, r.dir);
    float c = dot(oc, oc) - radius * radius;
    float disc = halfB * halfB - a * c;
    // Проверка на численную стабильность: касательный луч или численная погрешность
    if (disc < 1e-8) return false;
    float sqrtD = sqrt(disc);

    float root = (-halfB - sqrtD) / a;
    if (root < tMin || root > tMax) {
        root = (-halfB + sqrtD) / a;
        if (root < tMin || root > tMax) return false;
    }
    h.t = root;
    h.p = rayAt(r, root);
    vec3 outward = (h.p - center) / radius;
    setFaceNormal(r, h, outward);
    return true;
}

bool hitPlane(vec3 point, vec3 normal, Ray r, float tMin, float tMax, inout Hit h) {
    float denom = dot(normal, r.dir);
    if (abs(denom) < 1e-6) return false;
    float t = dot(point - r.orig, normal) / denom;
    if (t < tMin || t > tMax) return false;
    h.t = t;
    h.p = rayAt(r, t);
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
    float t0 = tMin;
    float t1 = tMax;
    
    for (int i = 0; i < 3; i++) {
        float invD = 1.0 / (i == 0 ? r.dir.x : (i == 1 ? r.dir.y : r.dir.z));
        float orig = (i == 0 ? r.orig.x : (i == 1 ? r.orig.y : r.orig.z));
        float minV = (i == 0 ? bmin.x : (i == 1 ? bmin.y : bmin.z));
        float maxV = (i == 0 ? bmax.x : (i == 1 ? bmax.y : bmax.z));
        
        float tNear = (minV - orig) * invD;
        float tFar  = (maxV - orig) * invD;
        
        if (invD < 0.0) {
            float tmp = tNear; tNear = tFar; tFar = tmp;
        }
        
        t0 = max(t0, tNear);  // Ближайшее пересечение
        t1 = min(t1, tFar);   // Дальнее пересечение
        
        if (t1 <= t0) return false;
    }
    
    // Выбираем правильное пересечение
    h.t = findExit ? t1 : t0;
    
    if (h.t < tMin || h.t > tMax) {
        return false;
    }
    
    h.p = rayAt(r, h.t);
    
    // УЛУЧШЕННОЕ ВЫЧИСЛЕНИЕ НОРМАЛИ
    vec3 center = (bmin + bmax) * 0.5;
    vec3 halfSize = (bmax - bmin) * 0.5;
    vec3 localPoint = h.p - center;
    
    // Находим ближайшую грань
    vec3 absLocal = abs(localPoint);
    float maxDist = max(absLocal.x, max(absLocal.y, absLocal.z));
    
    // Определяем нормаль по ближайшей грани
    vec3 outwardNormal;
    const float epsilon = 1e-4;
    
    if (abs(absLocal.x - halfSize.x) < epsilon) {
        outwardNormal = vec3(sign(localPoint.x), 0.0, 0.0);
    } else if (abs(absLocal.y - halfSize.y) < epsilon) {
        outwardNormal = vec3(0.0, sign(localPoint.y), 0.0);
    } else {
        outwardNormal = vec3(0.0, 0.0, sign(localPoint.z));
    }
    
    // Для выхода инвертируем нормаль
    if (findExit) {
        outwardNormal = -outwardNormal;
    }
    
    setFaceNormal(r, h, outwardNormal);
    return true;
}

// Перегрузка для обратной совместимости (по умолчанию ищем вход)
bool hitBox(vec3 bmin, vec3 bmax, Ray r, float tMin, float tMax, inout Hit h) {
    return hitBox(bmin, bmax, r, tMin, tMax, h, false);
}

// Функция для получения обоих пересечений (вход и выход)
bool hitBoxFull(vec3 bmin, vec3 bmax, Ray r, float tMin, float tMax, 
                out Hit entry, out Hit exit) {
    float t0 = tMin;
    float t1 = tMax;
    
    for (int i = 0; i < 3; i++) {
        float invD = 1.0 / (i == 0 ? r.dir.x : (i == 1 ? r.dir.y : r.dir.z));
        float orig = (i == 0 ? r.orig.x : (i == 1 ? r.orig.y : r.orig.z));
        float minV = (i == 0 ? bmin.x : (i == 1 ? bmin.y : bmin.z));
        float maxV = (i == 0 ? bmax.x : (i == 1 ? bmax.y : bmax.z));
        
        float tNear = (minV - orig) * invD;
        float tFar  = (maxV - orig) * invD;
        
        if (invD < 0.0) {
            float tmp = tNear; tNear = tFar; tFar = tmp;
        }
        
        t0 = max(t0, tNear);
        t1 = min(t1, tFar);
        
        if (t1 <= t0) return false;
    }
    
    if (t0 < tMin || t0 > tMax) return false;
    if (t1 < tMin || t1 > tMax) return false;
    
    // Заполняем точку входа
    entry.t = t0;
    entry.p = rayAt(r, t0);
    
    // Заполняем точку выхода
    exit.t = t1;
    exit.p = rayAt(r, t1);
    
    // Вычисляем нормали
    vec3 center = (bmin + bmax) * 0.5;
    vec3 halfSize = (bmax - bmin) * 0.5;
    
    // Нормаль для входа (направлена наружу)
    vec3 localEntry = entry.p - center;
    vec3 entryOutwardNormal = computeBoxNormal(localEntry, halfSize);
    setFaceNormal(r, entry, entryOutwardNormal);
    
    // Нормаль для выхода (направлена внутрь)
    vec3 localExit = exit.p - center;
    vec3 exitOutwardNormal = computeBoxNormal(localExit, halfSize);
    // Для выхода инвертируем нормаль
    setFaceNormal(r, exit, -exitOutwardNormal);
    
    return true;
}

bool hitWorld(Ray r, float tMin, float tMax, out Hit outHit) {
    bool hitAnything = false;
    float closest = tMax;
    Hit temp;
    int objCount = int(objData.length()) / OBJ_STRIDE;
    for (int i = 0; i < objCount; i++) {
        int typ, matIdx;
        vec3 pos, size;
        readObject(i, typ, matIdx, pos, size);
        temp.matIndex = matIdx;
        bool hitObj = false;
        if (typ == OBJ_SPHERE) {
            hitObj = hitSphere(pos, size.x, r, tMin, closest, temp);
        } else if (typ == OBJ_PLANE) {
            hitObj = hitPlane(pos, vec3(0, 1, 0), r, tMin, closest, temp);
        } else if (typ == OBJ_BOX) {
            vec3 bmin = pos - 0.5 * size;
            vec3 bmax = pos + 0.5 * size;
            hitObj = hitBox(bmin, bmax, r, tMin, closest, temp);
        }
        if (hitObj) {
            hitAnything = true;
            closest = temp.t;
            outHit = temp;
            outHit.objIndex = i;  // Запоминаем индекс объекта для правильного поиска выхода
        }
    }
    return hitAnything;
}

// Случайные направления
vec3 randomInUnitSphere(inout uint state) {
    for (int i = 0; i < 16; i++) {
        vec3 p = 2.0 * vec3(rng(state), rng(state), rng(state)) - vec3(1.0);
        if (dot(p, p) >= 1.0) continue;
        return p;
    }
    return vec3(0, 0, 1);
}

vec3 randomCosineDirection(vec3 normal, inout uint state) {
    float r1 = rng(state);
    float r2 = rng(state);
    float phi = 2.0 * 3.14159265359 * r1;
    float cosTheta = sqrt(r2);
    float sinTheta = sqrt(1.0 - r2);
    vec3 u = normalize(abs(normal.x) > 0.9 ? vec3(0, 1, 0) : vec3(1, 0, 0));
    vec3 v = normalize(cross(normal, u));
    vec3 w = normal;
    vec3 localDir = vec3(
        sinTheta * cos(phi),
        sinTheta * sin(phi),
        cosTheta
    );
    return normalize(localDir.x * u + localDir.y * v + localDir.z * w);
}

// GGX/Trowbridge-Reitz importance sampling для металлических материалов
// Генерирует направление отражения на основе GGX distribution
vec3 sampleGGX(vec3 viewDir, vec3 normal, float roughness, inout uint state) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    
    // Сэмплируем полусферу в локальной системе координат
    float r1 = rng(state);
    float r2 = rng(state);
    
    // GGX importance sampling для нормалей микроповерхности
    float cosTheta = sqrt((1.0 - r2) / (1.0 + (alpha2 - 1.0) * r2));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    float phi = 2.0 * PI * r1;
    
    // Локальная система координат
    vec3 tangent, bitangent;
    vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    tangent = normalize(cross(up, normal));
    bitangent = cross(normal, tangent);
    
    // Нормаль микроповерхности в локальной системе
    vec3 halfVecLocal = vec3(
        sinTheta * cos(phi),
        sinTheta * sin(phi),
        cosTheta
    );
    
    // Преобразуем в мировую систему координат
    vec3 halfVec = normalize(
        halfVecLocal.x * tangent +
        halfVecLocal.y * bitangent +
        halfVecLocal.z * normal
    );
    
    // Вычисляем направление отражения
    vec3 reflectDir = reflect(-viewDir, halfVec);
    
    // Проверяем, что отраженное направление находится в правильной полусфере
    if (dot(reflectDir, normal) <= 0.0) {
        // Если нет, используем идеальное отражение как fallback
        reflectDir = reflect(-viewDir, normal);
    }
    
    return normalize(reflectDir);
}

vec3 reflectVec(vec3 v, vec3 n) {
    return v - 2.0 * dot(v, n) * n;
}

// ОБНОВЛЕННАЯ ФИЗИЧЕСКИ КОРРЕКТНАЯ ФУНКЦИЯ ПРЕЛОМЛЕНИЯ
// Используем правильную формулу Снеллиуса с учетом IOR и нормалей
vec3 refractVec(vec3 v, vec3 n, float eta) {
    float cosTheta = min(dot(-v, n), 1.0);
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    
    if (eta * sinTheta > 1.0) {
        // Полное внутреннее отражение
        return reflectVec(v, n);
    }
    
    vec3 rOutPerp = eta * (v + cosTheta * n);
    vec3 rOutParallel = -sqrt(1.0 - min(dot(rOutPerp, rOutPerp), 1.0)) * n;
    return rOutPerp + rOutParallel;
}

// Улучшенная функция отражения Френеля с учетом обоих случаев (вход/выход)
float reflectance(float cosine, float relIOR) {
    // relIOR = n2/n1 (n1 - откуда луч, n2 - куда луч)
    
    // Используем аппроксимацию Шлика для большей точности
    float r0 = (relIOR - 1.0) / (relIOR + 1.0);
    r0 = r0 * r0;
    
    // Для углов, близких к скользящим, увеличиваем отражение
    float x = 1.0 - cosine;
    float x5 = x * x * x * x * x;
    
    return r0 + (1.0 - r0) * x5;
}

// Простейший диффузный BRDF (Lambertian)
vec3 brdfLambert(vec3 albedo) {
	return albedo / PI;
}

// Получение количества источников света (оптимизированная версия)
int getLightCount() {
	return lightIndices.length();
}

// Получение индекса объекта источника света по индексу в списке источников
int getLightObjectIndex(int lightIdx) {
	if (lightIdx < 0 || lightIdx >= lightIndices.length()) {
		return -1;
	}
	return lightIndices[lightIdx];
}

// Сэмплирование точки на источнике света и расчёт PDF по площади.
// Для простоты корректно обрабатываем только сферические источники (лампы).
bool sampleLightGeometry(
	int objIndex,
	out vec3 lightPos,
	out vec3 lightNormal,
	out float pdfArea,
	inout uint state
) {
	int oTyp, mIdx;
	vec3 pos, size;
	readObject(objIndex, oTyp, mIdx, pos, size);

	if (oTyp == OBJ_SPHERE) {
		float radius = size.x;
		// uniform на сфере
		float u1 = rng(state);
		float u2 = rng(state);
		float z = 1.0 - 2.0 * u1;
		float r = sqrt(max(0.0, 1.0 - z * z));
		float phi = 2.0 * PI * u2;
		vec3 local = vec3(r * cos(phi), r * sin(phi), z);

		lightNormal = normalize(local);
		lightPos = pos + radius * lightNormal;
		float area = 4.0 * PI * radius * radius;
		pdfArea = 1.0 / area;
		return true;
	}

	// Для остальных типов пока не реализуем area sampling
	return false;
}

// Оценка прямого освещения от одного источника света (next event estimation).
// Используется для сэмплирования отдельного источника.
vec3 estimateDirectLightSingle(
	int lightObjIndex,
	int lightMatIndex,
	Hit surfHit,
	vec3 albedo,
	inout uint state
) {
	vec3 lightPos, lightNormal;
	float pdfArea;
	if (!sampleLightGeometry(lightObjIndex, lightPos, lightNormal, pdfArea, state)) {
		return vec3(0.0);
	}
	if (pdfArea <= 0.0) {
		return vec3(0.0);
	}

	// Направление и расстояние до источника
	vec3 toLight = lightPos - surfHit.p;
	float distSq = dot(toLight, toLight);
	if (distSq <= 1e-6) {
		return vec3(0.0);
	}
	float dist = sqrt(distSq);
	vec3 wi = toLight / dist;

	// Проверка видимости: shadow ray до источника
	Ray shadowRay;
	shadowRay.orig = surfHit.p + surfHit.normal * 0.001;
	shadowRay.dir = wi;
	Hit shadowHit;
	if (hitWorld(shadowRay, 0.001, dist - 0.002, shadowHit)) {
		// что-то блокирует источник
		return vec3(0.0);
	}

	// Читаем светоотдачу источника
	int mTyp;
	float mRough, mIor, mSmoothness, mReflectivity, mAbsScale;
	vec3 mAlbedo, mEmit, mAbs, mTint;
	readMaterial(lightMatIndex, mTyp, mRough, mIor, mAlbedo, mEmit, mAbs, mSmoothness, mReflectivity, mTint, mAbsScale);
	if (mTyp != MAT_EMISSIVE) {
		return vec3(0.0);
	}

	// Косинусные множители
	float cosSurf = max(0.0, dot(surfHit.normal, wi));
	float cosLight = max(0.0, dot(lightNormal, -wi));
	if (cosSurf <= 0.0 || cosLight <= 0.0) {
		return vec3(0.0);
	}

	// BRDF для диффузной поверхности
	vec3 f = brdfLambert(albedo);

	// Геометрический термин и переход из pdf по площади в pdf по направлению
	float geometry = (cosSurf * cosLight) / max(1e-6, distSq);
	vec3 contrib = f * mEmit * geometry / max(1e-6, pdfArea);

	// Улучшенный firefly reduction: используем более умный подход
	// Вместо простого clamp, применяем мягкое ограничение на основе яркости
	float luminance = dot(contrib, vec3(0.2126, 0.7152, 0.0722));
	float maxLuminance = 500.0; // увеличиваем максимальную допустимую яркость для предотвращения затемнения
	if (luminance > maxLuminance) {
		float scale = maxLuminance / max(luminance, 1e-6);
		contrib *= scale;
	}
	
	return contrib;
}

// Оценка прямого освещения от ВСЕХ источников света (sample all lights).
// Это значительно улучшает качество освещения, особенно для сцен с несколькими источниками.
// Использует оптимизированный список источников света вместо сканирования всех объектов.
vec3 estimateDirectLight(
	Ray r,
	Hit surfHit,
	vec3 albedo,
	inout uint state
) {
	vec3 totalContrib = vec3(0.0);
	int lightCount = getLightCount();
	
	if (lightCount == 0) {
		return vec3(0.0);
	}
	
	// Оптимизация производительности: ограничиваем количество источников для сэмплирования
	// Сэмплируем максимум 8 источников для предотвращения падения производительности
	const int MAX_LIGHTS_TO_SAMPLE = 8;
	int lightsToSample = lightCount < MAX_LIGHTS_TO_SAMPLE ? lightCount : MAX_LIGHTS_TO_SAMPLE;
	
	// Если источников больше максимума, случайно выбираем подмножество
	if (lightCount > MAX_LIGHTS_TO_SAMPLE) {
		// Используем детерминированный выбор на основе состояния RNG
		// для обеспечения воспроизводимости
		int startIdx = int(rng(state) * float(lightCount)) % lightCount;
		for (int j = 0; j < lightsToSample; j++) {
			int i = (startIdx + j) % lightCount;
			int objIdx = getLightObjectIndex(i);
			if (objIdx < 0) {
				continue;
			}
			
			int oTyp, mIdx;
			vec3 pos, size;
			readObject(objIdx, oTyp, mIdx, pos, size);
			
			vec3 contrib = estimateDirectLightSingle(objIdx, mIdx, surfHit, albedo, state);
			totalContrib += contrib;
		}
		// Масштабируем для компенсации того, что мы сэмплировали не все источники
		totalContrib *= float(lightCount) / float(lightsToSample);
	} else {
		// Сэмплируем все источники, если их немного
		for (int i = 0; i < lightCount; i++) {
			int objIdx = getLightObjectIndex(i);
			if (objIdx < 0) {
				continue;
			}
			
			int oTyp, mIdx;
			vec3 pos, size;
			readObject(objIdx, oTyp, mIdx, pos, size);
			
			vec3 contrib = estimateDirectLightSingle(objIdx, mIdx, surfHit, albedo, state);
			totalContrib += contrib;
		}
	}
	
	// Усредняем по количеству реально сэмплированных источников
	// Если сэмплировали подмножество, масштабирование уже применено в строке 847
	if (lightCount > MAX_LIGHTS_TO_SAMPLE) {
		// Сэмплировали подмножество, уже масштабировано - делим на количество сэмплированных
		totalContrib /= float(lightsToSample);
	} else {
		// Сэмплировали все источники - делим на общее количество
		totalContrib /= float(lightCount);
	}
	
	return totalContrib;
}

// Фон / небо
vec3 backgroundColor(Ray r) {
    vec3 dir = normalize(r.dir);
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

    vec3 w = normalize(origin - target);
    vec3 u = normalize(cross(up, w));
    vec3 v = cross(w, u);

    float focusDist = camFocusDist != 0.0 ? camFocusDist : length(origin - target);
    vec3 horizontal = viewportWidth * focusDist * u;
    vec3 vertical   = viewportHeight * focusDist * v;
    vec3 lowerLeftCorner = origin - 0.5 * horizontal - 0.5 * vertical - w * focusDist;

    float lensRadius = camAperture * 0.5;
    if (lensRadius > 0.0) {
        vec3 rd = lensRadius * randomInUnitSphere(state);
        vec3 offset = u * rd.x + v * rd.y;
        vec3 dir = lowerLeftCorner + uv.x * horizontal + uv.y * vertical - origin - offset;
        r.orig = origin + offset;
        r.dir = normalize(dir);
    } else {
        vec3 dir = lowerLeftCorner + uv.x * horizontal + uv.y * vertical - origin;
        r.orig = origin;
        r.dir = normalize(dir);
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
    return (1.0 - gg) / (4.0 * PI * denom * sqrt(max(denom, 1e-6)));
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

// Фрактальный 3D‑шум для неоднородного тумана.
float volumeNoise(vec3 p) {
    float amp = 1.0;
    float freq = fogNoiseScale;
    float sum = 0.0;
    float norm = 0.0;
    int oct = int(clamp(fogNoiseOctaves, 1.0, 5.0));
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

// Оценка освещения тумана в точке pos (single scattering) от эмиссивных источников.
// Проходимся по ВСЕМ эмиссивным объектам (особенно сферам-лампам), чтобы усилить
// видимость лучей. Это дороже, но даёт гораздо более заметный объёмный свет.
vec3 estimateVolumeLight(vec3 pos, vec3 viewDir, inout uint state) {
    if (fogScatter <= 0.0) {
        return vec3(0.0);
    }

    vec3 sum = vec3(0.0);
    int objCount = int(objData.length()) / OBJ_STRIDE;

    // Используем оптимизированный список источников света
    int lightCount = getLightCount();
    for (int i = 0; i < lightCount; i++) {
        int objIdx = getLightObjectIndex(i);
        if (objIdx < 0) {
            continue;
        }
        
        int oTyp, mIdx;
        vec3 oPos, oSize;
        readObject(objIdx, oTyp, mIdx, oPos, oSize);

        int mTyp;
        float mRough, mIor, mSmoothness, mReflectivity, mAbsScale;
        vec3 mAlbedo, mEmit, mAbs, mTint;
        readMaterial(mIdx, mTyp, mRough, mIor, mAlbedo, mEmit, mAbs, mSmoothness, mReflectivity, mTint, mAbsScale);
        if (mTyp != MAT_EMISSIVE || (mEmit.r <= 0.0 && mEmit.g <= 0.0 && mEmit.b <= 0.0)) {
            continue;
        }

        vec3 lightPos, lightNormal;
        float pdfArea;
        if (!sampleLightGeometry(objIdx, lightPos, lightNormal, pdfArea, state)) {
            continue;
        }
        if (pdfArea <= 0.0) {
            continue;
        }

        vec3 toLight = lightPos - pos;
        float distSq = dot(toLight, toLight);
        if (distSq <= 1e-6) {
            continue;
        }
        float dist = sqrt(distSq);
        vec3 wi = toLight / dist;

        // Теневой луч до источника (без учёта тумана по пути — только геометрия).
        Ray shadowRay;
        shadowRay.orig = pos;
        shadowRay.dir = wi;
        Hit shadowHit;
        if (hitWorld(shadowRay, 0.001, dist - 0.002, shadowHit)) {
            continue;
        }

        // Косинусы: со стороны источника и для фазовой функции.
        float cosLight = max(0.0, dot(lightNormal, -wi));
        if (cosLight <= 0.0) {
            continue;
        }

        // Угол между направлением на источник и направлением к камере.
        float cosTheta = dot(-wi, viewDir);
        float phase = phaseHG(cosTheta, fogG);

        // Геометрический термин (источник считаем площадочным).
        float geometry = cosLight / max(1e-6, distSq);
        vec3 Le = mEmit; // интенсивность источника

        // Переход от pdf по площади к направлению.
        vec3 contrib = Le * geometry * phase / max(1e-6, pdfArea);

        sum += contrib;
    }

    // Немного усиливаем объёмный свет, чтобы лучи были явно видны.
    vec3 result = sum * 2.0;
    
    // Улучшенный firefly reduction для объёмного света
    float luminance = dot(result, vec3(0.2126, 0.7152, 0.0722));
    float maxLuminance = 500.0; // максимальная допустимая яркость для объёмного света
    if (luminance > maxLuminance) {
        float scale = maxLuminance / max(luminance, 1e-6);
        result *= scale;
    }
    
    return result;
}

// Основной path tracing для одного луча (исправленная версия с правильными отражениями)
vec3 rayColor(Ray r, inout uint state) {
    vec3 throughput = vec3(1.0);
    vec3 radiance = vec3(0.0);

    int depth = uMaxDepth;
    bool firstSegment = true;
    float firstHitT = 0.0;
    int currentGlassObject = -1; // Отслеживаем текущий стеклянный объект
    float accumulatedTravelDistance = 0.0; // Суммарное расстояние внутри стекла

    // Объёмное рассеяние (single scattering) вдоль первичного луча
    if (fogGpuVolumetric > 0.5 && depth > 0) {
        Ray primary = r;
        Hit hFirst;
        float tMax = 40.0;
        if (hitWorld(primary, 0.001, tMax, hFirst)) {
            tMax = hFirst.t;
        }

        int steps = 24;
        float step = tMax / float(steps);
        if (step > 0.0) {
            for (int i = 0; i < steps; i++) {
                float t = (float(i) + 0.5) * step;
                vec3 pos = rayAt(primary, t);

                float sigma_s, sigma_a, sigma_t;
                mediumCoeffs(pos, sigma_s, sigma_a, sigma_t);
                if (sigma_t <= 0.0 || sigma_s <= 0.0) {
                    continue;
                }

                float Tr = exp(-sigma_t * t);
                vec3 Ls = estimateVolumeLight(pos, primary.dir, state);
                vec3 dL = fogColor.rgb * Ls * sigma_s * Tr * step;
                radiance += dL;
            }
        }
    }

    while (depth > 0) {
        Hit h;
        
        // Исключаем текущий стеклянный объект при поиске пересечений
        bool hitFound = false;
        float closest = 1e20;
        Hit temp;
        int objCount = int(objData.length()) / OBJ_STRIDE;
        
        for (int i = 0; i < objCount; i++) {
            // Пропускаем текущий стеклянный объект, если мы внутри него
            if (currentGlassObject >= 0 && i == currentGlassObject) {
                continue;
            }
            
            int typ, matIdx;
            vec3 pos, size;
            readObject(i, typ, matIdx, pos, size);
            temp.matIndex = matIdx;
            temp.objIndex = i;
            
            bool hitObj = false;
            if (typ == OBJ_SPHERE) {
                hitObj = hitSphere(pos, size.x, r, 0.001, closest, temp);
            } else if (typ == OBJ_PLANE) {
                hitObj = hitPlane(pos, vec3(0, 1, 0), r, 0.001, closest, temp);
            } else if (typ == OBJ_BOX) {
                vec3 bmin = pos - 0.5 * size;
                vec3 bmax = pos + 0.5 * size;
                
                // Если мы внутри куба, ищем выход
                if (currentGlassObject == i) {
                    hitObj = hitBox(bmin, bmax, r, 0.001, closest, temp, true);
                } else {
                    hitObj = hitBox(bmin, bmax, r, 0.001, closest, temp, false);
                }
            }
            
            if (hitObj) {
                hitFound = true;
                closest = temp.t;
                h = temp;
            }
        }
        
        if (!hitFound) {
            vec3 bg = backgroundColor(r);
            float missDist = 50.0;
            vec3 outBg = bg;
            if (fogDensity > 0.0 && fogAffectSky > 0.5) {
                outBg = applyFog(bg, missDist);
            }
            radiance += throughput * outBg;
            break;
        }

        int typ;
        float rough, ior, smoothness, reflectivity, absorptionScale;
        vec3 albedo, emit, absorption, tint;
        readMaterial(h.matIndex, typ, rough, ior, albedo, emit, absorption, smoothness, reflectivity, tint, absorptionScale);

        if (firstSegment) {
            firstSegment = false;
            firstHitT = h.t;
        }

        vec3 emitted = (typ == MAT_EMISSIVE) ? emit : vec3(0.0);
        radiance += throughput * emitted;

        // Случайные рассеяния по типу материала
        vec3 newDir;
        vec3 attenuation = albedo;
        bool scattered = true;

        // Диффузные материалы
        if (typ == MAT_LAMBERT) {
            newDir = randomCosineDirection(h.normal, state);
            attenuation = albedo;
            
            // Прямое освещение для диффузных материалов
            vec3 direct = estimateDirectLight(r, h, albedo, state);
            radiance += throughput * direct;
            
        } else if (typ == MAT_METAL || typ == MAT_MIRROR) {
            vec3 viewDir = normalize(r.dir);
            float metalRough = rough;
            if (smoothness > 0.0) {
                metalRough = 1.0 - smoothness;
            }
            
            // Используем reflectivity для металлов
            float effectiveReflectivity = reflectivity > 0.0 ? reflectivity : 1.0;
            
            if (typ == MAT_METAL && metalRough > 1e-4) {
                // Используем GGX importance sampling для шероховатых металлов
                newDir = sampleGGX(viewDir, h.normal, metalRough, state);
                
                // Для шероховатых металлов учитываем, что не весь свет отражается
                // Используем комбинацию specular и диффузной компоненты
                float specularWeight = 1.0 / (1.0 + metalRough * metalRough * 2.0);
                specularWeight = clamp(specularWeight, 0.1, 0.9);
                float diffuseWeight = 1.0 - specularWeight;
                
                // Добавляем вклад от диффузной компоненты (для шероховатых металлов)
                vec3 diffuseDirect = estimateDirectLight(r, h, albedo, state);
                radiance += throughput * diffuseDirect * diffuseWeight * effectiveReflectivity * 0.5;
                
                // Корректируем attenuation для учета reflectivity и шероховатости
                attenuation = albedo * (specularWeight * effectiveReflectivity + diffuseWeight * 0.3);
            } else {
                // Идеальное зеркальное отражение для зеркал и гладких металлов
                newDir = reflectVec(viewDir, h.normal);
                newDir = normalize(newDir);
                attenuation = albedo * effectiveReflectivity;
            }
            
            float dotNorm = dot(newDir, h.normal);
            if (dotNorm <= 1e-6) {
                scattered = false;
            }
            
            // Для металлов также добавляем прямое освещение через отражения
            if (scattered && metalRough > 1e-4) {
                // Сэмплируем направление отражения для прямого освещения
                vec3 reflectDir = reflectVec(normalize(r.dir), h.normal);
                Ray reflectRay;
                reflectRay.orig = h.p + h.normal * 0.001;
                reflectRay.dir = reflectDir;
                
                Hit reflectHit;
                if (hitWorld(reflectRay, 0.001, 1e20, reflectHit)) {
                    int rTyp;
                    float rRough, rIor, rSmoothness, rReflectivity, rAbsScale;
                    vec3 rAlbedo, rEmit, rAbsorption, rTint;
                    readMaterial(reflectHit.matIndex, rTyp, rRough, rIor, rAlbedo, rEmit, rAbsorption, rSmoothness, rReflectivity, rTint, rAbsScale);
                    
                    // Если отражаемся от эмиссивного объекта
                    if (rTyp == MAT_EMISSIVE) {
                        vec3 directReflect = rEmit * max(0.0, dot(reflectHit.normal, -reflectDir)) / (reflectHit.t * reflectHit.t);
                        radiance += throughput * directReflect * albedo * 0.5;
                    }
                }
            }
            
        } else if (typ == MAT_DIELECTRIC) {
            attenuation = vec3(1.0);
            vec3 unitDir = normalize(r.dir);
            float cosTheta = min(dot(-unitDir, h.normal), 1.0);
            float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
            
            // Определяем, входим мы в объект или выходим
            bool entering = h.frontFace;
            float eta = entering ? (1.0 / ior) : ior;
            
            // ПРАВИЛЬНЫЙ РАСЧЕТ ВЕРОЯТНОСТИ ОТРАЖЕНИЯ ПО ФРЕНЕЛЮ
            // Функция reflectance ожидает относительный индекс преломления n2/n1
            // где n1 - среда откуда идет луч, n2 - среда куда идет луч
            float relIOR = entering ? ior : (1.0 / ior); // n2/n1
            
            // Проверяем полное внутреннее отражение
            if (eta * sinTheta > 1.0) {
                newDir = reflectVec(unitDir, h.normal);
            } else {
                // ПРАВИЛЬНО: вероятность отражения зависит от угла и относительного IOR
                float reflectProb = reflectance(cosTheta, relIOR);
                
                // При переходе из более плотной в менее плотную среду вероятность отражения выше
                // Это физически корректно: полное внутреннее отражение происходит только при переходе из плотной в менее плотную
                if (!entering) {
                    // При выходе из стекла вероятность отражения выше для больших углов
                    reflectProb = max(reflectProb, 0.05); // Минимальная вероятность отражения
                }
                
                if (rng(state) < reflectProb) {
                    newDir = reflectVec(unitDir, h.normal);
                } else {
                    newDir = refractVec(unitDir, h.normal, eta);
                    
                    // Применяем поглощение света внутри стекла
                    float travelDistance = 0.0;
                    
                    // Для куба вычисляем расстояние до выхода
                    if (entering) {
                        // Запоминаем, что мы вошли в стеклянный объект
                        currentGlassObject = h.objIndex;
                        
                        // Находим точку выхода
                        Hit exitHit;
                        int oTyp, mIdx;
                        vec3 oPos, oSize;
                        readObject(h.objIndex, oTyp, mIdx, oPos, oSize);
                        
                        if (oTyp == OBJ_BOX) {
                            vec3 bmin = oPos - 0.5 * oSize;
                            vec3 bmax = oPos + 0.5 * oSize;
                            
                            // Ищем выход из куба
                            Ray exitRay;
                            exitRay.orig = h.p + newDir * 0.001;
                            exitRay.dir = newDir;
                            if (hitBox(bmin, bmax, exitRay, 0.001, 1e20, exitHit, true)) {
                                travelDistance = exitHit.t;
                            }
                        } else if (oTyp == OBJ_SPHERE) {
                            // Для сферы вычисляем расстояние до противоположной стороны
                            float radius = oSize.x;
                            vec3 center = oPos;
                            
                            // Находим второе пересечение со сферой
                            vec3 oc = (h.p + newDir * 0.001) - center;
                            float a = dot(newDir, newDir);
                            float halfB = dot(oc, newDir);
                            float c = dot(oc, oc) - radius * radius;
                            float disc = halfB * halfB - a * c;
                            
                            if (disc > 0) {
                                float sqrtD = sqrt(disc);
                                float root1 = (-halfB - sqrtD) / a;
                                float root2 = (-halfB + sqrtD) / a;
                                
                                // Используем второй корень как точку выхода
                                float exitT = max(root1, root2);
                                if (exitT > 0.001) {
                                    travelDistance = exitT;
                                }
                            }
                        }
                        
                        // Применяем поглощение на основе пройденного расстояния
                        // Формула Beer-Lambert: I = I0 * exp(-sigma * d)
                        // где sigma = absorption * absorptionScale, d = travelDistance (в см)
                        if (travelDistance > 0.0) {
                            accumulatedTravelDistance = travelDistance;
                            vec3 effectiveAbsorption = absorption * absorptionScale;
                            vec3 absorb = exp(-effectiveAbsorption * travelDistance);
                            attenuation *= mix(vec3(1.0), absorb, 0.9);
                            
                            // Также применяем tint для цвета стекла
                            if (tint.r > 0.0 || tint.g > 0.0 || tint.b > 0.0) {
                                attenuation *= tint;
                            }
                        }
                    } else {
                        // Выходим из стеклянного объекта
                        currentGlassObject = -1;
                        
                        // Применяем поглощение для выходящего луча
                        // Формула Beer-Lambert: I = I0 * exp(-sigma * d)
                        // где sigma = absorption * absorptionScale, d = accumulatedTravelDistance (в см)
                        if (accumulatedTravelDistance > 0.0) {
                            vec3 effectiveAbsorption = absorption * absorptionScale;
                            vec3 absorb = exp(-effectiveAbsorption * accumulatedTravelDistance);
                            attenuation *= mix(vec3(1.0), absorb, 0.9);
                            
                            // Также применяем tint для цвета стекла
                            if (tint.r > 0.0 || tint.g > 0.0 || tint.b > 0.0) {
                                attenuation *= tint;
                            }
                        }
                        accumulatedTravelDistance = 0.0;
                    }
                }
            }
            newDir = normalize(newDir);
            
        } else {
            scattered = false;
        }

        // Russian roulette
        const int rrThreshold = 3;
        if (depth <= rrThreshold) {
            float maxComp = max(attenuation.r, max(attenuation.g, attenuation.b));
            if (maxComp < 1e-6) {
                break;
            }
            float rrProb = min(maxComp, 0.95);
            if (rng(state) > rrProb) {
                break;
            }
            attenuation /= rrProb;
        }

        throughput *= attenuation;
        
        // Обновляем луч
        r.orig = h.p + h.normal * 0.001; // Сдвигаем точку, чтобы избежать самопересечения
        r.dir = newDir;
        depth--;
    }

    return radiance;
}

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    if (pix.x >= uWidth || pix.y >= uHeight) {
        return;
    }

    uint state = hash_u(uint(pix.x) * 1973u ^ uint(pix.y) * 9277u ^ uFrameSeed);

    vec3 col = vec3(0.0);
    
    // Stratified sampling: используем NxN сетку для первого сэмпла
    // Вычисляем размер страты (например, 4x4 = 16 страт)
    int strataSize = 4; // можно сделать настраиваемым
    int totalStrata = strataSize * strataSize;
    int samplesPerStratum = max(1, uSamplesPerPx / totalStrata);
    int extraSamples = uSamplesPerPx - samplesPerStratum * totalStrata;
    
    int sampleIdx = 0;
    
    // Проходим по всем стратам
    for (int sy = 0; sy < strataSize; sy++) {
        for (int sx = 0; sx < strataSize; sx++) {
            int samplesInStratum = samplesPerStratum;
            if (sy * strataSize + sx < extraSamples) {
                samplesInStratum++;
            }
            
            for (int s = 0; s < samplesInStratum; s++) {
                // Случайная точка внутри страты
                float jitterX = rng(state);
                float jitterY = rng(state);
                
                // Координаты страты в [0,1]
                float stratumU = (float(sx) + jitterX) / float(strataSize);
                float stratumV = (float(sy) + jitterY) / float(strataSize);
                
                // Преобразуем в координаты пикселя
                float u = (float(pix.x) + stratumU) / float(uWidth - 1);
                float fy = float(uHeight - 1 - pix.y);
                float v = (fy + stratumV) / float(uHeight - 1);
                
                Ray r;
                buildCamera(vec2(u, v), r, state);
                col += rayColor(r, state);
                sampleIdx++;
            }
        }
    }
    
    // Добавляем оставшиеся сэмплы (если uSamplesPerPx не кратно totalStrata)
    for (int s = sampleIdx; s < uSamplesPerPx; s++) {
        float u = (float(pix.x) + rng(state)) / float(uWidth - 1);
        float fy = float(uHeight - 1 - pix.y);
        float v = (fy + rng(state)) / float(uHeight - 1);
        Ray r;
        buildCamera(vec2(u, v), r, state);
        col += rayColor(r, state);
    }
    
    col /= float(uSamplesPerPx);

    // Накопление на GPU: добавляем текущий сэмпл к накопительному буферу
    int pixIdx = pix.y * uWidth + pix.x;
    int accumBase = pixIdx * 3;
    accumData[accumBase + 0] += col.r;
    accumData[accumBase + 1] += col.g;
    accumData[accumBase + 2] += col.b;
    
    // Читаем накопленное значение и усредняем по количеству сэмплов
    float sampleCount = float(max(1, uSampleCount));
    vec3 accumCol = vec3(
        accumData[accumBase + 0],
        accumData[accumBase + 1],
        accumData[accumBase + 2]
    ) / sampleCount;
    
    // Пишем усредненный ЛИНЕЙНЫЙ цвет (0..1) без гамма-коррекции.
    vec3 finalCol = max(accumCol, vec3(0.0));
    imageStore(destTex, pix, vec4(finalCol, 1.0));
}
`
		cs, err := compileShader(computeSrc, gl.COMPUTE_SHADER)
		if err != nil {
			r.initErr = fmt.Errorf("compile compute shader: %w", err)
			return
		}
		r.program = gl.CreateProgram()
		gl.AttachShader(r.program, cs)
		gl.LinkProgram(r.program)

		var status int32
		gl.GetProgramiv(r.program, gl.LINK_STATUS, &status)
		if status == gl.FALSE {
			var logLen int32
			gl.GetProgramiv(r.program, gl.INFO_LOG_LENGTH, &logLen)
			log := make([]byte, logLen+1)
			gl.GetProgramInfoLog(r.program, logLen, nil, &log[0])
			r.initErr = fmt.Errorf("link compute program: %s", string(log))
			return
		}
	})

	return r.initErr
}

func compileShader(src string, shaderType uint32) (uint32, error) {
	shader := gl.CreateShader(shaderType)
	csources, free := gl.Strs(src + "\x00")
	defer free()
	gl.ShaderSource(shader, 1, csources, nil)
	gl.CompileShader(shader)

	var status int32
	gl.GetShaderiv(shader, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		var logLen int32
		gl.GetShaderiv(shader, gl.INFO_LOG_LENGTH, &logLen)
		log := make([]byte, logLen+1)
		gl.GetShaderInfoLog(shader, logLen, nil, &log[0])
		return 0, fmt.Errorf("shader compile: %s", string(log))
	}
	return shader, nil
}

// renderOnce executes a very simple GPU render into img using the already
// initialized GL context owned by the worker goroutine.
func (r *gpuRenderer) renderOnce(sc *scene.Scene, cfg RenderConfig, img *image.RGBA, progress func()) error {
	// --- Инициализация накопительного буфера ---
	pixelCount := cfg.Width * cfg.Height
	if pixelCount <= 0 {
		return nil
	}
	if len(r.accum) != pixelCount*3 {
		r.accum = make([]float32, pixelCount*3)
	} else {
		for i := range r.accum {
			r.accum[i] = 0
		}
	}

	// --- Пакуем материалы ---
	matCount := len(sc.Materials)
	if matCount == 0 {
		// Без материалов смысла рендерить нет - заполняем чёрным.
		for i := range img.Pix {
			if i%4 == 3 {
				img.Pix[i] = 255
			} else {
				img.Pix[i] = 0
			}
		}
		if progress != nil {
			progress()
		}
		return nil
	}
	matStride := 20
	matData := make([]float32, matCount*matStride)
	for i, m := range sc.Materials {
		base := i * matStride

		var typ float32
		switch m.Type {
		case scene.MaterialLambert:
			typ = float32(MAT_LAMBERT)
		case scene.MaterialMetal:
			typ = float32(MAT_METAL)
		case scene.MaterialDielectric:
			typ = float32(MAT_DIELECTRIC)
		case scene.MaterialEmissive:
			typ = float32(MAT_EMISSIVE)
		case scene.MaterialMirror:
			typ = float32(MAT_MIRROR)
		default:
			typ = float32(MAT_LAMBERT)
		}

		matData[base+0] = typ
		matData[base+1] = float32(m.Rough)
		matData[base+2] = float32(m.IOR)

		// Smoothness для металлов: если не задан, вычисляем из Rough (обратная совместимость)
		smoothness := m.Smoothness
		if smoothness == 0 && m.Type == scene.MaterialMetal {
			// Если smoothness не задан, вычисляем из rough для обратной совместимости
			smoothness = 1.0 - m.Rough
		}
		if smoothness < 0 {
			smoothness = 0
		}
		if smoothness > 1 {
			smoothness = 1
		}
		matData[base+3] = float32(smoothness)

		matData[base+4] = float32(m.Albedo.R)
		matData[base+5] = float32(m.Albedo.G)
		matData[base+6] = float32(m.Albedo.B)

		// Reflectivity для металлов: если не задан, используем 1.0
		reflectivity := m.Reflectivity
		if reflectivity == 0 && m.Type == scene.MaterialMetal {
			reflectivity = 1.0
		}
		if reflectivity < 0 {
			reflectivity = 0
		}
		if reflectivity > 1 {
			reflectivity = 1
		}
		matData[base+7] = float32(reflectivity)

		// emit * power
		matData[base+8] = float32(m.Emit.R * m.Power)
		matData[base+9] = float32(m.Emit.G * m.Power)
		matData[base+10] = float32(m.Emit.B * m.Power)

		matData[base+12] = float32(m.Absorption.R)
		matData[base+13] = float32(m.Absorption.G)
		matData[base+14] = float32(m.Absorption.B)

		// AbsorptionScale для диэлектриков: если не задан, используем 0.01 (разумное значение для см)
		absorptionScale := m.AbsorptionScale
		if absorptionScale == 0 && m.Type == scene.MaterialDielectric {
			absorptionScale = 0.01 // По умолчанию 0.01 для см
		}
		matData[base+15] = float32(absorptionScale)

		// Tint для диэлектриков: если не задан, используем белый (1,1,1)
		tintR := m.Tint.R
		tintG := m.Tint.G
		tintB := m.Tint.B
		if tintR == 0 && tintG == 0 && tintB == 0 && m.Type == scene.MaterialDielectric {
			tintR = 1.0
			tintG = 1.0
			tintB = 1.0
		}
		matData[base+16] = float32(tintR)
		matData[base+17] = float32(tintG)
		matData[base+18] = float32(tintB)
	}

	// --- Пакуем объекты ---
	objCount := len(sc.Objects)
	objStride := 12
	objData := make([]float32, objCount*objStride)
	// карта materialID -> индекс материала
	matIndex := make(map[string]int, matCount)
	for i, m := range sc.Materials {
		matIndex[m.ID] = i
	}

	// Предвычисляем список индексов эмиссивных объектов для оптимизации
	lightIndices := make([]int32, 0, objCount)

	for i, o := range sc.Objects {
		base := i * objStride

		var typ float32
		switch o.Type {
		case scene.ObjectSphere, scene.ObjectSphereLight:
			typ = float32(OBJ_SPHERE)
		case scene.ObjectPlane:
			typ = float32(OBJ_PLANE)
		case scene.ObjectBox:
			typ = float32(OBJ_BOX)
		default:
			typ = float32(OBJ_SPHERE)
		}

		objData[base+0] = typ
		var matIdx int
		if idx, ok := matIndex[o.MaterialID]; ok {
			objData[base+1] = float32(idx)
			matIdx = idx
		} else {
			objData[base+1] = 0
			matIdx = 0
		}

		objData[base+4] = float32(o.Position.X)
		objData[base+5] = float32(o.Position.Y)
		objData[base+6] = float32(o.Position.Z)

		objData[base+8] = float32(o.Size.X)
		objData[base+9] = float32(o.Size.Y)
		objData[base+10] = float32(o.Size.Z)

		// Проверяем, является ли объект источником света
		if matIdx < len(sc.Materials) {
			m := sc.Materials[matIdx]
			if m.Type == scene.MaterialEmissive && (m.Emit.R > 0 || m.Emit.G > 0 || m.Emit.B > 0) {
				lightIndices = append(lightIndices, int32(i))
			}
		}
	}

	// --- Пакуем камеру ---
	cam := sc.Camera
	aspect := float32(cfg.Width) / float32(cfg.Height)
	if cam.AspectRatio != 0 {
		aspect = float32(cam.AspectRatio)
	}

	camBlock := [16]float32{
		float32(cam.Position.X), float32(cam.Position.Y), float32(cam.Position.Z), 0,
		float32(cam.Target.X), float32(cam.Target.Y), float32(cam.Target.Z), 0,
		float32(cam.Up.X), float32(cam.Up.Y), float32(cam.Up.Z), 0,
		float32(cam.FOV),
		float32(cam.Aperture),
		float32(cam.FocusDist),
		aspect,
	}

	// --- Пакуем небо ---
	skyType := 0
	var skyColor, skyHorizon, skyZenith [4]float32
	if sc.Sky != nil {
		if sc.Sky.Type == "gradient" {
			skyType = 2
			skyHorizon = [4]float32{float32(sc.Sky.Horizon.R), float32(sc.Sky.Horizon.G), float32(sc.Sky.Horizon.B), 1}
			skyZenith = [4]float32{float32(sc.Sky.Zenith.R), float32(sc.Sky.Zenith.G), float32(sc.Sky.Zenith.B), 1}
		} else { // solid
			skyType = 1
			skyColor = [4]float32{float32(sc.Sky.Color.R), float32(sc.Sky.Color.G), float32(sc.Sky.Color.B), 1}
		}
	} else {
		skyType = 0
		skyColor = [4]float32{float32(sc.Background.R), float32(sc.Background.G), float32(sc.Background.B), 1}
	}

	skyBlock := [16]float32{
		float32(skyType), 0, 0, 0,
		skyColor[0], skyColor[1], skyColor[2], skyColor[3],
		skyHorizon[0], skyHorizon[1], skyHorizon[2], skyHorizon[3],
		skyZenith[0], skyZenith[1], skyZenith[2], skyZenith[3],
	}

	// --- Пакуем туман (fog) ---
	fogDensity := float32(0)
	fogScatter := float32(0)
	fogAffectSky := float32(0)
	fogColor := [4]float32{0, 0, 0, 0}
	fogSigmaS := float32(0)
	fogSigmaA := float32(0)
	fogG := float32(0)
	fogHeteroStrength := float32(0)
	fogNoiseScale := float32(4.0)
	fogNoiseOctaves := float32(3)
	fogGpuVolumetric := float32(0)

	if sc.Fog != nil {
		// Базовые параметры (обратная совместимость).
		if sc.Fog.Density > 0 {
			fogDensity = float32(sc.Fog.Density)
		}
		if sc.Fog.Scatter > 0 {
			fogScatter = float32(sc.Fog.Scatter)
		} else if fogDensity > 0 {
			fogScatter = 1.0
		}
		if sc.Fog.AffectSky {
			fogAffectSky = 1.0
		}
		fogColor = [4]float32{
			float32(sc.Fog.Color.R),
			float32(sc.Fog.Color.G),
			float32(sc.Fog.Color.B),
			1,
		}

		// Физическая модель: sigma_s / sigma_a.
		if sc.Fog.SigmaS > 0 || sc.Fog.SigmaA > 0 {
			fogSigmaS = float32(sc.Fog.SigmaS)
			fogSigmaA = float32(sc.Fog.SigmaA)
		} else if fogDensity > 0 {
			// Если явно не заданы, выводим из плотности и scatter.
			baseSigmaT := fogDensity
			sMul := fogScatter
			if sMul < 0 {
				sMul = 0
			}
			if sMul > 1 {
				sMul = 1
			}
			fogSigmaS = baseSigmaT * sMul
			fogSigmaA = baseSigmaT - fogSigmaS
			if fogSigmaA < 0 {
				fogSigmaA = 0
			}
		}

		// Анизотропия HG‑фазы.
		if sc.Fog.G < -0.9 {
			fogG = -0.9
		} else if sc.Fog.G > 0.9 {
			fogG = 0.9
		} else {
			fogG = float32(sc.Fog.G)
		}

		// Неоднородный туман (шум).
		if sc.Fog.HeteroStrength > 0 {
			if sc.Fog.HeteroStrength > 1 {
				fogHeteroStrength = 1
			} else {
				fogHeteroStrength = float32(sc.Fog.HeteroStrength)
			}
		}
		if sc.Fog.NoiseScale > 0 {
			fogNoiseScale = float32(sc.Fog.NoiseScale)
		}
		if sc.Fog.NoiseOctaves > 0 {
			if sc.Fog.NoiseOctaves > 5 {
				fogNoiseOctaves = 5
			} else {
				fogNoiseOctaves = float32(sc.Fog.NoiseOctaves)
			}
		}

		if sc.Fog.GPUVolumetric {
			fogGpuVolumetric = 1.0
		}
	}

	// std140: выравниваем до кратности vec4.
	fogBlock := [16]float32{
		fogDensity, fogScatter, fogAffectSky, fogGpuVolumetric,
		fogColor[0], fogColor[1], fogColor[2], fogColor[3],
		fogSigmaS, fogSigmaA, fogG, fogHeteroStrength,
		fogNoiseScale, fogNoiseOctaves, 0, 0,
	}

	// Resize texture if needed.
	if r.width != cfg.Width || r.height != cfg.Height {
		r.width = cfg.Width
		r.height = cfg.Height

		gl.BindTexture(gl.TEXTURE_2D, r.imgTexture)
		// HDR-подобный формат: RGBA16F, чтобы compute шейдер писал линейный цвет
		// с меньшим квантованием. Читаем как float32.
		gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, int32(cfg.Width), int32(cfg.Height), 0, gl.RGBA, gl.FLOAT, nil)

		gl.BindBuffer(gl.PIXEL_PACK_BUFFER, r.pbo)
		// 4 компоненты по 4 байта (float32) на пиксель
		gl.BufferData(gl.PIXEL_PACK_BUFFER, cfg.Width*cfg.Height*4*4, nil, gl.STREAM_READ)
		gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)

		// Инициализируем накопительный буфер на GPU
		accumSize := pixelCount * 3 * 4 // 3 компоненты (RGB) по 4 байта (float32)
		gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, r.accumSSBO)
		gl.BufferData(gl.SHADER_STORAGE_BUFFER, accumSize, nil, gl.DYNAMIC_DRAW)
		// Инициализируем нулями
		zeroData := make([]float32, pixelCount*3)
		gl.BufferSubData(gl.SHADER_STORAGE_BUFFER, 0, len(zeroData)*4, gl.Ptr(zeroData))
		gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, 0)
	}

	// Загружаем материалы/объекты/камеру/небо
	gl.BindBufferBase(gl.SHADER_STORAGE_BUFFER, 3, r.matSSBO)
	gl.BufferData(gl.SHADER_STORAGE_BUFFER, len(matData)*4, gl.Ptr(matData), gl.DYNAMIC_DRAW)

	gl.BindBufferBase(gl.SHADER_STORAGE_BUFFER, 4, r.objSSBO)
	if len(objData) > 0 {
		gl.BufferData(gl.SHADER_STORAGE_BUFFER, len(objData)*4, gl.Ptr(objData), gl.DYNAMIC_DRAW)
	} else {
		// хотя бы маленький буфер, чтобы avoid GL errors при нуле объектов
		tmp := []float32{0}
		gl.BufferData(gl.SHADER_STORAGE_BUFFER, len(tmp)*4, gl.Ptr(tmp), gl.DYNAMIC_DRAW)
	}

	// Загружаем список индексов эмиссивных объектов
	gl.BindBufferBase(gl.SHADER_STORAGE_BUFFER, 6, r.lightSSBO)
	if len(lightIndices) > 0 {
		gl.BufferData(gl.SHADER_STORAGE_BUFFER, len(lightIndices)*4, gl.Ptr(lightIndices), gl.DYNAMIC_DRAW)
	} else {
		// хотя бы маленький буфер, чтобы avoid GL errors при нуле источников света
		tmp := []int32{0}
		gl.BufferData(gl.SHADER_STORAGE_BUFFER, len(tmp)*4, gl.Ptr(tmp), gl.DYNAMIC_DRAW)
	}

	gl.BindBufferBase(gl.UNIFORM_BUFFER, 1, r.camUBO)
	gl.BufferData(gl.UNIFORM_BUFFER, len(camBlock)*4, gl.Ptr(camBlock[:]), gl.DYNAMIC_DRAW)

	gl.BindBufferBase(gl.UNIFORM_BUFFER, 2, r.skyUBO)
	gl.BufferData(gl.UNIFORM_BUFFER, len(skyBlock)*4, gl.Ptr(skyBlock[:]), gl.DYNAMIC_DRAW)

	// Fog UBO (binding = 5)
	gl.BindBufferBase(gl.UNIFORM_BUFFER, 5, r.fogUBO)
	gl.BufferData(gl.UNIFORM_BUFFER, len(fogBlock)*4, gl.Ptr(fogBlock[:]), gl.DYNAMIC_DRAW)

	gl.UseProgram(r.program)

	// Bind image for compute shader: формат должен совпадать с внутренним форматом текстуры (RGBA16F).
	gl.BindImageTexture(0, r.imgTexture, 0, false, 0, gl.WRITE_ONLY, gl.RGBA16F)

	// Bind накопительный буфер
	gl.BindBufferBase(gl.SHADER_STORAGE_BUFFER, 7, r.accumSSBO)

	// Set uniforms.
	locW := gl.GetUniformLocation(r.program, gl.Str("uWidth\x00"))
	locH := gl.GetUniformLocation(r.program, gl.Str("uHeight\x00"))
	locSpp := gl.GetUniformLocation(r.program, gl.Str("uSamplesPerPx\x00"))
	locDepth := gl.GetUniformLocation(r.program, gl.Str("uMaxDepth\x00"))
	locSeed := gl.GetUniformLocation(r.program, gl.Str("uFrameSeed\x00"))
	locSampleCount := gl.GetUniformLocation(r.program, gl.Str("uSampleCount\x00"))

	gl.Uniform1i(locW, int32(cfg.Width))
	gl.Uniform1i(locH, int32(cfg.Height))
	gl.Uniform1i(locDepth, int32(cfg.MaxDepth))

	// Инициализируем накопительный буфер нулями при начале нового рендера
	if r.width == cfg.Width && r.height == cfg.Height {
		// Размер не изменился - просто очищаем существующий буфер
		accumSize := pixelCount * 3 * 4
		gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, r.accumSSBO)
		// Проверяем, что буфер существует и правильного размера
		var bufSize int32
		gl.GetBufferParameteriv(gl.SHADER_STORAGE_BUFFER, gl.BUFFER_SIZE, &bufSize)
		if int(bufSize) != accumSize {
			// Пересоздаем буфер нужного размера
			gl.BufferData(gl.SHADER_STORAGE_BUFFER, accumSize, nil, gl.DYNAMIC_DRAW)
		}
		zeroData := make([]float32, pixelCount*3)
		gl.BufferSubData(gl.SHADER_STORAGE_BUFFER, 0, len(zeroData)*4, gl.Ptr(zeroData))
		gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, 0)
	}

	// Временный буфер для чтения данных с GPU (RGBA16F -> float32)
	tmp := make([]float32, pixelCount*4)

	// Сколько проходов выполнять (по одному сэмплу за проход)
	passes := cfg.SamplesPerPx
	if passes < 1 {
		passes = 1
	}
	updateEvery := passes / 10
	if updateEvery < 1 {
		updateEvery = 1
	}

	for s := 0; s < passes; s++ {
		// Один сэмпл на проход
		gl.Uniform1i(locSpp, 1)
		gl.Uniform1i(locSampleCount, int32(s+1)) // Количество накопленных сэмплов
		gl.Uniform1ui(locSeed, uint32(time.Now().UnixNano())+uint32(s))

		// Запуск compute shader
		groupsX := (cfg.Width + 15) / 16
		groupsY := (cfg.Height + 15) / 16
		gl.DispatchCompute(uint32(groupsX), uint32(groupsY), 1)

		gl.MemoryBarrier(gl.SHADER_IMAGE_ACCESS_BARRIER_BIT | gl.SHADER_STORAGE_BARRIER_BIT | gl.TEXTURE_FETCH_BARRIER_BIT)

		// Читаем результат только при обновлении UI (не на каждом проходе!)
		// Периодически обновляем img и вызываем progress()
		if (s%updateEvery) == updateEvery-1 || s == passes-1 {
			// Читаем накопленный результат с GPU
			gl.BindBuffer(gl.PIXEL_PACK_BUFFER, r.pbo)
			gl.BindTexture(gl.TEXTURE_2D, r.imgTexture)
			gl.GetTexImage(gl.TEXTURE_2D, 0, gl.RGBA, gl.FLOAT, nil)

			ptr := gl.MapBuffer(gl.PIXEL_PACK_BUFFER, gl.READ_ONLY)
			if ptr != nil {
				src := ((*[1 << 28]float32)(ptr))[:len(tmp)]
				copy(tmp, src)
				gl.UnmapBuffer(gl.PIXEL_PACK_BUFFER)
			}
			gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)

			// Обрабатываем данные из текстуры (уже усредненные на GPU)
			dst := img.Pix
			for i := 0; i < pixelCount; i++ {
				off := i * 4
				rLin := tmp[off]
				gLin := tmp[off+1]
				bLin := tmp[off+2]

				// Ограничиваем линейный цвет
				if rLin < 0 {
					rLin = 0
				}
				if gLin < 0 {
					gLin = 0
				}
				if bLin < 0 {
					bLin = 0
				}

				// ACES-подобный тон-маппинг из HDR в [0,1]
				rTm := acesTonemap(rLin)
				gTm := acesTonemap(gLin)
				bTm := acesTonemap(bLin)

				// Гамма-коррекция (переход в sRGB, gamma ~2.0 как в CPU-рендере)
				rGamma := float32(math.Sqrt(float64(rTm)))
				gGamma := float32(math.Sqrt(float64(gTm)))
				bGamma := float32(math.Sqrt(float64(bTm)))
				if rGamma > 1 {
					rGamma = 1
				}
				if gGamma > 1 {
					gGamma = 1
				}
				if bGamma > 1 {
					bGamma = 1
				}

				dst[off] = uint8(rGamma*255.0 + 0.5)
				dst[off+1] = uint8(gGamma*255.0 + 0.5)
				dst[off+2] = uint8(bGamma*255.0 + 0.5)
				dst[off+3] = 255
			}

			if progress != nil {
				progress()
			}

			// Денойзинг выполняется только в конце рендеринга для производительности
			// (пропускаем промежуточные обновления)
		}
	}

	// Финальная обработка: денойзинг и сглаживание только в конце
	// Читаем финальный результат с GPU
	gl.BindBuffer(gl.PIXEL_PACK_BUFFER, r.pbo)
	gl.BindTexture(gl.TEXTURE_2D, r.imgTexture)
	gl.GetTexImage(gl.TEXTURE_2D, 0, gl.RGBA, gl.FLOAT, nil)

	ptr := gl.MapBuffer(gl.PIXEL_PACK_BUFFER, gl.READ_ONLY)
	dst := img.Pix
	if ptr != nil {
		tmp := make([]float32, pixelCount*4)
		src := ((*[1 << 28]float32)(ptr))[:len(tmp)]
		copy(tmp, src)
		gl.UnmapBuffer(gl.PIXEL_PACK_BUFFER)

		// Обрабатываем финальные данные
		for i := 0; i < pixelCount; i++ {
			off := i * 4
			rLin := tmp[off]
			gLin := tmp[off+1]
			bLin := tmp[off+2]

			// Ограничиваем линейный цвет
			if rLin < 0 {
				rLin = 0
			}
			if gLin < 0 {
				gLin = 0
			}
			if bLin < 0 {
				bLin = 0
			}

			// ACES-подобный тон-маппинг из HDR в [0,1]
			rTm := acesTonemap(rLin)
			gTm := acesTonemap(gLin)
			bTm := acesTonemap(bLin)

			// Гамма-коррекция (переход в sRGB, gamma ~2.0 как в CPU-рендере)
			rGamma := float32(math.Sqrt(float64(rTm)))
			gGamma := float32(math.Sqrt(float64(gTm)))
			bGamma := float32(math.Sqrt(float64(bTm)))
			if rGamma > 1 {
				rGamma = 1
			}
			if gGamma > 1 {
				gGamma = 1
			}
			if bGamma > 1 {
				bGamma = 1
			}

			dst[off] = uint8(rGamma*255.0 + 0.5)
			dst[off+1] = uint8(gGamma*255.0 + 0.5)
			dst[off+2] = uint8(bGamma*255.0 + 0.5)
			dst[off+3] = 255
		}
	}
	gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)

	cfgDN := getDenoiseConfig()
	w := cfg.Width
	h := cfg.Height
	if cfgDN.Enabled && w > 2 && h > 2 {
		smoothed := make([]byte, len(dst))
		sigmaS := cfgDN.SigmaS
		sigmaR := cfgDN.SigmaR
		twoSigmaS2 := 2 * sigmaS * sigmaS
		twoSigmaR2 := 2 * sigmaR * sigmaR

		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				centerIdx := (y*w + x) * 4
				cr := float64(dst[centerIdx]) / 255.0
				cg := float64(dst[centerIdx+1]) / 255.0
				cb := float64(dst[centerIdx+2]) / 255.0

				var sumR, sumG, sumB, sumW float64
				for ky := -1; ky <= 1; ky++ {
					ny := y + ky
					if ny < 0 || ny >= h {
						continue
					}
					for kx := -1; kx <= 1; kx++ {
						nx := x + kx
						if nx < 0 || nx >= w {
							continue
						}
						ni := (ny*w + nx) * 4
						nr := float64(dst[ni]) / 255.0
						ng := float64(dst[ni+1]) / 255.0
						nb := float64(dst[ni+2]) / 255.0

						ds2 := float64(kx*kx + ky*ky)
						// расстояние в цветовом пространстве (sRGB 0..1)
						dr := cr - nr
						dg := cg - ng
						dbb := cb - nb
						dr2 := dr*dr + dg*dg + dbb*dbb

						ws := math.Exp(-ds2 / twoSigmaS2)
						wr := math.Exp(-dr2 / twoSigmaR2)
						wgt := ws * wr

						sumW += wgt
						sumR += nr * wgt
						sumG += ng * wgt
						sumB += nb * wgt
					}
				}

				if sumW > 0 {
					nr := sumR / sumW
					ng := sumG / sumW
					nb := sumB / sumW
					if nr < 0 {
						nr = 0
					} else if nr > 1 {
						nr = 1
					}
					if ng < 0 {
						ng = 0
					} else if ng > 1 {
						ng = 1
					}
					if nb < 0 {
						nb = 0
					} else if nb > 1 {
						nb = 1
					}
					smoothed[centerIdx] = uint8(nr*255.0 + 0.5)
					smoothed[centerIdx+1] = uint8(ng*255.0 + 0.5)
					smoothed[centerIdx+2] = uint8(nb*255.0 + 0.5)
					smoothed[centerIdx+3] = 255
				} else {
					// fallback: копируем исходный пиксель
					smoothed[centerIdx] = dst[centerIdx]
					smoothed[centerIdx+1] = dst[centerIdx+1]
					smoothed[centerIdx+2] = dst[centerIdx+2]
					smoothed[centerIdx+3] = dst[centerIdx+3]
				}
			}
		}
		copy(dst, smoothed)
	}

	// Дополнительное, более сильное сглаживание (blur) по всему изображению.
	// Используется простое box-усреднение с радиусом 1..5, затем смешивание
	// с оригиналом по коэффициенту strength (0..1).
	cfgSM := getSmoothConfig()
	if cfgSM.Enabled && w > 2 && h > 2 && cfgSM.Radius > 0 && cfgSM.Strength > 0 {
		rad := cfgSM.Radius
		if rad < 1 {
			rad = 1
		}
		if rad > 5 {
			rad = 5
		}
		str := cfgSM.Strength
		if str < 0 {
			str = 0
		}
		if str > 1 {
			str = 1
		}

		blurred := make([]byte, len(dst))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				var sumR, sumG, sumB float64
				var count float64
				for ky := -rad; ky <= rad; ky++ {
					ny := y + ky
					if ny < 0 || ny >= h {
						continue
					}
					for kx := -rad; kx <= rad; kx++ {
						nx := x + kx
						if nx < 0 || nx >= w {
							continue
						}
						ni := (ny*w + nx) * 4
						sumR += float64(dst[ni])
						sumG += float64(dst[ni+1])
						sumB += float64(dst[ni+2])
						count++
					}
				}
				if count > 0 {
					avgR := sumR / count
					avgG := sumG / count
					avgB := sumB / count
					ci := (y*w + x) * 4
					origR := float64(dst[ci])
					origG := float64(dst[ci+1])
					origB := float64(dst[ci+2])

					outR := (1-str)*origR + str*avgR
					outG := (1-str)*origG + str*avgG
					outB := (1-str)*origB + str*avgB

					if outR < 0 {
						outR = 0
					} else if outR > 255 {
						outR = 255
					}
					if outG < 0 {
						outG = 0
					} else if outG > 255 {
						outG = 255
					}
					if outB < 0 {
						outB = 0
					} else if outB > 255 {
						outB = 255
					}

					blurred[ci] = byte(outR + 0.5)
					blurred[ci+1] = byte(outG + 0.5)
					blurred[ci+2] = byte(outB + 0.5)
					blurred[ci+3] = 255
				}
			}
		}
		copy(dst, blurred)
	}

	// Финальное обновление UI после всех обработок
	if progress != nil {
		progress()
	}

	return nil
}

// Render is the public entry point that schedules a GPU render on the
// dedicated GL worker and waits for completion.
// For now it just draws a test gradient using a compute shader to verify GPU path.
// Later this can be replaced with a full path tracer that uses sc and cfg.
func Render(sc *scene.Scene, cfg RenderConfig, img *image.RGBA, progress func()) error {
	ensureWorker()
	done := make(chan error, 1)
	req := renderRequest{
		sc:       sc,
		cfg:      cfg,
		img:      img,
		progress: progress,
		done:     done,
	}
	renderCh <- req
	return <-done
}

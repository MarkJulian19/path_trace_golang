package scene

import (
	"image"
	"image/color"
)

// Vec3 represents a simple 3D vector or point.
type Vec3 struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	Z float64 `json:"z"`
}

// Color is an RGB color in linear space.
type Color struct {
	R float64 `json:"r"`
	G float64 `json:"g"`
	B float64 `json:"b"`
}

// Camera describes the viewpoint for the renderer.
type Camera struct {
	Position Vec3   `json:"position"`
	Target   Vec3   `json:"target"`
	Up       Vec3   `json:"up"`
	FOV      float64`json:"fov"`

	Aperture   float64 `json:"aperture"`
	FocusDist  float64 `json:"focus_dist"`
	AspectRatio float64`json:"aspect_ratio"`
}

// MaterialType enumerates supported material kinds.
type MaterialType string

const (
	MaterialLambert   MaterialType = "lambert"
	MaterialMetal     MaterialType = "metal"
	MaterialDielectric MaterialType = "dielectric"
	MaterialEmissive  MaterialType = "emissive"
	MaterialMirror    MaterialType = "mirror"
)

// Material describes surface properties.
type Material struct {
	ID   string       `json:"id"`
	Type MaterialType `json:"type"`

	Albedo Color   `json:"albedo"`
	Rough  float64 `json:"rough"` // for metal and lambert (roughness), deprecated for metals (use Smoothness)
	IOR    float64 `json:"ior"`   // for dielectric

	Emit Color   `json:"emit"`  // emissive color
	Power float64`json:"power"` // emissive intensity multiplier
	
	// Absorption for dielectric materials (color tinting through glass)
	// Higher values mean more absorption (darker color)
	Absorption Color `json:"absorption"` // for dielectric (default: 0,0,0 = no absorption)
	
	// Advanced parameters for metals
	Smoothness  float64 `json:"smoothness"`  // for metals: 0.0 = matte, 1.0 = perfect mirror (default: 1.0)
	Reflectivity float64 `json:"reflectivity"` // for metals: 0.0-1.0, controls reflection intensity (default: 1.0)
	
	// Advanced parameters for dielectrics (glass)
	Tint Color `json:"tint"` // for dielectric: color tint for light passing through glass (default: 1,1,1 = no tint)
	AbsorptionScale float64 `json:"absorption_scale"` // for dielectric: scale factor for absorption coefficient (default: 0.01, units are in cm)
}

// ObjectType enumerates supported geometric primitives.
type ObjectType string

const (
	ObjectSphere     ObjectType = "sphere"
	ObjectPlane      ObjectType = "plane"
	ObjectBox        ObjectType = "box"
	ObjectSphereLight ObjectType = "sphere_light"
)

// Object is a single entity in the scene.
type Object struct {
	ID string     `json:"id"`
	Type ObjectType `json:"type"`

	Position Vec3 `json:"position"`
	Size     Vec3 `json:"size"` // radius for sphere: use X, for planes/boxes: extents

	MaterialID string `json:"material_id"`
}

// RenderSettings defines quality/performance parameters.
type RenderSettings struct {
	Width        int `json:"width"`
	Height       int `json:"height"`
	SamplesPerPx int `json:"samples_per_px"`
	MaxDepth     int `json:"max_depth"`
}

// Fog описывает туман / участвующую среду.
// Новый объёмный туман используется только в GPU-рендере, CPU по‑прежнему игнорирует эти поля.
type Fog struct {
	// Простой параметр плотности для обратной совместимости.
	// Если заданы SigmaS/SigmaA, то Density рассматривается как базовая экстинкция и может быть 0.
	Density float64 `json:"density"` // 0 = нет тумана, >0 = базовая экстинкция

	// Базовый цвет тумана/лучей.
	Color Color `json:"color"` // базовый цвет тумана/лучей

	// Насколько сильно туман рассеивает свет в сторону камеры (0..1).
	// При 0 — чистое поглощение, при 1 — сильные видимые лучи. Используется как множитель для SigmaS.
	Scatter float64 `json:"scatter"`

	// Физические коэффициенты объёмной среды (опционально).
	// Если оба равны 0, они будут выведены из Density/Scatter.
	// sigma_t = sigma_a + sigma_s определяет общее затухание вдоль луча.
	SigmaS float64 `json:"sigma_s"` // коэффициент рассеяния
	SigmaA float64 `json:"sigma_a"` // коэффициент поглощения

	// Анизотропия фазовой функции Хеньи–Грина (−0.9..0.9).
	// g > 0 даёт направленные лучи от источников (god rays), g < 0 — обратное рассеяние.
	G float64 `json:"g"`

	// Параметры неоднородного (шумового) тумана.
	// Если HeteroStrength = 0, туман считается однородным.
	HeteroStrength float64 `json:"hetero_strength"` // 0..1, сила модификации плотности шумом
	NoiseScale     float64 `json:"noise_scale"`     // базовая частота шума (меньше = более крупные «облака»)
	NoiseOctaves   int     `json:"noise_octaves"`   // количество октав фрактального шума (1..5)

	// Влияет ли туман на фон/небо. Если false, sky почти не глушится.
	AffectSky bool `json:"affect_sky"`

	// Включение нового физически‑реалистичного объёмного тумана только для GPU.
	// Если false, GPU может использовать упрощённую модель или полностью выключить объёмный туман.
	GPUVolumetric bool `json:"gpu_volumetric"`
}

// Sky describes sky/environment settings.
type Sky struct {
	Type      string  `json:"type"`      // "solid" or "gradient"
	Color     Color   `json:"color"`     // for solid type
	Horizon   Color   `json:"horizon"`   // for gradient type
	Zenith    Color   `json:"zenith"`    // for gradient type
}

// Scene holds everything needed to render an image.
type Scene struct {
	Name      string          `json:"name"`
	Camera    Camera          `json:"camera"`
	Objects   []Object        `json:"objects"`
	Materials []Material      `json:"materials"`
	Settings  RenderSettings  `json:"settings"`

	Background Color `json:"background"` // deprecated, use Sky instead
	Sky        *Sky  `json:"sky"`        // new sky system

	// Необязательный однородный туман для GPU path tracer.
	Fog *Fog `json:"fog,omitempty"`
}

// ToImageColor converts linear Color to sRGB image color.
func (c Color) ToImageColor() color.Color {
	clamp := func(v float64) uint8 {
		if v < 0 {
			v = 0
		}
		if v > 1 {
			v = 1
		}
		// gamma correction
		v = pow(v, 1.0/2.2)
		return uint8(v * 255.999)
	}
	return color.RGBA{
		R: clamp(c.R),
		G: clamp(c.G),
		B: clamp(c.B),
		A: 255,
	}
}

// NewImage allocates an RGBA image for the scene.
func NewImage(w, h int) *image.RGBA {
	return image.NewRGBA(image.Rect(0, 0, w, h))
}

// tiny pow implementation to avoid importing math here for now.
func pow(x, y float64) float64 {
	// naive exponentiation via math.Pow will be used in engine;
	// here we keep it simple and delegate to engine if needed.
	// This function is intentionally tiny and not heavily used.
	if y == 1 {
		return x
	}
	return x
}



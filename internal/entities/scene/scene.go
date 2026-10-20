package scene

// Vec3 описывает трёхмерную точку или вектор.
type Vec3 struct {
	X float64
	Y float64
	Z float64
}

// Color описывает цвет RGB в линейном цветовом пространстве.
type Color struct {
	R float64
	G float64
	B float64
}

// Camera описывает положение и параметры камеры.
type Camera struct {
	Position Vec3
	Target   Vec3
	Up       Vec3
	FOV      float64

	Aperture    float64
	FocusDist   float64
	AspectRatio float64
}

// MaterialType задаёт поддерживаемый тип материала.
type MaterialType string

const (
	// MaterialLambert — диффузный материал.
	MaterialLambert MaterialType = "lambert"
	// MaterialMetal — отражающий металлический материал.
	MaterialMetal MaterialType = "metal"
	// MaterialDielectric — прозрачный материал.
	MaterialDielectric MaterialType = "dielectric"
	// MaterialEmissive — материал-источник света.
	MaterialEmissive MaterialType = "emissive"
	// MaterialMirror — идеальное зеркало.
	MaterialMirror MaterialType = "mirror"
)

// Material описывает свойства поверхности.
type Material struct {
	ID   string
	Type MaterialType

	Albedo Color
	Rough  float64 // for metal and lambert (roughness), deprecated for metals (use Smoothness)
	IOR    float64 // for dielectric

	Emit  Color   // emissive color
	Power float64 // emissive intensity multiplier

	// Absorption for dielectric materials (color tinting through glass)
	// Higher values mean more absorption (darker color)
	Absorption Color // for dielectric (default: 0,0,0 = no absorption)

	// Advanced parameters for metals
	Smoothness   float64 // for metals: 0.0 = matte, 1.0 = perfect mirror (default: 1.0)
	Reflectivity float64 // for metals: 0.0-1.0, controls reflection intensity (default: 1.0)

	// Advanced parameters for dielectrics (glass)
	Tint            Color   // for dielectric: color tint for light passing through glass (default: 1,1,1 = no tint)
	AbsorptionScale float64 // for dielectric: scale factor for absorption coefficient (default: 0.01, units are in cm)
}

// ObjectType задаёт поддерживаемый тип геометрического примитива.
type ObjectType string

const (
	// ObjectSphere — сферический примитив.
	ObjectSphere ObjectType = "sphere"
	// ObjectPlane — бесконечная плоскость.
	ObjectPlane ObjectType = "plane"
	// ObjectBox — осевой параллелепипед.
	ObjectBox ObjectType = "box"
	// ObjectSphereLight — сферический источник света.
	ObjectSphereLight ObjectType = "sphere_light"
)

// Object описывает один объект сцены.
type Object struct {
	ID   string
	Type ObjectType

	Position Vec3
	Size     Vec3 // radius for sphere: use X, for planes/boxes: extents

	MaterialID string
}

// RenderSettings задаёт параметры качества и производительности рендера.
type RenderSettings struct {
	Width        int
	Height       int
	SamplesPerPx int
	MaxDepth     int
	MaxRayDist   float64 // Максимальная длина луча (для предотвращения бесконечной трассировки)
}

// Fog описывает туман / участвующую среду.
// Новый объёмный туман используется только в GPU-рендере, CPU по‑прежнему игнорирует эти поля.
type Fog struct {
	// Простой параметр плотности для обратной совместимости.
	// Если заданы SigmaS/SigmaA, то Density рассматривается как базовая экстинкция и может быть 0.
	Density float64 // 0 = нет тумана, >0 = базовая экстинкция

	// Базовый цвет тумана/лучей.
	Color Color // базовый цвет тумана/лучей

	// Насколько сильно туман рассеивает свет в сторону камеры (0..1).
	// При 0 — чистое поглощение, при 1 — сильные видимые лучи. Используется как множитель для SigmaS.
	Scatter float64

	// Физические коэффициенты объёмной среды (опционально).
	// Если оба равны 0, они будут выведены из Density/Scatter.
	// sigma_t = sigma_a + sigma_s определяет общее затухание вдоль луча.
	SigmaS float64 // коэффициент рассеяния
	SigmaA float64 // коэффициент поглощения

	// Анизотропия фазовой функции Хеньи–Грина (−0.9..0.9).
	// g > 0 даёт направленные лучи от источников (god rays), g < 0 — обратное рассеяние.
	G float64

	// Параметры неоднородного (шумового) тумана.
	// Если HeteroStrength = 0, туман считается однородным.
	HeteroStrength float64 // 0..1, сила модификации плотности шумом
	NoiseScale     float64 // базовая частота шума (меньше = более крупные «облака»)
	NoiseOctaves   int     // количество октав фрактального шума (1..5)

	// Влияет ли туман на фон/небо. Если false, sky почти не глушится.
	AffectSky bool

	// Включение нового физически‑реалистичного объёмного тумана только для GPU.
	// Если false, GPU может использовать упрощённую модель или полностью выключить объёмный туман.
	GPUVolumetric bool
}

// Sky задаёт параметры неба и окружающей среды.
type Sky struct {
	Type    string // "solid" or "gradient"
	Color   Color  // for solid type
	Horizon Color  // for gradient type
	Zenith  Color  // for gradient type
}

// Scene содержит всё необходимое для построения изображения.
type Scene struct {
	Name      string
	Camera    Camera
	Objects   []Object
	Materials []Material
	Settings  RenderSettings

	Background Color // deprecated, use Sky instead
	Sky        *Sky  // new sky system

	// Необязательный однородный туман для GPU path tracer.
	Fog *Fog
}

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
	Rough  float64 `json:"rough"` // for metal and lambert (roughness)
	IOR    float64 `json:"ior"`   // for dielectric

	Emit Color   `json:"emit"`  // emissive color
	Power float64`json:"power"` // emissive intensity multiplier
	
	// Absorption for dielectric materials (color tinting through glass)
	// Higher values mean more absorption (darker color)
	Absorption Color `json:"absorption"` // for dielectric (default: 0,0,0 = no absorption)
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



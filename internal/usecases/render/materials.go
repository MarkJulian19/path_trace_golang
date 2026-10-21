package render

import (
	"math"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

type materialType int

const (
	matLambert materialType = iota
	matMetal
	matDielectric
	matEmissive
	matMirror
)

type material struct {
	typ             materialType
	albedo          vec3
	rough           float64
	ior             float64
	emit            vec3
	absorption      vec3 // для диэлектриков: поглощение света при прохождении через материал
	smoothness      float64
	reflectivity    float64
	tint            vec3
	absorptionScale float64
}

func convertMaterial(m scene.Material) material {
	al := v(clamp01(m.Albedo.R), clamp01(m.Albedo.G), clamp01(m.Albedo.B))

	switch m.Type {
	case scene.MaterialLambert:
		return material{typ: matLambert, albedo: al, rough: clamp01(m.Rough), tint: v(1, 1, 1)}
	case scene.MaterialMetal:
		return convertMetalMaterial(m, al)
	case scene.MaterialDielectric:
		return convertDielectricMaterial(m, al)
	case scene.MaterialEmissive:
		return material{typ: matEmissive, emit: emissiveColor(m), tint: v(1, 1, 1)}
	case scene.MaterialMirror:
		return material{typ: matMirror, albedo: al, reflectivity: materialReflectivity(m), tint: v(1, 1, 1)}
	default:
		return material{typ: matLambert, albedo: al, rough: clamp01(m.Rough), tint: v(1, 1, 1)}
	}
}

func convertMetalMaterial(m scene.Material, albedo vec3) material {
	return material{
		typ:          matMetal,
		albedo:       albedo,
		rough:        clamp01(m.Rough),
		smoothness:   materialSmoothness(m),
		reflectivity: materialReflectivity(m),
		tint:         v(1, 1, 1),
	}
}

func convertDielectricMaterial(m scene.Material, albedo vec3) material {
	ior := m.IOR
	if ior <= 0 {
		ior = renderDefaultIOR
	}
	return material{
		typ:             matDielectric,
		albedo:          albedo,
		ior:             ior,
		absorption:      v(m.Absorption.R, m.Absorption.G, m.Absorption.B),
		tint:            materialTint(m),
		absorptionScale: materialAbsorptionScale(m),
	}
}

func emissiveColor(m scene.Material) vec3 {
	return v(max(0, m.Emit.R)*max(0, m.Power), max(0, m.Emit.G)*max(0, m.Power), max(0, m.Emit.B)*max(0, m.Power))
}

func materialSmoothness(m scene.Material) float64 {
	value := m.Smoothness
	if value == 0 && m.Type == scene.MaterialMetal {
		value = renderUnitValue - m.Rough
	}
	return clamp01(value)
}

func materialReflectivity(m scene.Material) float64 {
	value := m.Reflectivity
	if value == 0 && (m.Type == scene.MaterialMetal || m.Type == scene.MaterialMirror) {
		value = renderUnitValue
	}
	return clamp01(value)
}

func materialAbsorptionScale(m scene.Material) float64 {
	if m.AbsorptionScale == 0 && m.Type == scene.MaterialDielectric {
		return renderThroughputCutoff
	}
	return m.AbsorptionScale
}

func materialTint(m scene.Material) vec3 {
	if m.Tint.R == 0 && m.Tint.G == 0 && m.Tint.B == 0 && m.Type == scene.MaterialDielectric {
		return v(1, 1, 1)
	}
	return v(m.Tint.R, m.Tint.G, m.Tint.B)
}

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func reflectance(cosine, refIdx float64) float64 {
	if refIdx == 1 {
		return 0
	}
	cosine = clamp01(cosine)
	sinT2 := refIdx * refIdx * (1 - cosine*cosine)
	if sinT2 >= 1 {
		return 1
	}
	cosT := math.Sqrt(1 - sinT2)
	rS := (refIdx*cosine - cosT) / (refIdx*cosine + cosT)
	rP := (cosine - refIdx*cosT) / (cosine + refIdx*cosT)
	return (rS*rS + rP*rP) * renderHalf
}

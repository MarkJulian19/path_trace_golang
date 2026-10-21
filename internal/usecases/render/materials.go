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
	al := v(m.Albedo.R, m.Albedo.G, m.Albedo.B)

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
	if ior == 0 {
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
	return v(m.Emit.R*m.Power, m.Emit.G*m.Power, m.Emit.B*m.Power)
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

func (m *material) emitted() vec3 {
	if m.typ == matEmissive {
		return m.emit
	}
	return vec3{x: 0, y: 0, z: 0} // Прямое создание вместо v()
}

func (m *material) scatterEmissive() (bool, vec3, ray) {
	return false, vec3{}, ray{}
}

func (m *material) scatterLambert(rng *randSource, rec *hitRecord) (bool, vec3, ray) {
	scatteredDir := randomCosineDirection(rec.normal, rng)
	if m.rough > renderRoughnessThreshold {
		randomOffset := randomInUnitSphere(rng)
		scatteredDir.x += randomOffset.x * m.rough * renderRoughnessScale
		scatteredDir.y += randomOffset.y * m.rough * renderRoughnessScale
		scatteredDir.z += randomOffset.z * m.rough * renderRoughnessScale
		scatteredDir = scatteredDir.unit()
	}
	return true, m.albedo, ray{orig: rec.p, dir: scatteredDir}
}

func (m *material) scatterMetal(rng *randSource, rIn ray, rec *hitRecord) (bool, vec3, ray) {
	dirLen := rIn.dir.length()
	if dirLen == 0 {
		return false, vec3{}, ray{orig: rec.p, dir: rIn.dir}
	}
	unitDir := rIn.dir.div(dirLen)
	reflected := reflectVec(unitDir, rec.normal)
	if m.rough <= renderRoughnessThreshold {
		return true, m.albedo, ray{orig: rec.p, dir: reflected}
	}
	scatteredDir := randomCosineDirection(reflected, rng)
	alpha := m.rough * m.rough
	scatteredDir = vec3{
		x: reflected.x*(1-renderUnitValue*alpha) + scatteredDir.x*alpha,
		y: reflected.y*(1-renderUnitValue*alpha) + scatteredDir.y*alpha,
		z: reflected.z*(1-renderUnitValue*alpha) + scatteredDir.z*alpha,
	}.unit()
	if scatteredDir.dot(rec.normal) <= 0 {
		scatteredDir = reflected
	}
	return true, m.albedo, ray{orig: rec.p, dir: scatteredDir}
}

func (m *material) scatterDielectric(rng *randSource, rIn ray, rec *hitRecord) (bool, vec3, ray) {
	attenuation := vec3{x: 1, y: 1, z: 1}
	refractionRatio := m.ior
	if rec.frontFace {
		refractionRatio = 1 / m.ior
	}
	dirLen := rIn.dir.length()
	if dirLen == 0 {
		return false, attenuation, ray{orig: rec.p, dir: rIn.dir}
	}
	unitDir := rIn.dir.div(dirLen)
	cosTheta := math.Min(-unitDir.dot(rec.normal), 1)
	sinTheta := math.Sqrt(1 - cosTheta*cosTheta)
	reflectProb := reflectance(cosTheta, refractionRatio)
	direction := refractVec(unitDir, rec.normal, refractionRatio)
	if refractionRatio*sinTheta > 1 || reflectProb > rng.Float64() {
		direction = reflectVec(unitDir, rec.normal)
	}
	return true, attenuation, ray{orig: rec.p, dir: direction}
}

func (m *material) scatterMirror(rIn ray, rec *hitRecord) (bool, vec3, ray) {
	dirLen := rIn.dir.length()
	if dirLen == 0 {
		return false, vec3{}, ray{orig: rec.p, dir: rIn.dir}
	}
	reflected := reflectVec(rIn.dir.div(dirLen), rec.normal)
	return true, m.albedo, ray{orig: rec.p, dir: reflected}
}

func reflectance(cosine, refIdx float64) float64 {
	// Schlick approximation
	r0 := (1 - refIdx) / (1 + refIdx)
	r0 *= r0
	return r0 + (1-r0)*math.Pow(1-cosine, renderReflectanceExponent)
}

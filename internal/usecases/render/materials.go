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
	typ        materialType
	albedo     vec3
	rough      float64
	ior        float64
	emit       vec3
	absorption vec3 // для диэлектриков: поглощение света при прохождении через материал
}

func convertMaterial(m scene.Material) material {
	al := v(m.Albedo.R, m.Albedo.G, m.Albedo.B)
	em := v(m.Emit.R*m.Power, m.Emit.G*m.Power, m.Emit.B*m.Power)
	abs := v(m.Absorption.R, m.Absorption.G, m.Absorption.B)

	switch m.Type {
	case scene.MaterialLambert:
		return material{typ: matLambert, albedo: al, rough: clamp01(m.Rough), absorption: vec3{x: 0, y: 0, z: 0}}
	case scene.MaterialMetal:
		// Для металлов используем Smoothness, если задан, иначе вычисляем из Rough
		rough := m.Rough
		if m.Smoothness > 0 {
			rough = 1.0 - clamp01(m.Smoothness)
		}
		return material{typ: matMetal, albedo: al, rough: clamp01(rough), absorption: vec3{x: 0, y: 0, z: 0}}
	case scene.MaterialDielectric:
		ior := m.IOR
		if ior == 0 {
			ior = 1.5
		}
		return material{typ: matDielectric, albedo: al, ior: ior, absorption: abs}
	case scene.MaterialEmissive:
		return material{typ: matEmissive, emit: em, absorption: vec3{x: 0, y: 0, z: 0}}
	case scene.MaterialMirror:
		return material{typ: matMirror, albedo: al, absorption: vec3{x: 0, y: 0, z: 0}}
	default:
		return material{typ: matLambert, albedo: al, rough: clamp01(m.Rough), absorption: vec3{x: 0, y: 0, z: 0}}
	}
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

func (m material) emitted() vec3 {
	if m.typ == matEmissive {
		return m.emit
	}
	return vec3{x: 0, y: 0, z: 0} // Прямое создание вместо v()
}

func (m material) scatter(rng *randSource, rIn ray, rec *hitRecord) (bool, vec3, ray) {
	switch m.typ {
	case matLambert:
		return m.scatterLambert(rng, rec)

	case matMetal:
		return m.scatterMetal(rng, rIn, rec)

	case matDielectric:
		return m.scatterDielectric(rng, rIn, rec)

	case matEmissive:
		return false, vec3{x: 0, y: 0, z: 0}, ray{}

	case matMirror:
		return m.scatterMirror(rIn, rec)
	}
	return false, vec3{x: 0, y: 0, z: 0}, ray{}
}

func (m material) scatterLambert(rng *randSource, rec *hitRecord) (bool, vec3, ray) {
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

func (m material) scatterMetal(rng *randSource, rIn ray, rec *hitRecord) (bool, vec3, ray) {
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

func (m material) scatterDielectric(rng *randSource, rIn ray, rec *hitRecord) (bool, vec3, ray) {
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

func (m material) scatterMirror(rIn ray, rec *hitRecord) (bool, vec3, ray) {
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

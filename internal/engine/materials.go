package engine

import (
	"math"

	"github.com/user/pathtracer/internal/scene"
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
	typ    materialType
	albedo vec3
	rough  float64
	ior    float64
	emit   vec3
}

func convertMaterial(m scene.Material) material {
	al := v(m.Albedo.R, m.Albedo.G, m.Albedo.B)
	em := v(m.Emit.R*m.Power, m.Emit.G*m.Power, m.Emit.B*m.Power)

	switch m.Type {
	case scene.MaterialMetal:
		return material{typ: matMetal, albedo: al, rough: clamp(m.Rough, 0, 1)}
	case scene.MaterialDielectric:
		ior := m.IOR
		if ior == 0 {
			ior = 1.5
		}
		return material{typ: matDielectric, albedo: al, ior: ior}
	case scene.MaterialEmissive:
		return material{typ: matEmissive, emit: em}
	case scene.MaterialMirror:
		return material{typ: matMirror, albedo: al}
	default:
		return material{typ: matLambert, albedo: al}
	}
}

func clamp(x, minVal, maxVal float64) float64 {
	if x < minVal {
		return minVal
	}
	if x > maxVal {
		return maxVal
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
		// Правильное косинусное распределение для корректного path tracing
		// Используем cosine-weighted hemisphere sampling
		scatteredDir := randomCosineDirection(rec.normal, rng)
		scattered := ray{
			orig: rec.p,
			dir:  scatteredDir,
		}
		return true, m.albedo, scattered

	case matMetal:
		// Металлическое отражение с шероховатостью для path tracing
		// Оптимизация: предвычисляем unit direction
		dirLen := math.Sqrt(rIn.dir.x*rIn.dir.x + rIn.dir.y*rIn.dir.y + rIn.dir.z*rIn.dir.z)
		if dirLen == 0 {
			return false, vec3{x: 0, y: 0, z: 0}, ray{orig: rec.p, dir: rIn.dir}
		}
		invLen := 1.0 / dirLen
		unitDirX := rIn.dir.x * invLen
		unitDirY := rIn.dir.y * invLen
		unitDirZ := rIn.dir.z * invLen

		reflected := reflectVec(vec3{x: unitDirX, y: unitDirY, z: unitDirZ}, rec.normal)
		randomSphere := randomInUnitSphere(rng)
		scatteredDirX := reflected.x + randomSphere.x*m.rough
		scatteredDirY := reflected.y + randomSphere.y*m.rough
		scatteredDirZ := reflected.z + randomSphere.z*m.rough

		// Нормализуем направление после добавления шероховатости
		scatteredLenSq := scatteredDirX*scatteredDirX + scatteredDirY*scatteredDirY + scatteredDirZ*scatteredDirZ
		if scatteredLenSq < 1e-8 {
			// Если направление слишком близко к нулю, используем отраженное направление
			scatteredDirX = reflected.x
			scatteredDirY = reflected.y
			scatteredDirZ = reflected.z
		} else {
			scatteredLen := math.Sqrt(scatteredLenSq)
			invScatteredLen := 1.0 / scatteredLen
			scatteredDirX *= invScatteredLen
			scatteredDirY *= invScatteredLen
			scatteredDirZ *= invScatteredLen
		}

		// Проверка dot product - направление должно быть в правильной полусфере
		dot := scatteredDirX*rec.normal.x + scatteredDirY*rec.normal.y + scatteredDirZ*rec.normal.z
		if dot <= 0 {
			return false, vec3{x: 0, y: 0, z: 0}, ray{
				orig: rec.p,
				dir:  vec3{x: scatteredDirX, y: scatteredDirY, z: scatteredDirZ},
			}
		}
		return true, m.albedo, ray{
			orig: rec.p,
			dir:  vec3{x: scatteredDirX, y: scatteredDirY, z: scatteredDirZ},
		}

	case matDielectric:
		attenuation := vec3{x: 1, y: 1, z: 1}
		var refractionRatio float64
		if rec.frontFace {
			refractionRatio = 1.0 / m.ior
		} else {
			refractionRatio = m.ior
		}

		// Оптимизация: прямое вычисление unit direction
		dirLen := math.Sqrt(rIn.dir.x*rIn.dir.x + rIn.dir.y*rIn.dir.y + rIn.dir.z*rIn.dir.z)
		if dirLen == 0 {
			return false, attenuation, ray{orig: rec.p, dir: rIn.dir}
		}
		invLen := 1.0 / dirLen
		unitDirX := rIn.dir.x * invLen
		unitDirY := rIn.dir.y * invLen
		unitDirZ := rIn.dir.z * invLen

		// Оптимизация: прямое вычисление cosTheta
		cosTheta := math.Min(-(unitDirX*rec.normal.x + unitDirY*rec.normal.y + unitDirZ*rec.normal.z), 1.0)
		sinTheta := math.Sqrt(1.0 - cosTheta*cosTheta)

		cannotRefract := refractionRatio*sinTheta > 1.0
		var direction vec3

		if cannotRefract || reflectance(cosTheta, refractionRatio) > rng.Float64() {
			direction = reflectVec(vec3{x: unitDirX, y: unitDirY, z: unitDirZ}, rec.normal)
		} else {
			direction = refractVec(vec3{x: unitDirX, y: unitDirY, z: unitDirZ}, rec.normal, refractionRatio)
		}

		scattered := ray{orig: rec.p, dir: direction}
		return true, attenuation, scattered

	case matEmissive:
		return false, vec3{x: 0, y: 0, z: 0}, ray{}

	case matMirror:
		// Идеальное зеркальное отражение без шероховатости
		dirLen := math.Sqrt(rIn.dir.x*rIn.dir.x + rIn.dir.y*rIn.dir.y + rIn.dir.z*rIn.dir.z)
		if dirLen == 0 {
			return false, vec3{x: 0, y: 0, z: 0}, ray{orig: rec.p, dir: rIn.dir}
		}
		invLen := 1.0 / dirLen
		unitDirX := rIn.dir.x * invLen
		unitDirY := rIn.dir.y * invLen
		unitDirZ := rIn.dir.z * invLen

		reflected := reflectVec(vec3{x: unitDirX, y: unitDirY, z: unitDirZ}, rec.normal)
		scattered := ray{
			orig: rec.p,
			dir:  reflected,
		}
		return true, m.albedo, scattered
	}
	return false, vec3{x: 0, y: 0, z: 0}, ray{}
}

func reflectance(cosine, refIdx float64) float64 {
	// Schlick approximation
	r0 := (1 - refIdx) / (1 + refIdx)
	r0 = r0 * r0
	return r0 + (1-r0)*math.Pow(1-cosine, 5)
}

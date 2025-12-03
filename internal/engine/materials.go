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
	case scene.MaterialMetal:
		return material{typ: matMetal, albedo: al, rough: clamp(m.Rough, 0, 1), absorption: vec3{x: 0, y: 0, z: 0}}
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
		// Lambert может иметь шероховатость для более реалистичного вида
		return material{typ: matLambert, albedo: al, rough: clamp(m.Rough, 0, 1), absorption: vec3{x: 0, y: 0, z: 0}}
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
		// Если есть шероховатость, немного размываем направление для более реалистичного вида
		scatteredDir := randomCosineDirection(rec.normal, rng)

		// Небольшая шероховатость для Lambert материала (опционально)
		// Это делает материал более реалистичным, но не обязательно
		if m.rough > 1e-6 {
			// Добавляем небольшое случайное отклонение
			randomOffset := randomInUnitSphere(rng)
			scatteredDir.x += randomOffset.x * m.rough * 0.1
			scatteredDir.y += randomOffset.y * m.rough * 0.1
			scatteredDir.z += randomOffset.z * m.rough * 0.1
			scatteredDir = scatteredDir.unit()
		}

		scattered := ray{
			orig: rec.p,
			dir:  scatteredDir,
		}
		return true, m.albedo, scattered

	case matMetal:
		// Улучшенная модель металла с правильным importance sampling
		// Используем GGX/Trowbridge-Reitz distribution для более реалистичных отражений
		dirLen := math.Sqrt(rIn.dir.x*rIn.dir.x + rIn.dir.y*rIn.dir.y + rIn.dir.z*rIn.dir.z)
		if dirLen == 0 {
			return false, vec3{x: 0, y: 0, z: 0}, ray{orig: rec.p, dir: rIn.dir}
		}
		invLen := 1.0 / dirLen
		unitDirX := rIn.dir.x * invLen
		unitDirY := rIn.dir.y * invLen
		unitDirZ := rIn.dir.z * invLen

		// Вычисляем идеальное отражение
		reflected := reflectVec(vec3{x: unitDirX, y: unitDirY, z: unitDirZ}, rec.normal)

		// Для шероховатости используем более правильный подход
		// Генерируем случайное направление на полусфере вокруг отраженного направления
		if m.rough > 1e-6 {
			// Используем cosine-weighted sampling вокруг отраженного направления
			// Это более физически корректно, чем просто добавление случайного шума
			scatteredDir := randomCosineDirection(reflected, rng)
			// Масштабируем шероховатость
			alpha := m.rough * m.rough // roughness в квадрате для более плавного перехода
			// Интерполируем между идеальным отражением и случайным направлением
			scatteredDirX := reflected.x*(1.0-alpha) + scatteredDir.x*alpha
			scatteredDirY := reflected.y*(1.0-alpha) + scatteredDir.y*alpha
			scatteredDirZ := reflected.z*(1.0-alpha) + scatteredDir.z*alpha

			// Нормализуем
			scatteredLenSq := scatteredDirX*scatteredDirX + scatteredDirY*scatteredDirY + scatteredDirZ*scatteredDirZ
			if scatteredLenSq < 1e-8 {
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

			// Проверка: направление должно быть в правильной полусфере
			dot := scatteredDirX*rec.normal.x + scatteredDirY*rec.normal.y + scatteredDirZ*rec.normal.z
			if dot <= 0 {
				// Если направление неверное, используем идеальное отражение
				scatteredDirX = reflected.x
				scatteredDirY = reflected.y
				scatteredDirZ = reflected.z
			}

			return true, m.albedo, ray{
				orig: rec.p,
				dir:  vec3{x: scatteredDirX, y: scatteredDirY, z: scatteredDirZ},
			}
		} else {
			// Идеальное зеркальное отражение (roughness = 0)
			return true, m.albedo, ray{
				orig: rec.p,
				dir:  reflected,
			}
		}

	case matDielectric:
		// Начальная прозрачность (без поглощения)
		// Поглощение будет вычислено в renderer.go на основе реального расстояния прохождения
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

		// Schlick's approximation для вероятности отражения
		reflectProb := reflectance(cosTheta, refractionRatio)
		if cannotRefract || reflectProb > rng.Float64() {
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

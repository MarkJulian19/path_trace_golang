package render

import "math"

func sphereLightPDFCPU(center, reference vec3, radius float64) float64 {
	delta := center.sub(reference)
	distanceSquared := delta.dot(delta)
	if distanceSquared <= radius*radius {
		return 0
	}
	sineSquared := radius * radius / distanceSquared
	oneMinusCosine := sineSquared / (1 + math.Sqrt(max(0, 1-sineSquared)))
	return 1 / (2 * math.Pi * oneMinusCosine)
}

func sampleCPULight(light *cpuLight, reference vec3, rng *randSource) (cpuAreaLightSample, bool) {
	if !light.valid {
		return cpuAreaLightSample{}, false
	}
	if light.kind == hittableBox {
		return sampleVisibleBoxCPU(light, reference, rng)
	}
	delta := light.center.sub(reference)
	distanceSquared := delta.dot(delta)
	if distanceSquared <= light.radius*light.radius {
		return cpuAreaLightSample{}, false
	}
	distance := math.Sqrt(distanceSquared)
	axis := delta.div(distance)
	sineSquared := light.radius * light.radius / distanceSquared
	oneMinusCosine := sineSquared / (1 + math.Sqrt(max(0, 1-sineSquared)))
	pdf := 1 / (2 * math.Pi * oneMinusCosine)
	deltaCosine := rng.Float64() * oneMinusCosine
	cosine := 1 - deltaCosine
	sine := math.Sqrt(max(0, deltaCosine*(2-deltaCosine)))
	phi := 2 * math.Pi * rng.Float64()
	tangentAxis := v(0, 0, 1)
	if math.Abs(axis.z) >= cpuBasisCutoff {
		tangentAxis = v(1, 0, 0)
	}
	tangent := tangentAxis.cross(axis).unit()
	direction := axis.mul(cosine).add(tangent.mul(sine * math.Cos(phi))).
		add(axis.cross(tangent).mul(sine * math.Sin(phi)))
	projection := distance * cosine
	halfChord := math.Sqrt(max(0, light.radius*light.radius-distanceSquared*sine*sine))
	point := reference.add(direction.mul((distanceSquared - light.radius*light.radius) / (projection + halfChord)))
	return cpuAreaLightSample{
		position:     point,
		normal:       point.sub(light.center).div(light.radius),
		pdfDirection: pdf,
	}, true
}

func visibleBoxAreasCPU(center, size, reference vec3) [3]float64 {
	delta := reference.sub(center)
	var areas [3]float64
	if math.Abs(delta.x) > size.x*renderHalf {
		areas[0] = size.y * size.z
	}
	if math.Abs(delta.y) > size.y*renderHalf {
		areas[1] = size.x * size.z
	}
	if math.Abs(delta.z) > size.z*renderHalf {
		areas[2] = size.x * size.y
	}
	return areas
}

func sampleVisibleBoxCPU(light *cpuLight, reference vec3, rng *randSource) (cpuAreaLightSample, bool) {
	areas := visibleBoxAreasCPU(light.center, light.size, reference)
	area := areas[0] + areas[1] + areas[2]
	if area <= 0 {
		return cpuAreaLightSample{}, false
	}
	choice := rng.Float64() * area
	axis := 0
	for axis < 2 && choice >= areas[axis] {
		choice -= areas[axis]
		axis++
	}
	positions := [3]float64{light.center.x, light.center.y, light.center.z}
	references := [3]float64{reference.x, reference.y, reference.z}
	sizes := [3]float64{light.size.x, light.size.y, light.size.z}
	normal := [3]float64{}
	normal[axis] = 1
	if references[axis] < positions[axis] {
		normal[axis] = -1
	}
	for i := range positions {
		if i == axis {
			positions[i] += normal[axis] * sizes[i] * renderHalf
		} else {
			positions[i] += (rng.Float64() - renderHalf) * sizes[i]
		}
	}
	return cpuAreaLightSample{
		position: v(positions[0], positions[1], positions[2]),
		normal:   v(normal[0], normal[1], normal[2]), pdfArea: 1 / area,
	}, true
}

func lightPDFAtHitCPU(object *hittable, reference vec3, hit hitRecord) float64 {
	if object.kind == hittableSphere {
		return sphereLightPDFCPU(object.sphere.center, reference, object.sphere.radius)
	}
	if object.kind != hittableBox {
		return 0
	}
	size := object.box.max.sub(object.box.min)
	areas := visibleBoxAreasCPU(object.box.min.add(object.box.max).mul(renderHalf), size, reference)
	area := areas[0] + areas[1] + areas[2]
	delta := hit.p.sub(reference)
	cosine := math.Abs(delta.unit().dot(hit.normal))
	if area <= 0 || cosine <= 0 {
		return 0
	}
	return delta.dot(delta) / (area * cosine)
}

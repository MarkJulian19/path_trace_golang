package render

import "math"

const cpuMicrofacetJacobian = 4

func objectMaterialCPU(object *hittable) *material {
	switch object.kind {
	case hittableSphere:
		return &object.sphere.mat
	case hittablePlane:
		return &object.pln.mat
	default:
		return &object.box.mat
	}
}

func containsPointCPU(object *hittable, point vec3) bool {
	switch object.kind {
	case hittableSphere:
		delta := point.sub(object.sphere.center)
		return delta.dot(delta) < object.sphere.radius*object.sphere.radius
	case hittableBox:
		return point.x > object.box.min.x && point.x < object.box.max.x &&
			point.y > object.box.min.y && point.y < object.box.max.y &&
			point.z > object.box.min.z && point.z < object.box.max.z
	default:
		return point.y < object.pln.point.y
	}
}

// mediumAtCPU выбирает внутреннюю вложенную среду независимо от направления луча.
func mediumAtCPU(world *cpuRenderWorld, r ray, exclude int) int {
	medium, nearest := -1, math.Inf(1)
	for i := range world.objects {
		object := &world.objects[i]
		if i == exclude || objectMaterialCPU(object).typ != matDielectric || !containsPointCPU(object, r.orig) {
			continue
		}
		distance := mediumBoundaryDistanceCPU(object, r.orig)
		if distance < nearest {
			nearest, medium = distance, i
		}
	}
	return medium
}

func mediumBoundaryDistanceCPU(object *hittable, point vec3) float64 {
	switch object.kind {
	case hittableSphere:
		return object.sphere.radius - point.sub(object.sphere.center).length()
	case hittableBox:
		lower, upper := point.sub(object.box.min), object.box.max.sub(point)
		return min(lower.x, lower.y, lower.z, upper.x, upper.y, upper.z)
	default:
		return object.pln.point.y - point.y
	}
}

func offsetRayCPU(point, normal, direction vec3) ray {
	const originULPs = 64
	scale := max(1, math.Abs(point.x), math.Abs(point.y), math.Abs(point.z))
	epsilon := originULPs * (math.Nextafter(scale, math.Inf(1)) - scale)
	if direction.dot(normal) < 0 {
		epsilon = -epsilon
	}
	return ray{orig: point.add(normal.mul(epsilon)), dir: direction.unit()}
}

func dielectricInterfaceCPU(world *cpuRenderWorld, r ray, rec hitRecord, current int) (float64, int) {
	etaI := renderUnitValue
	if current >= 0 {
		etaI = objectMaterialCPU(&world.objects[current]).ior
	}
	next := rec.objectIdx
	if !rec.frontFace {
		next = mediumAtCPU(world, offsetRayCPU(rec.p, rec.normal, r.dir), rec.objectIdx)
	}
	etaT := renderUnitValue
	if next >= 0 {
		etaT = objectMaterialCPU(&world.objects[next]).ior
	}
	return etaI / etaT, next
}

func ggxG1CPU(cosine, alpha float64) float64 {
	if cosine <= 0 {
		return 0
	}
	return 2 * cosine / (cosine + math.Sqrt(alpha*alpha+(1-alpha*alpha)*cosine*cosine))
}

func fresnelMetalCPU(cosine float64, f0 vec3) vec3 {
	x := 1 - clamp01(cosine)
	x5 := x * x * x * x * x
	return f0.add(v(1-f0.x, 1-f0.y, 1-f0.z).mul(x5))
}

func surfaceBSDFCPU(m *material, incident, wi, normal vec3) (vec3, float64) {
	nv, nl := -incident.dot(normal), wi.dot(normal)
	if nv <= 0 || nl <= 0 {
		return vec3{}, 0
	}
	if m.typ == matLambert {
		return m.albedo.mul(cpuInvPi), nl * cpuInvPi
	}
	roughness := 1 - m.smoothness
	if m.typ != matMetal || roughness <= renderMetalRoughThreshold {
		return vec3{}, 0
	}
	half := wi.sub(incident).unit()
	nh, vh := clamp01(half.dot(normal)), math.Abs(half.dot(incident))
	if nh <= 0 || vh <= cpuGGXDenominatorCutoff {
		return vec3{}, 0
	}
	alpha := roughness * roughness
	denominator := (1 - nh*nh) + nh*nh*alpha*alpha
	d := alpha * alpha / (math.Pi * denominator * denominator)
	g1View := ggxG1CPU(nv, alpha)
	g := g1View * ggxG1CPU(nl, alpha)
	f := fresnelMetalCPU(vh, m.albedo.mul(m.reflectivity))
	return f.mul(d * g / (cpuMicrofacetJacobian * nv * nl)), d * g1View / (cpuMicrofacetJacobian * nv)
}

func emissionWeightCPU(world *cpuRenderWorld, rec hitRecord, previous vec3, pdf float64) float64 {
	if pdf <= 0 || len(world.lights) == 0 {
		return 1
	}
	lightPDF := lightPDFAtHitCPU(&world.objects[rec.objectIdx], previous, rec)
	lightPDF *= min(1, float64(cpuMaxLightSamples)/float64(len(world.lights)))
	return pdf / (pdf + lightPDF)
}

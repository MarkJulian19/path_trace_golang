package render

import (
	"math"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

type hitRecord struct {
	p         vec3
	normal    vec3
	t         float64
	frontFace bool
	mat       material
}

func (h *hitRecord) setFaceNormal(r ray, outwardNormal vec3) {
	h.frontFace = r.dir.dot(outwardNormal) < 0
	if h.frontFace {
		h.normal = outwardNormal
	} else {
		h.normal = outwardNormal.mul(-1)
	}
}

type hittable interface {
	hit(r ray, tMin, tMax float64, rec *hitRecord) bool
}

// Sphere primitive.
type sphere struct {
	center vec3
	radius float64
	mat    material
}

func (s sphere) hit(r ray, tMin, tMax float64, rec *hitRecord) bool {
	// Оптимизация: прямое вычисление без промежуточных структур
	ocX := r.orig.x - s.center.x
	ocY := r.orig.y - s.center.y
	ocZ := r.orig.z - s.center.z

	a := r.dir.x*r.dir.x + r.dir.y*r.dir.y + r.dir.z*r.dir.z
	halfB := ocX*r.dir.x + ocY*r.dir.y + ocZ*r.dir.z
	ocLenSq := ocX*ocX + ocY*ocY + ocZ*ocZ
	radiusSq := s.radius * s.radius
	c := ocLenSq - radiusSq

	root, ok := sphereRoot(a, halfB, c, tMin, tMax)
	if !ok {
		return false
	}

	rec.t = root
	// Оптимизация: прямое вычисление точки без вызова r.at()
	rec.p.x = r.orig.x + r.dir.x*root
	rec.p.y = r.orig.y + r.dir.y*root
	rec.p.z = r.orig.z + r.dir.z*root

	// Оптимизация: прямое вычисление нормали
	invRadius := 1.0 / s.radius
	outwardNormalX := (rec.p.x - s.center.x) * invRadius
	outwardNormalY := (rec.p.y - s.center.y) * invRadius
	outwardNormalZ := (rec.p.z - s.center.z) * invRadius

	// Оптимизация: прямое вычисление frontFace и normal
	dot := r.dir.x*outwardNormalX + r.dir.y*outwardNormalY + r.dir.z*outwardNormalZ
	rec.frontFace = dot < 0
	if rec.frontFace {
		rec.normal.x = outwardNormalX
		rec.normal.y = outwardNormalY
		rec.normal.z = outwardNormalZ
	} else {
		rec.normal.x = -outwardNormalX
		rec.normal.y = -outwardNormalY
		rec.normal.z = -outwardNormalZ
	}
	rec.mat = s.mat
	return true
}

func sphereRoot(a, halfB, c, tMin, tMax float64) (float64, bool) {
	discriminant := halfB*halfB - a*c
	if discriminant < 0 {
		return 0, false
	}
	sqrtD := math.Sqrt(discriminant)
	root := (-halfB - sqrtD) / a
	if root >= tMin && root <= tMax {
		return root, true
	}
	root = (-halfB + sqrtD) / a
	return root, root >= tMin && root <= tMax
}

// Infinite plane (used as ground).
type plane struct {
	point  vec3
	normal vec3
	mat    material
}

func (p plane) hit(r ray, tMin, tMax float64, rec *hitRecord) bool {
	// Оптимизация: прямое вычисление без промежуточных структур
	denom := p.normal.x*r.dir.x + p.normal.y*r.dir.y + p.normal.z*r.dir.z
	if math.Abs(denom) < renderRoughnessThreshold {
		return false
	}

	pointMinusOrigX := p.point.x - r.orig.x
	pointMinusOrigY := p.point.y - r.orig.y
	pointMinusOrigZ := p.point.z - r.orig.z
	t := (pointMinusOrigX*p.normal.x + pointMinusOrigY*p.normal.y + pointMinusOrigZ*p.normal.z) / denom

	if t < tMin || t > tMax {
		return false
	}

	rec.t = t
	// Оптимизация: прямое вычисление точки
	rec.p.x = r.orig.x + r.dir.x*t
	rec.p.y = r.orig.y + r.dir.y*t
	rec.p.z = r.orig.z + r.dir.z*t

	// Оптимизация: прямое вычисление frontFace и normal
	rec.frontFace = denom < 0
	if rec.frontFace {
		rec.normal.x = p.normal.x
		rec.normal.y = p.normal.y
		rec.normal.z = p.normal.z
	} else {
		rec.normal.x = -p.normal.x
		rec.normal.y = -p.normal.y
		rec.normal.z = -p.normal.z
	}
	rec.mat = p.mat
	return true
}

// Axis-aligned box defined by min and max points.
type box struct {
	min, max vec3
	mat      material
}

func (b box) hit(r ray, tMin, tMax float64, rec *hitRecord) bool {
	t0 := tMin
	t1 := tMax

	for axis := range 3 {
		invD, tNear, tFar := b.axisInterval(r, axis)
		if invD < 0 {
			tNear, tFar = tFar, tNear
		}
		if tNear > t0 {
			t0 = tNear
		}
		if tFar < t1 {
			t1 = tFar
		}
		if t1 <= t0 {
			return false
		}
	}

	rec.t = t0
	rec.p = r.at(t0)

	rec.setFaceNormal(r, b.normalAt(rec.p))
	rec.mat = b.mat
	return true
}

func (b box) normalAt(point vec3) vec3 {
	dxMin := point.x - b.min.x
	dxMax := b.max.x - point.x
	dyMin := point.y - b.min.y
	dyMax := b.max.y - point.y
	dzMin := point.z - b.min.z
	dzMax := b.max.z - point.z
	minDist := dxMin
	normal := v(-1, 0, 0)
	if dxMax < minDist {
		minDist = dxMax
		normal = v(1, 0, 0)
	}
	if dyMin < minDist {
		minDist = dyMin
		normal = v(0, -1, 0)
	}
	if dyMax < minDist {
		minDist = dyMax
		normal = v(0, 1, 0)
	}
	if dzMin < minDist {
		minDist = dzMin
		normal = v(0, 0, -1)
	}
	if dzMax < minDist {
		normal = v(0, 0, 1)
	}
	return normal
}

func (b box) axisInterval(r ray, axis int) (float64, float64, float64) {
	switch axis {
	case 0:
		invD := 1 / r.dir.x
		return invD, (b.min.x - r.orig.x) * invD, (b.max.x - r.orig.x) * invD
	case 1:
		invD := 1 / r.dir.y
		return invD, (b.min.y - r.orig.y) * invD, (b.max.y - r.orig.y) * invD
	default:
		invD := 1 / r.dir.z
		return invD, (b.min.z - r.orig.z) * invD, (b.max.z - r.orig.z) * invD
	}
}

// sceneToWorld builds hittable list from scene description.
func sceneToWorld(sc *scene.Scene) []hittable {
	materials := make(map[string]material)
	for _, m := range sc.Materials {
		materials[m.ID] = convertMaterial(m)
	}

	world := make([]hittable, 0, len(sc.Objects))
	for _, o := range sc.Objects {
		object, ok := objectToHittable(o, materials[o.MaterialID])
		if ok {
			world = append(world, object)
		}
	}
	return world
}

func objectToHittable(value scene.Object, mat material) (hittable, bool) {
	pos := v(value.Position.X, value.Position.Y, value.Position.Z)
	size := v(value.Size.X, value.Size.Y, value.Size.Z)
	switch value.Type {
	case scene.ObjectSphere, scene.ObjectSphereLight:
		return sphere{center: pos, radius: size.x, mat: mat}, true
	case scene.ObjectPlane:
		return plane{point: pos, normal: v(0, 1, 0), mat: mat}, true
	case scene.ObjectBox:
		return box{min: pos.sub(size.mul(renderHalf)), max: pos.add(size.mul(renderHalf)), mat: mat}, true
	default:
		return nil, false
	}
}

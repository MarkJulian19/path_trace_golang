package engine

import (
	"math"

	"github.com/user/pathtracer/internal/scene"
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

	discriminant := halfB*halfB - a*c
	if discriminant < 0 {
		return false
	}
	sqrtD := math.Sqrt(discriminant)

	root := (-halfB - sqrtD) / a
	if root < tMin || root > tMax {
		root = (-halfB + sqrtD) / a
		if root < tMin || root > tMax {
			return false
		}
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

// Infinite plane (used as ground).
type plane struct {
	point  vec3
	normal vec3
	mat    material
}

func (p plane) hit(r ray, tMin, tMax float64, rec *hitRecord) bool {
	// Оптимизация: прямое вычисление без промежуточных структур
	denom := p.normal.x*r.dir.x + p.normal.y*r.dir.y + p.normal.z*r.dir.z
	if math.Abs(denom) < 1e-6 {
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

	for i := 0; i < 3; i++ {
		var invD, orig, minV, maxV float64
		switch i {
		case 0:
			invD = 1 / r.dir.x
			orig = r.orig.x
			minV = b.min.x
			maxV = b.max.x
		case 1:
			invD = 1 / r.dir.y
			orig = r.orig.y
			minV = b.min.y
			maxV = b.max.y
		default:
			invD = 1 / r.dir.z
			orig = r.orig.z
			minV = b.min.z
			maxV = b.max.z
		}

		tNear := (minV - orig) * invD
		tFar := (maxV - orig) * invD
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

	// approximate normal: determine which face was hit
	const eps = 1e-4
	var n vec3
	if math.Abs(rec.p.x-b.min.x) < eps {
		n = v(-1, 0, 0)
	} else if math.Abs(rec.p.x-b.max.x) < eps {
		n = v(1, 0, 0)
	} else if math.Abs(rec.p.y-b.min.y) < eps {
		n = v(0, -1, 0)
	} else if math.Abs(rec.p.y-b.max.y) < eps {
		n = v(0, 1, 0)
	} else if math.Abs(rec.p.z-b.min.z) < eps {
		n = v(0, 0, -1)
	} else {
		n = v(0, 0, 1)
	}

	rec.setFaceNormal(r, n)
	rec.mat = b.mat
	return true
}

// sceneToWorld builds hittable list from scene description.
func sceneToWorld(sc *scene.Scene) []hittable {
	materials := make(map[string]material)
	for _, m := range sc.Materials {
		materials[m.ID] = convertMaterial(m)
	}

	world := make([]hittable, 0, len(sc.Objects))
	for _, o := range sc.Objects {
		mat := materials[o.MaterialID]
		pos := v(o.Position.X, o.Position.Y, o.Position.Z)
		size := v(o.Size.X, o.Size.Y, o.Size.Z)

		switch o.Type {
		case scene.ObjectSphere:
			world = append(world, sphere{
				center: pos,
				radius: size.x,
				mat:    mat,
			})
		case scene.ObjectSphereLight:
			// Сферический источник света - это сфера с emissive материалом
			world = append(world, sphere{
				center: pos,
				radius: size.x,
				mat:    mat,
			})
		case scene.ObjectPlane:
			n := v(0, 1, 0)
			world = append(world, plane{
				point:  pos,
				normal: n,
				mat:    mat,
			})
		case scene.ObjectBox:
			min := pos.sub(size.mul(0.5))
			max := pos.add(size.mul(0.5))
			world = append(world, box{
				min: min,
				max: max,
				mat: mat,
			})
		}
	}
	return world
}

package render

import (
	"math"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

type hitRecord struct {
	mat       *material
	p         vec3
	normal    vec3
	t         float64
	objectIdx int
	frontFace bool
}

func (h *hitRecord) setFaceNormal(r ray, outwardNormal vec3) {
	h.frontFace = r.dir.dot(outwardNormal) < 0
	if h.frontFace {
		h.normal = outwardNormal
	} else {
		h.normal = outwardNormal.mul(-1)
	}
}

type hittableKind uint8

const (
	hittableSphere hittableKind = iota
	hittablePlane
	hittableBox
)

type hittable struct {
	box       box
	sphere    sphere
	pln       plane
	objectIdx int
	sceneIdx  int
	kind      hittableKind
}

func (h *hittable) hit(r ray, tMin, tMax float64, rec *hitRecord) bool {
	switch h.kind {
	case hittableSphere:
		return h.sphere.hit(r, tMin, tMax, rec)
	case hittablePlane:
		return h.pln.hit(r, tMin, tMax, rec)
	default:
		return h.box.hit(r, tMin, tMax, rec)
	}
}

// Sphere primitive.
type sphere struct {
	mat    material
	center vec3
	radius float64
}

func (s *sphere) hit(r ray, tMin, tMax float64, rec *hitRecord) bool {
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

	// Нормализуем и восстанавливаем поверхность до смещения вторичного луча.
	outwardNormal := rec.p.sub(s.center).unit()
	rec.p = s.center.add(outwardNormal.mul(s.radius))
	rec.setFaceNormal(r, outwardNormal)
	rec.mat = &s.mat
	return true
}

func sphereRoot(a, halfB, c, tMin, tMax float64) (float64, bool) {
	if a <= 0 {
		return 0, false
	}
	discriminant := math.FMA(-a, c, halfB*halfB)
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
	mat   material
	point vec3
}

func (p *plane) hit(r ray, tMin, tMax float64, rec *hitRecord) bool {
	// Оптимизация: прямое вычисление без промежуточных структур
	// Все planes создаются горизонтальными, с нормалью (0, 1, 0).
	denom := r.dir.y
	if math.Abs(denom) < renderRoughnessThreshold {
		return false
	}

	pointMinusOrigY := p.point.y - r.orig.y
	t := pointMinusOrigY / denom

	if t < tMin || t > tMax {
		return false
	}

	rec.t = t
	// Оптимизация: прямое вычисление точки
	rec.p.x = r.orig.x + r.dir.x*t
	rec.p.y = p.point.y
	rec.p.z = r.orig.z + r.dir.z*t

	// Оптимизация: прямое вычисление frontFace и normal
	rec.frontFace = denom < 0
	if rec.frontFace {
		rec.normal = vec3{x: 0, y: 1, z: 0}
	} else {
		rec.normal = vec3{x: 0, y: -1, z: 0}
	}
	rec.mat = &p.mat
	return true
}

// Axis-aligned box defined by min and max points.
type box struct {
	mat material
	min vec3
	max vec3
}

func (b *box) hit(r ray, tMin, tMax float64, rec *hitRecord) bool {
	return b.hitWithMode(r, tMin, tMax, rec, false)
}

func (b *box) hitWithMode(r ray, tMin, tMax float64, rec *hitRecord, findExit bool) bool {
	t0 := math.Inf(-1)
	t1 := math.Inf(1)

	var ok bool
	t0, t1, ok = updateBoxInterval(t0, t1, b.min.x, b.max.x, r.orig.x, r.dir.x)
	if !ok {
		return false
	}
	t0, t1, ok = updateBoxInterval(t0, t1, b.min.y, b.max.y, r.orig.y, r.dir.y)
	if !ok {
		return false
	}
	t0, t1, ok = updateBoxInterval(t0, t1, b.min.z, b.max.z, r.orig.z, r.dir.z)
	if !ok {
		return false
	}

	t := t0
	if findExit || t < tMin {
		t = t1
	}
	if t < tMin || t > tMax {
		return false
	}
	rec.t = t
	rec.p = r.at(rec.t)

	outwardNormal := b.normalAt(rec.p)
	rec.p = b.projectSurfacePoint(rec.p, outwardNormal)
	rec.setFaceNormal(r, outwardNormal)
	rec.mat = &b.mat
	return true
}

// projectSurfacePoint сохраняет точную координату грани для начала вторичного луча.
func (b *box) projectSurfacePoint(point, normal vec3) vec3 {
	switch {
	case normal.x < 0:
		point.x = b.min.x
	case normal.x > 0:
		point.x = b.max.x
	case normal.y < 0:
		point.y = b.min.y
	case normal.y > 0:
		point.y = b.max.y
	case normal.z < 0:
		point.z = b.min.z
	default:
		point.z = b.max.z
	}
	return point
}

func updateBoxInterval(t0, t1, minValue, maxValue, origin, direction float64) (float64, float64, bool) {
	if direction == 0 {
		return t0, t1, origin >= minValue && origin <= maxValue
	}
	invD := 1 / direction
	tNear := (minValue - origin) * invD
	tFar := (maxValue - origin) * invD
	if invD < 0 {
		tNear, tFar = tFar, tNear
	}
	if tNear > t0 {
		t0 = tNear
	}
	if tFar < t1 {
		t1 = tFar
	}
	return t0, t1, t1 >= t0
}

func (b *box) normalAt(point vec3) vec3 {
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

// sceneToWorld builds hittable list from scene description.
func sceneToWorld(sc *scene.Scene) []hittable {
	materialIndex := make(map[string]int, len(sc.Materials))
	for i, m := range sc.Materials {
		materialIndex[m.ID] = i
	}

	world := make([]hittable, 0, len(sc.Objects))
	for sceneIdx, o := range sc.Objects {
		material := materialAtScene(sc.Materials, materialIndex[o.MaterialID])
		object := objectToHittable(o, convertMaterial(material))
		object.objectIdx = len(world)
		object.sceneIdx = sceneIdx
		world = append(world, object)
	}
	return world
}

func materialAtScene(materials []scene.Material, index int) scene.Material {
	if index < 0 || index >= len(materials) {
		return scene.Material{}
	}
	return materials[index]
}

func objectToHittable(value scene.Object, mat material) hittable {
	pos := v(value.Position.X, value.Position.Y, value.Position.Z)
	size := v(value.Size.X, value.Size.Y, value.Size.Z)
	switch value.Type {
	case scene.ObjectSphere, scene.ObjectSphereLight:
		return hittable{kind: hittableSphere, sphere: sphere{center: pos, radius: size.x, mat: mat}}
	case scene.ObjectPlane:
		return hittable{kind: hittablePlane, pln: plane{point: pos, mat: mat}}
	case scene.ObjectBox:
		return hittable{
			kind: hittableBox,
			box:  box{min: pos.sub(size.mul(renderHalf)), max: pos.add(size.mul(renderHalf)), mat: mat},
		}
	default:
		// The GPU packing path treats unknown object types as spheres. Keep the
		// CPU world index-compatible with that fallback instead of dropping the
		// object and shifting every following object/light index.
		return hittable{kind: hittableSphere, sphere: sphere{center: pos, radius: size.x, mat: mat}}
	}
}

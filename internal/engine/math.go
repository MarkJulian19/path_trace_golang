package engine

import "math"

type vec3 struct {
	x, y, z float64
}

func v(x, y, z float64) vec3 { return vec3{x, y, z} }

func (a vec3) add(b vec3) vec3    { return vec3{x: a.x + b.x, y: a.y + b.y, z: a.z + b.z} }
func (a vec3) sub(b vec3) vec3    { return vec3{x: a.x - b.x, y: a.y - b.y, z: a.z - b.z} }
func (a vec3) mul(t float64) vec3 { return vec3{x: a.x * t, y: a.y * t, z: a.z * t} }
func (a vec3) div(t float64) vec3 {
	invT := 1.0 / t
	return vec3{x: a.x * invT, y: a.y * invT, z: a.z * invT}
}

func (a vec3) dot(b vec3) float64 { return a.x*b.x + a.y*b.y + a.z*b.z }

func (a vec3) cross(b vec3) vec3 {
	return v(
		a.y*b.z-a.z*b.y,
		a.z*b.x-a.x*b.z,
		a.x*b.y-a.y*b.x,
	)
}

func (a vec3) length() float64 { return math.Sqrt(a.dot(a)) }

func (a vec3) unit() vec3 {
	l := a.length()
	if l == 0 {
		return a
	}
	return a.div(l)
}

func reflectVec(v, n vec3) vec3 {
	dot := v.dot(n)
	return vec3{
		x: v.x - n.x*2*dot,
		y: v.y - n.y*2*dot,
		z: v.z - n.z*2*dot,
	}
}

func refractVec(uv, n vec3, etaiOverEtat float64) vec3 {
	cosTheta := math.Min(-uv.x*n.x-uv.y*n.y-uv.z*n.z, 1.0)
	rOutPerpX := uv.x + n.x*cosTheta
	rOutPerpY := uv.y + n.y*cosTheta
	rOutPerpZ := uv.z + n.z*cosTheta
	rOutPerpX *= etaiOverEtat
	rOutPerpY *= etaiOverEtat
	rOutPerpZ *= etaiOverEtat

	perpLenSq := rOutPerpX*rOutPerpX + rOutPerpY*rOutPerpY + rOutPerpZ*rOutPerpZ
	rOutParallel := -math.Sqrt(math.Abs(1.0 - perpLenSq))
	return vec3{
		x: rOutPerpX + n.x*rOutParallel,
		y: rOutPerpY + n.y*rOutParallel,
		z: rOutPerpZ + n.z*rOutParallel,
	}
}

func randomInUnitSphere(rng *randSource) vec3 {
	// Оптимизированная версия с меньшим количеством вызовов функций
	for {
		x := rng.Float64()*2 - 1
		y := rng.Float64()*2 - 1
		z := rng.Float64()*2 - 1
		// Проверка длины без создания vec3
		lenSq := x*x + y*y + z*z
		if lenSq >= 1.0 {
			continue
		}
		return vec3{x: x, y: y, z: z}
	}
}

func randomUnitVector(rng *randSource) vec3 {
	return randomInUnitSphere(rng).unit()
}

// randomCosineDirection generates a random direction on the hemisphere
// with cosine-weighted distribution relative to the normal.
// This is the correct distribution for Lambertian (diffuse) materials in path tracing.
func randomCosineDirection(normal vec3, rng *randSource) vec3 {
	// Generate a random point in unit sphere
	r1 := rng.Float64()
	r2 := rng.Float64()

	// Spherical coordinates for cosine-weighted hemisphere sampling
	phi := 2.0 * math.Pi * r1
	cosTheta := math.Sqrt(r2)
	sinTheta := math.Sqrt(1.0 - r2)

	// Create a local coordinate system aligned with the normal
	// Choose an arbitrary vector not parallel to normal
	var u vec3
	if math.Abs(normal.x) > 0.9 {
		u = v(0, 1, 0)
	} else {
		u = v(1, 0, 0)
	}

	// Build orthonormal basis
	w := normal
	vVec := w.cross(u).unit()
	uVec := vVec.cross(w)

	// Generate direction in local coordinate system
	localDir := vec3{
		x: sinTheta * math.Cos(phi),
		y: sinTheta * math.Sin(phi),
		z: cosTheta,
	}

	// Transform to world space
	return vec3{
		x: localDir.x*uVec.x + localDir.y*vVec.x + localDir.z*w.x,
		y: localDir.x*uVec.y + localDir.y*vVec.y + localDir.z*w.y,
		z: localDir.x*uVec.z + localDir.y*vVec.z + localDir.z*w.z,
	}
}

type ray struct {
	orig vec3
	dir  vec3
}

func (r ray) at(t float64) vec3 {
	return r.orig.add(r.dir.mul(t))
}

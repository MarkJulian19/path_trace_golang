package render

import "math"

type vec3 struct {
	x, y, z float64
}

func v(x, y, z float64) vec3 { return vec3{x, y, z} }

func (a vec3) add(b vec3) vec3    { return vec3{x: a.x + b.x, y: a.y + b.y, z: a.z + b.z} }
func (a vec3) sub(b vec3) vec3    { return vec3{x: a.x - b.x, y: a.y - b.y, z: a.z - b.z} }
func (a vec3) mul(t float64) vec3 { return vec3{x: a.x * t, y: a.y * t, z: a.z * t} }

func mulVec3(a, b vec3) vec3 { return vec3{x: a.x * b.x, y: a.y * b.y, z: a.z * b.z} }

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
	if perpLenSq > 1 {
		return reflectVec(uv, n)
	}
	rOutParallel := -math.Sqrt(max(0, 1.0-perpLenSq))
	return vec3{
		x: rOutPerpX + n.x*rOutParallel,
		y: rOutPerpY + n.y*rOutParallel,
		z: rOutPerpZ + n.z*rOutParallel,
	}
}

type ray struct {
	orig vec3
	dir  vec3
}

func (r ray) at(t float64) vec3 {
	return r.orig.add(r.dir.mul(t))
}

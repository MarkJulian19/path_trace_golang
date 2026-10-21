package render

import (
	"math"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

const (
	cpuPi                   = math.Pi
	cpuInvPi                = 1 / math.Pi
	cpuMaxLightSamples      = 8
	cpuFogMaxAnisotropy     = 0.9
	cpuFogMaxNoiseOctaves   = 5
	cpuFogSkyDistance       = 50
	cpuGGXDenominatorCutoff = 1e-12
	cpuRRDepth              = 3
	cpuBasisCutoff          = 0.999
	cpuShadowDistanceOffset = 2 * renderMinRayDistance
	cpuHashX                = 510.1
	cpuHashY                = 766.9
	cpuHashZ                = 445.4
	cpuHashScale            = 43758.5453
)

// cpuRenderWorld contains the data that is shared by every CPU worker.  It is
// deliberately immutable after construction, so workers can trace rays
// without locks or per-ray allocations.
type cpuRenderWorld struct {
	objects    []hittable
	lights     []cpuLight
	fog        cpuFog
	maxRayDist float64
}

type cpuLight struct {
	sceneIdx int
	center   vec3
	size     vec3
	radius   float64
	emit     vec3
	kind     hittableKind
	valid    bool
}

type cpuFog struct {
	density        float64
	color          vec3
	scatter        float64
	sigmaS         float64
	sigmaA         float64
	g              float64
	heteroStrength float64
	noiseScale     float64
	noiseOctaves   int
	affectSky      bool
	volumetric     bool
}

func newCPURenderWorld(sc *scene.Scene, cfg Config) *cpuRenderWorld {
	world := &cpuRenderWorld{
		objects:    sceneToWorld(sc),
		maxRayDist: float64(cfg.MaxRayDist),
	}
	if world.maxRayDist <= 0 {
		world.maxRayDist = defaultMaxRayDistance
	}

	world.lights = buildCPULightSlots(sc, world.objects)
	if sc.Fog != nil {
		world.fog = newCPUFog(*sc.Fog)
	}
	return world
}

func buildCPULightSlots(sc *scene.Scene, objects []hittable) []cpuLight {
	materialIndex := make(map[string]int, len(sc.Materials))
	for i, material := range sc.Materials {
		materialIndex[material.ID] = i
	}
	worldIndexBySceneIndex := make([]int, len(sc.Objects))
	for i := range worldIndexBySceneIndex {
		worldIndexBySceneIndex[i] = -1
	}
	for worldIdx := range objects {
		sceneIdx := objects[worldIdx].sceneIdx
		if sceneIdx >= 0 && sceneIdx < len(worldIndexBySceneIndex) {
			worldIndexBySceneIndex[sceneIdx] = worldIdx
		}
	}

	// This deliberately mirrors gpu.findLightIndices: source discovery uses
	// the original scene material, not the multiplied/converted CPU material.
	// The resulting slots must stay in scene-object order because the shader
	// uses the same order for light sampling and inclusion probabilities.
	lights := make([]cpuLight, 0, len(sc.Objects))
	for sceneIdx, objectValue := range sc.Objects {
		sceneMaterial := materialAtScene(sc.Materials, materialIndex[objectValue.MaterialID])
		if !isGPULightMaterial(sceneMaterial) {
			continue
		}
		light := cpuLight{}
		light.sceneIdx = sceneIdx
		light.emit = emissiveColor(sceneMaterial)
		worldIdx := worldIndexBySceneIndex[sceneIdx]
		if worldIdx < 0 {
			lights = append(lights, light)
			continue
		}
		setCPULightGeometry(&light, &objects[worldIdx])
		lights = append(lights, light)
	}
	return lights
}

func newCPUFog(fog scene.Fog) cpuFog {
	result := cpuFog{
		density:        maxFloat(fog.Density, 0),
		color:          v(clamp01(fog.Color.R), clamp01(fog.Color.G), clamp01(fog.Color.B)),
		scatter:        clamp01(fog.Scatter),
		g:              clampFloat(fog.G, -cpuFogMaxAnisotropy, cpuFogMaxAnisotropy),
		heteroStrength: clamp01(fog.HeteroStrength),
		noiseScale:     fog.NoiseScale,
		noiseOctaves:   fog.NoiseOctaves,
		affectSky:      fog.AffectSky,
		volumetric:     fog.GPUVolumetric,
	}
	if result.noiseScale <= 0 {
		result.noiseScale = 4
	}
	if result.noiseOctaves <= 0 {
		result.noiseOctaves = 3
	}
	if result.noiseOctaves > cpuFogMaxNoiseOctaves {
		result.noiseOctaves = cpuFogMaxNoiseOctaves
	}

	result.sigmaS = maxFloat(fog.SigmaS, 0)
	result.sigmaA = maxFloat(fog.SigmaA, 0)
	if result.sigmaS <= 0 && result.sigmaA <= 0 && result.density > 0 {
		result.sigmaS = result.density * result.scatter
		result.sigmaA = maxFloat(result.density-result.sigmaS, 0)
	}
	return result
}

func clampFloat(value, low, high float64) float64 {
	if value < low {
		return low
	}
	if value > high {
		return high
	}
	return value
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func isGPULightMaterial(material scene.Material) bool {
	return material.Type == scene.MaterialEmissive &&
		(material.Emit.R > 0 || material.Emit.G > 0 || material.Emit.B > 0)
}

func hitClosestCPU(world *cpuRenderWorld, r ray, tMax float64, rec *hitRecord) bool {
	hitAnything := false
	closest := tMax
	var candidate hitRecord
	for i := range world.objects {
		object := &world.objects[i]
		var hit bool
		switch object.kind {
		case hittableSphere:
			hit = object.sphere.hit(r, renderMinRayDistance, closest, &candidate)
		case hittablePlane:
			hit = object.pln.hit(r, renderMinRayDistance, closest, &candidate)
		case hittableBox:
			hit = object.box.hit(r, renderMinRayDistance, closest, &candidate)
		default:
			hit = false
		}
		if !hit {
			continue
		}
		hitAnything = true
		*rec = candidate
		closest = rec.t
		rec.objectIdx = object.objectIdx
	}
	return hitAnything
}

// rayColorGPUParity mirrors the iterative GPU path tracer.  The CPU keeps its
// own float64 arithmetic, worker pool, and per-worker RNG; only the rendering
// semantics are shared with the shader.
func rayColorGPUParity(r ray, world *cpuRenderWorld, background func(ray) vec3, depth int, rng *randSource) vec3 {
	path := cpuPath{r: r, throughput: v(1, 1, 1), medium: mediumAtCPU(world, r, -1), previousPoint: r.orig, etaScale: 1}
	for bounce := range depth {
		if !path.traceBounce(world, background, bounce, rng, bounce+1 < depth) {
			break
		}
	}
	return path.radiance
}

func advanceCPUPath(
	r *ray,
	throughput *vec3,
	rec hitRecord,
	result cpuScatterResult,
	remaining int,
	etaScale float64,
	rng *randSource,
) (int, bool) {
	if !result.scattered {
		return result.glassObject, false
	}
	weighted := mulVec3(*throughput, result.attenuation)
	weighted, canContinue := applyCPURussianRoulette(weighted, remaining, etaScale, rng)
	if !canContinue {
		return result.glassObject, false
	}
	*throughput = weighted
	*r = offsetRayCPU(rec.p, rec.normal, result.direction)
	return result.glassObject, true
}

func cpuMissRadiance(r ray, world *cpuRenderWorld, background func(ray) vec3, throughput vec3) vec3 {
	bg := background(r)
	if !world.fog.volumetric && world.fog.density > 0 && world.fog.affectSky {
		bg = applySimpleFogCPU(bg, cpuFogSkyDistance, world.fog)
	}
	return mulVec3(throughput, bg)
}

func applyCPURussianRoulette(attenuation vec3, remaining int, etaScale float64, rng *randSource) (vec3, bool) {
	if remaining < cpuRRDepth {
		return attenuation, true
	}
	maxAttenuation := maxComponent(attenuation)
	if maxAttenuation <= 0 {
		return vec3{}, false
	}
	rrProbability := min(maxAttenuation*etaScale, renderUnitValue)
	if rrProbability >= 1 {
		return attenuation, true
	}
	if rng.Float64() > rrProbability {
		return vec3{}, false
	}
	return attenuation.div(rrProbability), true
}

type cpuScatterResult struct {
	direction   vec3
	attenuation vec3
	radiance    vec3
	glassObject int
	pdf         float64
	etaScale    float64
	scattered   bool
}

func scatterCPUSurface(
	r ray,
	world *cpuRenderWorld,
	rec hitRecord,
	rng *randSource,
	currentGlassObject int,
	continuation bool,
) cpuScatterResult {
	switch rec.mat.typ {
	case matLambert:
		return scatterLambertCPU(r, world, rec, rng, currentGlassObject, continuation)
	case matMetal, matMirror:
		return scatterMetalCPU(r, world, rec, rng, currentGlassObject, continuation)
	case matDielectric:
		return scatterDielectricCPU(r, world, rec, rng, currentGlassObject)
	default:
		return cpuScatterResult{glassObject: currentGlassObject}
	}
}

func scatterLambertCPU(
	r ray,
	world *cpuRenderWorld,
	rec hitRecord,
	rng *randSource,
	currentGlassObject int,
	continuation bool,
) cpuScatterResult {
	direction := randomCosineDirection(rec.normal, rng)
	return cpuScatterResult{
		pdf:         max(0, direction.dot(rec.normal)) * cpuInvPi,
		direction:   direction,
		attenuation: rec.mat.albedo,
		radiance:    estimateDirectLightCPU(r, world, rec, rec.mat.albedo, rng, continuation),
		glassObject: currentGlassObject,
		scattered:   true,
	}
}

func scatterMetalCPU(
	r ray,
	world *cpuRenderWorld,
	rec hitRecord,
	rng *randSource,
	currentGlassObject int,
	continuation bool,
) cpuScatterResult {
	mat := rec.mat
	viewDir := r.dir.mul(-1)
	metalRough := 1 - mat.smoothness
	effectiveReflectivity := mat.reflectivity
	if effectiveReflectivity <= 0 {
		effectiveReflectivity = 1
	}
	isRoughMetal := mat.typ == matMetal && metalRough > renderMetalRoughThreshold
	result := cpuScatterResult{glassObject: currentGlassObject, scattered: true}
	if isRoughMetal {
		result.direction = sampleGGXCPU(viewDir, rec.normal, metalRough, rng)
		f, pdf := surfaceBSDFCPU(mat, r.dir, result.direction, rec.normal)
		result.pdf = pdf
		if pdf > 0 {
			result.attenuation = f.mul(max(0, result.direction.dot(rec.normal)) / pdf)
		}
		result.radiance = estimateDirectLightCPU(r, world, rec, mat.albedo, rng, continuation)
	} else {
		result.direction = reflectVec(r.dir, rec.normal)
		result.attenuation = mat.albedo.mul(effectiveReflectivity)
		if mat.typ == matMetal {
			result.attenuation = fresnelMetalCPU(viewDir.dot(rec.normal), result.attenuation)
		}
	}
	if result.direction.dot(rec.normal) <= 0 {
		result.scattered = false
		return result
	}
	return result
}

func scatterDielectricCPU(
	r ray,
	world *cpuRenderWorld,
	rec hitRecord,
	rng *randSource,
	currentGlassObject int,
) cpuScatterResult {
	eta, next := dielectricInterfaceCPU(world, r, rec, currentGlassObject)
	newDir, shouldReflect := dielectricDirectionCPU(r, rec, eta, rng)
	if shouldReflect {
		return cpuScatterResult{
			direction:   newDir,
			attenuation: v(1, 1, 1),
			glassObject: currentGlassObject,
			scattered:   true,
		}
	}

	return cpuScatterResult{
		direction:   newDir,
		attenuation: v(eta*eta, eta*eta, eta*eta),
		glassObject: next,
		etaScale:    1 / (eta * eta),
		scattered:   true,
	}
}

func dielectricDirectionCPU(
	r ray,
	rec hitRecord,
	eta float64,
	rng *randSource,
) (vec3, bool) {
	viewDir := r.dir.mul(-1)
	cosTheta := clampFloat(viewDir.dot(rec.normal), 0, 1)
	sinTheta := math.Sqrt(maxFloat(1-cosTheta*cosTheta, 0))
	reflectProb := reflectance(cosTheta, eta)
	shouldReflect := eta*sinTheta > 1
	if !shouldReflect {
		shouldReflect = rng.Float64() < reflectProb
	}
	if shouldReflect {
		return reflectVec(r.dir, rec.normal), true
	}
	return refractVec(r.dir, rec.normal, eta).unit(), false
}

func maxComponent(value vec3) float64 {
	return maxFloat(value.x, maxFloat(value.y, value.z))
}

func randomCosineDirection(normal vec3, rng *randSource) vec3 {
	r1 := rng.Float64()
	r2 := rng.Float64()
	phi := 2 * cpuPi * r1
	cosTheta := math.Sqrt(r2)
	sinTheta := math.Sqrt(maxFloat(1-r2, 0))
	axis := v(0, 0, 1)
	if math.Abs(normal.z) >= cpuBasisCutoff {
		axis = v(1, 0, 0)
	}
	u := axis.cross(normal).unit()
	vVec := normal.cross(u)
	sinPhi, cosPhi := math.Sincos(phi)
	return u.mul(sinTheta * cosPhi).add(vVec.mul(sinTheta * sinPhi)).add(normal.mul(cosTheta)).unit()
}

// sampleGGXCPU выбирает видимую микронормаль; её PDF согласован с surfaceBSDFCPU.
func sampleGGXCPU(viewDir, normal vec3, roughness float64, rng *randSource) vec3 {
	tangentAxis := v(0, 0, 1)
	if math.Abs(normal.z) >= cpuBasisCutoff {
		tangentAxis = v(1, 0, 0)
	}
	tangent := tangentAxis.cross(normal).unit()
	bitangent := normal.cross(tangent)
	alpha := roughness * roughness
	stretched := v(alpha*viewDir.dot(tangent), alpha*viewDir.dot(bitangent), viewDir.dot(normal)).unit()
	t1 := v(1, 0, 0)
	lengthSquared := stretched.x*stretched.x + stretched.y*stretched.y
	if lengthSquared > 0 {
		t1 = v(-stretched.y, stretched.x, 0).div(math.Sqrt(lengthSquared))
	}
	t2 := stretched.cross(t1)
	radius := math.Sqrt(rng.Float64())
	phi := 2 * math.Pi * rng.Float64()
	x, y := radius*math.Cos(phi), radius*math.Sin(phi)
	blend := (1 + stretched.z) * renderHalf
	y = (1-blend)*math.Sqrt(max(0, 1-x*x)) + blend*y
	sampled := t1.mul(x).add(t2.mul(y)).add(stretched.mul(math.Sqrt(max(0, 1-x*x-y*y))))
	half := v(alpha*sampled.x, alpha*sampled.y, max(0, sampled.z)).unit()
	worldHalf := tangent.mul(half.x).add(bitangent.mul(half.y)).add(normal.mul(half.z))
	return reflectVec(viewDir.mul(-1), worldHalf).unit()
}

func estimateDirectLightCPU(
	r ray,
	world *cpuRenderWorld,
	surfHit hitRecord,
	albedo vec3,
	rng *randSource,
	continuation bool,
) vec3 {
	lightCount := len(world.lights)
	if lightCount == 0 {
		return vec3{}
	}

	lightsToSample := lightCount
	start := 0
	if lightCount > cpuMaxLightSamples {
		lightsToSample = cpuMaxLightSamples
		start = int(rng.Float64()*float64(lightCount)) % lightCount
	}

	total := vec3{}
	for j := range lightsToSample {
		lightIdx := (start + j) % lightCount
		contrib := estimateDirectLightSingleCPU(r, world, &world.lights[lightIdx], surfHit, albedo, rng, continuation)
		total = total.add(contrib)
	}
	return total
}

type cpuAreaLightSample struct {
	position     vec3
	normal       vec3
	pdfArea      float64
	pdfDirection float64
}

func estimateDirectLightSingleCPU(
	r ray,
	world *cpuRenderWorld,
	light *cpuLight,
	surfHit hitRecord,
	albedo vec3,
	rng *randSource,
	continuation bool,
) vec3 {
	sample, ok := sampleCPULight(light, surfHit.p, rng)
	if !ok {
		return vec3{}
	}

	toLight := sample.position.sub(surfHit.p)
	distSq := toLight.dot(toLight)
	if distSq <= 0 {
		return vec3{}
	}
	dist := math.Sqrt(distSq)
	wi := toLight.div(dist)
	if !directLightVisibleCPU(world, surfHit.p, surfHit.normal, wi, dist, light.sceneIdx) {
		return vec3{}
	}

	cosSurf := maxFloat(surfHit.normal.dot(wi), 0)
	cosLight := maxFloat(sample.normal.dot(wi.mul(-1)), 0)
	if cosSurf <= 0 || cosLight <= 0 {
		return vec3{}
	}

	connection := cpuLightConnection{sample: sample, wi: wi, distance: dist, cosSurf: cosSurf, cosLight: cosLight}
	return directLightContributionCPU(r, world, surfHit, albedo, light, connection, rng, continuation)
}

func directLightContributionCPU(r ray, world *cpuRenderWorld, hit hitRecord, albedo vec3, light *cpuLight,
	connection cpuLightConnection, rng *randSource, continuation bool,
) vec3 {
	lightPDF := connection.sample.pdfDirection
	if lightPDF <= 0 {
		lightPDF = connection.sample.pdfArea * connection.distance * connection.distance / connection.cosLight
	}
	inclusion := min(1, float64(cpuMaxLightSamples)/float64(len(world.lights)))
	mat := hit.mat
	if mat == nil {
		mat = &material{typ: matLambert, albedo: albedo}
	}
	f, bsdfPDF := surfaceBSDFCPU(mat, r.dir, connection.wi, hit.normal)
	denominator := lightPDF * inclusion
	if continuation {
		denominator += bsdfPDF
	}
	transmittance := shadowTransmittanceCPU(world, ray{orig: hit.p, dir: connection.wi}, connection.distance, rng)
	return mulVec3(mulVec3(f, light.emit), transmittance).mul(connection.cosSurf / denominator)
}

func directLightVisibleCPU(world *cpuRenderWorld, point, normal, direction vec3, distance float64, emitter int) bool {
	shadowRay := offsetRayCPU(point, normal, direction)
	delta := point.add(direction.mul(distance)).sub(shadowRay.orig)
	shadowRay.dir = delta.unit()
	for i := range world.objects {
		if world.objects[i].sceneIdx == emitter {
			continue
		}
		var hit hitRecord
		if world.objects[i].hit(shadowRay, renderMinRayDistance, delta.length()-cpuShadowDistanceOffset, &hit) {
			return false
		}
	}
	return true
}

func dielectricAttenuation(mat *material, distance float64) vec3 {
	absorb := v(
		math.Exp(-max(0, mat.absorption.x)*max(0, mat.absorptionScale)*distance),
		math.Exp(-max(0, mat.absorption.y)*max(0, mat.absorptionScale)*distance),
		math.Exp(-max(0, mat.absorption.z)*max(0, mat.absorptionScale)*distance),
	)
	tint := v(math.Pow(clamp01(mat.tint.x), distance), math.Pow(clamp01(mat.tint.y), distance),
		math.Pow(clamp01(mat.tint.z), distance))
	return mulVec3(absorb, tint)
}

func (fog cpuFog) mediumCoefficients(pos vec3) (sigmaS, sigmaT float64) {
	sigmaS = fog.sigmaS
	sigmaA := fog.sigmaA
	if sigmaS <= 0 && sigmaA <= 0 && fog.density > 0 {
		sigmaS = fog.density * fog.scatter
		sigmaA = maxFloat(fog.density-sigmaS, 0)
	}
	sigmaT = sigmaS + sigmaA
	if sigmaT <= 0 {
		return 0, 0
	}
	if fog.heteroStrength > 0 {
		scale := (1 - fog.heteroStrength) + 2*fog.heteroStrength*volumeNoiseCPU(pos, fog)
		sigmaS *= scale
		sigmaA *= scale
		sigmaT = sigmaS + sigmaA
	}
	return sigmaS, sigmaT
}

func applySimpleFogCPU(radiance vec3, distance float64, fog cpuFog) vec3 {
	if fog.density <= 0 || distance <= 0 {
		return radiance
	}
	attenuation := math.Exp(-fog.density * distance)
	return radiance.mul(attenuation).add(fog.color.mul(1 - attenuation))
}

func phaseHGCPU(cosTheta, g float64) float64 {
	gg := g * g
	denom := 1 + gg - 2*g*cosTheta
	return (1 - gg) / (4 * math.Pi * denom * math.Sqrt(denom))
}

func volumeNoiseCPU(pos vec3, fog cpuFog) float64 {
	if fog.noiseOctaves <= 1 {
		return hash31CPU(pos.mul(fog.noiseScale))
	}
	if fog.noiseOctaves == 2 {
		return (hash31CPU(pos.mul(fog.noiseScale)) + hash31CPU(pos.mul(fog.noiseScale*2))) * renderHalf
	}
	amp := 1.0
	frequency := fog.noiseScale
	var sum, norm float64
	for range fog.noiseOctaves {
		sum += hash31CPU(pos.mul(frequency)) * amp
		norm += amp
		amp *= renderHalf
		frequency *= 2
	}
	if norm <= 0 {
		return 1
	}
	return sum / norm
}

func hash31CPU(pos vec3) float64 {
	value := pos.x*cpuHashX + pos.y*cpuHashY + pos.z*cpuHashZ
	fract := math.Sin(value) * cpuHashScale
	return fract - math.Floor(fract)
}

func cpuACES(x float64) float64 {
	if x <= 0 || math.IsNaN(x) || math.IsInf(x, 0) {
		return 0
	}
	const (
		a = 2.51
		b = 0.03
		c = 2.43
		d = 0.59
		e = 0.14
	)
	numerator := x * (a*x + b)
	denominator := x*(c*x+d) + e
	if denominator <= 0 {
		return 0
	}
	return clampFloat(numerator/denominator, 0, 1)
}

func cpuTonemapGamma(value float64) float64 {
	return math.Sqrt(cpuACES(value))
}

func setCPULightGeometry(light *cpuLight, object *hittable) {
	light.kind = object.kind
	switch object.kind {
	case hittableSphere:
		light.center = object.sphere.center
		light.radius = object.sphere.radius
		light.valid = light.radius > 0
	case hittableBox:
		light.center = object.box.min.add(object.box.max).mul(renderHalf)
		light.size = object.box.max.sub(object.box.min)
		light.valid = light.size.x > 0 && light.size.y > 0 && light.size.z > 0
	default:
		light.valid = false
	}
}

type cpuLightConnection struct {
	sample                      cpuAreaLightSample
	wi                          vec3
	distance, cosSurf, cosLight float64
}

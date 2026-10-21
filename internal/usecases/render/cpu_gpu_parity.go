package render

import (
	"math"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

const (
	cpuPi                    = math.Pi
	cpuInvPi                 = 1 / math.Pi
	cpuMaxLightSamples       = 8
	cpuVolumeSteps           = 24
	cpuFogMaxAnisotropy      = 0.9
	cpuFogMaxNoiseOctaves    = 5
	cpuFogSkyDistance        = 50
	cpuGGXDenominatorCutoff  = 1e-12
	cpuDirectPDFCutoff       = 1e-10
	cpuDirectMaxLuminance    = 500
	cpuMetalDiffuseScale     = 0.3
	cpuGlassReflectanceFloor = 0.05
	cpuRRDepth               = 3
	cpuBasisCutoff           = 0.999
	cpuVolumeMaxDistance     = 40
	cpuShadowDistanceOffset  = 0.002
	cpuHashX                 = 510.1
	cpuHashY                 = 766.9
	cpuHashZ                 = 445.4
	cpuHashScale             = 43758.5453
	cpuLumaR                 = 0.2126
	cpuLumaG                 = 0.7152
	cpuLumaB                 = 0.0722
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
	radius   float64
	emit     vec3
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
	// uses the same order for light sampling and light-count normalization.
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
		object := &objects[worldIdx]
		if object.kind == hittableSphere {
			light.center = object.sphere.center
			light.radius = object.sphere.radius
			light.valid = light.radius > 0
		}
		lights = append(lights, light)
	}
	return lights
}

func newCPUFog(fog scene.Fog) cpuFog {
	result := cpuFog{
		density:        maxFloat(fog.Density, 0),
		color:          v(fog.Color.R, fog.Color.G, fog.Color.B),
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

func hasPositiveComponent(value vec3) bool {
	return value.x > 0 || value.y > 0 || value.z > 0
}

func isGPULightMaterial(material scene.Material) bool {
	return material.Type == scene.MaterialEmissive &&
		(material.Emit.R > 0 || material.Emit.G > 0 || material.Emit.B > 0)
}

func hitClosestCPU(world *cpuRenderWorld, r ray, tMax float64, exitObject int, rec *hitRecord) bool {
	hitAnything := false
	closest := tMax
	for i := range world.objects {
		object := &world.objects[i]
		findExit := object.objectIdx == exitObject
		var hit bool
		switch object.kind {
		case hittableSphere:
			hit = object.sphere.hit(r, renderMinRayDistance, closest, rec)
		case hittablePlane:
			hit = object.pln.hit(r, renderMinRayDistance, closest, rec)
		case hittableBox:
			hit = object.box.hitWithMode(r, renderMinRayDistance, closest, rec, findExit)
		default:
			hit = false
		}
		if !hit {
			continue
		}
		hitAnything = true
		closest = rec.t
		rec.objectIdx = object.objectIdx
	}
	return hitAnything
}

// rayColorGPUParity mirrors the iterative GPU path tracer.  The CPU keeps its
// own float64 arithmetic, worker pool, and per-worker RNG; only the rendering
// semantics are shared with the shader.
func rayColorGPUParity(r ray, world *cpuRenderWorld, background func(ray) vec3, depth int, rng *randSource) vec3 {
	if depth <= 0 {
		return vec3{}
	}

	throughput := v(1, 1, 1)
	radiance := vec3{}
	currentGlassObject := -1
	var accumulatedTravelDistance float64

	if world.fog.volumetric {
		radiance = radiance.add(estimatePrimaryVolume(r, world, rng))
	}

	for remaining := depth; remaining > 0; remaining-- {
		if maxComponent(throughput) < renderThroughputCutoff {
			break
		}

		var rec hitRecord
		if !hitClosestCPU(world, r, world.maxRayDist, currentGlassObject, &rec) {
			radiance = radiance.add(cpuMissRadiance(r, world, background, throughput))
			break
		}

		mat := rec.mat
		if mat.typ == matEmissive {
			radiance = radiance.add(mulVec3(throughput, mat.emit))
			break
		}

		result := scatterCPUSurface(r, world, rec, rng, currentGlassObject, &accumulatedTravelDistance)
		radiance = radiance.add(mulVec3(throughput, result.radiance))
		var canContinue bool
		currentGlassObject, canContinue = advanceCPUPath(&r, &throughput, rec, result, remaining, rng)
		if !canContinue {
			break
		}
	}

	return radiance
}

func advanceCPUPath(
	r *ray,
	throughput *vec3,
	rec hitRecord,
	result cpuScatterResult,
	remaining int,
	rng *randSource,
) (int, bool) {
	if !result.scattered {
		return result.glassObject, false
	}
	attenuation, canContinue := applyCPURussianRoulette(result.attenuation, remaining, rng)
	if !canContinue {
		return result.glassObject, false
	}
	*throughput = mulVec3(*throughput, attenuation)
	r.orig = rec.p.add(result.direction.mul(renderMinRayDistance))
	r.dir = result.direction.unit()
	return result.glassObject, true
}

func cpuMissRadiance(r ray, world *cpuRenderWorld, background func(ray) vec3, throughput vec3) vec3 {
	bg := background(r)
	if world.fog.density > 0 && world.fog.affectSky {
		bg = applySimpleFogCPU(bg, cpuFogSkyDistance, world.fog)
	}
	return mulVec3(throughput, bg)
}

func applyCPURussianRoulette(attenuation vec3, remaining int, rng *randSource) (vec3, bool) {
	if remaining > cpuRRDepth {
		return attenuation, true
	}
	maxAttenuation := maxComponent(attenuation)
	if maxAttenuation < renderRoughnessThreshold {
		return vec3{}, false
	}
	rrProbability := min(maxAttenuation, renderEdgeThreshold)
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
	scattered   bool
}

func scatterCPUSurface(
	r ray,
	world *cpuRenderWorld,
	rec hitRecord,
	rng *randSource,
	currentGlassObject int,
	travelDistance *float64,
) cpuScatterResult {
	switch rec.mat.typ {
	case matLambert:
		return scatterLambertCPU(world, rec, rng, currentGlassObject)
	case matMetal, matMirror:
		return scatterMetalCPU(r, world, rec, rng, currentGlassObject)
	case matDielectric:
		return scatterDielectricCPU(r, world, rec, rng, currentGlassObject, travelDistance)
	default:
		return cpuScatterResult{glassObject: currentGlassObject}
	}
}

func scatterLambertCPU(world *cpuRenderWorld, rec hitRecord, rng *randSource, currentGlassObject int) cpuScatterResult {
	return cpuScatterResult{
		direction:   randomCosineDirectionGPU(rec.normal, rng),
		attenuation: rec.mat.albedo,
		radiance:    estimateDirectLightCPU(world, rec, rec.mat.albedo, rng),
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
) cpuScatterResult {
	mat := rec.mat
	viewDir := r.dir.mul(-1)
	metalRough := mat.rough
	if mat.smoothness > 0 {
		metalRough = 1 - mat.smoothness
	}
	effectiveReflectivity := mat.reflectivity
	if effectiveReflectivity <= 0 {
		effectiveReflectivity = 1
	}
	isRoughMetal := mat.typ == matMetal && metalRough > renderMetalRoughThreshold
	result := cpuScatterResult{glassObject: currentGlassObject, scattered: true}
	if isRoughMetal {
		result.direction = sampleGGXCPU(viewDir, rec.normal, metalRough, rng)
		roughnessSquared := metalRough * metalRough
		specularWeight := clampFloat(1/(1+roughnessSquared*2), renderRoughnessScale, cpuFogMaxAnisotropy)
		diffuseWeight := 1 - specularWeight
		direct := estimateDirectLightCPU(world, rec, mat.albedo, rng)
		result.radiance = direct.mul(diffuseWeight * effectiveReflectivity * renderHalf)
		result.attenuation = mat.albedo.mul(specularWeight*effectiveReflectivity + diffuseWeight*cpuMetalDiffuseScale)
	} else {
		result.direction = reflectVec(r.dir, rec.normal)
		result.attenuation = mat.albedo.mul(effectiveReflectivity)
	}
	if result.direction.dot(rec.normal) <= renderRoughnessThreshold {
		result.scattered = false
		return result
	}
	if isRoughMetal {
		result.radiance = result.radiance.add(roughMetalEmissionCPU(r, world, rec, mat.albedo))
	}
	return result
}

func roughMetalEmissionCPU(r ray, world *cpuRenderWorld, rec hitRecord, albedo vec3) vec3 {
	reflectDir := reflectVec(r.dir, rec.normal)
	reflectRay := ray{orig: rec.p.add(rec.normal.mul(renderMinRayDistance)), dir: reflectDir}
	var reflected hitRecord
	if !hitClosestCPU(world, reflectRay, world.maxRayDist, -1, &reflected) || reflected.mat.typ != matEmissive {
		return vec3{}
	}
	distanceSquared := reflected.t * reflected.t
	if distanceSquared <= cpuGGXDenominatorCutoff {
		return vec3{}
	}
	cosLight := maxFloat(reflected.normal.dot(reflectRay.dir.mul(-1)), 0)
	directReflect := reflected.mat.emit.mul(cosLight / distanceSquared)
	return mulVec3(directReflect, albedo).mul(renderHalf)
}

func scatterDielectricCPU(
	r ray,
	world *cpuRenderWorld,
	rec hitRecord,
	rng *randSource,
	currentGlassObject int,
	travelDistance *float64,
) cpuScatterResult {
	mat := rec.mat
	newDir, entering, shouldReflect := dielectricDirectionCPU(r, rec, mat.ior, rng, currentGlassObject)
	if shouldReflect {
		return cpuScatterResult{
			direction:   newDir,
			attenuation: v(1, 1, 1),
			glassObject: currentGlassObject,
			scattered:   true,
		}
	}

	result := cpuScatterResult{direction: newDir, attenuation: v(1, 1, 1), scattered: true}
	if entering {
		result.glassObject = rec.objectIdx
		insideDistance := glassExitDistanceCPU(world, rec, newDir)
		if insideDistance > 0 {
			*travelDistance = insideDistance
			result.attenuation = dielectricAttenuation(mat, insideDistance)
		}
		return result
	}
	result.glassObject = -1
	if *travelDistance > 0 {
		result.attenuation = dielectricAttenuation(mat, *travelDistance)
	}
	*travelDistance = 0
	return result
}

func dielectricDirectionCPU(
	r ray,
	rec hitRecord,
	ior float64,
	rng *randSource,
	currentGlassObject int,
) (vec3, bool, bool) {
	entering := rec.frontFace
	if rec.objectIdx == currentGlassObject {
		entering = false
	}
	if ior <= 0 {
		ior = renderDefaultIOR
	}
	eta := ior
	if entering {
		eta = 1 / ior
	}
	viewDir := r.dir.mul(-1)
	cosTheta := clampFloat(viewDir.dot(rec.normal), 0, 1)
	sinTheta := math.Sqrt(maxFloat(1-cosTheta*cosTheta, 0))
	reflectProb := reflectance(cosTheta, eta)
	if !entering {
		reflectProb = maxFloat(reflectProb, cpuGlassReflectanceFloor)
	}
	shouldReflect := eta*sinTheta > 1
	if !shouldReflect {
		shouldReflect = rng.Float64() < reflectProb
	}
	if shouldReflect {
		return reflectVec(r.dir, rec.normal), entering, true
	}
	return refractVec(r.dir, rec.normal, eta).unit(), entering, false
}

func maxComponent(value vec3) float64 {
	return maxFloat(value.x, maxFloat(value.y, value.z))
}

func randomCosineDirectionGPU(normal vec3, rng *randSource) vec3 {
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

func sampleGGXCPU(viewDir, normal vec3, roughness float64, rng *randSource) vec3 {
	alpha := roughness * roughness
	r1 := rng.Float64()
	r2 := rng.Float64()
	phi := 2 * cpuPi * r1
	denominator := 1 + (alpha*alpha-1)*r2
	cosTheta := math.Sqrt(maxFloat((1-r2)/maxFloat(denominator, cpuGGXDenominatorCutoff), 0))
	sinTheta := math.Sqrt(maxFloat(1-cosTheta*cosTheta, 0))
	halfVec := v(sinTheta*math.Cos(phi), sinTheta*math.Sin(phi), cosTheta)
	up := v(0, 0, 1)
	if math.Abs(normal.z) >= cpuBasisCutoff {
		up = v(1, 0, 0)
	}
	tangent := up.cross(normal).unit()
	bitangent := normal.cross(tangent)
	halfVec = tangent.mul(halfVec.x).add(bitangent.mul(halfVec.y)).add(normal.mul(halfVec.z)).unit()
	return reflectVec(viewDir.mul(-1), halfVec).unit()
}

func estimateDirectLightCPU(world *cpuRenderWorld, surfHit hitRecord, albedo vec3, rng *randSource) vec3 {
	lightCount := len(world.lights)
	if lightCount == 0 {
		return vec3{}
	}

	lightsToSample := lightCount
	start := 0
	scaleFactor := renderUnitValue
	if lightCount > cpuMaxLightSamples {
		lightsToSample = cpuMaxLightSamples
		start = int(rng.Float64()*float64(lightCount)) % lightCount
		scaleFactor = float64(lightCount) / float64(lightsToSample)
	}

	total := vec3{}
	for j := range lightsToSample {
		lightIdx := (start + j) % lightCount
		contrib := estimateDirectLightSingleCPU(world, &world.lights[lightIdx], surfHit, albedo, rng)
		total = total.add(contrib)
	}
	return total.mul(scaleFactor / float64(lightCount))
}

type cpuAreaLightSample struct {
	position vec3
	normal   vec3
	pdfArea  float64
}

func sampleCPUAreaLight(light *cpuLight, rng *randSource) (cpuAreaLightSample, bool) {
	if !light.valid || light.radius <= 0 {
		return cpuAreaLightSample{}, false
	}
	u1 := rng.Float64()
	u2 := rng.Float64()
	z := 1 - 2*u1
	radial := math.Sqrt(maxFloat(1-z*z, 0))
	phi := 2 * cpuPi * u2
	normal := v(radial*math.Cos(phi), radial*math.Sin(phi), z).unit()
	return cpuAreaLightSample{
		position: light.center.add(normal.mul(light.radius)),
		normal:   normal,
		pdfArea:  1 / (4 * cpuPi * light.radius * light.radius),
	}, true
}

func estimateDirectLightSingleCPU(
	world *cpuRenderWorld,
	light *cpuLight,
	surfHit hitRecord,
	albedo vec3,
	rng *randSource,
) vec3 {
	sample, ok := sampleCPUAreaLight(light, rng)
	if !ok {
		return vec3{}
	}

	toLight := sample.position.sub(surfHit.p)
	distSq := toLight.dot(toLight)
	if distSq <= renderRoughnessThreshold {
		return vec3{}
	}
	dist := math.Sqrt(distSq)
	wi := toLight.div(dist)
	if !directLightVisibleCPU(world, surfHit.p, surfHit.normal, wi, dist) {
		return vec3{}
	}

	cosSurf := maxFloat(surfHit.normal.dot(wi), 0)
	cosLight := maxFloat(sample.normal.dot(wi.mul(-1)), 0)
	if cosSurf <= 0 || cosLight <= 0 {
		return vec3{}
	}

	geometry := cosSurf * cosLight / maxFloat(distSq, renderRoughnessThreshold)
	lightPdf := pdfLightDirectionCPU(toLight, light.radius, surfHit.normal)
	bsdfPdf := cosSurf * cpuInvPi
	misWeight := 1.0
	if lightPdf > cpuDirectPDFCutoff && bsdfPdf > cpuDirectPDFCutoff {
		misWeight = lightPdf / (lightPdf + bsdfPdf)
	}
	contrib := albedo.mul(cpuInvPi)
	contrib = mulVec3(contrib, light.emit).mul(
		geometry / maxFloat(sample.pdfArea, renderRoughnessThreshold) * misWeight,
	)
	return clampLuminanceCPU(contrib, cpuDirectMaxLuminance)
}

func directLightVisibleCPU(world *cpuRenderWorld, point, normal, direction vec3, distance float64) bool {
	shadowRay := ray{orig: point.add(normal.mul(renderMinRayDistance)), dir: direction}
	var shadowHit hitRecord
	return !hitClosestCPU(world, shadowRay, distance-cpuShadowDistanceOffset, -1, &shadowHit)
}

func pdfLightDirectionCPU(toLight vec3, radius float64, normal vec3) float64 {
	distSq := toLight.dot(toLight)
	if distSq < cpuDirectPDFCutoff {
		return 0
	}
	wi := toLight.div(math.Sqrt(distSq))
	cosLight := maxFloat(normal.dot(wi), 0)
	if cosLight <= 0 || radius <= 0 {
		return 0
	}
	pdfArea := 1 / (4 * cpuPi * radius * radius)
	return pdfArea / maxFloat(cosLight/distSq, cpuDirectPDFCutoff)
}

func glassExitDistanceCPU(world *cpuRenderWorld, rec hitRecord, direction vec3) float64 {
	if rec.objectIdx < 0 || rec.objectIdx >= len(world.objects) {
		return 0
	}
	object := &world.objects[rec.objectIdx]
	exitRay := ray{orig: rec.p.add(direction.mul(renderMinRayDistance)), dir: direction}
	switch object.kind {
	case hittableSphere:
		oc := exitRay.orig.sub(object.sphere.center)
		a := exitRay.dir.dot(exitRay.dir)
		halfB := oc.dot(exitRay.dir)
		c := oc.dot(oc) - object.sphere.radius*object.sphere.radius
		disc := halfB*halfB - a*c
		if disc <= 0 {
			return 0
		}
		sqrtDisc := math.Sqrt(disc)
		root := (-halfB + sqrtDisc) / a
		if root > renderMinRayDistance && root < world.maxRayDist {
			return root
		}
	case hittableBox:
		var exit hitRecord
		if object.box.hitWithMode(exitRay, renderMinRayDistance, world.maxRayDist, &exit, true) {
			return exit.t
		}
	default:
		return 0
	}
	return 0
}

func dielectricAttenuation(mat *material, distance float64) vec3 {
	absorb := v(
		math.Exp(-mat.absorption.x*mat.absorptionScale*distance),
		math.Exp(-mat.absorption.y*mat.absorptionScale*distance),
		math.Exp(-mat.absorption.z*mat.absorptionScale*distance),
	)
	base := v(renderRoughnessScale, renderRoughnessScale, renderRoughnessScale)
	return mulVec3(base.add(absorb.mul(cpuFogMaxAnisotropy)), mat.tint)
}

func estimatePrimaryVolume(primary ray, world *cpuRenderWorld, rng *randSource) vec3 {
	var firstHit hitRecord
	tMax := float64(cpuVolumeMaxDistance)
	if hitClosestCPU(world, primary, tMax, -1, &firstHit) {
		tMax = firstHit.t
	}
	if tMax <= 0 {
		return vec3{}
	}
	step := tMax / cpuVolumeSteps
	result := vec3{}
	for i := range cpuVolumeSteps {
		t := (float64(i) + renderHalf) * step
		pos := primary.at(t)
		sigmaS, _, sigmaT := world.fog.mediumCoefficients(pos)
		if sigmaT <= 0 || sigmaS <= 0 {
			continue
		}
		transmittance := math.Exp(-sigmaT * t)
		light := estimateVolumeLightCPU(pos, primary.dir, world, rng)
		result = result.add(mulVec3(world.fog.color, light).mul(sigmaS * transmittance * step))
	}
	return result
}

func (fog cpuFog) mediumCoefficients(pos vec3) (sigmaS, sigmaA, sigmaT float64) {
	sigmaS = fog.sigmaS
	sigmaA = fog.sigmaA
	if sigmaS <= 0 && sigmaA <= 0 && fog.density > 0 {
		sigmaS = fog.density * fog.scatter
		sigmaA = maxFloat(fog.density-sigmaS, 0)
	}
	sigmaT = sigmaS + sigmaA
	if sigmaT <= 0 {
		return 0, 0, 0
	}
	if fog.heteroStrength > 0 {
		scale := (1 - fog.heteroStrength) + 2*fog.heteroStrength*volumeNoiseCPU(pos, fog)
		sigmaS *= scale
		sigmaA *= scale
		sigmaT = sigmaS + sigmaA
	}
	return sigmaS, sigmaA, sigmaT
}

func estimateVolumeLightCPU(pos, viewDir vec3, world *cpuRenderWorld, rng *randSource) vec3 {
	if world.fog.scatter <= 0 {
		return vec3{}
	}
	sum := vec3{}
	for i := range world.lights {
		light := &world.lights[i]
		if !light.valid || !hasPositiveComponent(light.emit) {
			continue
		}
		radius := light.radius
		u1 := rng.Float64()
		u2 := rng.Float64()
		z := 1 - 2*u1
		radial := math.Sqrt(maxFloat(1-z*z, 0))
		phi := 2 * cpuPi * u2
		lightNormal := v(radial*math.Cos(phi), radial*math.Sin(phi), z).unit()
		lightPos := light.center.add(lightNormal.mul(radius))
		pdfArea := 1 / (4 * cpuPi * radius * radius)
		toLight := lightPos.sub(pos)
		distSq := toLight.dot(toLight)
		if distSq <= renderRoughnessThreshold {
			continue
		}
		dist := math.Sqrt(distSq)
		wi := toLight.div(dist)
		if !directLightVisibleCPU(world, pos, vec3{}, wi, dist) {
			continue
		}
		cosLight := maxFloat(lightNormal.dot(wi.mul(-1)), 0)
		if cosLight <= 0 {
			continue
		}
		phase := phaseHGCPU(wi.mul(-1).dot(viewDir), world.fog.g)
		contrib := light.emit.mul(
			cosLight / maxFloat(distSq, renderRoughnessThreshold) * phase / maxFloat(pdfArea, renderRoughnessThreshold),
		)
		sum = sum.add(contrib)
	}
	return clampLuminanceCPU(sum.mul(2), cpuDirectMaxLuminance)
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
	return (1 - gg) * cpuInvPi * renderHalf / (maxFloat(denom, renderRoughnessThreshold) * math.Sqrt(maxFloat(denom, renderRoughnessThreshold)))
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

func clampLuminanceCPU(value vec3, maxLuminance float64) vec3 {
	luminance := value.x*cpuLumaR + value.y*cpuLumaG + value.z*cpuLumaB
	if luminance > maxLuminance {
		return value.mul(maxLuminance / maxFloat(luminance, renderRoughnessThreshold))
	}
	return value
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

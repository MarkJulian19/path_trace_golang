package render

type cpuPath struct {
	r             ray
	throughput    vec3
	radiance      vec3
	previousPoint vec3
	previousPDF   float64
	etaScale      float64
	medium        int
}

func (path *cpuPath) traceBounce(world *cpuRenderWorld, background func(ray) vec3, bounce int,
	rng *randSource, continuation bool,
) bool {
	if maxComponent(path.throughput) <= 0 {
		return false
	}
	var rec hitRecord
	found := hitClosestCPU(world, path.r, world.maxRayDist, &rec)
	distance := world.maxRayDist
	if found {
		distance = rec.t
	}
	if path.medium < 0 && world.fog.volumetric && (found || world.fog.affectSky) {
		if path.scatterFog(world, distance, rng, continuation) {
			return path.roulette(bounce, rng)
		}
	}
	if path.medium >= 0 {
		glass := objectMaterialCPU(&world.objects[path.medium])
		path.throughput = mulVec3(path.throughput, dielectricAttenuation(glass, distance))
	}
	if !found {
		path.radiance = path.radiance.add(cpuMissRadiance(path.r, world, background, path.throughput))
		return false
	}
	return path.scatterSurface(world, rec, bounce, rng, continuation)
}

func (path *cpuPath) scatterSurface(world *cpuRenderWorld, rec hitRecord, bounce int,
	rng *randSource, continuation bool,
) bool {
	if rec.mat.typ == matEmissive {
		if rec.frontFace {
			weight := emissionWeightCPU(world, rec, path.previousPoint, path.previousPDF)
			path.radiance = path.radiance.add(mulVec3(path.throughput, rec.mat.emit).mul(weight))
		}
		return false
	}
	result := scatterCPUSurface(path.r, world, rec, rng, path.medium, continuation)
	path.radiance = path.radiance.add(mulVec3(path.throughput, result.radiance))
	path.previousPDF = result.pdf
	path.previousPoint = rec.p
	if result.etaScale > 0 {
		path.etaScale *= result.etaScale
	}
	var next bool
	path.medium, next = advanceCPUPath(&path.r, &path.throughput, rec, result, bounce, path.etaScale, rng)
	return next
}

func (path *cpuPath) scatterFog(world *cpuRenderWorld, distance float64, rng *randSource, continuation bool) bool {
	t, weight, scattered := sampleFogDistanceCPU(path.r, distance, world.fog, rng)
	if !scattered {
		return false
	}
	point := path.r.at(t)
	path.throughput = mulVec3(path.throughput, weight)
	light := estimateFogDirectCPU(point, path.r.dir, world, rng, continuation)
	path.radiance = path.radiance.add(mulVec3(path.throughput, light))
	direction := samplePhaseHGCPU(path.r.dir, world.fog.g, rng)
	path.previousPDF = phaseHGCPU(path.r.dir.dot(direction), world.fog.g)
	path.previousPoint = point
	path.r = ray{orig: point, dir: direction}
	return true
}

func (path *cpuPath) roulette(bounce int, rng *randSource) bool {
	var next bool
	path.throughput, next = applyCPURussianRoulette(path.throughput, bounce, path.etaScale, rng)
	return next
}

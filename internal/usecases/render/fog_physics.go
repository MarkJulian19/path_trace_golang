package render

import "math"

const hgInverseCDFConstant = 3

func (fog cpuFog) majorant() float64 {
	extinction := max(0, fog.sigmaS) + max(0, fog.sigmaA)
	if extinction <= 0 {
		extinction = max(0, fog.density)
	}
	return extinction * (1 + fog.heteroStrength)
}

func fogTransmittanceCPU(r ray, distance float64, fog cpuFog, rng *randSource) float64 {
	majorant := fog.majorant()
	if !fog.volumetric || majorant <= 0 {
		return 1
	}
	if fog.heteroStrength <= 0 {
		return math.Exp(-majorant * distance)
	}
	t, weight := float64(0), renderUnitValue
	for {
		t += -math.Log1p(-rng.Float64()) / majorant
		if t >= distance {
			return weight
		}
		_, extinction := fog.mediumCoefficients(r.at(t))
		weight *= max(0, 1-extinction/majorant)
		if weight <= 0 {
			return 0
		}
	}
}

func sampleFogDistanceCPU(r ray, distance float64, fog cpuFog, rng *randSource) (float64, vec3, bool) {
	majorant := fog.majorant()
	if majorant <= 0 {
		return 0, vec3{}, false
	}
	var t float64
	for {
		t += -math.Log1p(-rng.Float64()) / majorant
		if t >= distance {
			return 0, vec3{}, false
		}
		scattering, extinction := fog.mediumCoefficients(r.at(t))
		if rng.Float64()*majorant < extinction {
			return t, fog.color.mul(scattering / extinction), true
		}
	}
}

func samplePhaseHGCPU(axis vec3, g float64, rng *randSource) vec3 {
	u := 2*rng.Float64() - 1
	denominator := 1 + g*u
	// Algebraic inverse HG CDF without cancellation or an approximation near g=0.
	numerator := u + renderHalf*g*(u*u+hgInverseCDFConstant) + g*g*u + renderHalf*g*g*g*(u*u-1)
	cosine := clampFloat(numerator/(denominator*denominator), -1, 1)
	phi := 2 * math.Pi * rng.Float64()
	sine := math.Sqrt(max(0, 1-cosine*cosine))
	tangentAxis := v(1, 0, 0)
	if math.Abs(axis.x) > cpuFogMaxAnisotropy {
		tangentAxis = v(0, 1, 0)
	}
	tangent := tangentAxis.cross(axis).unit()
	return tangent.mul(sine * math.Cos(phi)).
		add(axis.cross(tangent).mul(sine * math.Sin(phi))).
		add(axis.mul(cosine)).
		unit()
}

func estimateFogDirectCPU(point, incident vec3, world *cpuRenderWorld, rng *randSource, continuation bool) vec3 {
	count := len(world.lights)
	if count == 0 {
		return vec3{}
	}
	samples := min(count, cpuMaxLightSamples)
	start := 0
	if count > samples {
		start = int(rng.Float64() * float64(count))
	}
	sum := vec3{}
	for i := range samples {
		light := &world.lights[(start+i)%count]
		sum = sum.add(fogLightContributionCPU(point, incident, world, light, rng, continuation))
	}
	return sum
}

func fogLightContributionCPU(point, incident vec3, world *cpuRenderWorld, light *cpuLight,
	rng *randSource, continuation bool,
) vec3 {
	sample, ok := sampleCPULight(light, point, rng)
	if !ok {
		return vec3{}
	}
	delta := sample.position.sub(point)
	distance := delta.length()
	if distance <= 0 {
		return vec3{}
	}
	wi := delta.div(distance)
	cosine := max(0, sample.normal.dot(wi.mul(-1)))
	if cosine <= 0 || !directLightVisibleCPU(world, point, vec3{}, wi, distance, light.sceneIdx) {
		return vec3{}
	}
	lightPDF := sample.pdfDirection
	if lightPDF <= 0 {
		lightPDF = sample.pdfArea * distance * distance / cosine
	}
	lightPDF *= min(1, float64(cpuMaxLightSamples)/float64(len(world.lights)))
	phase := phaseHGCPU(incident.dot(wi), world.fog.g)
	denominator := lightPDF
	if continuation {
		denominator += phase
	}
	transmittance := fogTransmittanceCPU(ray{orig: point, dir: wi}, distance, world.fog, rng)
	return light.emit.mul(phase / denominator * transmittance)
}

func shadowTransmittanceCPU(world *cpuRenderWorld, r ray, distance float64, rng *randSource) vec3 {
	medium := mediumAtCPU(world, r, -1)
	if medium >= 0 {
		return dielectricAttenuation(objectMaterialCPU(&world.objects[medium]), distance)
	}
	value := fogTransmittanceCPU(r, distance, world.fog, rng)
	return v(value, value, value)
}

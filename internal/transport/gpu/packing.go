package gpu

import (
	"image"

	renderentity "github.com/MarkJulian19/path_trace_golang/internal/entities/render"
	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

const (
	gpuMaterialStride = 20
	gpuObjectStride   = 12
)

type gpuSceneData struct {
	matData, objData []float32
	lightIndices     []int32
	camBlock         [16]float32
	skyBlock         [16]float32
	fogBlock         [16]float32
}

func clearImageWithoutMaterials(img *image.RGBA, progress func(currentSample, totalSamples int)) {
	for i := range img.Pix {
		if i%gpuRGBAComponents == gpuRGBComponents {
			img.Pix[i] = 255
			continue
		}
		img.Pix[i] = 0
	}
	if progress != nil {
		progress(1, 1)
	}
}

func prepareGPUScene(sc *scene.Scene, cfg renderentity.Config) gpuSceneData {
	return gpuSceneData{
		matData:      packMaterials(sc.Materials),
		objData:      packObjects(sc.Materials, sc.Objects),
		lightIndices: findLightIndices(sc.Materials, sc.Objects),
		camBlock:     packCamera(sc.Camera, cfg),
		skyBlock:     packSky(sc),
		fogBlock:     packFogBlock(sc),
	}
}

func packMaterials(materials []scene.Material) []float32 {
	data := make([]float32, len(materials)*gpuMaterialStride)
	for i, material := range materials {
		base := i * gpuMaterialStride
		data[base] = gpuMaterialType(material.Type)
		data[base+1] = float32(material.Rough)
		data[base+2] = float32(material.IOR)
		data[base+3] = float32(materialSmoothness(material))
		data[base+4] = float32(material.Albedo.R)
		data[base+5] = float32(material.Albedo.G)
		data[base+6] = float32(material.Albedo.B)
		data[base+7] = float32(materialReflectivity(material))
		data[base+8] = float32(material.Emit.R * material.Power)
		data[base+9] = float32(material.Emit.G * material.Power)
		data[base+10] = float32(material.Emit.B * material.Power)
		data[base+12] = float32(material.Absorption.R)
		data[base+13] = float32(material.Absorption.G)
		data[base+14] = float32(material.Absorption.B)
		data[base+15] = float32(materialAbsorptionScale(material))
		tintR, tintG, tintB := materialTint(material)
		data[base+16] = float32(tintR)
		data[base+17] = float32(tintG)
		data[base+18] = float32(tintB)
	}
	return data
}

func gpuMaterialType(materialType scene.MaterialType) float32 {
	switch materialType {
	case scene.MaterialLambert:
		return float32(gpuMatLambert)
	case scene.MaterialMetal:
		return float32(gpuMatMetal)
	case scene.MaterialDielectric:
		return float32(gpuMatDielectric)
	case scene.MaterialEmissive:
		return float32(gpuMatEmissive)
	case scene.MaterialMirror:
		return float32(gpuMatMirror)
	default:
		return float32(gpuMatLambert)
	}
}

func materialSmoothness(material scene.Material) float64 {
	value := material.Smoothness
	if value == 0 && material.Type == scene.MaterialMetal {
		value = gpuUnitValue - material.Rough
	}
	return clampFloat(value, 0, gpuUnitValue)
}

func materialReflectivity(material scene.Material) float64 {
	value := material.Reflectivity
	if value == 0 && material.Type == scene.MaterialMetal {
		value = gpuUnitValue
	}
	return clampFloat(value, 0, gpuUnitValue)
}

func materialAbsorptionScale(material scene.Material) float64 {
	if material.AbsorptionScale == 0 && material.Type == scene.MaterialDielectric {
		return gpuAbsorptionScaleDefault
	}
	return material.AbsorptionScale
}

func materialTint(material scene.Material) (float64, float64, float64) {
	tintR, tintG, tintB := material.Tint.R, material.Tint.G, material.Tint.B
	if tintR == 0 && tintG == 0 && tintB == 0 && material.Type == scene.MaterialDielectric {
		return gpuUnitValue, gpuUnitValue, gpuUnitValue
	}
	return tintR, tintG, tintB
}

func packObjects(materials []scene.Material, objects []scene.Object) []float32 {
	materialIndex := make(map[string]int, len(materials))
	for i, material := range materials {
		materialIndex[material.ID] = i
	}
	data := make([]float32, len(objects)*gpuObjectStride)
	for i, object := range objects {
		base := i * gpuObjectStride
		data[base] = gpuObjectType(object.Type)
		data[base+1] = float32(materialIndex[object.MaterialID])
		data[base+4] = float32(object.Position.X)
		data[base+5] = float32(object.Position.Y)
		data[base+6] = float32(object.Position.Z)
		data[base+8] = float32(object.Size.X)
		data[base+9] = float32(object.Size.Y)
		data[base+10] = float32(object.Size.Z)
	}
	return data
}

func gpuObjectType(objectType scene.ObjectType) float32 {
	switch objectType {
	case scene.ObjectSphere, scene.ObjectSphereLight:
		return float32(gpuObjSphere)
	case scene.ObjectPlane:
		return float32(gpuObjPlane)
	case scene.ObjectBox:
		return float32(gpuObjBox)
	default:
		return float32(gpuObjSphere)
	}
}

func findLightIndices(materials []scene.Material, objects []scene.Object) []int32 {
	materialIndex := make(map[string]int, len(materials))
	for i, material := range materials {
		materialIndex[material.ID] = i
	}
	indices := make([]int32, 0, len(objects))
	for i, object := range objects {
		material, ok := materialAt(materials, materialIndex[object.MaterialID])
		if ok && isEmissive(material) {
			indices = append(indices, int32(i))
		}
	}
	return indices
}

func materialAt(materials []scene.Material, index int) (scene.Material, bool) {
	if index < 0 || index >= len(materials) {
		return scene.Material{}, false
	}
	return materials[index], true
}

func isEmissive(material scene.Material) bool {
	return material.Type == scene.MaterialEmissive &&
		(material.Emit.R > 0 || material.Emit.G > 0 || material.Emit.B > 0)
}

func packCamera(camera scene.Camera, cfg renderentity.Config) [16]float32 {
	aspect := float32(cfg.Width) / float32(cfg.Height)
	if camera.AspectRatio != 0 {
		aspect = float32(camera.AspectRatio)
	}
	return [16]float32{
		float32(camera.Position.X), float32(camera.Position.Y), float32(camera.Position.Z), 0,
		float32(camera.Target.X), float32(camera.Target.Y), float32(camera.Target.Z), 0,
		float32(camera.Up.X), float32(camera.Up.Y), float32(camera.Up.Z), 0,
		float32(camera.FOV), float32(camera.Aperture), float32(camera.FocusDist), aspect,
	}
}

func packSky(sc *scene.Scene) [16]float32 {
	var skyType int
	var color, horizon, zenith [4]float32
	switch {
	case sc.Sky == nil:
		color = [4]float32{float32(sc.Background.R), float32(sc.Background.G), float32(sc.Background.B), 1}
	case sc.Sky.Type == "gradient":
		skyType = 2
		horizon = [4]float32{float32(sc.Sky.Horizon.R), float32(sc.Sky.Horizon.G), float32(sc.Sky.Horizon.B), 1}
		zenith = [4]float32{float32(sc.Sky.Zenith.R), float32(sc.Sky.Zenith.G), float32(sc.Sky.Zenith.B), 1}
	default:
		skyType = 1
		color = [4]float32{float32(sc.Sky.Color.R), float32(sc.Sky.Color.G), float32(sc.Sky.Color.B), 1}
	}
	return [16]float32{
		float32(skyType), 0, 0, 0,
		color[0], color[1], color[2], color[3],
		horizon[0], horizon[1], horizon[2], horizon[3],
		zenith[0], zenith[1], zenith[2], zenith[3],
	}
}

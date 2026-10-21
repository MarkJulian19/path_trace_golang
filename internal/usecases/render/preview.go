package render

import (
	"image"
	"sync"
)

const (
	previewAmbient       = 0.6
	previewShadow        = 0.4
	previewEmissionScale = 3
	previewMetalScale    = 1.24
	previewLightX        = 0.4472136
	previewLightY        = 0.8944272
	previewLightZ        = 0.2683282
	previewGlassRed      = 0.85
)

// cpuFramePublication передаёт только законченные плитки и защищает снимки
// изображения внутри progress от записей остальных работников.
type cpuFramePublication struct {
	target   *image.RGBA
	progress func(int, int)
	pixels   []byte
	cfg      Config
	mu       sync.Mutex
	done     int
}

func (p *cpuFramePublication) commit(tile renderTile) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for y := tile.y0; y < tile.y1; y++ {
		start := y*p.target.Stride + tile.x0*renderRGBAComponents
		end := y*p.target.Stride + tile.x1*renderRGBAComponents
		copy(p.target.Pix[start:end], p.pixels[start:end])
	}
	p.done += (tile.x1 - tile.x0) * (tile.y1 - tile.y0)
	current := int(int64(p.done) * int64(p.cfg.SamplesPerPx) / int64(p.cfg.Width*p.cfg.Height))
	p.progress(current, p.cfg.SamplesPerPx)
}

func wireframeColor(rec *hitRecord) vec3 {
	brightness := previewAmbient
	if rec.normal.y > renderUnitValue-renderRoughnessScale {
		brightness = 1
	} else if rec.normal.y < renderRoughnessScale-renderUnitValue {
		brightness = previewShadow
	}
	return wireframeSurfaceColor(rec, brightness)
}

func wireframeSurfaceColor(rec *hitRecord, brightness float64) vec3 {
	col := rec.mat.albedo.mul(brightness)
	switch rec.mat.typ {
	case matEmissive:
		col = rec.mat.emit.mul(previewEmissionScale)
	case matDielectric:
		tint := v(previewGlassRed, renderUnitValue-renderRoughnessScale, renderEdgeThreshold)
		col = col.mul(previewAmbient).add(tint.mul(brightness * previewShadow))
	case matMetal, matMirror:
		col = col.mul(previewMetalScale)
	default:
	}
	return v(clamp01(col.x), clamp01(col.y), clamp01(col.z))
}

func previewRayColor(r ray, world *cpuRenderWorld) vec3 {
	var rec hitRecord
	if !hitClosestCPU(world, r, world.maxRayDist, -1, &rec) {
		return v(renderRoughnessScale, renderRoughnessScale, renderRoughnessScale)
	}
	if world.objects[rec.objectIdx].kind == hittableSphere {
		light := v(previewLightX, previewLightY, previewLightZ)
		brightness := renderHalf + max(0, rec.normal.dot(light))*renderHalf
		return wireframeSurfaceColor(&rec, brightness)
	}
	return wireframeColor(&rec)
}

func writePreviewPixel(pixels []byte, index int, color vec3) {
	pixels[index] = uint8(clamp01(color.x)*renderColorMax + renderHalf)
	pixels[index+1] = uint8(clamp01(color.y)*renderColorMax + renderHalf)
	pixels[index+2] = uint8(clamp01(color.z)*renderColorMax + renderHalf)
	pixels[index+3] = 255
}

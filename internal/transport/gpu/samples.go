package gpu

import (
	"image"
	"time"

	"github.com/go-gl/gl/v3.3-core/gl"
)

type gpuSampleRequest struct {
	cfg                                        configForSamples
	img                                        *image.RGBA
	tmp                                        []float32
	pixelCount, passes, updateInterval         int
	isWireframe, wantLiveReadback              bool
	locSpp, locSampleCount, locSeed, locObject int32
	progress                                   func(currentSample, totalSamples int)
}

type configForSamples struct {
	width, height int
}

type gpuSampleStats struct {
	lastReadback bool
	dispatchTime time.Duration
	readbackTime time.Duration
	readbacks    int
}

func (r *gpuRenderer) renderSamples(request gpuSampleRequest) gpuSampleStats {
	SetRenderCancel(false)
	stats := gpuSampleStats{}
	for sample := range request.passes {
		if IsRenderCancelled() {
			gpuLogf("cancelled before sample %d/%d", sample, request.passes)
			break
		}
		stats.dispatchTime += r.dispatchSample(request, sample)
		isLast := sample+1 == request.passes
		if shouldReadSample(request, sample, isLast) {
			lastReadback, elapsed := r.readSample(request, sample, isLast)
			stats.lastReadback = lastReadback
			stats.readbackTime += elapsed
			stats.readbacks++
		}
		if !request.isWireframe && r.prevFrameTexture != 0 && TemporalEnabled() {
			r.copyFrameToPrev(request.cfg.width, request.cfg.height)
		}
		if request.progress != nil {
			request.progress(sample+1, request.passes)
		}
		if IsRenderCancelled() {
			gpuLogf("cancelled after sample %d/%d", sample+1, request.passes)
			break
		}
	}
	return stats
}

func (r *gpuRenderer) dispatchSample(request gpuSampleRequest, sample int) time.Duration {
	if request.isWireframe {
		gl.Uniform1i(request.locSpp, 1)
		gl.Uniform1i(request.locSampleCount, 1)
		gl.Uniform1ui(request.locSeed, glSeed(time.Now().UnixNano()))
	} else {
		gl.Uniform1i(request.locSpp, 1)
		gl.Uniform1i(request.locSampleCount, glInt32(sample+1))
		gl.Uniform1ui(request.locSeed, glSeed(time.Now().UnixNano()+int64(sample)))
	}
	gl.Uniform1i(request.locObject, glInt32(GetSelectedObject()))
	start := time.Now()
	groupsX := (request.cfg.width + gpuWorkgroupOffset) / gpuWorkgroupSize
	groupsY := (request.cfg.height + gpuWorkgroupOffset) / gpuWorkgroupSize
	gl.DispatchCompute(glUint32(groupsX), glUint32(groupsY), 1)
	gl.MemoryBarrier(gl.SHADER_IMAGE_ACCESS_BARRIER_BIT)
	if !gpuDebugEnabled() {
		return 0
	}
	return time.Since(start)
}

func shouldReadSample(request gpuSampleRequest, sample int, isLast bool) bool {
	return request.isWireframe ||
		(request.wantLiveReadback && ((sample+1)%request.updateInterval == 0 || isLast))
}

func (r *gpuRenderer) readSample(request gpuSampleRequest, sample int, isLast bool) (bool, time.Duration) {
	start := time.Now()
	if request.isWireframe {
		r.readWireframe(request.tmp)
	} else {
		r.readPathSample(request.tmp, sample, isLast)
	}
	actualPixelCount := min(len(request.img.Pix)/gpuRGBAComponents, request.pixelCount)
	if request.isWireframe {
		writeWireframePixels(request.img.Pix, request.tmp, actualPixelCount)
	} else {
		tonemapHDRToRGBA(request.img.Pix, request.tmp, actualPixelCount)
	}
	return isLast || !request.wantLiveReadback, time.Since(start)
}

func (r *gpuRenderer) readWireframe(tmp []float32) {
	gl.MemoryBarrier(gl.SHADER_IMAGE_ACCESS_BARRIER_BIT | gl.TEXTURE_FETCH_BARRIER_BIT)
	gl.BindBuffer(gl.PIXEL_PACK_BUFFER, r.pbo)
	gl.BindTexture(gl.TEXTURE_2D, r.imgTexture)
	gl.GetTexImage(gl.TEXTURE_2D, 0, gl.RGBA, gl.FLOAT, nil)
	gl.MemoryBarrier(gl.PIXEL_BUFFER_BARRIER_BIT)
	readMappedPixels(tmp, gpuWireframeFallback, true)
	gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)
}

func (r *gpuRenderer) readPathSample(tmp []float32, sample int, isLast bool) {
	currentPBO := r.pbo
	if r.pboIndex == 1 {
		currentPBO = r.pboBack
	}
	if sample > 0 {
		readPBO := r.pboBack
		if r.pboIndex == 1 {
			readPBO = r.pbo
		}
		gl.BindBuffer(gl.PIXEL_PACK_BUFFER, readPBO)
		readMappedPixels(tmp, 0, false)
	}
	gl.BindBuffer(gl.PIXEL_PACK_BUFFER, currentPBO)
	gl.BindTexture(gl.TEXTURE_2D, r.imgTexture)
	gl.GetTexImage(gl.TEXTURE_2D, 0, gl.RGBA, gl.FLOAT, nil)
	if isLast {
		gl.MemoryBarrier(gl.PIXEL_BUFFER_BARRIER_BIT)
		readMappedPixels(tmp, 0, false)
	}
	r.pboIndex = 1 - r.pboIndex
	gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)
}

func readMappedPixels(tmp []float32, fallback float32, fillFallback bool) {
	ptr := gl.MapBuffer(gl.PIXEL_PACK_BUFFER, gl.READ_ONLY)
	if ptr != nil {
		src := (*[1 << 28]float32)(ptr)[:len(tmp)]
		copy(tmp, src)
		gl.UnmapBuffer(gl.PIXEL_PACK_BUFFER)
		return
	}
	if !fillFallback {
		return
	}
	for i := 0; i < len(tmp); i += gpuRGBAComponents {
		if i+3 < len(tmp) {
			tmp[i], tmp[i+1], tmp[i+2], tmp[i+3] = fallback, fallback, fallback, gpuUnitValue
		}
	}
}

func writeWireframePixels(dst []byte, src []float32, pixelCount int) {
	for i := range pixelCount {
		offset := i * gpuRGBAComponents
		if offset+3 >= len(dst) || offset+3 >= len(src) {
			break
		}
		dst[offset] = uint8(clampFloat(float64(src[offset]), 0, 1)*gpuByteMax + gpuHalf)
		dst[offset+1] = uint8(clampFloat(float64(src[offset+1]), 0, 1)*gpuByteMax + gpuHalf)
		dst[offset+2] = uint8(clampFloat(float64(src[offset+2]), 0, 1)*gpuByteMax + gpuHalf)
		dst[offset+3] = 255
	}
}

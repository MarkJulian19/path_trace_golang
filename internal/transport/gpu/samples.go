package gpu

import (
	"image"
	"time"

	"github.com/go-gl/gl/v3.3-core/gl"

	renderentity "github.com/MarkJulian19/path_trace_golang/internal/entities/render"
)

type gpuSampleRequest struct {
	img *image.RGBA

	cancelled func() bool
	progress  func(currentSample, totalSamples int)
	tmp       []float32
	cfg       configForSamples

	pixelCount     int
	passes         int
	updateInterval int

	locSpp            int32
	locSampleCount    int32
	locSeed           int32
	locObject         int32
	locDispatchOffset int32
	isWireframe       bool
	wantLiveReadback  bool
}

type configForSamples struct {
	width, height int
}

type gpuSampleStats struct {
	dispatchTime time.Duration
	readbackTime time.Duration
	readbacks    int
	lastReadback bool
	cancelled    bool
}

func newGPUSampleRequest(
	state gpuFrameState, cfg renderentity.Config, img *image.RGBA,
	progress func(int, int), cancelled func() bool,
) gpuSampleRequest {
	return gpuSampleRequest{
		cfg:               configForSamples{width: cfg.Width, height: cfg.Height},
		img:               img,
		tmp:               state.tmp,
		pixelCount:        state.pixelCount,
		passes:            state.passes,
		updateInterval:    state.updateInterval,
		isWireframe:       state.isWireframe,
		wantLiveReadback:  state.wantLiveReadback,
		locSpp:            state.uniforms.locSpp,
		locSampleCount:    state.uniforms.locSampleCount,
		locSeed:           state.uniforms.locSeed,
		locObject:         state.uniforms.locSelectedObject,
		locDispatchOffset: state.uniforms.locDispatchOffset,
		progress:          progress,
		cancelled:         cancelled,
	}
}

func (r *gpuRenderer) renderSamples(request gpuSampleRequest) gpuSampleStats {
	if !beginSampleRender(request) {
		return gpuSampleStats{cancelled: true}
	}
	stats := gpuSampleStats{}
	for sample := range request.passes {
		if sampleCancelled(request) {
			stats.cancelled = true
			gpuLogf("cancelled before sample %d/%d", sample, request.passes)
			break
		}
		stats.dispatchTime += r.dispatchSample(request, sample)
		if sampleCancelled(request) {
			stats.cancelled = true
			gpuLogf("cancelled after dispatch %d/%d", sample+1, request.passes)
			break
		}
		if !r.finishSample(request, sample, &stats) {
			stats.cancelled = true
			break
		}
	}
	return stats
}

// beginSampleRender отделяет локальный признак отмены запроса от устаревшего
// глобального флага. Для интерактивного запроса локальный признак имеет приоритет.
func beginSampleRender(request gpuSampleRequest) bool {
	if request.cancelled != nil {
		if request.cancelled() {
			SetRenderCancel(true)
			return false
		}
		SetRenderCancel(false)
		return true
	}
	return !IsRenderCancelled()
}

func (r *gpuRenderer) finishSample(request gpuSampleRequest, sample int, stats *gpuSampleStats) bool {
	isLast := sample+1 == request.passes
	if shouldReadSample(request, sample, isLast) {
		lastReadback, elapsed := r.readSample(request, isLast)
		stats.lastReadback = lastReadback
		stats.readbackTime += elapsed
		stats.readbacks++
	}
	if !isLast && !request.isWireframe && r.prevFrameTexture != 0 && TemporalEnabled() {
		r.swapTemporalTargets()
	}
	if request.progress != nil {
		request.progress(sample+1, request.passes)
	}
	if sampleCancelled(request) {
		gpuLogf("cancelled after sample %d/%d", sample+1, request.passes)
		return false
	}
	return true
}

// swapTemporalTargets avoids copying the full HDR frame after every dispatch.
// The texture written by the current pass becomes the sampled previous frame,
// while the other texture becomes the next output target.
func (r *gpuRenderer) swapTemporalTargets() {
	gl.MemoryBarrier(
		gl.SHADER_IMAGE_ACCESS_BARRIER_BIT |
			gl.SHADER_STORAGE_BARRIER_BIT |
			gl.TEXTURE_FETCH_BARRIER_BIT,
	)
	r.imgTexture, r.prevFrameTexture = r.prevFrameTexture, r.imgTexture
	gl.BindImageTexture(0, r.imgTexture, 0, false, 0, gl.WRITE_ONLY, gl.RGBA16F)
	gl.ActiveTexture(gl.TEXTURE9)
	gl.BindTexture(gl.TEXTURE_2D, r.prevFrameTexture)
}

func (r *gpuRenderer) dispatchSample(request gpuSampleRequest, sample int) time.Duration {
	if request.isWireframe {
		gl.Uniform1i(request.locSpp, 1)
		gl.Uniform1i(request.locSampleCount, 1)
		gl.Uniform1ui(request.locSeed, glSeed(time.Now().UnixNano()))
	} else {
		gl.Uniform1i(request.locSpp, glInt32(samplesPerPass()))
		gl.Uniform1i(request.locSampleCount, glInt32(sample+1))
		gl.Uniform1ui(request.locSeed, glSeed(time.Now().UnixNano()+int64(sample)))
	}
	gl.Uniform1i(request.locObject, glInt32(GetSelectedObject()))
	start := time.Now()
	workgroupSize := gpuWorkgroupSize
	if request.isWireframe {
		workgroupSize = gpuWireframeWorkgroupSize
	}
	tileSize := dispatchTileSize()
	for y := 0; y < request.cfg.height; y += tileSize {
		for x := 0; x < request.cfg.width; x += tileSize {
			if sampleCancelled(request) {
				break
			}
			gl.Uniform2i(request.locDispatchOffset, glInt32(x), glInt32(y))
			groupsX := dispatchGroups(min(tileSize, request.cfg.width-x), workgroupSize)
			groupsY := dispatchGroups(min(tileSize, request.cfg.height-y), workgroupSize)
			gl.DispatchCompute(glUint32(groupsX), glUint32(groupsY), 1)
		}
		if sampleCancelled(request) {
			break
		}
	}
	gl.MemoryBarrier(gl.SHADER_IMAGE_ACCESS_BARRIER_BIT | gl.SHADER_STORAGE_BARRIER_BIT)
	if !gpuDebugEnabled() {
		return 0
	}
	return time.Since(start)
}

func sampleCancelled(request gpuSampleRequest) bool {
	if request.cancelled != nil {
		return request.cancelled()
	}
	return IsRenderCancelled()
}

func dispatchGroups(size, workgroupSize int) int {
	return (size + workgroupSize - 1) / workgroupSize
}

func shouldReadSample(request gpuSampleRequest, sample int, isLast bool) bool {
	return request.isWireframe ||
		(request.wantLiveReadback && ((sample+1)%request.updateInterval == 0 || isLast))
}

func (r *gpuRenderer) readSample(request gpuSampleRequest, isLast bool) (bool, time.Duration) {
	start := time.Now()
	if request.isWireframe {
		r.readWireframe(request.tmp)
	} else {
		r.readPathSample(request.tmp, isLast)
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

func (r *gpuRenderer) readPathSample(tmp []float32, isLast bool) {
	currentPBO := r.pbo
	if r.pboIndex == 1 {
		currentPBO = r.pboBack
	}
	if r.pboPending {
		readPBO := r.pboBack
		if r.pboIndex == 1 {
			readPBO = r.pbo
		}
		gl.BindBuffer(gl.PIXEL_PACK_BUFFER, readPBO)
		readMappedPixels(tmp, 0, false)
	}
	gl.MemoryBarrier(gl.SHADER_IMAGE_ACCESS_BARRIER_BIT | gl.SHADER_STORAGE_BARRIER_BIT)
	gl.BindBuffer(gl.PIXEL_PACK_BUFFER, currentPBO)
	gl.BindTexture(gl.TEXTURE_2D, r.imgTexture)
	gl.GetTexImage(gl.TEXTURE_2D, 0, gl.RGBA, gl.FLOAT, nil)
	if isLast || !r.pboPending {
		gl.MemoryBarrier(gl.PIXEL_BUFFER_BARRIER_BIT)
		readMappedPixels(tmp, 0, false)
	}
	r.pboIndex = 1 - r.pboIndex
	r.pboPending = true
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

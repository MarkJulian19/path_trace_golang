package gpu

import (
	"image"

	"github.com/go-gl/gl/v3.3-core/gl"

	renderentity "github.com/MarkJulian19/path_trace_golang/internal/entities/render"
	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

type gpuUniforms struct {
	locSpp, locSampleCount, locSeed, locSelectedObject int32
}

type gpuFrameState struct {
	pixelCount, passes, updateInterval int
	isWireframe, wantLiveReadback      bool
	tmp                                []float32
	uniforms                           gpuUniforms
}

func (r *gpuRenderer) prepareGPUFrame(
	sc *scene.Scene,
	cfg renderentity.Config,
	img *image.RGBA,
	progress func(int, int),
) (gpuFrameState, bool) {
	pixelCount := cfg.Width * cfg.Height
	if pixelCount <= 0 {
		return gpuFrameState{}, false
	}
	pixelCount = min(pixelCount, len(img.Pix)/gpuRGBAComponents)
	gpuLogf("begin mats=%d objs=%d", len(sc.Materials), len(sc.Objects))
	if len(sc.Materials) == 0 {
		clearImageWithoutMaterials(img, progress)
		return gpuFrameState{}, false
	}
	sceneData := prepareGPUScene(sc, cfg)
	r.resizeGPUResources(cfg, pixelCount)
	r.uploadGPUScene(sceneData)
	isWireframe := GetRenderMode() == 1
	uniforms := r.configureGPUProgram(cfg, isWireframe, pixelCount)
	return gpuFrameState{
		pixelCount:       pixelCount,
		passes:           samplePasses(isWireframe, cfg.SamplesPerPx),
		updateInterval:   sampleUpdateInterval(pixelCount),
		isWireframe:      isWireframe,
		wantLiveReadback: (LivePreviewEnabled() && progress != nil) || isWireframe,
		tmp:              r.ensureReadTmp(pixelCount),
		uniforms:         uniforms,
	}, true
}

func samplePasses(isWireframe bool, samplesPerPixel int) int {
	if isWireframe {
		return 1
	}
	return max(samplesPerPixel, 1)
}

func applyPostprocess(img *image.RGBA, cfg renderentity.Config) {
	width, height := cfg.Width, cfg.Height
	denoise := getDenoiseConfig()
	applyDenoisePostprocess(img, cfg, denoise)
	smooth := getSmoothConfig()
	if shouldApplySmooth(width, height, cfg.SamplesPerPx, smooth) {
		applySmooth(img.Pix, width, height, smooth)
	}
}

func applyDenoisePostprocess(img *image.RGBA, cfg renderentity.Config, denoise denoiseConfig) {
	if denoise.Enabled && cfg.SamplesPerPx < gpuRGBAComponents {
		gpuLogf("skip cpu_denoise for preview spp=%d", cfg.SamplesPerPx)
	}
	if denoise.Enabled && cfg.Width > 2 && cfg.Height > 2 && cfg.SamplesPerPx >= gpuRGBAComponents {
		applyDenoise(img.Pix, cfg.Width, cfg.Height, denoise)
	}
}

func shouldApplySmooth(width, height, samples int, smooth smoothConfig) bool {
	return smooth.Enabled && width > 2 && height > 2 && smooth.Radius > 0 &&
		smooth.Strength > 0 && samples >= gpuRGBAComponents
}

func (r *gpuRenderer) resizeGPUResources(cfg renderentity.Config, pixelCount int) {
	if r.width == cfg.Width && r.height == cfg.Height {
		return
	}
	gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)
	gl.MemoryBarrier(gl.ALL_BARRIER_BITS)
	r.width, r.height = cfg.Width, cfg.Height
	r.resizeTextures(cfg)
	pboSize := cfg.Width * cfg.Height * gpuRGBAComponents * gpuFloatBytes
	r.resizePBOs(pboSize)
	accumSize := pixelCount * gpuRGBComponents * gpuFloatBytes
	varianceSize := pixelCount * gpuVarianceComponents * gpuFloatBytes
	resizeSSBO(r.accumSSBO, accumSize, pixelCount*gpuRGBComponents)
	resizeSSBO(r.varianceSSBO, varianceSize, pixelCount*gpuVarianceComponents)
	gl.MemoryBarrier(gl.ALL_BARRIER_BITS)
	gpuLogf("resized GPU buffers to %dx%d (accum=%dB variance=%dB)", cfg.Width, cfg.Height, accumSize, varianceSize)
}

func (r *gpuRenderer) resizeTextures(cfg renderentity.Config) {
	setTextureSize(r.imgTexture, cfg)
	if r.prevFrameTexture != 0 {
		setTextureSize(r.prevFrameTexture, cfg)
	}
}

func setTextureSize(texture uint32, cfg renderentity.Config) {
	gl.BindTexture(gl.TEXTURE_2D, texture)
	gl.TexImage2D(
		gl.TEXTURE_2D,
		0,
		gl.RGBA16F,
		glInt32(cfg.Width),
		glInt32(cfg.Height),
		0,
		gl.RGBA,
		gl.FLOAT,
		nil,
	)
	clearTexture(texture)
}

func (r *gpuRenderer) resizePBOs(size int) {
	gl.BindBuffer(gl.PIXEL_PACK_BUFFER, r.pbo)
	gl.BufferData(gl.PIXEL_PACK_BUFFER, size, nil, gl.STREAM_READ)
	gl.BindBuffer(gl.PIXEL_PACK_BUFFER, r.pboBack)
	gl.BufferData(gl.PIXEL_PACK_BUFFER, size, nil, gl.STREAM_READ)
	gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)
	r.pboIndex = 0
}

func resizeSSBO(buffer uint32, size, values int) {
	gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, buffer)
	gl.BufferData(gl.SHADER_STORAGE_BUFFER, size, nil, gl.DYNAMIC_DRAW)
	gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, 0)
	clearSSBOFloats(buffer, values)
}

func (r *gpuRenderer) uploadGPUScene(data gpuSceneData) {
	gl.BindBufferBase(gl.SHADER_STORAGE_BUFFER, gpuMaterialBinding, r.matSSBO)
	bufferDataFloat(gl.SHADER_STORAGE_BUFFER, data.matData)
	uploadFloatBuffer(r.objSSBO, gpuObjectBinding, data.objData)
	uploadIntBuffer(r.lightSSBO, gpuLightBinding, data.lightIndices)
	uploadUniformBuffer(r.camUBO, 1, data.camBlock[:])
	uploadUniformBuffer(r.skyUBO, 2, data.skyBlock[:])
	gl.BindBufferBase(gl.UNIFORM_BUFFER, gpuFogBinding, r.fogUBO)
	bufferDataFloat(gl.UNIFORM_BUFFER, data.fogBlock[:])
}

func uploadFloatBuffer(buffer, binding uint32, values []float32) {
	gl.BindBufferBase(gl.SHADER_STORAGE_BUFFER, binding, buffer)
	if len(values) == 0 {
		values = []float32{0}
	}
	bufferDataFloat(gl.SHADER_STORAGE_BUFFER, values)
}

func uploadIntBuffer(buffer, binding uint32, values []int32) {
	gl.BindBufferBase(gl.SHADER_STORAGE_BUFFER, binding, buffer)
	if len(values) == 0 {
		values = []int32{0}
	}
	bufferDataInt(gl.SHADER_STORAGE_BUFFER, values)
}

func uploadUniformBuffer(buffer, binding uint32, values []float32) {
	gl.BindBufferBase(gl.UNIFORM_BUFFER, binding, buffer)
	bufferDataFloat(gl.UNIFORM_BUFFER, values)
}

func bufferDataFloat(target uint32, values []float32) {
	gl.BufferData(target, len(values)*gpuFloatBytes, nil, gl.DYNAMIC_DRAW)
	copyMappedFloatBuffer(target, values)
}

func bufferDataInt(target uint32, values []int32) {
	gl.BufferData(target, len(values)*gpuFloatBytes, nil, gl.DYNAMIC_DRAW)
	copyMappedIntBuffer(target, values)
}

func copyMappedFloatBuffer(target uint32, values []float32) {
	if len(values) == 0 {
		return
	}
	data := gl.MapBuffer(target, gl.WRITE_ONLY)
	if data == nil {
		gpuLogf("failed to map OpenGL buffer target=%d for float upload", target)
		return
	}
	destination := (*[1 << 28]float32)(data)[:len(values):len(values)]
	copy(destination, values)
	if !gl.UnmapBuffer(target) {
		gpuLogf("failed to unmap OpenGL buffer target=%d after float upload", target)
	}
}

func copyMappedIntBuffer(target uint32, values []int32) {
	if len(values) == 0 {
		return
	}
	data := gl.MapBuffer(target, gl.WRITE_ONLY)
	if data == nil {
		gpuLogf("failed to map OpenGL buffer target=%d for int upload", target)
		return
	}
	destination := (*[1 << 28]int32)(data)[:len(values):len(values)]
	copy(destination, values)
	if !gl.UnmapBuffer(target) {
		gpuLogf("failed to unmap OpenGL buffer target=%d after int upload", target)
	}
}

func clearTexture(texture uint32) {
	var framebuffer uint32
	gl.GenFramebuffers(1, &framebuffer)
	gl.BindFramebuffer(gl.FRAMEBUFFER, framebuffer)
	gl.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0)
	if gl.CheckFramebufferStatus(gl.FRAMEBUFFER) != gl.FRAMEBUFFER_COMPLETE {
		gpuLogf("failed to create framebuffer for texture %d clear", texture)
	} else {
		gl.ClearColor(0, 0, 0, 0)
		gl.Clear(gl.COLOR_BUFFER_BIT)
	}
	gl.BindFramebuffer(gl.FRAMEBUFFER, 0)
	gl.DeleteFramebuffers(1, &framebuffer)
}

func (r *gpuRenderer) configureGPUProgram(cfg renderentity.Config, isWireframe bool, pixelCount int) gpuUniforms {
	program := r.program
	if isWireframe {
		program = r.wireframeProgram
	}
	gl.UseProgram(program)
	gl.BindImageTexture(0, r.imgTexture, 0, false, 0, gl.WRITE_ONLY, gl.RGBA16F)
	if !isWireframe && r.prevFrameTexture != 0 {
		gl.ActiveTexture(gl.TEXTURE9)
		gl.BindTexture(gl.TEXTURE_2D, r.prevFrameTexture)
	}
	gl.BindBufferBase(gl.SHADER_STORAGE_BUFFER, gpuAccumBinding, r.accumSSBO)
	if !isWireframe {
		gl.BindBufferBase(gl.SHADER_STORAGE_BUFFER, gpuVarianceBinding, r.varianceSSBO)
	}
	r.cacheUniforms(program)
	uniforms := gpuUniforms{
		locSpp:            r.locs.locSpp,
		locSampleCount:    r.locs.locSampleCount,
		locSeed:           r.locs.locSeed,
		locSelectedObject: r.locs.locSelectedObject,
	}
	gl.Uniform1i(r.locs.locW, glInt32(cfg.Width))
	gl.Uniform1i(r.locs.locH, glInt32(cfg.Height))
	gl.Uniform1i(r.locs.locDepth, glInt32(cfg.MaxDepth))
	maxRayDist := cfg.MaxRayDist
	if maxRayDist <= 0 {
		maxRayDist = 1000
	}
	gl.Uniform1f(r.locs.locMaxRayDist, maxRayDist)
	gl.Uniform1i(uniforms.locSelectedObject, glInt32(GetSelectedObject()))
	if !isWireframe && r.width == cfg.Width && r.height == cfg.Height {
		clearAccumulationBuffers(r, pixelCount)
	}
	return uniforms
}

func clearAccumulationBuffers(r *gpuRenderer, pixelCount int) {
	timer := startGPUTimer("clear_ssbo")
	clearSizedSSBO(r.accumSSBO, pixelCount*gpuRGBComponents*gpuFloatBytes, pixelCount*gpuRGBComponents)
	clearSizedSSBO(r.varianceSSBO, pixelCount*gpuVarianceComponents*gpuFloatBytes, pixelCount*gpuVarianceComponents)
	timer.stop()
}

func clearSizedSSBO(buffer uint32, size, values int) {
	gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, buffer)
	var currentSize int32
	gl.GetBufferParameteriv(gl.SHADER_STORAGE_BUFFER, gl.BUFFER_SIZE, &currentSize)
	if int(currentSize) != size {
		gl.BufferData(gl.SHADER_STORAGE_BUFFER, size, nil, gl.DYNAMIC_DRAW)
	}
	gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, 0)
	clearSSBOFloats(buffer, values)
}

func needsFinalReadback(isWireframe, wantLiveReadback, lastReadback bool) bool {
	if isWireframe || lastReadback && wantLiveReadback {
		return false
	}
	return !wantLiveReadback || !lastReadback
}

func (r *gpuRenderer) readFinalFrame(img *image.RGBA, pixelCount int) {
	timer := startGPUTimer("final_readback")
	gl.MemoryBarrier(gl.PIXEL_BUFFER_BARRIER_BIT | gl.TEXTURE_FETCH_BARRIER_BIT | gl.SHADER_IMAGE_ACCESS_BARRIER_BIT)
	currentPBO := r.pbo
	if r.pboIndex == 1 {
		currentPBO = r.pboBack
	}
	gl.BindBuffer(gl.PIXEL_PACK_BUFFER, currentPBO)
	gl.BindTexture(gl.TEXTURE_2D, r.imgTexture)
	gl.GetTexImage(gl.TEXTURE_2D, 0, gl.RGBA, gl.FLOAT, nil)
	gl.MemoryBarrier(gl.PIXEL_BUFFER_BARRIER_BIT)
	tmp := r.ensureReadTmp(pixelCount)
	if readMappedFrame(tmp) {
		tonemapHDRToRGBA(img.Pix, tmp, pixelCount)
	}
	gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)
	timer.stop()
}

func readMappedFrame(tmp []float32) bool {
	ptr := gl.MapBuffer(gl.PIXEL_PACK_BUFFER, gl.READ_ONLY)
	if ptr == nil {
		return false
	}
	src := (*[1 << 28]float32)(ptr)[:len(tmp)]
	copy(tmp, src)
	gl.UnmapBuffer(gl.PIXEL_PACK_BUFFER)
	return true
}

package gpu

import (
	"errors"
	"fmt"
	"image"
	"math"
	"os"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-gl/gl/v3.3-core/gl"
	"github.com/go-gl/glfw/v3.4/glfw"

	renderentity "github.com/MarkJulian19/path_trace_golang/internal/entities/render"
	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

var (
	errShaderLink    = errors.New("shader link error")
	errShaderCompile = errors.New("shader compile error")
)

const (
	gpuDenoiseSigmaRDefault   = 0.15
	gpuUnitValue              = 1.0
	gpuWireframeFallback      = 0.1
	gpuHalf                   = 0.5
	gpuMaxSmoothRadius        = 5
	gpuRGBAComponents         = 4
	gpuRGBComponents          = 3
	gpuVarianceComponents     = 9
	gpuFloatBytes             = 4
	gpuByteMax                = 255.0
	gpuAnisotropyLimit        = 0.9
	gpuFogNoiseScaleDefault   = 4.0
	gpuFogNoiseOctavesDefault = 3
	gpuGLMajorVersion         = 4
	gpuGLMinorVersion         = 3
	gpuMaterialBinding        = 3
	gpuObjectBinding          = 4
	gpuLightBinding           = 6
	gpuFogBinding             = 5
	gpuAccumBinding           = 7
	gpuVarianceBinding        = 8
	gpuWorkgroupSize          = 32
	gpuWorkgroupOffset        = gpuWorkgroupSize - 1
	gpuSampleIntervalMedium   = 3
	gpuSampleIntervalDefault  = 5
	gpuAbsorptionScaleDefault = 0.01
)

// acesTonemap применяет простую аппроксимацию ACES-фильма к одному
// каналу линейного HDR-цвета (0..+inf) и возвращает значение в диапазоне [0,1].
func acesTonemap(x float32) float32 {
	if x <= 0 {
		return 0
	}
	// Классическая аппроксимация ACES:
	// x*(a*x + b) / (x*(c*x + d) + e)
	const a = 2.51
	const b = 0.03
	const c = 2.43
	const d = 0.59
	const e = 0.14

	y := float64(x)
	num := y * (a*y + b)
	den := y*(c*y+d) + e
	if den <= 0 {
		return 0
	}
	r := num / den
	if r < 0 {
		r = 0
	} else if r > 1 {
		r = 1
	}
	return float32(r)
}

// denoiseConfig описывает параметры пост-денойзинга для GPU-рендера.
type denoiseConfig struct {
	Enabled bool
	SigmaS  float64
	SigmaR  float64
}

var (
	denoiseCfgOnce  sync.Once
	denoiseCfg      denoiseConfig
	denoiseOverride bool
)

// getDenoiseConfig читает настройки денойзера из переменных окружения один раз.
// PATHTRACER_GPU_DENOISE: "0"/"false"/"off" — выкл, "1"/"true"/"on" — вкл (по умолчанию: вкл).
// PATHTRACER_GPU_DENOISE_SIGMA_S: пространственный радиус (по умолчанию 1.0).
// PATHTRACER_GPU_DENOISE_SIGMA_R: радиус по цвету в sRGB (0..1), по умолчанию 0.15.
func getDenoiseConfig() denoiseConfig {
	if denoiseOverride {
		return denoiseCfg
	}
	denoiseCfgOnce.Do(func() {
		cfg := denoiseConfig{
			Enabled: true,
			SigmaS:  gpuUnitValue,
			SigmaR:  gpuDenoiseSigmaRDefault,
		}
		cfg.Enabled = readEnvBool("PATHTRACER_GPU_DENOISE", cfg.Enabled)
		cfg.SigmaS = readPositiveFloatEnv("PATHTRACER_GPU_DENOISE_SIGMA_S", cfg.SigmaS)
		cfg.SigmaR = readPositiveFloatEnv("PATHTRACER_GPU_DENOISE_SIGMA_R", cfg.SigmaR)
		denoiseCfg = cfg
	})
	return denoiseCfg
}

// SetDenoiseConfigFromUI позволяет UI/CLI переопределить настройки денойзинга
// во время работы приложения, не полагаясь только на переменные окружения.
func SetDenoiseConfigFromUI(enabled bool, sigmaS, sigmaR float64) {
	if sigmaS <= 0 {
		sigmaS = gpuUnitValue
	}
	if sigmaR <= 0 {
		sigmaR = gpuDenoiseSigmaRDefault
	}
	denoiseCfg = denoiseConfig{
		Enabled: enabled,
		SigmaS:  sigmaS,
		SigmaR:  sigmaR,
	}
	denoiseOverride = true
}

// Дополнительный конфиг для более сильного сглаживания (blur) после денойза.
type smoothConfig struct {
	Enabled  bool
	Radius   int     // радиус бокса (в пикселях), 1..5
	Strength float64 // сила смешивания 0..1 (0 = нет эффекта, 1 = только blur)
}

var (
	smoothCfgOnce  sync.Once
	smoothCfg      smoothConfig
	smoothOverride bool
)

// Режим рендеринга: 0 = normal, 1 = wireframe.
var (
	renderMode      int
	renderModeMutex sync.RWMutex
)

// Выбранный объект для подсветки.
var (
	selectedObject      = -1
	selectedObjectMutex sync.RWMutex
)

// Глобальный флаг отмены рендеринга используется для прерывания длительных операций.
var (
	renderCancelFlag atomic.Int32 // 0 = продолжать, 1 = отменить.
)

// SetRenderCancel устанавливает флаг отмены рендеринга.
func SetRenderCancel(cancel bool) {
	if cancel {
		renderCancelFlag.Store(1)
	} else {
		renderCancelFlag.Store(0)
	}
}

// IsRenderCancelled проверяет, была ли запрошена отмена рендеринга.
func IsRenderCancelled() bool {
	return renderCancelFlag.Load() == 1
}

// getSmoothConfig читает настройки сглаживания из переменных окружения один раз.
// PATHTRACER_GPU_SMOOTH: "0"/"false"/"off" — выкл, "1"/"true"/"on" — вкл (по умолчанию: выкл).
// PATHTRACER_GPU_SMOOTH_RADIUS: целое 1..5 (по умолчанию 2).
// PATHTRACER_GPU_SMOOTH_STRENGTH: 0..1 (по умолчанию 0.5).
func getSmoothConfig() smoothConfig {
	if smoothOverride {
		return smoothCfg
	}
	smoothCfgOnce.Do(func() {
		cfg := smoothConfig{
			Enabled:  false,
			Radius:   2,
			Strength: gpuHalf,
		}
		cfg.Enabled = readEnvBool("PATHTRACER_GPU_SMOOTH", cfg.Enabled)
		cfg.Radius = readIntEnv("PATHTRACER_GPU_SMOOTH_RADIUS", cfg.Radius, 1, gpuMaxSmoothRadius)
		cfg.Strength = readFloatEnv("PATHTRACER_GPU_SMOOTH_STRENGTH", cfg.Strength, 0, gpuUnitValue)
		smoothCfg = cfg
	})
	return smoothCfg
}

// SetSmoothConfigFromUI переопределяет настройки сглаживания из UI.
func SetSmoothConfigFromUI(enabled bool, radius int, strength float64) {
	radius = clampInt(radius, 1, gpuMaxSmoothRadius)
	strength = clampFloat(strength, 0, gpuUnitValue)
	smoothCfg = smoothConfig{
		Enabled:  enabled,
		Radius:   radius,
		Strength: strength,
	}
	smoothOverride = true
}

func readPositiveFloatEnv(name string, fallback float64) float64 {
	return readFloatEnv(name, fallback, 0, math.MaxFloat64)
}

func readFloatEnv(name string, fallback, minValue, maxValue float64) float64 {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return fallback
	}
	return clampFloat(parsed, minValue, maxValue)
}

func readIntEnv(name string, fallback, minValue, maxValue int) int {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return clampInt(parsed, minValue, maxValue)
}

func clampFloat(value, minValue, maxValue float64) float64 {
	return math.Min(math.Max(value, minValue), maxValue)
}

func clampInt(value, minValue, maxValue int) int {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

// SetRenderModeFromUI устанавливает режим рендеринга из UI.
// mode: 0 = normal, 1 = wireframe
func SetRenderModeFromUI(mode int) {
	renderModeMutex.Lock()
	defer renderModeMutex.Unlock()
	if mode < 0 {
		mode = 0
	}
	if mode > 1 {
		mode = 1
	}
	renderMode = mode
}

// GetRenderMode возвращает текущий режим рендеринга.
func GetRenderMode() int {
	renderModeMutex.RLock()
	defer renderModeMutex.RUnlock()
	return renderMode
}

// SetSelectedObjectFromUI устанавливает выбранный объект для подсветки из UI.
// ObjIndex: индекс объекта (-1 = нет выбора).
func SetSelectedObjectFromUI(objIndex int) {
	selectedObjectMutex.Lock()
	defer selectedObjectMutex.Unlock()
	selectedObject = objIndex
}

// GetSelectedObject возвращает индекс выбранного объекта.
func GetSelectedObject() int {
	selectedObjectMutex.RLock()
	defer selectedObjectMutex.RUnlock()
	return selectedObject
}

// programLocs кэширует GetUniformLocation для compute-программы.
type programLocs struct {
	prog              uint32
	locW              int32
	locH              int32
	locSpp            int32
	locDepth          int32
	locMaxRayDist     int32
	locSeed           int32
	locSampleCount    int32
	locSelectedObject int32
}

// gpuRenderer owns a hidden GLFW window and GL resources used for compute rendering.
type gpuRenderer struct {
	initOnce         sync.Once
	initErr          error
	window           *glfw.Window
	program          uint32
	wireframeProgram uint32 // Отдельный шейдер для wireframe режима
	imgTexture       uint32
	prevFrameTexture uint32 // Текстура для предыдущего кадра (Temporal Accumulation)
	pbo              uint32
	pboBack          uint32 // Второй PBO для асинхронного чтения (двойная буферизация)
	pboIndex         int    // Индекс текущего PBO (0 или 1)
	matSSBO          uint32
	objSSBO          uint32
	lightSSBO        uint32 // SSBO для списка индексов эмиссивных объектов
	accumSSBO        uint32 // SSBO для накопительного буфера на GPU (width * height * 3 * float32)
	varianceSSBO     uint32 // SSBO для variance данных (width * height * 9 * float32) - [mean, M2, count] × 3 канала
	camUBO           uint32
	skyUBO           uint32
	fogUBO           uint32
	width            int
	height           int
	// readTmp — переиспользуемый CPU-буфер для PBO→float readback (RGBA).
	readTmp []float32
	locs    programLocs
}

// Go-side copies of material / object type constants used in GLSL.
// Must stay in sync with values in the compute shader.
const (
	gpuMatLambert    = 0
	gpuMatMetal      = 1
	gpuMatDielectric = 2
	gpuMatEmissive   = 3
	gpuMatMirror     = 4
)

const (
	gpuObjSphere = 0
	gpuObjPlane  = 1
	gpuObjBox    = 2
)

// cacheUniforms обновляет кэш GetUniformLocation при смене программы.
func (r *gpuRenderer) cacheUniforms(prog uint32) {
	if r.locs.prog == prog {
		return
	}
	r.locs = programLocs{
		prog:              prog,
		locW:              gl.GetUniformLocation(prog, gl.Str("uWidth\x00")),
		locH:              gl.GetUniformLocation(prog, gl.Str("uHeight\x00")),
		locSpp:            gl.GetUniformLocation(prog, gl.Str("uSamplesPerPx\x00")),
		locDepth:          gl.GetUniformLocation(prog, gl.Str("uMaxDepth\x00")),
		locMaxRayDist:     gl.GetUniformLocation(prog, gl.Str("uMaxRayDist\x00")),
		locSeed:           gl.GetUniformLocation(prog, gl.Str("uFrameSeed\x00")),
		locSampleCount:    gl.GetUniformLocation(prog, gl.Str("uSampleCount\x00")),
		locSelectedObject: gl.GetUniformLocation(prog, gl.Str("uSelectedObject\x00")),
	}
}

// clearSSBOFloats заполняет SSBO нулями на GPU без загрузки гигантского CPU-слайса.
func clearSSBOFloats(ssbo uint32, floatCount int) {
	if floatCount <= 0 {
		return
	}
	gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, ssbo)
	data := gl.MapBuffer(gl.SHADER_STORAGE_BUFFER, gl.WRITE_ONLY)
	if data == nil {
		gpuLogf("failed to map SSBO %d for zero fill", ssbo)
		gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, 0)
		return
	}
	values := (*[1 << 28]float32)(data)[:floatCount:floatCount]
	clear(values)
	if !gl.UnmapBuffer(gl.SHADER_STORAGE_BUFFER) {
		gpuLogf("failed to unmap SSBO %d after zero fill", ssbo)
	}
	gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, 0)
}

// ensureReadTmp возвращает переиспользуемый буфер RGBA float для readback.
func (r *gpuRenderer) ensureReadTmp(pixelCount int) []float32 {
	need := pixelCount * gpuRGBAComponents
	if cap(r.readTmp) < need {
		r.readTmp = make([]float32, need)
	} else {
		r.readTmp = r.readTmp[:need]
	}
	return r.readTmp
}

// tonemapHDRToRGBA пишет ACES+gamma в dst из линейного RGBA float tmp.
func tonemapHDRToRGBA(dst []uint8, tmp []float32, pixelCount int) {
	maxPixels := len(dst) / gpuRGBAComponents
	n := min(maxPixels, pixelCount)
	for i := range n {
		off := i * gpuRGBAComponents
		if off+3 >= len(dst) || off+3 >= len(tmp) {
			break
		}
		writeTonemappedPixel(dst, tmp, off)
	}
}

func writeTonemappedPixel(dst []uint8, tmp []float32, off int) {
	rGamma := tonemapGamma(tmp[off])
	gGamma := tonemapGamma(tmp[off+1])
	bGamma := tonemapGamma(tmp[off+2])
	dst[off] = uint8(rGamma*gpuByteMax + gpuHalf)
	dst[off+1] = uint8(gGamma*gpuByteMax + gpuHalf)
	dst[off+2] = uint8(bGamma*gpuByteMax + gpuHalf)
	dst[off+3] = 255
}

func tonemapGamma(value float32) float32 {
	if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || value < 0 {
		return 0
	}
	return min(float32(1), float32(math.Sqrt(float64(acesTonemap(value)))))
}

// copyFrameToPrev копирует текущий HDR-кадр в prevFrameTexture для temporal blend.
func (r *gpuRenderer) copyFrameToPrev(w, h int) {
	if r.prevFrameTexture == 0 || w <= 0 || h <= 0 {
		return
	}
	gl.CopyImageSubData(
		r.imgTexture, gl.TEXTURE_2D, 0, 0, 0, 0,
		r.prevFrameTexture, gl.TEXTURE_2D, 0, 0, 0, 0,
		glInt32(w), glInt32(h), 1,
	)
}

// renderRequest is sent from callers to the dedicated GL worker goroutine.
type renderRequest struct {
	sc       *scene.Scene
	cfg      renderentity.Config
	img      *image.RGBA
	progress func(currentSample, totalSamples int)
	done     chan error
}

var (
	renderer   gpuRenderer
	renderCh   chan renderRequest
	workerOnce sync.Once
)

// ensureWorker starts the dedicated GL worker goroutine exactly once.
func ensureWorker() {
	workerOnce.Do(func() {
		renderCh = make(chan renderRequest)
		go renderWorker()
	})
}

// renderWorker owns the GL context and processes all GPU render requests.
// It always runs on a single locked OS thread, which is required by OpenGL.
func renderWorker() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	err := renderer.initGL()
	if err != nil {
		// Если не удалось инициализировать GL, логируем ошибку и отвечаем ошибкой на все запросы.
		fmt.Fprintf(os.Stderr, "GPU initialization failed: %v\n", err)
		for req := range renderCh {
			req.done <- err
		}
		return
	}

	fmt.Fprint(os.Stderr, "GPU renderer initialized successfully\n")
	if IsSoftwareGL() {
		fmt.Fprintf(
			os.Stderr,
			"WARNING: %s — OpenGL compute is software (CPU). Discrete GPU will not show load.\n",
			glRenderer,
		)
	}
	gpuLogf("GL worker ready (debug=%v live_readback=%v temporal=%v softGL=%v)",
		gpuDebugEnabled(), LivePreviewEnabled(), TemporalEnabled(), IsSoftwareGL())

	for req := range renderCh {
		err := renderer.renderOnce(req.sc, req.cfg, req.img, req.progress)
		if err != nil {
			fmt.Fprintf(os.Stderr, "GPU render error: %v\n", err)
		}
		req.done <- err
	}
}

// initGL must be called from the GL worker goroutine (locked OS thread).
func (r *gpuRenderer) initGL() error {
	r.initOnce.Do(func() {
		r.initErr = r.initializeGL()
	})

	return r.initErr
}

func (r *gpuRenderer) initializeGL() error {
	err := r.initGLContext()
	if err != nil {
		return err
	}

	r.initGLResources()
	return r.initGLPrograms()
}

func (r *gpuRenderer) initGLContext() error {
	err := glfw.Init()
	if err != nil {
		return fmt.Errorf("glfw init: %w", err)
	}

	glfw.WindowHint(glfw.Visible, glfw.False)
	glfw.WindowHint(glfw.ContextVersionMajor, gpuGLMajorVersion)
	glfw.WindowHint(glfw.ContextVersionMinor, gpuGLMinorVersion)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)

	w, err := glfw.CreateWindow(1, 1, "pathtracer-gpu-hidden", nil, nil)
	if err != nil {
		return fmt.Errorf("glfw create window: %w", err)
	}
	r.window = w
	w.MakeContextCurrent()

	err = gl.Init()
	if err != nil {
		return fmt.Errorf("gl init: %w", err)
	}

	vendor := gl.GoStr(gl.GetString(gl.VENDOR))
	rendererStr := gl.GoStr(gl.GetString(gl.RENDERER))
	version := gl.GoStr(gl.GetString(gl.VERSION))
	setGLInfo(vendor, rendererStr, version)
	return nil
}

func (r *gpuRenderer) initGLResources() {
	gl.GenTextures(1, &r.imgTexture)
	gl.BindTexture(gl.TEXTURE_2D, r.imgTexture)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)

	gl.GenTextures(1, &r.prevFrameTexture)
	gl.BindTexture(gl.TEXTURE_2D, r.prevFrameTexture)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

	gl.GenBuffers(1, &r.pbo)
	gl.GenBuffers(1, &r.pboBack)
	r.pboIndex = 0
	gl.GenBuffers(1, &r.matSSBO)
	gl.GenBuffers(1, &r.objSSBO)
	gl.GenBuffers(1, &r.lightSSBO)
	gl.GenBuffers(1, &r.accumSSBO)
	gl.GenBuffers(1, &r.varianceSSBO)
	gl.GenBuffers(1, &r.camUBO)
	gl.GenBuffers(1, &r.skyUBO)
	gl.GenBuffers(1, &r.fogUBO)
}

func (r *gpuRenderer) initGLPrograms() error {
	program, err := createComputeProgram(computeShader, "compute")
	if err != nil {
		return err
	}
	r.program = program

	program, err = createComputeProgram(wireframeShaderSource, "wireframe")
	if err != nil {
		return err
	}
	r.wireframeProgram = program
	return nil
}

func createComputeProgram(source, name string) (uint32, error) {
	shader, err := compileShader(source, gl.COMPUTE_SHADER)
	if err != nil {
		return 0, fmt.Errorf("compile %s shader: %w", name, err)
	}

	program := gl.CreateProgram()
	gl.AttachShader(program, shader)
	gl.LinkProgram(program)

	var status int32
	gl.GetProgramiv(program, gl.LINK_STATUS, &status)
	if status == gl.FALSE {
		var logLen int32
		gl.GetProgramiv(program, gl.INFO_LOG_LENGTH, &logLen)
		log := make([]byte, logLen+1)
		gl.GetProgramInfoLog(program, logLen, nil, &log[0])
		return 0, fmt.Errorf("%w: link %s program: %s", errShaderLink, name, string(log))
	}

	return program, nil
}

func compileShader(src string, shaderType uint32) (uint32, error) {
	shader := gl.CreateShader(shaderType)
	csources, free := gl.Strs(src + "\x00")
	defer free()
	gl.ShaderSource(shader, 1, csources, nil)
	gl.CompileShader(shader)

	var status int32
	gl.GetShaderiv(shader, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		var logLen int32
		gl.GetShaderiv(shader, gl.INFO_LOG_LENGTH, &logLen)
		log := make([]byte, logLen+1)
		gl.GetShaderInfoLog(shader, logLen, nil, &log[0])
		return 0, fmt.Errorf("%w: %s", errShaderCompile, string(log))
	}
	return shader, nil
}

// renderOnce executes a very simple GPU render into img using the already
// initialized GL context owned by the worker goroutine.
func (r *gpuRenderer) renderOnce(
	sc *scene.Scene,
	cfg renderentity.Config,
	img *image.RGBA,
	progress func(currentSample, totalSamples int),
) error {
	totalT := startGPUTimer("renderOnce")
	defer totalT.stopf("%dx%d spp=%d live=%v", cfg.Width, cfg.Height, cfg.SamplesPerPx, LivePreviewEnabled())

	state, ok := r.prepareGPUFrame(sc, cfg, img, progress)
	if !ok {
		return nil
	}
	r.renderSamplesAndReadback(state, cfg, img, progress)

	applyPostprocess(img, cfg)

	if progress != nil {
		progress(cfg.SamplesPerPx, cfg.SamplesPerPx)
	}

	return nil
}

func (r *gpuRenderer) renderSamplesAndReadback(
	state gpuFrameState,
	cfg renderentity.Config,
	img *image.RGBA,
	progress func(int, int),
) {
	gpuLogf(
		"sample loop passes=%d updateInterval=%d live_readback=%v",
		state.passes,
		state.updateInterval,
		state.wantLiveReadback,
	)
	stats := r.renderSamples(gpuSampleRequest{
		cfg:              configForSamples{width: cfg.Width, height: cfg.Height},
		img:              img,
		tmp:              state.tmp,
		pixelCount:       state.pixelCount,
		passes:           state.passes,
		updateInterval:   state.updateInterval,
		isWireframe:      state.isWireframe,
		wantLiveReadback: state.wantLiveReadback,
		locSpp:           state.uniforms.locSpp,
		locSampleCount:   state.uniforms.locSampleCount,
		locSeed:          state.uniforms.locSeed,
		locObject:        state.uniforms.locSelectedObject,
		progress:         progress,
	})
	if gpuDebugEnabled() {
		gpuLogf("dispatch_total=%s readbacks=%d readback_total=%s",
			stats.dispatchTime.Round(time.Microsecond), stats.readbacks, stats.readbackTime.Round(time.Microsecond))
	}
	if needsFinalReadback(state.isWireframe, state.wantLiveReadback, stats.lastReadback) {
		r.readFinalFrame(img, state.pixelCount)
		return
	}
	gpuLogf("skip duplicate final_readback (already have last sample)")
}

type fogBlockValues struct {
	density, scatter, affectSky, gpuVolumetric float32
	color                                      [4]float32
	sigmaS, sigmaA, anisotropy, heteroStrength float32
	noiseScale, noiseOctaves                   float32
}

func packFogBlock(sc *scene.Scene) [16]float32 {
	values := fogBlockValues{
		color:        [4]float32{},
		noiseScale:   gpuFogNoiseScaleDefault,
		noiseOctaves: gpuFogNoiseOctavesDefault,
	}
	if sc.Fog != nil {
		setFogBase(&values, sc.Fog)
		setFogPhysics(&values, sc.Fog)
		setFogNoise(&values, sc.Fog)
	}
	return [16]float32{
		values.density, values.scatter, values.affectSky, values.gpuVolumetric,
		values.color[0], values.color[1], values.color[2], values.color[3],
		values.sigmaS, values.sigmaA, values.anisotropy, values.heteroStrength,
		values.noiseScale, values.noiseOctaves, 0, 0,
	}
}

func applyDenoise(dst []byte, width, height int, cfg denoiseConfig) {
	timer := startGPUTimer("cpu_denoise")
	smoothed := make([]byte, len(dst))
	for y := range height {
		for x := range width {
			writeDenoisedPixel(smoothed, dst, width, height, x, y, cfg)
		}
	}
	copy(dst, smoothed)
	timer.stop()
}

func writeDenoisedPixel(dst, src []byte, width, height, x, y int, cfg denoiseConfig) {
	centerIndex := (y*width + x) * gpuRGBAComponents
	center := [3]float64{
		float64(src[centerIndex]) / gpuByteMax,
		float64(src[centerIndex+1]) / gpuByteMax,
		float64(src[centerIndex+2]) / gpuByteMax,
	}
	sumR, sumG, sumB, sumWeight := denoiseNeighborhood(src, width, height, x, y, center, cfg)
	if sumWeight <= 0 {
		copy(dst[centerIndex:centerIndex+gpuRGBAComponents], src[centerIndex:centerIndex+gpuRGBAComponents])
		return
	}
	dst[centerIndex] = uint8(clampFloat(sumR/sumWeight, 0, 1)*gpuByteMax + gpuHalf)
	dst[centerIndex+1] = uint8(clampFloat(sumG/sumWeight, 0, 1)*gpuByteMax + gpuHalf)
	dst[centerIndex+2] = uint8(clampFloat(sumB/sumWeight, 0, 1)*gpuByteMax + gpuHalf)
	dst[centerIndex+3] = 255
}

func denoiseNeighborhood(
	src []byte,
	width, height, x, y int,
	center [3]float64,
	cfg denoiseConfig,
) (float64, float64, float64, float64) {
	var sumR, sumG, sumB, sumWeight float64
	twoSigmaS2 := 2 * cfg.SigmaS * cfg.SigmaS
	twoSigmaR2 := 2 * cfg.SigmaR * cfg.SigmaR
	for ky := -1; ky <= 1; ky++ {
		for kx := -1; kx <= 1; kx++ {
			nx, ny := x+kx, y+ky
			if nx < 0 || nx >= width || ny < 0 || ny >= height {
				continue
			}
			index := (ny*width + nx) * gpuRGBAComponents
			nr := float64(src[index]) / gpuByteMax
			ng := float64(src[index+1]) / gpuByteMax
			nb := float64(src[index+2]) / gpuByteMax
			dr, dg, db := center[0]-nr, center[1]-ng, center[2]-nb
			distanceColor := dr*dr + dg*dg + db*db
			spatialWeight := math.Exp(-float64(kx*kx+ky*ky) / twoSigmaS2)
			colorWeight := math.Exp(-distanceColor / twoSigmaR2)
			weight := spatialWeight * colorWeight
			sumWeight += weight
			sumR += nr * weight
			sumG += ng * weight
			sumB += nb * weight
		}
	}
	return sumR, sumG, sumB, sumWeight
}

func applySmooth(dst []byte, width, height int, cfg smoothConfig) {
	timer := startGPUTimer("cpu_smooth")
	radius := min(max(cfg.Radius, 1), gpuMaxSmoothRadius)
	strength := clampFloat(cfg.Strength, 0, 1)
	blurred := make([]byte, len(dst))
	for y := range height {
		for x := range width {
			writeSmoothedPixel(blurred, dst, width, height, x, y, radius, strength)
		}
	}
	copy(dst, blurred)
	timer.stop()
}

func writeSmoothedPixel(dst, src []byte, width, height, x, y, radius int, strength float64) {
	avgR, avgG, avgB, count := smoothNeighborhood(src, width, height, x, y, radius)
	if count <= 0 {
		return
	}
	index := (y*width + x) * gpuRGBAComponents
	origR, origG, origB := float64(src[index]), float64(src[index+1]), float64(src[index+2])
	dst[index] = byte(clampFloat((1-strength)*origR+strength*avgR/count, 0, gpuByteMax) + gpuHalf)
	dst[index+1] = byte(clampFloat((1-strength)*origG+strength*avgG/count, 0, gpuByteMax) + gpuHalf)
	dst[index+2] = byte(clampFloat((1-strength)*origB+strength*avgB/count, 0, gpuByteMax) + gpuHalf)
	dst[index+3] = 255
}

func smoothNeighborhood(src []byte, width, height, x, y, radius int) (float64, float64, float64, float64) {
	var sumR, sumG, sumB, count float64
	for ky := -radius; ky <= radius; ky++ {
		for kx := -radius; kx <= radius; kx++ {
			nx, ny := x+kx, y+ky
			if nx < 0 || nx >= width || ny < 0 || ny >= height {
				continue
			}
			index := (ny*width + nx) * gpuRGBAComponents
			sumR += float64(src[index])
			sumG += float64(src[index+1])
			sumB += float64(src[index+2])
			count++
		}
	}
	return sumR, sumG, sumB, count
}

func setFogBase(values *fogBlockValues, fog *scene.Fog) {
	if fog.Density > 0 {
		values.density = float32(fog.Density)
	}
	if fog.Scatter > 0 {
		values.scatter = float32(fog.Scatter)
	} else if values.density > 0 {
		values.scatter = gpuUnitValue
	}
	if fog.AffectSky {
		values.affectSky = gpuUnitValue
	}
	values.color = [4]float32{float32(fog.Color.R), float32(fog.Color.G), float32(fog.Color.B), 1}
}

func setFogPhysics(values *fogBlockValues, fog *scene.Fog) {
	if fog.SigmaS > 0 || fog.SigmaA > 0 {
		values.sigmaS = float32(fog.SigmaS)
		values.sigmaA = float32(fog.SigmaA)
		return
	}
	if values.density <= 0 {
		return
	}
	scatter := min(max(values.scatter, 0), gpuUnitValue)
	values.sigmaS = values.density * scatter
	values.sigmaA = max(values.density-values.sigmaS, 0)
}

func setFogNoise(values *fogBlockValues, fog *scene.Fog) {
	values.anisotropy = float32(clampFloat(fog.G, -gpuAnisotropyLimit, gpuAnisotropyLimit))
	if fog.HeteroStrength > 0 {
		values.heteroStrength = float32(min(fog.HeteroStrength, gpuUnitValue))
	}
	if fog.NoiseScale > 0 {
		values.noiseScale = float32(fog.NoiseScale)
	}
	if fog.NoiseOctaves > 0 {
		values.noiseOctaves = float32(min(fog.NoiseOctaves, gpuMaxSmoothRadius))
	}
	if fog.GPUVolumetric {
		values.gpuVolumetric = gpuUnitValue
	}
}

func sampleUpdateInterval(pixelCount int) int {
	switch {
	case pixelCount > 2000*2000:
		return 1
	case pixelCount > 1000*1000:
		return 2
	case pixelCount > 500*500:
		return gpuSampleIntervalMedium
	default:
		return gpuSampleIntervalDefault
	}
}

// Render is the public entry point that schedules a GPU render on the
// dedicated GL worker and waits for completion.
// For now it just draws a test gradient using a compute shader to verify GPU path.
// Later this can be replaced with a full path tracer that uses sc and cfg.
func Render(sc *scene.Scene, cfg renderentity.Config, img *image.RGBA, progress renderentity.Progress) error {
	ensureWorker()
	done := make(chan error, 1)
	req := renderRequest{
		sc:       sc,
		cfg:      cfg,
		img:      img,
		progress: progress,
		done:     done,
	}
	renderCh <- req
	return <-done
}

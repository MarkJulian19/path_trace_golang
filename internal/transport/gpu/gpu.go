package gpu

import (
	"errors"
	"fmt"
	"image"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/go-gl/gl/v3.3-core/gl"
	"github.com/go-gl/glfw/v3.4/glfw"

	renderentity "github.com/MarkJulian19/path_trace_golang/internal/entities/render"
	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

var (
	errShaderLink    = errors.New("shader link error")
	errShaderCompile = errors.New("shader compile error")
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
			SigmaS:  1.0,
			SigmaR:  0.15,
		}

		if v := os.Getenv("PATHTRACER_GPU_DENOISE"); v != "" {
			switch vLower := strings.ToLower(v); vLower {
			case "0", disabledValue, negativeValue, "no":
				cfg.Enabled = false
			case "1", enabledValue, "on", positiveValue:
				cfg.Enabled = true
			default:
				// Сохраняем значение по умолчанию.
			}
		}
		if v := os.Getenv("PATHTRACER_GPU_DENOISE_SIGMA_S"); v != "" {
			f, err := strconv.ParseFloat(v, 64)
			if err == nil && f > 0 {
				cfg.SigmaS = f
			}
		}
		if v := os.Getenv("PATHTRACER_GPU_DENOISE_SIGMA_R"); v != "" {
			f, err := strconv.ParseFloat(v, 64)
			if err == nil && f > 0 {
				cfg.SigmaR = f
			}
		}

		denoiseCfg = cfg
	})
	return denoiseCfg
}

// SetDenoiseConfigFromUI позволяет UI/CLI переопределить настройки денойзинга
// во время работы приложения, не полагаясь только на переменные окружения.
func SetDenoiseConfigFromUI(enabled bool, sigmaS, sigmaR float64) {
	if sigmaS <= 0 {
		sigmaS = 1.0
	}
	if sigmaR <= 0 {
		sigmaR = 0.15
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
			Strength: 0.5,
		}

		if v := os.Getenv("PATHTRACER_GPU_SMOOTH"); v != "" {
			switch strings.ToLower(v) {
			case "1", enabledValue, "on", positiveValue:
				cfg.Enabled = true
			case "0", disabledValue, negativeValue, "no":
				cfg.Enabled = false
			default:
				// Сохраняем значение по умолчанию.
			}
		}
		if v := os.Getenv("PATHTRACER_GPU_SMOOTH_RADIUS"); v != "" {
			r, err := strconv.Atoi(v)
			if err == nil {
				if r < 1 {
					r = 1
				}
				if r > 5 {
					r = 5
				}
				cfg.Radius = r
			}
		}
		if v := os.Getenv("PATHTRACER_GPU_SMOOTH_STRENGTH"); v != "" {
			f, err := strconv.ParseFloat(v, 64)
			if err == nil {
				if f < 0 {
					f = 0
				}
				if f > 1 {
					f = 1
				}
				cfg.Strength = f
			}
		}

		smoothCfg = cfg
	})
	return smoothCfg
}

// SetSmoothConfigFromUI переопределяет настройки сглаживания из UI.
func SetSmoothConfigFromUI(enabled bool, radius int, strength float64) {
	if radius < 1 {
		radius = 1
	}
	if radius > 5 {
		radius = 5
	}
	if strength < 0 {
		strength = 0
	}
	if strength > 1 {
		strength = 1
	}
	smoothCfg = smoothConfig{
		Enabled:  enabled,
		Radius:   radius,
		Strength: strength,
	}
	smoothOverride = true
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
	zero := float32(0)
	gl.ClearBufferData(gl.SHADER_STORAGE_BUFFER, gl.R32F, gl.RED, gl.FLOAT, unsafe.Pointer(&zero))
	gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, 0)
}

// ensureReadTmp возвращает переиспользуемый буфер RGBA float для readback.
func (r *gpuRenderer) ensureReadTmp(pixelCount int) []float32 {
	need := pixelCount * 4
	if cap(r.readTmp) < need {
		r.readTmp = make([]float32, need)
	} else {
		r.readTmp = r.readTmp[:need]
	}
	return r.readTmp
}

// tonemapHDRToRGBA пишет ACES+gamma в dst из линейного RGBA float tmp.
func tonemapHDRToRGBA(dst []uint8, tmp []float32, pixelCount int) {
	maxPixels := len(dst) / 4
	n := min(maxPixels, pixelCount)
	for i := 0; i < n; i++ {
		off := i * 4
		if off+3 >= len(dst) || off+3 >= len(tmp) {
			break
		}
		rLin := tmp[off]
		gLin := tmp[off+1]
		bLin := tmp[off+2]
		if math.IsNaN(float64(rLin)) || math.IsInf(float64(rLin), 0) {
			rLin = 0
		}
		if math.IsNaN(float64(gLin)) || math.IsInf(float64(gLin), 0) {
			gLin = 0
		}
		if math.IsNaN(float64(bLin)) || math.IsInf(float64(bLin), 0) {
			bLin = 0
		}
		if rLin < 0 {
			rLin = 0
		}
		if gLin < 0 {
			gLin = 0
		}
		if bLin < 0 {
			bLin = 0
		}
		rTm := acesTonemap(rLin)
		gTm := acesTonemap(gLin)
		bTm := acesTonemap(bLin)
		rGamma := float32(math.Sqrt(float64(rTm)))
		gGamma := float32(math.Sqrt(float64(gTm)))
		bGamma := float32(math.Sqrt(float64(bTm)))
		if rGamma > 1 {
			rGamma = 1
		}
		if gGamma > 1 {
			gGamma = 1
		}
		if bGamma > 1 {
			bGamma = 1
		}
		dst[off] = uint8(rGamma*255.0 + 0.5)
		dst[off+1] = uint8(gGamma*255.0 + 0.5)
		dst[off+2] = uint8(bGamma*255.0 + 0.5)
		dst[off+3] = 255
	}
}

// copyFrameToPrev копирует текущий HDR-кадр в prevFrameTexture для temporal blend.
func (r *gpuRenderer) copyFrameToPrev(w, h int) {
	if r.prevFrameTexture == 0 || w <= 0 || h <= 0 {
		return
	}
	gl.CopyImageSubData(
		r.imgTexture, gl.TEXTURE_2D, 0, 0, 0, 0,
		r.prevFrameTexture, gl.TEXTURE_2D, 0, 0, 0, 0,
		int32(w), int32(h), 1,
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
	glfw.WindowHint(glfw.ContextVersionMajor, 4)
	glfw.WindowHint(glfw.ContextVersionMinor, 3)
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

	// --- Инициализация накопительного буфера ---
	pixelCount := cfg.Width * cfg.Height
	if pixelCount <= 0 {
		return nil
	}

	// Проверяем, что размер изображения соответствует разрешению
	expectedPixSize := pixelCount * 4 // RGBA = 4 байта на пиксель
	if len(img.Pix) != expectedPixSize {
		// Изображение имеет неправильный размер - ограничиваем pixelCount размером изображения для безопасности
		actualPixSize := len(img.Pix) / 4
		if actualPixSize < pixelCount {
			pixelCount = actualPixSize
		}
	}

	gpuLogf("begin mats=%d objs=%d", len(sc.Materials), len(sc.Objects))

	// --- Пакуем материалы ---
	matCount := len(sc.Materials)
	if matCount == 0 {
		// Без материалов смысла рендерить нет - заполняем чёрным.
		for i := range img.Pix {
			if i%4 == 3 {
				img.Pix[i] = 255
			} else {
				img.Pix[i] = 0
			}
		}
		if progress != nil {
			progress(1, 1)
		}
		return nil
	}
	matStride := 20
	matData := make([]float32, matCount*matStride)
	for i, m := range sc.Materials {
		base := i * matStride

		var typ float32
		switch m.Type {
		case scene.MaterialLambert:
			typ = float32(gpuMatLambert)
		case scene.MaterialMetal:
			typ = float32(gpuMatMetal)
		case scene.MaterialDielectric:
			typ = float32(gpuMatDielectric)
		case scene.MaterialEmissive:
			typ = float32(gpuMatEmissive)
		case scene.MaterialMirror:
			typ = float32(gpuMatMirror)
		default:
			typ = float32(gpuMatLambert)
		}

		matData[base+0] = typ
		matData[base+1] = float32(m.Rough)
		matData[base+2] = float32(m.IOR)

		// Smoothness для металлов: если не задан, вычисляем из Rough (обратная совместимость)
		smoothness := m.Smoothness
		if smoothness == 0 && m.Type == scene.MaterialMetal {
			// Если smoothness не задан, вычисляем из rough для обратной совместимости
			smoothness = 1.0 - m.Rough
		}
		if smoothness < 0 {
			smoothness = 0
		}
		if smoothness > 1 {
			smoothness = 1
		}
		matData[base+3] = float32(smoothness)

		matData[base+4] = float32(m.Albedo.R)
		matData[base+5] = float32(m.Albedo.G)
		matData[base+6] = float32(m.Albedo.B)

		// Reflectivity для металлов: если не задан, используем 1.0
		reflectivity := m.Reflectivity
		if reflectivity == 0 && m.Type == scene.MaterialMetal {
			reflectivity = 1.0
		}
		if reflectivity < 0 {
			reflectivity = 0
		}
		if reflectivity > 1 {
			reflectivity = 1
		}
		matData[base+7] = float32(reflectivity)

		// emit * power
		matData[base+8] = float32(m.Emit.R * m.Power)
		matData[base+9] = float32(m.Emit.G * m.Power)
		matData[base+10] = float32(m.Emit.B * m.Power)

		matData[base+12] = float32(m.Absorption.R)
		matData[base+13] = float32(m.Absorption.G)
		matData[base+14] = float32(m.Absorption.B)

		// AbsorptionScale для диэлектриков: если не задан, используем 0.01 (разумное значение для см)
		absorptionScale := m.AbsorptionScale
		if absorptionScale == 0 && m.Type == scene.MaterialDielectric {
			absorptionScale = 0.01 // По умолчанию 0.01 для см
		}
		matData[base+15] = float32(absorptionScale)

		// Tint для диэлектриков: если не задан, используем белый (1,1,1)
		tintR := m.Tint.R
		tintG := m.Tint.G
		tintB := m.Tint.B
		if tintR == 0 && tintG == 0 && tintB == 0 && m.Type == scene.MaterialDielectric {
			tintR = 1.0
			tintG = 1.0
			tintB = 1.0
		}
		matData[base+16] = float32(tintR)
		matData[base+17] = float32(tintG)
		matData[base+18] = float32(tintB)
	}

	// --- Пакуем объекты ---
	objCount := len(sc.Objects)
	objStride := 12
	objData := make([]float32, objCount*objStride)
	// карта materialID -> индекс материала
	matIndex := make(map[string]int, matCount)
	for i, m := range sc.Materials {
		matIndex[m.ID] = i
	}

	// Предвычисляем список индексов эмиссивных объектов для оптимизации
	lightIndices := make([]int32, 0, objCount)

	for i, o := range sc.Objects {
		base := i * objStride

		var typ float32
		switch o.Type {
		case scene.ObjectSphere, scene.ObjectSphereLight:
			typ = float32(gpuObjSphere)
		case scene.ObjectPlane:
			typ = float32(gpuObjPlane)
		case scene.ObjectBox:
			typ = float32(gpuObjBox)
		default:
			typ = float32(gpuObjSphere)
		}

		objData[base+0] = typ
		var matIdx int
		if idx, ok := matIndex[o.MaterialID]; ok {
			objData[base+1] = float32(idx)
			matIdx = idx
		} else {
			objData[base+1] = 0
			matIdx = 0
		}

		objData[base+4] = float32(o.Position.X)
		objData[base+5] = float32(o.Position.Y)
		objData[base+6] = float32(o.Position.Z)

		objData[base+8] = float32(o.Size.X)
		objData[base+9] = float32(o.Size.Y)
		objData[base+10] = float32(o.Size.Z)

		// Проверяем, является ли объект источником света
		if matIdx < len(sc.Materials) {
			m := sc.Materials[matIdx]
			if m.Type == scene.MaterialEmissive && (m.Emit.R > 0 || m.Emit.G > 0 || m.Emit.B > 0) {
				lightIndices = append(lightIndices, int32(i))
			}
		}
	}

	// --- Пакуем камеру ---
	cam := sc.Camera
	aspect := float32(cfg.Width) / float32(cfg.Height)
	if cam.AspectRatio != 0 {
		aspect = float32(cam.AspectRatio)
	}

	camBlock := [16]float32{
		float32(cam.Position.X), float32(cam.Position.Y), float32(cam.Position.Z), 0,
		float32(cam.Target.X), float32(cam.Target.Y), float32(cam.Target.Z), 0,
		float32(cam.Up.X), float32(cam.Up.Y), float32(cam.Up.Z), 0,
		float32(cam.FOV),
		float32(cam.Aperture),
		float32(cam.FocusDist),
		aspect,
	}

	// --- Пакуем небо ---
	var skyType int
	var skyColor, skyHorizon, skyZenith [4]float32
	if sc.Sky != nil {
		if sc.Sky.Type == "gradient" {
			skyType = 2
			skyHorizon = [4]float32{float32(sc.Sky.Horizon.R), float32(sc.Sky.Horizon.G), float32(sc.Sky.Horizon.B), 1}
			skyZenith = [4]float32{float32(sc.Sky.Zenith.R), float32(sc.Sky.Zenith.G), float32(sc.Sky.Zenith.B), 1}
		} else { // solid
			skyType = 1
			skyColor = [4]float32{float32(sc.Sky.Color.R), float32(sc.Sky.Color.G), float32(sc.Sky.Color.B), 1}
		}
	} else {
		skyType = 0
		skyColor = [4]float32{float32(sc.Background.R), float32(sc.Background.G), float32(sc.Background.B), 1}
	}

	skyBlock := [16]float32{
		float32(skyType), 0, 0, 0,
		skyColor[0], skyColor[1], skyColor[2], skyColor[3],
		skyHorizon[0], skyHorizon[1], skyHorizon[2], skyHorizon[3],
		skyZenith[0], skyZenith[1], skyZenith[2], skyZenith[3],
	}

	// --- Пакуем туман (fog) ---
	fogDensity := float32(0)
	fogScatter := float32(0)
	fogAffectSky := float32(0)
	fogColor := [4]float32{0, 0, 0, 0}
	fogSigmaS := float32(0)
	fogSigmaA := float32(0)
	fogG := float32(0)
	fogHeteroStrength := float32(0)
	fogNoiseScale := float32(4.0)
	fogNoiseOctaves := float32(3)
	fogGpuVolumetric := float32(0)

	if sc.Fog != nil {
		// Базовые параметры (обратная совместимость).
		if sc.Fog.Density > 0 {
			fogDensity = float32(sc.Fog.Density)
		}
		if sc.Fog.Scatter > 0 {
			fogScatter = float32(sc.Fog.Scatter)
		} else if fogDensity > 0 {
			fogScatter = 1.0
		}
		if sc.Fog.AffectSky {
			fogAffectSky = 1.0
		}
		fogColor = [4]float32{
			float32(sc.Fog.Color.R),
			float32(sc.Fog.Color.G),
			float32(sc.Fog.Color.B),
			1,
		}

		// Физическая модель: sigma_s / sigma_a.
		if sc.Fog.SigmaS > 0 || sc.Fog.SigmaA > 0 {
			fogSigmaS = float32(sc.Fog.SigmaS)
			fogSigmaA = float32(sc.Fog.SigmaA)
		} else if fogDensity > 0 {
			// Если явно не заданы, выводим из плотности и scatter.
			baseSigmaT := fogDensity
			sMul := fogScatter
			if sMul < 0 {
				sMul = 0
			}
			if sMul > 1 {
				sMul = 1
			}
			fogSigmaS = baseSigmaT * sMul
			fogSigmaA = baseSigmaT - fogSigmaS
			if fogSigmaA < 0 {
				fogSigmaA = 0
			}
		}

		// Анизотропия HG‑фазы.
		if sc.Fog.G < -0.9 {
			fogG = -0.9
		} else if sc.Fog.G > 0.9 {
			fogG = 0.9
		} else {
			fogG = float32(sc.Fog.G)
		}

		// Неоднородный туман (шум).
		if sc.Fog.HeteroStrength > 0 {
			if sc.Fog.HeteroStrength > 1 {
				fogHeteroStrength = 1
			} else {
				fogHeteroStrength = float32(sc.Fog.HeteroStrength)
			}
		}
		if sc.Fog.NoiseScale > 0 {
			fogNoiseScale = float32(sc.Fog.NoiseScale)
		}
		if sc.Fog.NoiseOctaves > 0 {
			if sc.Fog.NoiseOctaves > 5 {
				fogNoiseOctaves = 5
			} else {
				fogNoiseOctaves = float32(sc.Fog.NoiseOctaves)
			}
		}

		if sc.Fog.GPUVolumetric {
			fogGpuVolumetric = 1.0
		}
	}

	// std140: выравниваем до кратности vec4.
	fogBlock := [16]float32{
		fogDensity, fogScatter, fogAffectSky, fogGpuVolumetric,
		fogColor[0], fogColor[1], fogColor[2], fogColor[3],
		fogSigmaS, fogSigmaA, fogG, fogHeteroStrength,
		fogNoiseScale, fogNoiseOctaves, 0, 0,
	}

	// Resize texture if needed.
	if r.width != cfg.Width || r.height != cfg.Height {
		// Если размер изменился, нужно полностью переинициализировать все буферы
		// и дождаться завершения всех операций перед началом нового рендеринга

		// Сначала завершаем все ожидающие операции с PBO
		gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)
		gl.MemoryBarrier(gl.ALL_BARRIER_BITS)

		r.width = cfg.Width
		r.height = cfg.Height

		gl.BindTexture(gl.TEXTURE_2D, r.imgTexture)
		// HDR-подобный формат: RGBA16F, чтобы compute шейдер писал линейный цвет
		// с меньшим квантованием. Читаем как float32.
		// Инициализируем текстуру нулями (черный цвет) для предотвращения белой мути
		texZeroData := make([]float32, cfg.Width*cfg.Height*4)
		for i := range texZeroData {
			texZeroData[i] = 0.0
		}
		gl.TexImage2D(
			gl.TEXTURE_2D,
			0,
			gl.RGBA16F,
			int32(cfg.Width),
			int32(cfg.Height),
			0,
			gl.RGBA,
			gl.FLOAT,
			gl.Ptr(texZeroData),
		)

		// Инициализируем текстуру предыдущего кадра для Temporal Accumulation
		if r.prevFrameTexture != 0 {
			gl.BindTexture(gl.TEXTURE_2D, r.prevFrameTexture)
			gl.TexImage2D(
				gl.TEXTURE_2D,
				0,
				gl.RGBA16F,
				int32(cfg.Width),
				int32(cfg.Height),
				0,
				gl.RGBA,
				gl.FLOAT,
				gl.Ptr(texZeroData),
			)
		}

		// Инициализируем оба PBO для двойной буферизации
		// Важно: полностью пересоздаем буферы, чтобы избежать проблем с размером
		pboSize := cfg.Width * cfg.Height * 4 * 4 // 4 компоненты по 4 байта (float32) на пиксель

		// Пересоздаем первый PBO
		gl.BindBuffer(gl.PIXEL_PACK_BUFFER, r.pbo)
		gl.BufferData(gl.PIXEL_PACK_BUFFER, pboSize, nil, gl.STREAM_READ)

		// Пересоздаем второй PBO
		gl.BindBuffer(gl.PIXEL_PACK_BUFFER, r.pboBack)
		gl.BufferData(gl.PIXEL_PACK_BUFFER, pboSize, nil, gl.STREAM_READ)

		// Сбрасываем состояние PBO
		gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)
		r.pboIndex = 0 // Сбрасываем индекс - начинаем с первого PBO

		// Инициализируем накопительный буфер на GPU
		accumSize := pixelCount * 3 * 4 // 3 компоненты (RGB) по 4 байта (float32)
		gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, r.accumSSBO)
		gl.BufferData(gl.SHADER_STORAGE_BUFFER, accumSize, nil, gl.DYNAMIC_DRAW)
		gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, 0)
		clearSSBOFloats(r.accumSSBO, pixelCount*3)

		// Инициализируем variance buffer на GPU (9 float32 на пиксель: [mean, M2, count] × 3 канала)
		varianceSize := pixelCount * 9 * 4 // 9 значений по 4 байта (float32)
		gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, r.varianceSSBO)
		gl.BufferData(gl.SHADER_STORAGE_BUFFER, varianceSize, nil, gl.DYNAMIC_DRAW)
		gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, 0)
		clearSSBOFloats(r.varianceSSBO, pixelCount*9)

		// Убеждаемся, что все операции завершены перед началом нового рендеринга
		gl.MemoryBarrier(gl.ALL_BARRIER_BITS)
		gpuLogf("resized GPU buffers to %dx%d (accum=%dB variance=%dB)", cfg.Width, cfg.Height, accumSize, varianceSize)
	}

	// Загружаем материалы/объекты/камеру/небо
	gl.BindBufferBase(gl.SHADER_STORAGE_BUFFER, 3, r.matSSBO)
	gl.BufferData(gl.SHADER_STORAGE_BUFFER, len(matData)*4, gl.Ptr(matData), gl.DYNAMIC_DRAW)

	gl.BindBufferBase(gl.SHADER_STORAGE_BUFFER, 4, r.objSSBO)
	if len(objData) > 0 {
		gl.BufferData(gl.SHADER_STORAGE_BUFFER, len(objData)*4, gl.Ptr(objData), gl.DYNAMIC_DRAW)
	} else {
		// хотя бы маленький буфер, чтобы avoid GL errors при нуле объектов
		tmp := []float32{0}
		gl.BufferData(gl.SHADER_STORAGE_BUFFER, len(tmp)*4, gl.Ptr(tmp), gl.DYNAMIC_DRAW)
	}

	// Загружаем список индексов эмиссивных объектов
	gl.BindBufferBase(gl.SHADER_STORAGE_BUFFER, 6, r.lightSSBO)
	if len(lightIndices) > 0 {
		gl.BufferData(gl.SHADER_STORAGE_BUFFER, len(lightIndices)*4, gl.Ptr(lightIndices), gl.DYNAMIC_DRAW)
	} else {
		// хотя бы маленький буфер, чтобы avoid GL errors при нуле источников света
		tmp := []int32{0}
		gl.BufferData(gl.SHADER_STORAGE_BUFFER, len(tmp)*4, gl.Ptr(tmp), gl.DYNAMIC_DRAW)
	}

	gl.BindBufferBase(gl.UNIFORM_BUFFER, 1, r.camUBO)
	gl.BufferData(gl.UNIFORM_BUFFER, len(camBlock)*4, gl.Ptr(camBlock[:]), gl.DYNAMIC_DRAW)

	gl.BindBufferBase(gl.UNIFORM_BUFFER, 2, r.skyUBO)
	gl.BufferData(gl.UNIFORM_BUFFER, len(skyBlock)*4, gl.Ptr(skyBlock[:]), gl.DYNAMIC_DRAW)

	// Fog UBO (binding = 5)
	gl.BindBufferBase(gl.UNIFORM_BUFFER, 5, r.fogUBO)
	gl.BufferData(gl.UNIFORM_BUFFER, len(fogBlock)*4, gl.Ptr(fogBlock[:]), gl.DYNAMIC_DRAW)

	// Выбираем шейдер в зависимости от режима рендеринга
	isWireframe := GetRenderMode() == 1
	currentProgram := r.program
	if isWireframe {
		currentProgram = r.wireframeProgram
	}
	gl.UseProgram(currentProgram)

	// Bind image for compute shader: формат должен совпадать с внутренним форматом текстуры (RGBA16F).
	gl.BindImageTexture(0, r.imgTexture, 0, false, 0, gl.WRITE_ONLY, gl.RGBA16F)

	// Привязываем текстуру предыдущего кадра для Temporal Accumulation (только для path tracing)
	if !isWireframe && r.prevFrameTexture != 0 {
		gl.ActiveTexture(gl.TEXTURE9)
		gl.BindTexture(gl.TEXTURE_2D, r.prevFrameTexture)
	}

	// Bind накопительный буфер
	gl.BindBufferBase(gl.SHADER_STORAGE_BUFFER, 7, r.accumSSBO)

	// Bind variance buffer для адаптивного sampling (только для path tracing)
	if !isWireframe {
		gl.BindBufferBase(gl.SHADER_STORAGE_BUFFER, 8, r.varianceSSBO)
	}

	// Set uniforms (locations кэшируются на программу).
	r.cacheUniforms(currentProgram)
	locW := r.locs.locW
	locH := r.locs.locH
	locSpp := r.locs.locSpp
	locDepth := r.locs.locDepth
	locMaxRayDist := r.locs.locMaxRayDist
	locSeed := r.locs.locSeed
	locSampleCount := r.locs.locSampleCount
	locSelectedObject := r.locs.locSelectedObject

	gl.Uniform1i(locW, int32(cfg.Width))
	gl.Uniform1i(locH, int32(cfg.Height))
	gl.Uniform1i(locDepth, int32(cfg.MaxDepth))
	// Устанавливаем максимальную длину луча (по умолчанию 1000.0, если не задано)
	maxRayDist := cfg.MaxRayDist
	if maxRayDist <= 0 {
		maxRayDist = 1000.0 // Значение по умолчанию - достаточно для большинства сцен
	}
	gl.Uniform1f(locMaxRayDist, maxRayDist)
	gl.Uniform1i(locSelectedObject, int32(GetSelectedObject())) // Выбранный объект

	// Инициализируем накопительный буфер нулями при начале нового рендера
	// Для wireframe накопительный буфер не используется, поэтому не очищаем его
	if !isWireframe && r.width == cfg.Width && r.height == cfg.Height {
		clearT := startGPUTimer("clear_ssbo")
		accumSize := pixelCount * 3 * 4
		gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, r.accumSSBO)
		var bufSize int32
		gl.GetBufferParameteriv(gl.SHADER_STORAGE_BUFFER, gl.BUFFER_SIZE, &bufSize)
		if int(bufSize) != accumSize {
			gl.BufferData(gl.SHADER_STORAGE_BUFFER, accumSize, nil, gl.DYNAMIC_DRAW)
		}
		gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, 0)
		clearSSBOFloats(r.accumSSBO, pixelCount*3)

		varianceSize := pixelCount * 9 * 4
		gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, r.varianceSSBO)
		var varianceBufSize int32
		gl.GetBufferParameteriv(gl.SHADER_STORAGE_BUFFER, gl.BUFFER_SIZE, &varianceBufSize)
		if int(varianceBufSize) != varianceSize {
			gl.BufferData(gl.SHADER_STORAGE_BUFFER, varianceSize, nil, gl.DYNAMIC_DRAW)
		}
		gl.BindBuffer(gl.SHADER_STORAGE_BUFFER, 0)
		clearSSBOFloats(r.varianceSSBO, pixelCount*9)
		clearT.stop()
	}

	// Временный буфер для чтения данных с GPU (RGBA16F -> float32), переиспользуется
	tmp := r.ensureReadTmp(pixelCount)

	// Для wireframe режима семплов быть не должно - рендерим один кадр
	// Для path tracing используем семплы
	var passes int
	if isWireframe {
		passes = 1 // Wireframe рендерится одним кадром без семплов
	} else {
		passes = max(cfg.SamplesPerPx, 1)
	}

	// Вычисляем адаптивный интервал обновления в зависимости от размера изображения
	// Для больших разрешений обновляем чаще, чтобы UI не зависал
	updateInterval := 5 // По умолчанию каждый 5-й сэмпл
	if pixelCount > 2000*2000 {
		// Очень большие разрешения (> 4M пикселей) - обновляем каждый сэмпл
		updateInterval = 1
	} else if pixelCount > 1000*1000 {
		// Большие разрешения (1M-4M пикселей) - обновляем каждый 2-й сэмпл
		updateInterval = 2
	} else if pixelCount > 500*500 {
		// Средние разрешения (250K-1M пикселей) - обновляем каждый 3-й сэмпл
		updateInterval = 3
	}

	// Сбрасываем флаг отмены перед началом рендеринга
	SetRenderCancel(false)

	wantLiveReadback := (LivePreviewEnabled() && progress != nil) || isWireframe
	lastSampleHadReadback := false
	dispatchTotal := time.Duration(0)
	readbackTotal := time.Duration(0)
	readbackCount := 0

	gpuLogf("sample loop passes=%d updateInterval=%d live_readback=%v", passes, updateInterval, wantLiveReadback)

	for s := 0; s < passes; s++ {
		if IsRenderCancelled() {
			gpuLogf("cancelled before sample %d/%d", s, passes)
			break
		}

		if !isWireframe {
			gl.Uniform1i(locSpp, 1)
			gl.Uniform1i(locSampleCount, int32(s+1))
			gl.Uniform1ui(locSeed, uint32(time.Now().UnixNano())+uint32(s))
		} else {
			gl.Uniform1i(locSpp, 1)
			gl.Uniform1i(locSampleCount, 1)
			gl.Uniform1ui(locSeed, uint32(time.Now().UnixNano()))
		}
		gl.Uniform1i(locSelectedObject, int32(GetSelectedObject()))

		dispStart := time.Now()
		groupsX := (cfg.Width + 31) / 32
		groupsY := (cfg.Height + 31) / 32
		gl.DispatchCompute(uint32(groupsX), uint32(groupsY), 1)
		gl.MemoryBarrier(gl.SHADER_IMAGE_ACCESS_BARRIER_BIT)
		if gpuDebugEnabled() {
			dispatchTotal += time.Since(dispStart)
		}

		// Mid-render readback только для live UI / wireframe; иначе — один финальный.
		isLast := (s + 1) == passes
		shouldUpdate := isWireframe || (wantLiveReadback && ((s+1)%updateInterval == 0 || isLast))

		if shouldUpdate {
			rbStart := time.Now()
			if isWireframe {
				gl.MemoryBarrier(gl.SHADER_IMAGE_ACCESS_BARRIER_BIT | gl.TEXTURE_FETCH_BARRIER_BIT)
				gl.BindBuffer(gl.PIXEL_PACK_BUFFER, r.pbo)
				gl.BindTexture(gl.TEXTURE_2D, r.imgTexture)
				gl.GetTexImage(gl.TEXTURE_2D, 0, gl.RGBA, gl.FLOAT, nil)
				gl.MemoryBarrier(gl.PIXEL_BUFFER_BARRIER_BIT)
				ptr := gl.MapBuffer(gl.PIXEL_PACK_BUFFER, gl.READ_ONLY)
				if ptr != nil {
					src := (*[1 << 28]float32)(ptr)[:len(tmp)]
					copy(tmp, src)
					gl.UnmapBuffer(gl.PIXEL_PACK_BUFFER)
				} else {
					for i := 0; i < len(tmp); i += 4 {
						if i+3 < len(tmp) {
							tmp[i], tmp[i+1], tmp[i+2], tmp[i+3] = 0.1, 0.1, 0.1, 1.0
						}
					}
				}
				gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)
			} else {
				currentPBO := r.pbo
				if r.pboIndex == 1 {
					currentPBO = r.pboBack
				}
				if s > 0 {
					readPBO := r.pboBack
					if r.pboIndex == 1 {
						readPBO = r.pbo
					}
					gl.BindBuffer(gl.PIXEL_PACK_BUFFER, readPBO)
					ptr := gl.MapBuffer(gl.PIXEL_PACK_BUFFER, gl.READ_ONLY)
					if ptr != nil {
						src := (*[1 << 28]float32)(ptr)[:len(tmp)]
						copy(tmp, src)
						gl.UnmapBuffer(gl.PIXEL_PACK_BUFFER)
					}
				}
				gl.BindBuffer(gl.PIXEL_PACK_BUFFER, currentPBO)
				gl.BindTexture(gl.TEXTURE_2D, r.imgTexture)
				gl.GetTexImage(gl.TEXTURE_2D, 0, gl.RGBA, gl.FLOAT, nil)
				if isLast {
					gl.MemoryBarrier(gl.PIXEL_BUFFER_BARRIER_BIT)
					ptr := gl.MapBuffer(gl.PIXEL_PACK_BUFFER, gl.READ_ONLY)
					if ptr != nil {
						src := (*[1 << 28]float32)(ptr)[:len(tmp)]
						copy(tmp, src)
						gl.UnmapBuffer(gl.PIXEL_PACK_BUFFER)
					}
				}
				r.pboIndex = 1 - r.pboIndex
				gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)
			}

			dst := img.Pix
			maxPixels := len(dst) / 4
			actualPixelCount := min(maxPixels, pixelCount)

			if isWireframe {
				for i := 0; i < actualPixelCount; i++ {
					off := i * 4
					if off+3 >= len(dst) || off+3 >= len(tmp) {
						break
					}
					rVal, gVal, bVal := tmp[off], tmp[off+1], tmp[off+2]
					if rVal < 0 {
						rVal = 0
					} else if rVal > 1 {
						rVal = 1
					}
					if gVal < 0 {
						gVal = 0
					} else if gVal > 1 {
						gVal = 1
					}
					if bVal < 0 {
						bVal = 0
					} else if bVal > 1 {
						bVal = 1
					}
					dst[off] = uint8(rVal*255.0 + 0.5)
					dst[off+1] = uint8(gVal*255.0 + 0.5)
					dst[off+2] = uint8(bVal*255.0 + 0.5)
					dst[off+3] = 255
				}
			} else {
				tonemapHDRToRGBA(dst, tmp, actualPixelCount)
			}
			lastSampleHadReadback = isLast || !wantLiveReadback
			if isLast {
				lastSampleHadReadback = true
			}
			if gpuDebugEnabled() {
				readbackTotal += time.Since(rbStart)
				readbackCount++
			}
		}

		// Temporal: копирование HDR-кадра (на softGL отключено — иначе CPU копирует 1080p каждый сэмпл).
		if !isWireframe && r.prevFrameTexture != 0 && TemporalEnabled() {
			r.copyFrameToPrev(cfg.Width, cfg.Height)
		}

		if progress != nil {
			progress(s+1, passes)
		}
		if IsRenderCancelled() {
			gpuLogf("cancelled after sample %d/%d", s+1, passes)
			break
		}
	}

	if gpuDebugEnabled() {
		gpuLogf("dispatch_total=%s readbacks=%d readback_total=%s",
			dispatchTotal.Round(time.Microsecond), readbackCount, readbackTotal.Round(time.Microsecond))
	}

	// Финальный readback только если mid-loop его ещё не сделал на последнем сэмпле.
	needFinalReadback := !isWireframe || !lastSampleHadReadback
	if !wantLiveReadback {
		needFinalReadback = true
	}
	if lastSampleHadReadback && wantLiveReadback {
		needFinalReadback = false
	}
	// Wireframe всегда уже прочитан в цикле.
	if isWireframe {
		needFinalReadback = false
	}

	if needFinalReadback {
		finalRB := startGPUTimer("final_readback")
		gl.MemoryBarrier(
			gl.PIXEL_BUFFER_BARRIER_BIT | gl.TEXTURE_FETCH_BARRIER_BIT | gl.SHADER_IMAGE_ACCESS_BARRIER_BIT,
		)

		currentPBO := r.pbo
		if r.pboIndex == 1 {
			currentPBO = r.pboBack
		}

		gl.BindBuffer(gl.PIXEL_PACK_BUFFER, currentPBO)
		gl.BindTexture(gl.TEXTURE_2D, r.imgTexture)
		gl.GetTexImage(gl.TEXTURE_2D, 0, gl.RGBA, gl.FLOAT, nil)
		gl.MemoryBarrier(gl.PIXEL_BUFFER_BARRIER_BIT)

		ptr := gl.MapBuffer(gl.PIXEL_PACK_BUFFER, gl.READ_ONLY)
		dst := img.Pix
		if ptr != nil {
			tmp := r.ensureReadTmp(pixelCount)
			src := (*[1 << 28]float32)(ptr)[:len(tmp)]
			copy(tmp, src)
			gl.UnmapBuffer(gl.PIXEL_PACK_BUFFER)
			tonemapHDRToRGBA(dst, tmp, pixelCount)
		}
		gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0)
		finalRB.stop()
	} else {
		gpuLogf("skip duplicate final_readback (already have last sample)")
	}

	cfgDN := getDenoiseConfig()
	w := cfg.Width
	h := cfg.Height
	dst := img.Pix
	// Bilateral на CPU: на preview (мало spp) пропускаем — дорогой и мало полезен.
	runDenoise := cfgDN.Enabled && w > 2 && h > 2 && cfg.SamplesPerPx >= 4
	if cfgDN.Enabled && cfg.SamplesPerPx < 4 {
		gpuLogf("skip cpu_denoise for preview spp=%d", cfg.SamplesPerPx)
	}
	if runDenoise {
		dnT := startGPUTimer("cpu_denoise")
		smoothed := make([]byte, len(dst))
		sigmaS := cfgDN.SigmaS
		sigmaR := cfgDN.SigmaR
		twoSigmaS2 := 2 * sigmaS * sigmaS
		twoSigmaR2 := 2 * sigmaR * sigmaR

		for y := range h {
			for x := range w {
				centerIdx := (y*w + x) * 4
				cr := float64(dst[centerIdx]) / 255.0
				cg := float64(dst[centerIdx+1]) / 255.0
				cb := float64(dst[centerIdx+2]) / 255.0

				var sumR, sumG, sumB, sumW float64
				for ky := -1; ky <= 1; ky++ {
					ny := y + ky
					if ny < 0 || ny >= h {
						continue
					}
					for kx := -1; kx <= 1; kx++ {
						nx := x + kx
						if nx < 0 || nx >= w {
							continue
						}
						ni := (ny*w + nx) * 4
						nr := float64(dst[ni]) / 255.0
						ng := float64(dst[ni+1]) / 255.0
						nb := float64(dst[ni+2]) / 255.0

						ds2 := float64(kx*kx + ky*ky)
						dr := cr - nr
						dg := cg - ng
						dbb := cb - nb
						dr2 := dr*dr + dg*dg + dbb*dbb

						ws := math.Exp(-ds2 / twoSigmaS2)
						wr := math.Exp(-dr2 / twoSigmaR2)
						wgt := ws * wr

						sumW += wgt
						sumR += nr * wgt
						sumG += ng * wgt
						sumB += nb * wgt
					}
				}

				if sumW > 0 {
					nr := sumR / sumW
					ng := sumG / sumW
					nb := sumB / sumW
					if nr < 0 {
						nr = 0
					} else if nr > 1 {
						nr = 1
					}
					if ng < 0 {
						ng = 0
					} else if ng > 1 {
						ng = 1
					}
					if nb < 0 {
						nb = 0
					} else if nb > 1 {
						nb = 1
					}
					smoothed[centerIdx] = uint8(nr*255.0 + 0.5)
					smoothed[centerIdx+1] = uint8(ng*255.0 + 0.5)
					smoothed[centerIdx+2] = uint8(nb*255.0 + 0.5)
					smoothed[centerIdx+3] = 255
				} else {
					smoothed[centerIdx] = dst[centerIdx]
					smoothed[centerIdx+1] = dst[centerIdx+1]
					smoothed[centerIdx+2] = dst[centerIdx+2]
					smoothed[centerIdx+3] = dst[centerIdx+3]
				}
			}
		}
		copy(dst, smoothed)
		dnT.stop()
	}

	// Дополнительное, более сильное сглаживание (blur) по всему изображению.
	// Используется простое box-усреднение с радиусом 1..5, затем смешивание
	// с оригиналом по коэффициенту strength (0..1).
	cfgSM := getSmoothConfig()
	runSmooth := cfgSM.Enabled && w > 2 && h > 2 && cfgSM.Radius > 0 && cfgSM.Strength > 0 && cfg.SamplesPerPx >= 4
	if runSmooth {
		smT := startGPUTimer("cpu_smooth")
		rad := max(cfgSM.Radius, 1)
		if rad > 5 {
			rad = 5
		}
		str := cfgSM.Strength
		if str < 0 {
			str = 0
		}
		if str > 1 {
			str = 1
		}

		blurred := make([]byte, len(dst))
		for y := range h {
			for x := range w {
				var sumR, sumG, sumB float64
				var count float64
				for ky := -rad; ky <= rad; ky++ {
					ny := y + ky
					if ny < 0 || ny >= h {
						continue
					}
					for kx := -rad; kx <= rad; kx++ {
						nx := x + kx
						if nx < 0 || nx >= w {
							continue
						}
						ni := (ny*w + nx) * 4
						sumR += float64(dst[ni])
						sumG += float64(dst[ni+1])
						sumB += float64(dst[ni+2])
						count++
					}
				}
				if count > 0 {
					avgR := sumR / count
					avgG := sumG / count
					avgB := sumB / count
					ci := (y*w + x) * 4
					origR := float64(dst[ci])
					origG := float64(dst[ci+1])
					origB := float64(dst[ci+2])

					outR := (1-str)*origR + str*avgR
					outG := (1-str)*origG + str*avgG
					outB := (1-str)*origB + str*avgB

					if outR < 0 {
						outR = 0
					} else if outR > 255 {
						outR = 255
					}
					if outG < 0 {
						outG = 0
					} else if outG > 255 {
						outG = 255
					}
					if outB < 0 {
						outB = 0
					} else if outB > 255 {
						outB = 255
					}

					blurred[ci] = byte(outR + 0.5)
					blurred[ci+1] = byte(outG + 0.5)
					blurred[ci+2] = byte(outB + 0.5)
					blurred[ci+3] = 255
				}
			}
		}
		copy(dst, blurred)
		smT.stop()
	}

	if progress != nil {
		progress(cfg.SamplesPerPx, cfg.SamplesPerPx)
	}

	return nil
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

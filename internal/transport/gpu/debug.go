package gpu

import (
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	enabledValue  = "true"
	disabledValue = "false"
	positiveValue = "yes"
	negativeValue = "off"
)

// PATHTRACER_GPU_DEBUG: 1/true/on — подробные тайминги GPU-пайплайна в лог.
// PATHTRACER_GPU_LIVE_READBACK: 0 — не читать кадр на каждом N-м сэмпле (только финал).
// PATHTRACER_GPU_TEMPORAL: 0 — отключить temporal accumulation каждый сэмпл.
// PATHTRACER_GPU_TILE_SIZE: размер tile-dispatch (32..512, по умолчанию 256).
// PATHTRACER_GPU_SAMPLES_PER_PASS: бюджет лучей за один проход (1..16, по умолчанию 4).

var (
	gpuDebugOnce      sync.Once
	gpuDebugOn        atomic.Bool
	gpuTileSize       atomic.Int64
	gpuSamplesPerPass atomic.Int64

	livePreview atomic.Bool
	temporalOn  atomic.Bool
	softGL      atomic.Bool

	glVendor   string
	glRenderer string
	glVersion  string
	glInfoMu   sync.RWMutex
)

func ensureGPUDebugConfig() {
	gpuDebugOnce.Do(configureGPUDebug)
}

func configureGPUDebug() {
	livePreview.Store(readEnvBool("PATHTRACER_GPU_LIVE_READBACK", true))
	temporalOn.Store(readEnvBool("PATHTRACER_GPU_TEMPORAL", true))
	gpuDebugOn.Store(readEnvBool("PATHTRACER_GPU_DEBUG", false))
	gpuTileSize.Store(int64(readIntEnv(
		"PATHTRACER_GPU_TILE_SIZE",
		gpuDispatchTileSizeDefault,
		gpuDispatchTileSizeMin,
		gpuDispatchTileSizeMax,
	)))
	gpuSamplesPerPass.Store(int64(readIntEnv(
		"PATHTRACER_GPU_SAMPLES_PER_PASS",
		gpuSamplesPerPassDefault,
		gpuSamplesPerPassMin,
		gpuSamplesPerPassMax,
	)))
	if gpuDebugOn.Load() {
		log.Printf(
			"[gpu] debug logging enabled (live_readback=%v temporal=%v)",
			livePreview.Load(),
			temporalOn.Load(),
		)
	}
}

func dispatchTileSize() int {
	ensureGPUDebugConfig()
	return int(gpuTileSize.Load()) / gpuWorkgroupSize * gpuWorkgroupSize
}

func samplesPerPass() int {
	ensureGPUDebugConfig()
	return int(gpuSamplesPerPass.Load())
}

func readEnvBool(name string, fallback bool) bool {
	value := strings.ToLower(os.Getenv(name))
	if value == "" {
		return fallback
	}
	switch value {
	case "0", disabledValue, negativeValue, "no":
		return false
	case "1", enabledValue, "on", positiveValue:
		return true
	default:
		return fallback
	}
}

// SetLivePreview включает/выключает промежуточный readback кадра для live-UI.
func SetLivePreview(enabled bool) {
	ensureGPUDebugConfig()
	livePreview.Store(enabled)
}

// LivePreviewEnabled сообщает, нужен ли mid-render GPU→CPU transfer.
func LivePreviewEnabled() bool {
	ensureGPUDebugConfig()
	return livePreview.Load()
}

// TemporalEnabled включает temporal accumulation через ping-pong текстуры.
func TemporalEnabled() bool {
	ensureGPUDebugConfig()
	return temporalOn.Load()
}

// IsSoftwareGL сообщает, что OpenGL — software (llvmpipe и т.п.): compute идёт на CPU.
func IsSoftwareGL() bool {
	return softGL.Load()
}

// GLInfo возвращает строки драйвера после initGL.
func GLInfo() (vendor, renderer, version string) {
	glInfoMu.RLock()
	defer glInfoMu.RUnlock()
	return glVendor, glRenderer, glVersion
}

func setGLInfo(vendor, renderer, version string) {
	ensureGPUDebugConfig()
	glInfoMu.Lock()
	glVendor, glRenderer, glVersion = vendor, renderer, version
	glInfoMu.Unlock()
	lower := strings.ToLower(renderer + " " + vendor)
	soft := strings.Contains(lower, "llvmpipe") ||
		strings.Contains(lower, "softpipe") ||
		strings.Contains(lower, "swiftshader") ||
		strings.Contains(lower, "software") ||
		strings.Contains(lower, "mesa offscreen") ||
		strings.Contains(lower, "microsoft basic render")
	softGL.Store(soft)
	if soft {
		// Compute-rendering на softGL выполняется на CPU.
		if os.Getenv("PATHTRACER_GPU_TEMPORAL") == "" {
			temporalOn.Store(false)
		}
		log.Print("[gpu] WARNING: software OpenGL detected — compute runs on CPU, discrete GPU stays idle")
		log.Printf("[gpu]   GL_VENDOR   = %s", vendor)
		log.Printf("[gpu]   GL_RENDERER = %s", renderer)
		log.Printf("[gpu]   GL_VERSION  = %s", version)
		log.Print("[gpu]   On WSL use a Windows-native build or GPU passthrough for real GPU load")
	} else {
		log.Printf("[gpu] GL_VENDOR=%s", vendor)
		log.Printf("[gpu] GL_RENDERER=%s", renderer)
		log.Printf("[gpu] GL_VERSION=%s", version)
	}
}

func gpuDebugEnabled() bool {
	ensureGPUDebugConfig()
	return gpuDebugOn.Load()
}

func gpuLogf(format string, args ...any) {
	if !gpuDebugEnabled() {
		return
	}
	log.Printf("[gpu] "+format, args...)
}

type gpuTimer struct {
	start time.Time
	name  string
}

func startGPUTimer(name string) gpuTimer {
	if !gpuDebugEnabled() {
		return gpuTimer{}
	}
	return gpuTimer{name: name, start: time.Now()}
}

func (t gpuTimer) stop() {
	if t.name == "" || t.start.IsZero() {
		return
	}
	gpuLogf("%s: %s", t.name, time.Since(t.start).Round(time.Microsecond))
}

func (t gpuTimer) stopf(format string, args ...any) {
	if t.name == "" || t.start.IsZero() {
		return
	}
	suffix := fmt.Sprintf(format, args...)
	gpuLogf("%s %s: %s", t.name, suffix, time.Since(t.start).Round(time.Microsecond))
}

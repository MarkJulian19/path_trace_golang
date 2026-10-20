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
// PATHTRACER_GPU_TEMPORAL: 0 — отключить CopyImageSubData каждый сэмпл.

var (
	gpuDebugOnce sync.Once
	gpuDebugOn   atomic.Bool

	livePreview atomic.Bool
	temporalOn  atomic.Bool
	softGL      atomic.Bool

	glVendor   string
	glRenderer string
	glVersion  string
)

func ensureGPUDebugConfig() {
	gpuDebugOnce.Do(configureGPUDebug)
}

func configureGPUDebug() {
	livePreview.Store(readEnvBool("PATHTRACER_GPU_LIVE_READBACK", true))
	temporalOn.Store(readEnvBool("PATHTRACER_GPU_TEMPORAL", true))
	gpuDebugOn.Store(readEnvBool("PATHTRACER_GPU_DEBUG", false))
	if gpuDebugOn.Load() {
		log.Printf(
			"[gpu] debug logging enabled (live_readback=%v temporal=%v)",
			livePreview.Load(),
			temporalOn.Load(),
		)
	}
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

// TemporalEnabled — копирование кадра в prevFrameTexture каждый сэмпл.
func TemporalEnabled() bool {
	ensureGPUDebugConfig()
	return temporalOn.Load() && !softGL.Load()
}

// IsSoftwareGL сообщает, что OpenGL — software (llvmpipe и т.п.): compute идёт на CPU.
func IsSoftwareGL() bool {
	return softGL.Load()
}

// GLInfo возвращает строки драйвера после initGL.
func GLInfo() (vendor, renderer, version string) {
	return glVendor, glRenderer, glVersion
}

func setGLInfo(vendor, renderer, version string) {
	glVendor, glRenderer, glVersion = vendor, renderer, version
	lower := strings.ToLower(renderer + " " + vendor)
	soft := strings.Contains(lower, "llvmpipe") ||
		strings.Contains(lower, "softpipe") ||
		strings.Contains(lower, "swiftshader") ||
		strings.Contains(lower, "software") ||
		strings.Contains(lower, "mesa offscreen") ||
		strings.Contains(lower, "microsoft basic render")
	softGL.Store(soft)
	if soft {
		// CopyImageSubData 1080p×N на softGL — чистый CPU.
		temporalOn.Store(false)
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
	name  string
	start time.Time
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

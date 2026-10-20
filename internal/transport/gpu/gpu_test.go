package gpu_test

import (
	"testing"

	"github.com/MarkJulian19/path_trace_golang/internal/transport/gpu"
)

func BenchmarkGPUStateAPI(b *testing.B) {
	// Initialize sync.Once-backed state before timing the steady-state access path.
	gpu.LivePreviewEnabled()
	gpu.TemporalEnabled()

	b.Run("SetDenoiseConfigFromUI", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			gpu.SetDenoiseConfigFromUI(true, 1, 0.15)
		}
	})
	b.Run("SetSmoothConfigFromUI", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			gpu.SetSmoothConfigFromUI(true, 2, 0.5)
		}
	})
	b.Run("SetRenderCancel", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			gpu.SetRenderCancel(true)
		}
	})
	b.Run("IsRenderCancelled", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkGPUCancelled = gpu.IsRenderCancelled()
		}
	})
	b.Run("SetRenderModeFromUI", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			gpu.SetRenderModeFromUI(1)
		}
	})
	b.Run("GetRenderMode", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkGPUMode = gpu.GetRenderMode()
		}
	})
	b.Run("SetSelectedObjectFromUI", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			gpu.SetSelectedObjectFromUI(1)
		}
	})
	b.Run("GetSelectedObject", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkGPUSelectedObject = gpu.GetSelectedObject()
		}
	})
	b.Run("SetLivePreview", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			gpu.SetLivePreview(true)
		}
	})
	b.Run("LivePreviewEnabled", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkGPULivePreview = gpu.LivePreviewEnabled()
		}
	})
	b.Run("TemporalEnabled", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkGPUTemporal = gpu.TemporalEnabled()
		}
	})
	b.Run("IsSoftwareGL", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkGPUSoftware = gpu.IsSoftwareGL()
		}
	})
	b.Run("GLInfo", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkGPUVendor, benchmarkGPURenderer, benchmarkGPUVersion = gpu.GLInfo()
		}
	})
}

func BenchmarkGPUAdapterStateAPI(b *testing.B) {
	adapter := gpu.Adapter{}
	b.Run("SetSelectedObjectFromUI", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			adapter.SetSelectedObjectFromUI(1)
		}
	})
	b.Run("SetLivePreview", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			adapter.SetLivePreview(true)
		}
	})
	b.Run("GetRenderMode", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkGPUMode = adapter.GetRenderMode()
		}
	})
	b.Run("SetRenderCancel", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			adapter.SetRenderCancel(true)
		}
	})
	b.Run("SetDenoiseConfigFromUI", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			adapter.SetDenoiseConfigFromUI(true, 1, 0.15)
		}
	})
	b.Run("SetSmoothConfigFromUI", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			adapter.SetSmoothConfigFromUI(true, 2, 0.5)
		}
	})
	b.Run("SetRenderModeFromUI", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			adapter.SetRenderModeFromUI(1)
		}
	})
}

var (
	benchmarkGPUCancelled      bool
	benchmarkGPUMode           int
	benchmarkGPUSelectedObject int
	benchmarkGPULivePreview    bool
	benchmarkGPUTemporal       bool
	benchmarkGPUSoftware       bool
	benchmarkGPUVendor         string
	benchmarkGPURenderer       string
	benchmarkGPUVersion        string
)

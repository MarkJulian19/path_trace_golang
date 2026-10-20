//go:build benchinternal

package gpu

import "testing"

func BenchmarkInternalGPUPostprocess(b *testing.B) {
	b.Run("acesTonemap", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalTonemap = acesTonemap(2)
		}
	})
	b.Run("ensureReadTmp", func(b *testing.B) {
		renderer := &gpuRenderer{}
		renderer.ensureReadTmp(1920 * 1080)
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalReadTmp = renderer.ensureReadTmp(1920 * 1080)
		}
	})
	b.Run("tonemapHDRToRGBA", func(b *testing.B) {
		pixelCount := 32 * 18
		tmp := make([]float32, pixelCount*4)
		dst := make([]uint8, pixelCount*4)
		for i := range tmp {
			tmp[i] = 0.5
		}
		b.SetBytes(int64(pixelCount * 4))
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			tonemapHDRToRGBA(dst, tmp, pixelCount)
		}
	})
}

func BenchmarkInternalGPUConfigReads(b *testing.B) {
	SetDenoiseConfigFromUI(true, 1, 0.15)
	SetSmoothConfigFromUI(true, 2, 0.5)
	b.Run("getDenoiseConfig", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalDenoise = getDenoiseConfig()
		}
	})
	b.Run("getSmoothConfig", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			benchmarkInternalSmooth = getSmoothConfig()
		}
	})
}

var (
	benchmarkInternalTonemap float32
	benchmarkInternalReadTmp []float32
	benchmarkInternalDenoise denoiseConfig
	benchmarkInternalSmooth  smoothConfig
)

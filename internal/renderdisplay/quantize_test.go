package renderdisplay_test

import (
	"math"
	"testing"

	"github.com/MarkJulian19/path_trace_golang/internal/renderdisplay"
)

func TestQuantizationPreservesFractionalMean(t *testing.T) {
	t.Parallel()
	const count = 1 << 20
	for _, value := range []float64{0, 0.2, 12.25, 127.5, 254.9, 255} {
		var total float64
		for pixel := range count {
			threshold := renderdisplay.Dither(pixel)
			if threshold < 0 || threshold >= 1 || threshold != renderdisplay.Dither(pixel) {
				t.Fatal("dither is unstable or outside [0, 1)")
			}
			got := float64(renderdisplay.Byte(value, threshold))
			if math.Abs(got-value) >= 1 {
				t.Fatalf("quantization error exceeds one level: %v -> %v", value, got)
			}
			total += got
		}
		if math.Abs(total/count-value) > 0.003 {
			t.Errorf("fractional gradient was lost: input=%v mean=%v", value, total/count)
		}
	}
}

func TestQuantizationBounds(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		value float64
		want  byte
	}{
		{math.NaN(), 0}, {math.Inf(-1), 0}, {-1, 0}, {256, 255}, {math.Inf(1), 255},
	} {
		if got := renderdisplay.Byte(tc.value, 0.5); got != tc.want {
			t.Errorf("quantize(%v)=%d, want %d", tc.value, got, tc.want)
		}
	}
}

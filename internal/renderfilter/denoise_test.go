package renderfilter_test

import (
	"bytes"
	"image"
	"image/color"
	"runtime"
	"sync/atomic"
	"testing"

	"github.com/MarkJulian19/path_trace_golang/internal/renderfilter"
)

const (
	channels     = 3
	rgbaChannels = 4
)

func TestDenoiseReducesNoiseAndPreservesEdges(t *testing.T) {
	t.Parallel()
	const width, height = 65, 65
	pixels := make([]byte, width*height*rgbaChannels)
	before := 0.0
	for i := range width * height {
		value := 128 + (i*17%41 - 20)
		for c := range channels {
			pixels[i*rgbaChannels+c] = byte(value)
			before += float64((value - 128) * (value - 128))
		}
		pixels[i*rgbaChannels+3] = 255
	}
	renderfilter.Apply(pixels, width, height, renderfilter.Config{Enabled: true}, nil)
	after := 0.0
	for i := range width * height {
		for c := range channels {
			delta := float64(pixels[i*rgbaChannels+c]) - 128
			after += delta * delta
		}
	}
	if after > before*0.25 {
		t.Fatalf("filter failed to reduce noise: before=%v after=%v", before, after)
	}
	for y := range height {
		for x := range width {
			value := byte(0)
			if x >= width/2 {
				value = 255
			}
			for c := range channels {
				pixels[(y*width+x)*rgbaChannels+c] = value
			}
		}
	}
	renderfilter.Apply(pixels, width, height, renderfilter.Config{Enabled: true}, nil)
	if pixels[(width/2-1)*rgbaChannels] > 2 || pixels[(width/2)*rgbaChannels] < 253 {
		t.Fatal("high-contrast edge was blurred")
	}
}

func TestDenoisePreservesSubimagePadding(t *testing.T) {
	t.Parallel()
	parent := image.NewRGBA(image.Rect(0, 0, 12, 12))
	for i := range parent.Pix {
		parent.Pix[i] = 42
	}
	child, ok := parent.SubImage(image.Rect(2, 2, 10, 10)).(*image.RGBA)
	if !ok {
		t.Fatal("expected RGBA subimage")
	}
	for y := range 8 {
		for x := range 8 {
			child.SetRGBA(x+2, y+2, color.RGBA{R: 127, G: 127, B: 127, A: 255})
		}
	}
	original := bytes.Clone(parent.Pix)
	renderfilter.ApplyRGBA(child, renderfilter.Config{Enabled: true}, nil)
	if !bytes.Equal(original, parent.Pix) {
		t.Fatal("filter changed a constant subimage or the surrounding padding")
	}
}

func TestCancellationJoinsFilterWorkers(t *testing.T) {
	t.Parallel()
	pixels := bytes.Repeat([]byte{100, 150, 200, 255}, 128*128)
	original := bytes.Clone(pixels)
	var calls atomic.Int64
	complete := renderfilter.Apply(
		pixels,
		128,
		128,
		renderfilter.Config{Enabled: true},
		func() bool { calls.Add(1); return true },
	)
	if complete || !bytes.Equal(pixels, original) {
		t.Fatal("cancelled filter published a partial image")
	}
	joined := calls.Load()
	for range 100 {
		runtime.Gosched()
	}
	if calls.Load() != joined {
		t.Fatal("filter worker survived cancellation")
	}
}

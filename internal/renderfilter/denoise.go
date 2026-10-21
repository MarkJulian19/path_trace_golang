package renderfilter

import (
	"image"
	"math"
	"runtime"
	"sync"

	"github.com/MarkJulian19/path_trace_golang/internal/renderdisplay"
)

const (
	channels         = 3
	rgbaChannels     = 4
	byteScale        = 255
	filterPasses     = 3
	parallelMinRows  = 64
	defaultSigmaR    = 0.15
	minimumColorBand = 0.0001
)

// Config задаёт необязательный фильтр изображения после преобразования HDR.
type Config struct {
	SigmaS  float64
	SigmaR  float64
	Enabled bool
}

// Normalize сохраняет допустимую ширину пространственного и цветового фильтров.
func Normalize(cfg Config) Config {
	if cfg.SigmaS <= 0 || math.IsNaN(cfg.SigmaS) || math.IsInf(cfg.SigmaS, 0) {
		cfg.SigmaS = 1
	}
	if cfg.SigmaR <= 0 || math.IsNaN(cfg.SigmaR) || math.IsInf(cfg.SigmaR, 0) {
		cfg.SigmaR = defaultSigmaR
	}
	return cfg
}

type buffers struct{ first, second []float32 }

var bufferPool sync.Pool

// ApplyRGBA сохраняет границы и шаг строк изображения, включая SubImage.
func ApplyRGBA(img *image.RGBA, cfg Config, cancelled func() bool) bool {
	width, height := img.Bounds().Dx(), img.Bounds().Dy()
	if img.Stride == width*rgbaChannels {
		return Apply(img.Pix, width, height, cfg, cancelled)
	}
	packed := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		copy(packed.Pix[y*packed.Stride:(y+1)*packed.Stride], img.Pix[y*img.Stride:y*img.Stride+width*rgbaChannels])
	}
	if !Apply(packed.Pix, width, height, cfg, cancelled) {
		return false
	}
	for y := range height {
		copy(img.Pix[y*img.Stride:y*img.Stride+width*rgbaChannels], packed.Pix[y*packed.Stride:(y+1)*packed.Stride])
	}
	return true
}

// Apply подавляет зерно на трёх масштабах, сохраняя цветовые границы и исходный HDR.
// Отмена оставляет исходное изображение и дожидается всех рабочих горутин.
func Apply(pixels []byte, width, height int, cfg Config, cancelled func() bool) bool {
	if !cfg.Enabled || !validImage(pixels, width, height) {
		return true
	}
	cfg = Normalize(cfg)
	data := acquireBuffers(width * height * channels)
	defer bufferPool.Put(data)
	decodeRGB(data.first, pixels, width*height)
	source, destination := data.first, data.second
	weights := spatialWeights(cfg.SigmaS)
	band := float32(1 / (2 * max(cfg.SigmaR*cfg.SigmaR, minimumColorBand) * byteScale * byteScale))
	for pass := range filterPasses {
		filterRows(source, destination, width, height, 1<<pass, weights, band, cancelled)
		if cancelled != nil && cancelled() {
			return false
		}
		source, destination = destination, source
	}
	writeRGB(pixels, source, width*height)
	return true
}

func acquireBuffers(size int) *buffers {
	data, ok := bufferPool.Get().(*buffers)
	if !ok {
		data = &buffers{}
	}
	if cap(data.first) < size {
		data.first, data.second = make([]float32, size), make([]float32, size)
	}
	data.first, data.second = data.first[:size], data.second[:size]
	return data
}

func spatialWeights(sigma float64) [9]float32 {
	var weights [9]float32
	for y := -1; y <= 1; y++ {
		for x := -1; x <= 1; x++ {
			weights[(y+1)*channels+x+1] = float32(math.Exp(-float64(x*x+y*y) / (2 * sigma * sigma)))
		}
	}
	return weights
}

func filterRows(source, destination []float32, width, height, step int, weights [9]float32, band float32,
	cancelled func() bool,
) {
	workers := min(runtime.GOMAXPROCS(0), height)
	if height < parallelMinRows {
		workers = 1
	}
	var group sync.WaitGroup
	for worker := range workers {
		group.Go(func() {
			for y := worker * height / workers; y < (worker+1)*height/workers; y++ {
				if cancelled != nil && cancelled() {
					return
				}
				for x := range width {
					filterPixel(source, destination, width, height, x, y, step, weights, band)
				}
			}
		})
	}
	group.Wait()
}

func filterPixel(source, destination []float32, width, height, x, y, step int, weights [9]float32, band float32) {
	center := (y*width + x) * channels
	var sum [3]float32
	var sumWeight float32
	for ky := -1; ky <= 1; ky++ {
		for kx := -1; kx <= 1; kx++ {
			nx, ny := x+kx*step, y+ky*step
			if nx < 0 || nx >= width || ny < 0 || ny >= height {
				continue
			}
			index := (ny*width + nx) * channels
			dr, dg, db := source[index]-source[center], source[index+1]-source[center+1], source[index+2]-source[center+2]
			rangeWeight := 1 + (dr*dr+dg*dg+db*db)*band
			weight := weights[(ky+1)*channels+kx+1] / (rangeWeight * rangeWeight)
			sumWeight += weight
			for c := range channels {
				sum[c] += source[index+c] * weight
			}
		}
	}
	for c := range channels {
		destination[center+c] = sum[c] / sumWeight
	}
}

func validImage(pixels []byte, width, height int) bool {
	return width > 0 && height > 0 && width <= len(pixels)/rgbaChannels/height
}

func decodeRGB(destination []float32, pixels []byte, count int) {
	for i := range count {
		for c := range channels {
			destination[i*channels+c] = float32(pixels[i*rgbaChannels+c])
		}
	}
}

func writeRGB(pixels []byte, source []float32, count int) {
	for i := range count {
		threshold := renderdisplay.Dither(i)
		for c := range channels {
			pixels[i*rgbaChannels+c] = renderdisplay.Byte(float64(source[i*channels+c]), threshold)
		}
	}
}

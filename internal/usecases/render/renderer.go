package render

import (
	"image"
	"math"
	"os"
	"runtime"
	"strconv"
	"sync"

	renderentity "github.com/MarkJulian19/path_trace_golang/internal/entities/render"
	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	"github.com/MarkJulian19/path_trace_golang/internal/renderdisplay"
)

// Config задаёт параметры рендера на границе сценария.
type Config = renderentity.Config

type renderTile struct {
	x0, y0, x1, y1 int
}

// Render performs a simple path tracing render of the given scene and returns a new image.
func Render(sc *scene.Scene, cfg Config) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
	renderIntoCPU(sc, cfg, img, nil, 0)
	return img
}

// Into строит сцену в переданном изображении.
// If progress is not nil, it will be called periodically from worker goroutines
// after finishing a row to allow interactive preview.
// Progress(currentSample, totalSamples) - currentSample is the current sample being rendered (1-based), totalSamples is
// the total number of samples.
func Into(sc *scene.Scene, cfg Config, img *image.RGBA, progress func(currentSample, totalSamples int)) {
	renderIntoCPU(sc, cfg, img, progress, 0)
}

// IntoWithCancel строит сцену в переданном изображении и периодически
// проверяет cancelled. Старый Into сохраняет прежний контракт для batch-кода.
func IntoWithCancel(
	sc *scene.Scene,
	cfg Config,
	img *image.RGBA,
	progress func(currentSample, totalSamples int),
	cancelled func() bool,
) {
	renderIntoCPUWithCancel(sc, cfg, img, progress, 0, cancelled)
}

// renderIntoCPU строит изображение на CPU.
func renderIntoCPU(
	sc *scene.Scene,
	cfg Config,
	img *image.RGBA,
	progress func(currentSample, totalSamples int),
	wireframeMode int,
) {
	renderIntoCPUWithCancel(sc, cfg, img, progress, wireframeMode, nil)
}

func renderIntoCPUWithCancel(
	sc *scene.Scene,
	cfg Config,
	img *image.RGBA,
	progress func(currentSample, totalSamples int),
	wireframeMode int,
	cancelled func() bool,
) {
	b := img.Bounds()
	if b.Dx() != cfg.Width || b.Dy() != cfg.Height {
		// basic safety: resize not supported, just return
		return
	}
	if isCancelled(cancelled) {
		return
	}
	cfg.SamplesPerPx = max(cfg.SamplesPerPx, 1)
	if wireframeMode == 1 {
		cfg.SamplesPerPx = 1
	}

	world := newCPURenderWorld(sc, cfg)
	// Камера будет создаваться в каждой goroutine с локальным rng
	// для безопасности конкурентного доступа

	bgFunc := backgroundForScene(sc)

	renderCPUImage(sc, cfg, img, world, bgFunc, wireframeMode, progress, cancelled)

	// Финальное обновление предпросмотра после завершения рендеринга
	if progress != nil && !isCancelled(cancelled) {
		progress(cfg.SamplesPerPx, cfg.SamplesPerPx)
	}
}

func renderCPUImage(
	sc *scene.Scene,
	cfg Config,
	img *image.RGBA,
	world *cpuRenderWorld,
	bgFunc func(ray) vec3,
	wireframeMode int,
	progress func(currentSample, totalSamples int),
	cancelled func() bool,
) {
	invWidth := 1.0 / float64(max(cfg.Width, 1))
	invHeight := 1.0 / float64(max(cfg.Height, 1))
	invSamples := 1.0 / float64(cfg.SamplesPerPx)
	heightMinus1 := float64(cfg.Height - 1)
	clearPixels(img.Pix, cfg.Width*cfg.Height)
	if isCancelled(cancelled) {
		return
	}
	tiles := buildRenderTiles(cfg.Width, cfg.Height)
	var wg sync.WaitGroup
	workerCount := min(configuredWorkerCount(), len(tiles))
	pixels := img.Pix
	var publication *cpuFramePublication
	if progress != nil {
		pixels = make([]byte, len(img.Pix))
		publication = &cpuFramePublication{target: img, pixels: pixels, progress: progress, cfg: cfg}
	}
	startRenderWorkers(
		&wg, workerCount, tiles, sc, cfg, pixels, img.Stride,
		heightMinus1, invWidth, invHeight, invSamples, world, bgFunc,
		wireframeMode, publication, cancelled,
	)
	wg.Wait()
}

func isCancelled(cancelled func() bool) bool {
	return cancelled != nil && cancelled()
}

func clearPixels(pix []uint8, pixelCount int) {
	for index := 0; index < pixelCount*renderRGBAComponents; index += renderRGBAComponents {
		pix[index] = 0
		pix[index+1] = 0
		pix[index+2] = 0
		pix[index+3] = 255
	}
}

func buildRenderTiles(width, height int) chan renderTile {
	const tileSize = 16
	numTilesX := (width + tileSize - 1) / tileSize
	numTilesY := (height + tileSize - 1) / tileSize
	tiles := make(chan renderTile, numTilesX*numTilesY)
	for y := 0; y < height; y += tileSize {
		for x := 0; x < width; x += tileSize {
			x1 := min(x+tileSize, width)
			y1 := min(y+tileSize, height)
			if x1 > x && y1 > y {
				tiles <- renderTile{x0: x, y0: y, x1: x1, y1: y1}
			}
		}
	}
	close(tiles)
	return tiles
}

func configuredWorkerCount() int {
	workerCount := max(runtime.NumCPU(), 1)
	envWorkers := os.Getenv("PATHTRACER_WORKERS")
	if envWorkers == "" {
		return workerCount
	}
	customWorkers, err := strconv.Atoi(envWorkers)
	if err != nil || customWorkers <= 0 || customWorkers > renderMaxWorkers {
		return workerCount
	}
	return customWorkers
}

func startRenderWorkers(
	wg *sync.WaitGroup,
	workerCount int,
	tiles <-chan renderTile,
	sc *scene.Scene,
	cfg Config,
	pix []uint8,
	stride int,
	heightMinus1, invWidth, invHeight, invSamples float64,
	world *cpuRenderWorld,
	bgFunc func(ray) vec3,
	wireframeMode int,
	publication *cpuFramePublication,
	cancelled func() bool,
) {
	for range workerCount {
		wg.Go(func() {
			renderWorkerTiles(
				tiles, sc, cfg, pix, stride, heightMinus1, invWidth, invHeight,
				invSamples, world, bgFunc, wireframeMode, publication, cancelled,
			)
		})
	}
}

func renderWorkerTiles(
	tiles <-chan renderTile,
	sc *scene.Scene,
	cfg Config,
	pix []uint8,
	stride int,
	heightMinus1, invWidth, invHeight, invSamples float64,
	world *cpuRenderWorld,
	bgFunc func(ray) vec3,
	wireframeMode int,
	publication *cpuFramePublication,
	cancelled func() bool,
) {
	rng := newRandSource()
	camera := newCamera(sc.Camera, cfg, rng)
	if wireframeMode == 1 {
		camera.lensRadius = 0
	}
	for tile := range tiles {
		if isCancelled(cancelled) || !renderTilePixels(
			tile, pix, stride, heightMinus1, invWidth, invHeight, invSamples,
			camera, cfg, rng, world, bgFunc, wireframeMode, cancelled,
		) {
			return
		}
		if publication != nil {
			publication.commit(tile)
		}
	}
}

func renderTilePixels(
	tile renderTile,
	pix []uint8,
	stride int,
	heightMinus1, invWidth, invHeight, invSamples float64,
	camera camera,
	cfg Config,
	rng *randSource,
	world *cpuRenderWorld,
	bgFunc func(ray) vec3,
	wireframeMode int,
	cancelled func() bool,
) bool {
	for y := tile.y0; y < tile.y1; y++ {
		if isCancelled(cancelled) {
			return false
		}
		yIdx := y * stride
		flipY := heightMinus1 - float64(y)
		for x := tile.x0; x < tile.x1; x++ {
			if !renderPixel(
				pix,
				yIdx,
				x,
				flipY,
				invWidth,
				invHeight,
				invSamples,
				camera,
				cfg,
				rng,
				world,
				bgFunc,
				wireframeMode,
				cancelled,
			) {
				return false
			}
		}
	}
	return true
}

func renderPixel(
	pix []uint8,
	yIdx, x int,
	flipY, invWidth, invHeight, invSamples float64,
	camera camera,
	cfg Config,
	rng *randSource,
	world *cpuRenderWorld,
	bgFunc func(ray) vec3,
	wireframeMode int,
	cancelled func() bool,
) bool {
	if wireframeMode == 1 {
		if isCancelled(cancelled) {
			return false
		}
		currentRay := camera.getRay((float64(x)+renderHalf)*invWidth, (flipY+renderHalf)*invHeight)
		writePreviewPixel(pix, yIdx+x*renderRGBAComponents, previewRayColor(currentRay, world))
		return true
	}
	if cancelled == nil {
		renderPixelUncancelled(
			pix, yIdx, x, flipY, invWidth, invHeight, invSamples, camera, cfg,
			rng, world, bgFunc,
		)
		return true
	}
	return renderPixelCancelled(
		pix, yIdx, x, flipY, invWidth, invHeight, invSamples, camera, cfg,
		rng, world, bgFunc, cancelled,
	)
}

func renderPixelUncancelled(
	pix []uint8,
	yIdx, x int,
	flipY, invWidth, invHeight, invSamples float64,
	camera camera,
	cfg Config,
	rng *randSource,
	world *cpuRenderWorld,
	bgFunc func(ray) vec3,
) {
	col := vec3{}
	xFloat := float64(x)
	for sample := range cfg.SamplesPerPx {
		rng.startPixelSample(cfg.Seed, yIdx/renderRGBAComponents+x, sample)
		u := (xFloat + rng.Float64()) * invWidth
		vv := (flipY + rng.Float64()) * invHeight
		currentRay := camera.getRay(u, vv)
		col = col.add(rayColorGPUParity(currentRay, world, bgFunc, cfg.MaxDepth, rng))
	}
	col = col.mul(invSamples)
	writePixel(pix, yIdx+x*renderRGBAComponents, col)
}

func renderPixelCancelled(
	pix []uint8,
	yIdx, x int,
	flipY, invWidth, invHeight, invSamples float64,
	camera camera,
	cfg Config,
	rng *randSource,
	world *cpuRenderWorld,
	bgFunc func(ray) vec3,
	cancelled func() bool,
) bool {
	col := vec3{}
	xFloat := float64(x)
	for sample := range cfg.SamplesPerPx {
		rng.startPixelSample(cfg.Seed, yIdx/renderRGBAComponents+x, sample)
		if cancelled() {
			return false
		}
		u := (xFloat + rng.Float64()) * invWidth
		vv := (flipY + rng.Float64()) * invHeight
		currentRay := camera.getRay(u, vv)
		col = col.add(rayColorGPUParity(currentRay, world, bgFunc, cfg.MaxDepth, rng))
	}
	col = col.mul(invSamples)
	writePixel(pix, yIdx+x*renderRGBAComponents, col)
	return true
}

func writePixel(pix []uint8, index int, col vec3) {
	rVal := cpuTonemapGamma(col.x) * renderColorMax
	gVal := cpuTonemapGamma(col.y) * renderColorMax
	bVal := cpuTonemapGamma(col.z) * renderColorMax
	threshold := renderdisplay.Dither(index / renderRGBAComponents)
	pix[index] = renderdisplay.Byte(rVal, threshold)
	pix[index+1] = renderdisplay.Byte(gVal, threshold)
	pix[index+2] = renderdisplay.Byte(bVal, threshold)
	pix[index+3] = 255
}

func backgroundForScene(sc *scene.Scene) func(ray) vec3 {
	if sc.Sky != nil && sc.Sky.Type == "gradient" {
		horizon := v(sc.Sky.Horizon.R, sc.Sky.Horizon.G, sc.Sky.Horizon.B)
		zenith := v(sc.Sky.Zenith.R, sc.Sky.Zenith.G, sc.Sky.Zenith.B)
		return func(r ray) vec3 {
			dirLen := r.dir.length()
			if dirLen == 0 {
				return horizon
			}
			t := (r.dir.y/dirLen + renderUnitValue) * renderHalf
			t = min(max(t, 0), renderUnitValue)
			return vec3{
				x: horizon.x*(renderUnitValue-t) + zenith.x*t,
				y: horizon.y*(renderUnitValue-t) + zenith.y*t,
				z: horizon.z*(renderUnitValue-t) + zenith.z*t,
			}
		}
	}
	bgColor := v(sc.Background.R, sc.Background.G, sc.Background.B)
	if sc.Sky != nil && sc.Sky.Type == "solid" {
		bgColor = v(sc.Sky.Color.R, sc.Sky.Color.G, sc.Sky.Color.B)
	}
	return func(_ ray) vec3 { return bgColor }
}

// PickObject выполняет raycast для определения объекта под курсором
// Возвращает индекс объекта или -1, если объект не найден
// Строит луч из камеры через указанную точку экрана и находит первый пересеченный объект.
func PickObject(sc *scene.Scene, cfg Config, x, y int) int {
	x, y = clampPixel(x, y, cfg.Width, cfg.Height)
	ray := pickRay(sc, cfg, x, y)
	return closestObject(sc.Objects, sceneToWorld(sc), ray)
}

func clampPixel(x, y, width, height int) (int, int) {
	return clampPixelCoordinate(x, width), clampPixelCoordinate(y, height)
}

func clampPixelCoordinate(value, limit int) int {
	if value < 0 {
		return 0
	}
	if value >= limit {
		return limit - 1
	}
	return value
}

func pickRay(sc *scene.Scene, cfg Config, x, y int) ray {
	camera := newCamera(sc.Camera, cfg, nil)
	u := (float64(x) + renderHalf) / float64(max(cfg.Width, 1))
	vv := (float64(cfg.Height-1-y) + renderHalf) / float64(max(cfg.Height, 1))
	return camera.getRay(u, vv)
}

func closestObject(objects []scene.Object, world []hittable, ray ray) int {
	closestT := math.MaxFloat64
	closestIndex := -1
	var rec hitRecord
	for i := range objects {
		if world[i].hit(ray, renderMinRayDistance, closestT, &rec) && rec.t < closestT {
			closestT = rec.t
			closestIndex = i
		}
	}
	return closestIndex
}

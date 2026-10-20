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

// renderIntoCPU строит изображение на CPU.
func renderIntoCPU(
	sc *scene.Scene,
	cfg Config,
	img *image.RGBA,
	progress func(currentSample, totalSamples int),
	wireframeMode int,
) {
	b := img.Bounds()
	if b.Dx() != cfg.Width || b.Dy() != cfg.Height {
		// basic safety: resize not supported, just return
		return
	}

	world := sceneToWorld(sc)
	// Камера будет создаваться в каждой goroutine с локальным rng
	// для безопасности конкурентного доступа

	bgFunc := backgroundForScene(sc)

	renderCPUImage(sc, cfg, img, world, bgFunc, wireframeMode, progress)

	// Финальное обновление предпросмотра после завершения рендеринга
	if progress != nil {
		progress(cfg.SamplesPerPx, cfg.SamplesPerPx)
	}
}

func renderCPUImage(
	sc *scene.Scene,
	cfg Config,
	img *image.RGBA,
	world []hittable,
	bgFunc func(ray) vec3,
	wireframeMode int,
	progress func(currentSample, totalSamples int),
) {
	invWidth := 1.0 / float64(cfg.Width-1)
	invHeight := 1.0 / float64(cfg.Height-1)
	invSamples := 1.0 / float64(cfg.SamplesPerPx)
	heightMinus1 := float64(cfg.Height - 1)
	clearPixels(img.Pix, cfg.Width*cfg.Height)
	tiles := buildRenderTiles(cfg.Width, cfg.Height)
	var wg sync.WaitGroup
	startRenderWorkers(
		&wg,
		configuredWorkerCount(),
		tiles,
		sc,
		cfg,
		img.Pix,
		img.Stride,
		heightMinus1,
		invWidth,
		invHeight,
		invSamples,
		world,
		bgFunc,
		wireframeMode,
		progress,
		cfg.Width*cfg.Height,
	)
	wg.Wait()
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
	const tileSize = 32
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
	world []hittable,
	bgFunc func(ray) vec3,
	wireframeMode int,
	progress func(currentSample, totalSamples int),
	pixelCount int,
) {
	var progressMu sync.Mutex
	processedPixels := int64(0)
	for range workerCount {
		wg.Go(func() {
			rng := newRandSource()
			camera := newCamera(sc.Camera, cfg, rng)
			for tile := range tiles {
				renderTilePixels(
					tile,
					pix,
					stride,
					heightMinus1,
					invWidth,
					invHeight,
					invSamples,
					camera,
					cfg,
					rng,
					world,
					bgFunc,
					wireframeMode,
				)
				updateRenderProgress(progress, &progressMu, &processedPixels, tile, cfg.SamplesPerPx, pixelCount)
			}
		})
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
	world []hittable,
	bgFunc func(ray) vec3,
	wireframeMode int,
) {
	for y := tile.y0; y < tile.y1; y++ {
		yIdx := y * stride
		flipY := heightMinus1 - float64(y)
		for x := tile.x0; x < tile.x1; x++ {
			renderPixel(
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
			)
		}
	}
}

func renderPixel(
	pix []uint8,
	yIdx, x int,
	flipY, invWidth, invHeight, invSamples float64,
	camera camera,
	cfg Config,
	rng *randSource,
	world []hittable,
	bgFunc func(ray) vec3,
	wireframeMode int,
) {
	col := vec3{}
	xFloat := float64(x)
	for range cfg.SamplesPerPx {
		u := (xFloat + rng.Float64()) * invWidth
		vv := (flipY + rng.Float64()) * invHeight
		currentRay := camera.getRay(u, vv)
		var sampleRec hitRecord
		col = col.add(rayColorOpt(currentRay, world, bgFunc, cfg.MaxDepth, rng, &sampleRec, wireframeMode))
	}
	col.x = math.Sqrt(col.x * invSamples)
	col.y = math.Sqrt(col.y * invSamples)
	col.z = math.Sqrt(col.z * invSamples)
	writePixel(pix, yIdx+x*renderRGBAComponents, col)
}

func writePixel(pix []uint8, index int, col vec3) {
	rVal := min(max(col.x*renderColorMax, 0), renderColorMax)
	gVal := min(max(col.y*renderColorMax, 0), renderColorMax)
	bVal := min(max(col.z*renderColorMax, 0), renderColorMax)
	pix[index] = uint8(rVal)
	pix[index+1] = uint8(gVal)
	pix[index+2] = uint8(bVal)
	pix[index+3] = 255
}

func updateRenderProgress(
	progress func(currentSample, totalSamples int),
	mu *sync.Mutex,
	processedPixels *int64,
	tile renderTile,
	samplesPerPixel, pixelCount int,
) {
	if progress == nil {
		return
	}
	mu.Lock()
	*processedPixels += int64((tile.x1 - tile.x0) * (tile.y1 - tile.y0) * samplesPerPixel)
	currentSample := min(int(float64(*processedPixels)/float64(pixelCount)), samplesPerPixel)
	mu.Unlock()
	progress(currentSample, samplesPerPixel)
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

func wireframeColor(r ray, rec *hitRecord) vec3 {
	dirLen := r.dir.length()
	if dirLen == 0 {
		return vec3{}
	}
	viewDir := r.dir.mul(-1 / dirLen)
	edgeFactor := math.Abs(rec.normal.dot(viewDir))
	if edgeFactor < renderRoughnessScale {
		return vec3{x: renderUnitValue, y: renderUnitValue, z: renderUnitValue}
	}
	maxAxis := max(math.Abs(rec.normal.x), math.Abs(rec.normal.y), math.Abs(rec.normal.z))
	if maxAxis > renderEdgeThreshold && edgeFactor < renderHalf {
		return vec3{x: renderGrayWireframe, y: renderGrayWireframe, z: renderGrayWireframe}
	}
	return vec3{}
}

// Оптимизированная версия rayColor с переиспользованием hitRecord
// rec используется только для текущего уровня, для рекурсивных вызовов создаются новые.
func rayColorOpt(
	r ray,
	world []hittable,
	background func(ray) vec3,
	depth int,
	rng *randSource,
	rec *hitRecord,
	wireframeMode int,
) vec3 {
	if depth <= 0 {
		return vec3{x: 0, y: 0, z: 0}
	}

	if !closestHit(r, world, rec) {
		if wireframeMode == 1 {
			return vec3{}
		}
		return background(r)
	}

	if wireframeMode == 1 {
		return wireframeColor(r, rec)
	}
	return shadeRay(r, world, background, depth, rng, rec, wireframeMode)
}

func shadeRay(
	r ray,
	world []hittable,
	background func(ray) vec3,
	depth int,
	rng *randSource,
	rec *hitRecord,
	wireframeMode int,
) vec3 {
	emitted := rec.mat.emitted()
	ok, attenuation, scattered := rec.mat.scatter(rng, r, rec)
	if !ok {
		return emitted
	}

	if rec.mat.typ == matDielectric && rec.frontFace {
		scattered, attenuation = applyDielectricPath(scattered, attenuation, world, rec)
	}

	// Russian Roulette: раннее завершение пути с вероятностью, основанной на яркости
	// Это ускоряет рендеринг без потери качества
	if depth <= 3 && !continueRussianRoulette(&attenuation, rng) {
		return emitted
	}

	// Создаём новый rec для рекурсивного вызова, чтобы избежать перезаписи
	var nextRec hitRecord
	// Оптимизация: вычисляем attenuation * rayColorOpt напрямую
	nextColor := rayColorOpt(scattered, world, background, depth-1, rng, &nextRec, wireframeMode)
	return vec3{
		x: emitted.x + attenuation.x*nextColor.x,
		y: emitted.y + attenuation.y*nextColor.y,
		z: emitted.z + attenuation.z*nextColor.z,
	}
}

func closestHit(r ray, world []hittable, rec *hitRecord) bool {
	hitAnything := false
	closest := math.MaxFloat64
	for i := range world {
		if world[i].hit(r, renderMinRayDistance, closest, rec) {
			hitAnything = true
			closest = rec.t
		}
	}
	return hitAnything
}

func applyDielectricPath(scattered ray, attenuation vec3, world []hittable, rec *hitRecord) (ray, vec3) {
	const exitTMin = 0.0001
	exitRec, hitExit := dielectricExit(
		scattered,
		world,
		rec,
		exitTMin,
		renderScatterLengthThreshold,
		defaultMaxRayDistance,
	)
	if !hitExit {
		return scattered, attenuation
	}
	dx := exitRec.p.x - rec.p.x
	dy := exitRec.p.y - rec.p.y
	dz := exitRec.p.z - rec.p.z
	distance := math.Sqrt(dx*dx + dy*dy + dz*dz)
	if rec.mat.absorption.x > 0 || rec.mat.absorption.y > 0 || rec.mat.absorption.z > 0 {
		attenuation.x = math.Exp(-rec.mat.absorption.x * distance)
		attenuation.y = math.Exp(-rec.mat.absorption.y * distance)
		attenuation.z = math.Exp(-rec.mat.absorption.z * distance)
	}
	scattered.orig = exitRec.p
	return scattered, attenuation
}

func dielectricExit(
	scattered ray,
	world []hittable,
	rec *hitRecord,
	exitTMin, minDistanceSq, maxDistanceSq float64,
) (hitRecord, bool) {
	var exitRec hitRecord
	exitT := math.MaxFloat64
	for i := range world {
		var tempRec hitRecord
		if !world[i].hit(scattered, exitTMin, exitT, &tempRec) {
			continue
		}
		if !validDielectricExit(tempRec, rec, exitT, minDistanceSq, maxDistanceSq) {
			continue
		}
		exitT = tempRec.t
		exitRec = tempRec
	}
	return exitRec, exitT < math.MaxFloat64
}

func validDielectricExit(tempRec hitRecord, rec *hitRecord, exitT, minDistanceSq, maxDistanceSq float64) bool {
	if tempRec.mat.typ != matDielectric || tempRec.frontFace || tempRec.t >= exitT {
		return false
	}
	dx := tempRec.p.x - rec.p.x
	dy := tempRec.p.y - rec.p.y
	dz := tempRec.p.z - rec.p.z
	distSq := dx*dx + dy*dy + dz*dz
	return distSq > minDistanceSq && distSq < maxDistanceSq
}

func continueRussianRoulette(attenuation *vec3, rng *randSource) bool {
	maxAttenuation := max(attenuation.x, max(attenuation.y, attenuation.z))
	if maxAttenuation < renderRoughnessThreshold {
		return false
	}
	rrProb := min(maxAttenuation, renderEdgeThreshold)
	if rng.Float64() > rrProb {
		return false
	}
	attenuation.x /= rrProb
	attenuation.y /= rrProb
	attenuation.z /= rrProb
	return true
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
	uCoord := float64(x) / float64(cfg.Width-1)
	vCoord := float64(cfg.Height-1-y) / float64(cfg.Height-1)
	aspect := float64(cfg.Width) / float64(cfg.Height)
	if sc.Camera.AspectRatio != 0 {
		aspect = sc.Camera.AspectRatio
	}
	theta := sc.Camera.FOV * math.Pi / renderDegreesPerHalfTurn
	viewportHeight := renderViewportDiameter * math.Tan(theta/renderViewportDiameter)
	viewportWidth := aspect * viewportHeight
	origin := v(sc.Camera.Position.X, sc.Camera.Position.Y, sc.Camera.Position.Z)
	target := v(sc.Camera.Target.X, sc.Camera.Target.Y, sc.Camera.Target.Z)
	up := v(sc.Camera.Up.X, sc.Camera.Up.Y, sc.Camera.Up.Z)
	w := origin.sub(target).unit()
	uVec := up.cross(w).unit()
	vVec := w.cross(uVec)
	focusDist := sc.Camera.FocusDist
	if focusDist == 0 {
		focusDist = origin.sub(target).length()
	}
	horizontal := uVec.mul(viewportWidth * focusDist)
	vertical := vVec.mul(viewportHeight * focusDist)
	lowerLeftCorner := origin.sub(horizontal.div(renderViewportDiameter)).
		sub(vertical.div(renderViewportDiameter)).
		sub(w.mul(focusDist))
	dir := lowerLeftCorner.add(horizontal.mul(uCoord)).add(vertical.mul(vCoord)).sub(origin)
	return ray{orig: origin, dir: dir.unit()}
}

func closestObject(objects []scene.Object, world []hittable, ray ray) int {
	closestT := math.MaxFloat64
	closestIndex := -1
	for i := range objects {
		var rec hitRecord
		if world[i].hit(ray, renderMinRayDistance, closestT, &rec) && rec.t < closestT {
			closestT = rec.t
			closestIndex = i
		}
	}
	return closestIndex
}

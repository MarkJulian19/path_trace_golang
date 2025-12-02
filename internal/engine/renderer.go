package engine

import (
	"image"
	"math"
	"os"
	"runtime"
	"strconv"
	"sync"

	"github.com/user/pathtracer/internal/scene"
)

// RenderConfig defines internal render parameters.
type RenderConfig struct {
	Width        int
	Height       int
	SamplesPerPx int
	MaxDepth     int
}

// Render performs a simple path tracing render of the given scene and returns a new image.
func Render(sc *scene.Scene, cfg RenderConfig) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
	RenderInto(sc, cfg, img, nil)
	return img
}

// RenderInto renders the scene into the provided image.
// If progress is not nil, it will be called periodically from worker goroutines
// after finishing a row to allow interactive preview.
func RenderInto(sc *scene.Scene, cfg RenderConfig, img *image.RGBA, progress func()) {
	b := img.Bounds()
	if b.Dx() != cfg.Width || b.Dy() != cfg.Height {
		// basic safety: resize not supported, just return
		return
	}

	world := sceneToWorld(sc)
	rng := newRandSource()
	cam := newCamera(sc.Camera, cfg, rng)

	// Поддержка нового градиентного неба или старого простого фона
	var bgFunc func(ray) vec3
	if sc.Sky != nil && sc.Sky.Type == "gradient" {
		horizon := v(sc.Sky.Horizon.R, sc.Sky.Horizon.G, sc.Sky.Horizon.B)
		zenith := v(sc.Sky.Zenith.R, sc.Sky.Zenith.G, sc.Sky.Zenith.B)
		bgFunc = func(r ray) vec3 {
			// Вычисляем градиент на основе направления луча (Y компонента)
			dirLen := math.Sqrt(r.dir.x*r.dir.x + r.dir.y*r.dir.y + r.dir.z*r.dir.z)
			if dirLen == 0 {
				return horizon
			}
			// Нормализуем Y компонент для интерполяции между горизонтом (y=0) и зенитом (y=1)
			t := (r.dir.y/dirLen + 1.0) * 0.5 // от -1..1 к 0..1
			if t < 0 {
				t = 0
			}
			if t > 1 {
				t = 1
			}
			// Интерполируем между горизонтом и зенитом
			return vec3{
				x: horizon.x*(1-t) + zenith.x*t,
				y: horizon.y*(1-t) + zenith.y*t,
				z: horizon.z*(1-t) + zenith.z*t,
			}
		}
	} else {
		// Простой цвет фона (старый способ или solid sky)
		var bgColor vec3
		if sc.Sky != nil && sc.Sky.Type == "solid" {
			bgColor = v(sc.Sky.Color.R, sc.Sky.Color.G, sc.Sky.Color.B)
		} else {
			bgColor = v(sc.Background.R, sc.Background.G, sc.Background.B)
		}
		bgFunc = func(r ray) vec3 {
			return bgColor
		}
	}

	// Предвычисление констант для оптимизации
	invWidth := 1.0 / float64(cfg.Width-1)
	invHeight := 1.0 / float64(cfg.Height-1)
	invSamples := 1.0 / float64(cfg.SamplesPerPx)
	heightMinus1 := float64(cfg.Height - 1)

	// Прямой доступ к пикселям для оптимизации записи
	pix := img.Pix
	stride := img.Stride

	// Инициализируем все пиксели в черный цвет перед рендерингом
	// Используем более эффективный метод для больших изображений
	totalPixels := cfg.Width * cfg.Height * 4
	for i := 0; i < totalPixels; i += 4 {
		pix[i] = 0     // R
		pix[i+1] = 0   // G
		pix[i+2] = 0   // B
		pix[i+3] = 255 // A
	}

	var wg sync.WaitGroup
	// Определяем количество воркеров: округленное вниз количество потоков * 1.2
	// Можно переопределить через переменную окружения PATHTRACER_WORKERS
	workerCount := runtime.NumCPU()
	if workerCount < 1 {
		workerCount = 1
	}

	// Проверяем переменную окружения для ручной настройки
	if envWorkers := os.Getenv("PATHTRACER_WORKERS"); envWorkers != "" {
		if customWorkers, err := strconv.Atoi(envWorkers); err == nil && customWorkers > 0 {
			if customWorkers <= 128 { // Разумный максимум
				workerCount = customWorkers
			}
		}
	}

	// Используем тайлы для лучшей балансировки и кэш-локальности
	const tileSize = 32
	type tile struct {
		x0, y0, x1, y1 int
	}
	// Увеличиваем размер буфера, чтобы гарантировать, что все тайлы поместятся
	numTilesX := (cfg.Width + tileSize - 1) / tileSize
	numTilesY := (cfg.Height + tileSize - 1) / tileSize
	tiles := make(chan tile, numTilesX*numTilesY)

	// Генерируем тайлы - гарантируем покрытие всех пикселей
	for ty := 0; ty < cfg.Height; ty += tileSize {
		for tx := 0; tx < cfg.Width; tx += tileSize {
			x1 := min(tx+tileSize, cfg.Width)
			y1 := min(ty+tileSize, cfg.Height)
			// Убеждаемся, что тайл не пустой
			if x1 > tx && y1 > ty {
				tiles <- tile{
					x0: tx,
					y0: ty,
					x1: x1,
					y1: y1,
				}
			}
		}
	}
	close(tiles)

	totalTiles := numTilesX * numTilesY
	var processedTiles int
	var progressMu sync.Mutex

	for i := 0; i < workerCount; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			localRng := newRandSource()

			for t := range tiles {
				for y := t.y0; y < t.y1; y++ {
					yIdx := y * stride
					flipY := heightMinus1 - float64(y)

					for x := t.x0; x < t.x1; x++ {
						col := vec3{x: 0, y: 0, z: 0}
						xFloat := float64(x)

						// Разворачиваем внутренний цикл для лучшей оптимизации компилятором
						for s := 0; s < cfg.SamplesPerPx; s++ {
							u := (xFloat + localRng.Float64()) * invWidth
							vv := (flipY + localRng.Float64()) * invHeight
							r := cam.getRay(u, vv)
							var sampleRec hitRecord
							col = col.add(rayColorOpt(r, world, bgFunc, cfg.MaxDepth, localRng, &sampleRec))
						}

						// Гамма-коррекция с предвычисленным invSamples
						col.x *= invSamples
						col.y *= invSamples
						col.z *= invSamples
						col.x = math.Sqrt(col.x)
						col.y = math.Sqrt(col.y)
						col.z = math.Sqrt(col.z)

						// Прямая запись в пиксели с предвычисленным индексом
						idx := yIdx + x*4
						// Inline clamp для лучшей производительности
						rVal := col.x * 255.999
						if rVal < 0 {
							rVal = 0
						} else if rVal > 255.999 {
							rVal = 255.999
						}
						gVal := col.y * 255.999
						if gVal < 0 {
							gVal = 0
						} else if gVal > 255.999 {
							gVal = 255.999
						}
						bVal := col.z * 255.999
						if bVal < 0 {
							bVal = 0
						} else if bVal > 255.999 {
							bVal = 255.999
						}
						pix[idx] = uint8(rVal)
						pix[idx+1] = uint8(gVal)
						pix[idx+2] = uint8(bVal)
						pix[idx+3] = 255
					}
				}

				// Обновляем прогресс после каждого тайла (примерно каждые 5% или при завершении)
				if progress != nil {
					progressMu.Lock()
					processedTiles++
					updateThreshold := max(1, totalTiles/20)
					shouldUpdate := processedTiles%updateThreshold == 0 || processedTiles == totalTiles
					progressMu.Unlock()
					if shouldUpdate {
						progress()
					}
				}
			}
		}()
	}

	wg.Wait()

	// Финальное обновление предпросмотра после завершения рендеринга
	if progress != nil {
		progress()
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func rayColor(r ray, world []hittable, background func(ray) vec3, depth int, rng *randSource) vec3 {
	var rec hitRecord
	return rayColorOpt(r, world, background, depth, rng, &rec)
}

// Оптимизированная версия rayColor с переиспользованием hitRecord
// rec используется только для текущего уровня, для рекурсивных вызовов создаются новые
func rayColorOpt(r ray, world []hittable, background func(ray) vec3, depth int, rng *randSource, rec *hitRecord) vec3 {
	if depth <= 0 {
		return vec3{x: 0, y: 0, z: 0}
	}

	// Предвычисляем константы для оптимизации
	const tMin = 0.001
	hitAnything := false
	closest := math.MaxFloat64

	// Оптимизация: проверяем объекты в порядке, который может дать ранний выход
	for i := range world {
		if world[i].hit(r, tMin, closest, rec) {
			hitAnything = true
			closest = rec.t
		}
	}

	if !hitAnything {
		return background(r)
	}

	emitted := rec.mat.emitted()
	ok, attenuation, scattered := rec.mat.scatter(rng, r, rec)
	if !ok {
		return emitted
	}
	// Создаём новый rec для рекурсивного вызова, чтобы избежать перезаписи
	var nextRec hitRecord
	// Оптимизация: вычисляем attenuation * rayColorOpt напрямую
	nextColor := rayColorOpt(scattered, world, background, depth-1, rng, &nextRec)
	return vec3{
		x: emitted.x + attenuation.x*nextColor.x,
		y: emitted.y + attenuation.y*nextColor.y,
		z: emitted.z + attenuation.z*nextColor.z,
	}
}

func (a vec3) mulVec(b vec3) vec3 {
	return vec3{x: a.x * b.x, y: a.y * b.y, z: a.z * b.z}
}

package engine

import (
	"fmt"
	"image"
	"math"
	"os"
	"runtime"
	"strconv"
	"sync"

	"github.com/user/pathtracer/internal/engine/gpu"
	"github.com/user/pathtracer/internal/scene"
)

// Режим рендеринга для CPU: 0 = normal, 1 = wireframe
var (
	cpuRenderMode      int = 0
	cpuRenderModeMutex sync.RWMutex
)

// SetCPURenderMode устанавливает режим рендеринга для CPU.
// mode: 0 = normal, 1 = wireframe
func SetCPURenderMode(mode int) {
	cpuRenderModeMutex.Lock()
	defer cpuRenderModeMutex.Unlock()
	if mode < 0 {
		mode = 0
	}
	if mode > 1 {
		mode = 1
	}
	cpuRenderMode = mode
}

// GetCPURenderMode возвращает текущий режим рендеринга для CPU.
func GetCPURenderMode() int {
	cpuRenderModeMutex.RLock()
	defer cpuRenderModeMutex.RUnlock()
	return cpuRenderMode
}

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
// progress(currentSample, totalSamples) - currentSample is the current sample being rendered (1-based), totalSamples is the total number of samples.
func RenderInto(sc *scene.Scene, cfg RenderConfig, img *image.RGBA, progress func(currentSample, totalSamples int)) {
	switch GetBackend() {
	case BackendGPU:
		renderIntoGPU(sc, cfg, img, progress)
	default:
		renderIntoCPU(sc, cfg, img, progress)
	}
}

// renderIntoCPU contains the original CPU implementation.
func renderIntoCPU(sc *scene.Scene, cfg RenderConfig, img *image.RGBA, progress func(currentSample, totalSamples int)) {
	b := img.Bounds()
	if b.Dx() != cfg.Width || b.Dy() != cfg.Height {
		// basic safety: resize not supported, just return
		return
	}

	world := sceneToWorld(sc)
	// Камера будет создаваться в каждой goroutine с локальным rng
	// для безопасности конкурентного доступа

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

	pixelCount := cfg.Width * cfg.Height
	processedPixels := int64(0)
	var progressMu sync.Mutex

	for i := 0; i < workerCount; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			localRng := newRandSource()
			// Создаем камеру с локальным rng для безопасности конкурентного доступа
			localCam := newCamera(sc.Camera, cfg, localRng)

			for t := range tiles {
				tilePixels := (t.x1 - t.x0) * (t.y1 - t.y0)
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
							r := localCam.getRay(u, vv)
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

				// Обновляем прогресс после каждого тайла
				if progress != nil {
					progressMu.Lock()
					processedPixels += int64(tilePixels * cfg.SamplesPerPx)
					currentSample := int(float64(processedPixels) / float64(pixelCount))
					if currentSample > cfg.SamplesPerPx {
						currentSample = cfg.SamplesPerPx
					}
					progressMu.Unlock()
					// Обновляем прогресс после каждого тайла
					progress(currentSample, cfg.SamplesPerPx)
				}
			}
		}()
	}

	wg.Wait()

	// Финальное обновление предпросмотра после завершения рендеринга
	if progress != nil {
		progress(cfg.SamplesPerPx, cfg.SamplesPerPx)
	}
}

// renderIntoGPU executes GPU rendering using compute shaders.
// If GPU path fails for any reason, it falls back to CPU renderer.
func renderIntoGPU(sc *scene.Scene, cfg RenderConfig, img *image.RGBA, progress func(currentSample, totalSamples int)) {
	gpuCfg := gpu.RenderConfig{
		Width:        cfg.Width,
		Height:       cfg.Height,
		SamplesPerPx: cfg.SamplesPerPx,
		MaxDepth:     cfg.MaxDepth,
	}
	if err := gpu.Render(sc, gpuCfg, img, progress); err != nil {
		// Логируем ошибку для отладки
		fmt.Fprintf(os.Stderr, "GPU render error: %v\nFalling back to CPU renderer.\n", err)
		// Если что-то пошло не так с OpenGL/GLFW, безопасно рендерим на CPU.
		renderIntoCPU(sc, cfg, img, progress)
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
		// Wireframe режим: черный фон
		if GetCPURenderMode() == 1 {
			return vec3{x: 0, y: 0, z: 0}
		}
		return background(r)
	}

	// Wireframe режим: рисуем только контуры объектов
	if GetCPURenderMode() == 1 {
		// Используем нормаль для определения краев
		dirLen := math.Sqrt(r.dir.x*r.dir.x + r.dir.y*r.dir.y + r.dir.z*r.dir.z)
		if dirLen > 0 {
			viewDir := vec3{x: -r.dir.x / dirLen, y: -r.dir.y / dirLen, z: -r.dir.z / dirLen} // Направление к камере
			edgeFactor := math.Abs(rec.normal.x*viewDir.x + rec.normal.y*viewDir.y + rec.normal.z*viewDir.z)
			
			// Для wireframe показываем только когда нормаль почти перпендикулярна направлению взгляда
			edgeThreshold := 0.1 // Более строгий порог для четких контуров
			if edgeFactor < edgeThreshold {
				return vec3{x: 1.0, y: 1.0, z: 1.0} // Белый цвет для контуров
			}
			
			// Также показываем грани кубов и плоскости
			absNormalX := math.Abs(rec.normal.x)
			absNormalY := math.Abs(rec.normal.y)
			absNormalZ := math.Abs(rec.normal.z)
			maxAxis := math.Max(math.Max(absNormalX, absNormalY), absNormalZ)
			if maxAxis > 0.95 {
				// Нормаль выровнена по одной из осей - это грань куба или плоскость
				if edgeFactor < 0.5 {
					return vec3{x: 0.8, y: 0.8, z: 0.8} // Светло-серый для граней
				}
			}
		}
		return vec3{x: 0, y: 0, z: 0} // Черный цвет для остального
	}

	emitted := rec.mat.emitted()
	ok, attenuation, scattered := rec.mat.scatter(rng, r, rec)
	if !ok {
		return emitted
	}

	// Для диэлектриков: если луч преломляется (не отражается), находим следующее пересечение
	// с тем же объектом для вычисления реального расстояния прохождения
	if rec.mat.typ == matDielectric {
		// Проверяем, преломляется ли луч (направление указывает внутрь объекта)
		// Если frontFace было true при входе, значит мы входим, и нужно найти выход
		if rec.frontFace {
			// Луч входит в объект - ищем выходную грань
			// Используем очень маленький tMin для выхода из объекта
			const exitTMin = 0.0001
			var exitRec hitRecord
			hitExit := false
			exitT := math.MaxFloat64

			// Ищем следующее пересечение с объектом того же типа материала
			// Проверяем, что это выходная грань (frontFace = false) и разумное расстояние
			for i := range world {
				var tempRec hitRecord
				if world[i].hit(scattered, exitTMin, exitT, &tempRec) {
					// Проверяем, что это тот же тип материала и выходная грань
					if tempRec.mat.typ == matDielectric && !tempRec.frontFace && tempRec.t < exitT {
						// Вычисляем расстояние для проверки разумности
						dx := tempRec.p.x - rec.p.x
						dy := tempRec.p.y - rec.p.y
						dz := tempRec.p.z - rec.p.z
						distSq := dx*dx + dy*dy + dz*dz

						// Проверяем, что расстояние разумное (не слишком большое, не слишком маленькое)
						// Это помогает убедиться, что мы нашли выход того же объекта
						if distSq > 1e-8 && distSq < 1000.0 {
							hitExit = true
							exitT = tempRec.t
							exitRec = tempRec
						}
					}
				}
			}

			// Если нашли выходную грань, вычисляем реальное расстояние и применяем поглощение
			if hitExit {
				// Вычисляем расстояние прохождения через материал
				dx := exitRec.p.x - rec.p.x
				dy := exitRec.p.y - rec.p.y
				dz := exitRec.p.z - rec.p.z
				distance := math.Sqrt(dx*dx + dy*dy + dz*dz)

				// Применяем Beer-Lambert law с реальным расстоянием
				if rec.mat.absorption.x > 0 || rec.mat.absorption.y > 0 || rec.mat.absorption.z > 0 {
					attenuation.x = math.Exp(-rec.mat.absorption.x * distance)
					attenuation.y = math.Exp(-rec.mat.absorption.y * distance)
					attenuation.z = math.Exp(-rec.mat.absorption.z * distance)
				}

				// Обновляем scattered луч на выходную точку с правильным направлением
				// Направление уже правильное (преломленное), просто обновляем точку
				scattered.orig = exitRec.p
			}
		}
	}

	// Russian Roulette: раннее завершение пути с вероятностью, основанной на яркости
	// Это ускоряет рендеринг без потери качества
	const rrThreshold = 3 // начинаем применять RR после 3 отскоков
	if depth <= rrThreshold {
		// Вычисляем максимальную яркость attenuation
		maxAttenuation := math.Max(attenuation.x, math.Max(attenuation.y, attenuation.z))
		if maxAttenuation < 1e-6 {
			return emitted
		}

		// Вероятность продолжения пути (чем меньше яркость, тем меньше вероятность продолжения)
		rrProb := math.Min(maxAttenuation, 0.95) // ограничиваем максимум 95%
		if rng.Float64() > rrProb {
			return emitted
		}

		// Компенсируем вероятность в attenuation (важно для unbiased рендеринга)
		attenuation.x /= rrProb
		attenuation.y /= rrProb
		attenuation.z /= rrProb
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

// PickObject выполняет raycast для определения объекта под курсором
// Возвращает индекс объекта или -1, если объект не найден
// Строит луч из камеры через указанную точку экрана и находит первый пересеченный объект
func PickObject(sc *scene.Scene, cfg RenderConfig, x, y int) int {
	// Ограничиваем координаты
	if x < 0 {
		x = 0
	}
	if x >= cfg.Width {
		x = cfg.Width - 1
	}
	if y < 0 {
		y = 0
	}
	if y >= cfg.Height {
		y = cfg.Height - 1
	}
	
	// Преобразуем координаты пикселя в UV координаты [0,1]
	// Важно: используем точное преобразование координат, как в шейдере
	// В шейдере: u = (float(pix.x) + stratumU) / float(uWidth - 1)
	//             fy = float(uHeight - 1 - pix.y)
	//             v = (fy + stratumV) / float(uHeight - 1)
	// Для центра пикселя (без jitter): u = pix.x / (uWidth - 1), v = (uHeight - 1 - pix.y) / (uHeight - 1)
	uCoord := float64(x) / float64(cfg.Width-1)
	fy := float64(cfg.Height - 1 - y) // Инвертируем Y точно как в шейдере
	vCoord := fy / float64(cfg.Height-1)
	
	// Создаем камеру для raycast - точно так же, как в buildCamera
	aspect := float64(cfg.Width) / float64(cfg.Height)
	if sc.Camera.AspectRatio != 0 {
		aspect = sc.Camera.AspectRatio
	}
	
	theta := sc.Camera.FOV * math.Pi / 180.0
	h := math.Tan(theta / 2)
	viewportHeight := 2.0 * h
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
	lowerLeftCorner := origin.sub(horizontal.div(2)).sub(vertical.div(2)).sub(w.mul(focusDist))
	
	// Создаем луч через точку (u, v) - точно так же, как в buildCamera
	dir := lowerLeftCorner.add(horizontal.mul(uCoord)).add(vertical.mul(vCoord)).sub(origin)
	ray := ray{orig: origin, dir: dir.unit()}
	
	// Преобразуем сцену в world для проверки пересечений
	world := sceneToWorld(sc)
	
	// Проверяем пересечение со всеми объектами и находим БЛИЖАЙШИЙ (первый пересеченный)
	closestT := math.MaxFloat64
	closestObjIndex := -1
	
	for i := range sc.Objects {
		var rec hitRecord
		// Используем closestT как tMax для раннего выхода
		if world[i].hit(ray, 0.001, closestT, &rec) {
			// Нашли пересечение - проверяем, ближе ли оно
			if rec.t < closestT && rec.t >= 0.001 {
				closestT = rec.t
				closestObjIndex = i
			}
		}
	}
	
	return closestObjIndex
}

func (a vec3) mulVec(b vec3) vec3 {
	return vec3{x: a.x * b.x, y: a.y * b.y, z: a.z * b.z}
}

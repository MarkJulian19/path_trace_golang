package ui

import (
	"fmt"
	"image"
	"image/color"
	"io"
	"log"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/widget"

	"github.com/user/pathtracer/internal/engine"
	"github.com/user/pathtracer/internal/engine/gpu"
	"github.com/user/pathtracer/internal/scene"
)

// logFilter фильтрует некритичные ошибки GLFW из логов
type logFilter struct {
	original io.Writer
}

func (f *logFilter) Write(p []byte) (n int, err error) {
	msg := string(p)
	// Пропускаем ошибки GLFW про Invalid scancode - это известная проблема
	// с нестандартными клавишами на Windows, не критична для работы
	if strings.Contains(msg, "Invalid scancode") {
		return len(p), nil // имитируем успешную запись, но ничего не пишем
	}
	return f.original.Write(p)
}

// Run starts the interactive UI with the given scene file.
func Run(scenePath, mode string) error {
	log.Printf("UI: starting with scene %q, mode=%s\n", scenePath, mode)

	// Подавляем некритичные ошибки GLFW (например, Invalid scancode на Windows)
	// Эти ошибки возникают из-за нестандартных клавиш и не влияют на работу приложения
	originalLogWriter := log.Writer()
	log.SetOutput(&logFilter{original: originalLogWriter})
	defer log.SetOutput(originalLogWriter)

	a := app.New()
	// Применяем темную тему в стиле Blender
	a.Settings().SetTheme(newBlenderDarkTheme())
	w := a.NewWindow("Go Path Tracer")

	sc, err := scene.Load(scenePath)
	if err != nil {
		return err
	}

	// базовые настройки рендера
	baseSettings := engine.RenderSettingsForMode(mode)
	if sc.Settings.Width > 0 && sc.Settings.Height > 0 {
		baseSettings.Width = sc.Settings.Width
		baseSettings.Height = sc.Settings.Height
		if sc.Settings.SamplesPerPx > 0 {
			baseSettings.SamplesPerPx = sc.Settings.SamplesPerPx
		}
		if sc.Settings.MaxDepth > 0 {
			baseSettings.MaxDepth = sc.Settings.MaxDepth
		}
		if sc.Settings.MaxRayDist > 0 {
			baseSettings.MaxRayDist = sc.Settings.MaxRayDist
		}
	}

	previewSettings := baseSettings
	finalSettings := baseSettings
	finalSettings.SamplesPerPx *= 4
	finalSettings.MaxDepth *= 2
	// MaxRayDist остается одинаковым для preview и final

	// максимальный размер области предпросмотра на экране (может быть изменён из UI)
	maxDisplayW := float32(1024.0)
	maxDisplayH := float32(768.0)

	// текущее изображение, в которое рендерит движок
	var img *image.RGBA
	img = image.NewRGBA(image.Rect(0, 0, previewSettings.Width, previewSettings.Height))
	for y := 0; y < previewSettings.Height; y++ {
		for x := 0; x < previewSettings.Width; x++ {
			img.Set(x, y, color.RGBA{0, 0, 0, 255})
		}
	}

	imgCanvas := canvas.NewImageFromImage(img)
	imgCanvas.FillMode = canvas.ImageFillContain
	// Отображаемое окно предпросмотра ограничиваем разумным максимумом,
	// чтобы большое логическое разрешение (2560x1440 и выше) не ломало UI.
	setCanvasSize := func() {
		aspect := float32(previewSettings.Width) / float32(previewSettings.Height)
		displayW := float32(maxDisplayW)
		displayH := displayW / aspect
		if displayH > maxDisplayH {
			displayH = maxDisplayH
			displayW = displayH * aspect
		}
		imgCanvas.SetMinSize(fyne.NewSize(displayW, displayH))
	}
	setCanvasSize()

	status := widget.NewLabel("Idle")
	renderProgressBar := widget.NewProgressBar()
	renderProgressBar.Hide()
	renderTimeLabel := widget.NewLabel("Render time: -")

	// Выбранный объект
	var selectedObjectIndex int = -1

	// Переменные для перетаскивания объектов
	var isDragging bool = false
	var dragStartX, dragStartY int
	var dragStartObjPos scene.Vec3

	// Объявляем startRender заранее, чтобы использовать в замыканиях
	var startRender func(final bool)

	// Создаем pickable canvas для обработки кликов и перетаскивания
	pickableCanvas := newPickableCanvas(imgCanvas, func(x, y int) {
		// Выполняем raycast для определения объекта
		cfg := engine.RenderConfig{
			Width:        previewSettings.Width,
			Height:       previewSettings.Height,
			SamplesPerPx: previewSettings.SamplesPerPx,
			MaxDepth:     previewSettings.MaxDepth,
			MaxRayDist:   float32(previewSettings.MaxRayDist),
		}
		objIndex := pickObject(sc, cfg, x, y)
		selectedObjectIndex = objIndex
		if objIndex >= 0 {
			status.SetText(fmt.Sprintf("Selected object: %d", objIndex))
			// Устанавливаем выбранный объект для подсветки
			if engine.GetBackend() == engine.BackendGPU {
				gpu.SetSelectedObjectFromUI(objIndex)
			}
			// Сохраняем начальную позицию для перетаскивания
			if objIndex < len(sc.Objects) {
				dragStartObjPos = sc.Objects[objIndex].Position
				dragStartX = x
				dragStartY = y
				isDragging = true
			}
			startRender(false) // Перезапускаем рендер для подсветки
		} else {
			status.SetText("No object selected")
			// Снимаем выделение
			if engine.GetBackend() == engine.BackendGPU {
				gpu.SetSelectedObjectFromUI(-1)
			}
			isDragging = false
			startRender(false)
		}
	}, func(x1, y1, x2, y2 int) {
		// Обработка перетаскивания объекта
		if !isDragging || selectedObjectIndex < 0 || selectedObjectIndex >= len(sc.Objects) {
			return
		}

		// Вычисляем смещение в 3D пространстве
		cfg := engine.RenderConfig{
			Width:        previewSettings.Width,
			Height:       previewSettings.Height,
			SamplesPerPx: previewSettings.SamplesPerPx,
			MaxDepth:     previewSettings.MaxDepth,
		}

		// Вычисляем смещение в мировых координатах на основе камеры
		aspect := float64(cfg.Width) / float64(cfg.Height)
		if sc.Camera.AspectRatio != 0 {
			aspect = sc.Camera.AspectRatio
		}
		theta := sc.Camera.FOV * math.Pi / 180.0
		h := math.Tan(theta / 2)
		viewportHeight := 2.0 * h
		viewportWidth := aspect * viewportHeight

		focusDist := sc.Camera.FocusDist
		if focusDist == 0 {
			focusDist = math.Sqrt(
				math.Pow(sc.Camera.Position.X-sc.Camera.Target.X, 2) +
					math.Pow(sc.Camera.Position.Y-sc.Camera.Target.Y, 2) +
					math.Pow(sc.Camera.Position.Z-sc.Camera.Target.Z, 2))
		}

		// Упрощенный подход: перемещаем объект в плоскости камеры
		// Вычисляем смещение пикселей
		dx := float64(x2 - dragStartX)
		dy := float64(y2 - dragStartY)

		// Вычисляем смещение в мировых координатах на основе viewport
		scaleX := viewportWidth * focusDist / float64(cfg.Width)
		scaleY := viewportHeight * focusDist / float64(cfg.Height)

		// Вычисляем векторы камеры вручную
		originX := sc.Camera.Position.X
		originY := sc.Camera.Position.Y
		originZ := sc.Camera.Position.Z
		targetX := sc.Camera.Target.X
		targetY := sc.Camera.Target.Y
		targetZ := sc.Camera.Target.Z
		upX := sc.Camera.Up.X
		upY := sc.Camera.Up.Y
		upZ := sc.Camera.Up.Z

		// w = normalize(origin - target)
		wX := originX - targetX
		wY := originY - targetY
		wZ := originZ - targetZ
		wLen := math.Sqrt(wX*wX + wY*wY + wZ*wZ)
		if wLen > 0 {
			wX /= wLen
			wY /= wLen
			wZ /= wLen
		}

		// u = normalize(cross(up, w))
		uX := upY*wZ - upZ*wY
		uY := upZ*wX - upX*wZ
		uZ := upX*wY - upY*wX
		uLen := math.Sqrt(uX*uX + uY*uY + uZ*uZ)
		if uLen > 0 {
			uX /= uLen
			uY /= uLen
			uZ /= uLen
		}

		// v = cross(w, u)
		var vX, vY, vZ float64
		vX = wY*uZ - wZ*uY
		vY = wZ*uX - wX*uZ
		vZ = wX*uY - wY*uX
		vLen := math.Sqrt(vX*vX + vY*vY + vZ*vZ)
		if vLen > 0 {
			vX /= vLen
			vY /= vLen
			vZ /= vLen
		}

		// Вычисляем смещение в мировых координатах
		worldDx := dx * scaleX
		worldDy := -dy * scaleY // Инвертируем Y

		deltaX := uX*worldDx + vX*worldDy
		deltaY := uY*worldDx + vY*worldDy
		deltaZ := uZ*worldDx + vZ*worldDy

		// Обновляем позицию объекта
		obj := &sc.Objects[selectedObjectIndex]
		obj.Position.X = dragStartObjPos.X + deltaX
		obj.Position.Y = dragStartObjPos.Y + deltaY
		obj.Position.Z = dragStartObjPos.Z + deltaZ

		// Перезапускаем рендер
		startRender(false)
	})

	var mu sync.Mutex
	var stopCh chan struct{}
	var renderTimer *time.Timer             // для debounce при быстрых изменениях
	var lastFinalImage image.Image          // последнее отрендеренное финальное изображение
	var lastFinalConfig engine.RenderConfig // параметры последнего финального рендера

	// Настройка FPS для wireframe стримминга
	wireframeFPS := 60.0

	liveUpdate := widget.NewCheck("Live update while rendering", func(bool) {})
	liveUpdate.SetChecked(true)

	camControlActive := false
	camControlCheck := widget.NewCheck("WASDQE camera control (preview)", func(b bool) {
		camControlActive = b
	})

	// Параметры GPU-денойзера (управляются из UI).
	denoiseEnabled := true
	denoiseSigmaS := 1.0
	denoiseSigmaR := 0.15

	// Параметры дополнительного сглаживания (сильный blur).
	smoothEnabled := false
	smoothRadius := 2
	smoothStrength := 0.5

	// Внутренняя функция, которая выполняет реальный рендеринг
	doRender := func(final bool) {
		go func() {
			log.Println("render goroutine started, final =", final)

			// Проверяем, включен ли wireframe режим для стримминга
			isWireframe := false
			if engine.GetBackend() == engine.BackendGPU {
				isWireframe = (gpu.GetRenderMode() == 1)
			} else {
				isWireframe = (engine.GetCPURenderMode() == 1)
			}

			// Для wireframe режима используем стримминг вместо покадрового рендеринга
			if isWireframe && !final {
				// Стримминг режим для wireframe
				status.SetText("Streaming wireframe...")
			cfg := engine.RenderConfig{
				Width:        previewSettings.Width,
				Height:       previewSettings.Height,
				SamplesPerPx: 1, // Для wireframe достаточно 1 сэмпла
				MaxDepth:     1, // Минимальная глубина для wireframe
				MaxRayDist:   float32(previewSettings.MaxRayDist),
			}

				// Переинициализируем основной буфер, если логическое разрешение изменилось
				mu.Lock()
				if img.Bounds().Dx() != cfg.Width || img.Bounds().Dy() != cfg.Height {
					img = image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
					imgCanvas.Image = img
					// Инициализируем темным серым (фон wireframe)
					for y := 0; y < cfg.Height; y++ {
						for x := 0; x < cfg.Width; x++ {
							img.Set(x, y, color.RGBA{25, 25, 25, 255})
						}
					}
				}
				mu.Unlock()

				// Вычисляем интервал между кадрами
				mu.Lock()
				currentFPS := wireframeFPS
				mu.Unlock()
				frameDuration := time.Duration(float64(time.Second) / currentFPS)
				ticker := time.NewTicker(frameDuration)
				defer ticker.Stop()

				// Стримминг цикл с правильной двойной буферизацией
				for {
					select {
					case <-stopCh:
						status.SetText("Streaming stopped")
						return
					case <-ticker.C:
						// Проверяем, не был ли рендер отменён
						select {
						case <-stopCh:
							status.SetText("Streaming stopped")
							return
						default:
						}

						// Проверяем, что мы всё ещё в wireframe режиме
						mu.Lock()
						currentIsWireframe := false
						if engine.GetBackend() == engine.BackendGPU {
							currentIsWireframe = (gpu.GetRenderMode() == 1)
						} else {
							currentIsWireframe = (engine.GetCPURenderMode() == 1)
						}
						mu.Unlock()

						// Если режим изменился, выходим из стримминга
						if !currentIsWireframe {
							status.SetText("Mode changed, stopping stream")
							return
						}

						progress := func(currentSample, totalSamples int) {
							// Не обновляем во время рендеринга для предотвращения мерцания
						}

						// Рендерим напрямую в основной буфер
						// Важно: получаем ссылку на изображение под блокировкой
						mu.Lock()
						renderImg := img // Сохраняем ссылку на изображение
						mu.Unlock()

						// Рендерим без блокировки мьютекса - это позволяет другим операциям работать
						engine.RenderInto(sc, cfg, renderImg, progress)

						// Обновляем UI безопасно - Refresh() потокобезопасен в Fyne
						// но нужно убедиться, что img не изменяется во время обновления
						mu.Lock()
						imgCanvas.Refresh()
						mu.Unlock()

						// Обновляем FPS если изменился
						mu.Lock()
						currentFPS := wireframeFPS
						mu.Unlock()
						newFrameDuration := time.Duration(float64(time.Second) / currentFPS)
						if newFrameDuration != frameDuration {
							frameDuration = newFrameDuration
							ticker.Stop()
							ticker = time.NewTicker(frameDuration)
						}
					}
				}
			}

			// Обычный покадровый рендеринг для normal режима или final рендеринга
			status.SetText("Rendering...")
			startTime := time.Now()
			var cfg engine.RenderConfig
			if final {
				cfg = engine.RenderConfig{
					Width:        finalSettings.Width,
					Height:       finalSettings.Height,
					SamplesPerPx: finalSettings.SamplesPerPx,
					MaxDepth:     finalSettings.MaxDepth,
					MaxRayDist:   float32(finalSettings.MaxRayDist),
				}
			} else {
				cfg = engine.RenderConfig{
					Width:        previewSettings.Width,
					Height:       previewSettings.Height,
					SamplesPerPx: previewSettings.SamplesPerPx,
					MaxDepth:     previewSettings.MaxDepth,
					MaxRayDist:   float32(previewSettings.MaxRayDist),
				}
			}

			// переинициализируем буфер, если логическое разрешение изменилось
			mu.Lock()
			if img.Bounds().Dx() != cfg.Width || img.Bounds().Dy() != cfg.Height {
				img = image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
				imgCanvas.Image = img
			}
			// Для wireframe режима не очищаем изображение, чтобы избежать мерцания
			// Для normal режима очищаем перед новым рендером
			// Проверяем режим рендеринга
			shouldClearImage := true
			if engine.GetBackend() == engine.BackendGPU {
				if gpu.GetRenderMode() == 1 {
					shouldClearImage = false
				}
			} else {
				if engine.GetCPURenderMode() == 1 {
					shouldClearImage = false
				}
			}
			if shouldClearImage {
				// очистить изображение перед новым рендером только для normal режима
				for y := 0; y < cfg.Height; y++ {
					for x := 0; x < cfg.Width; x++ {
						img.Set(x, y, color.RGBA{0, 0, 0, 255})
					}
				}
			}
			mu.Unlock()

			// Инициализируем прогресс-бар
			totalSamples := cfg.SamplesPerPx
			if totalSamples < 1 {
				totalSamples = 1
			}

			renderProgressBar.SetValue(0)
			renderProgressBar.Show()
			renderTimeLabel.SetText("Render time: 0.00s")

			var progress func(currentSample, totalSamples int)
			lastProgressUpdate := time.Now()
			const minProgressUpdateInterval = 100 * time.Millisecond // Минимальный интервал между обновлениями UI

			if liveUpdate.Checked {
				progress = func(currentSample, totalSamples int) {
					// проверяем, не был ли рендер отменён
					select {
					case <-stopCh:
						// Устанавливаем флаг отмены для GPU рендерера
						if engine.GetBackend() == engine.BackendGPU {
							gpu.SetRenderCancel(true)
						}
						return
					default:
					}

					// Обновляем прогресс-бар и время
					if totalSamples < 1 {
						totalSamples = 1
					}
					if currentSample > totalSamples {
						currentSample = totalSamples
					}
					progressValue := float64(currentSample) / float64(totalSamples)
					renderProgressBar.SetValue(progressValue)

					elapsed := time.Since(startTime).Seconds()
					renderTimeLabel.SetText(fmt.Sprintf("Render time: %.2fs", elapsed))

					// Адаптивный интервал обновления изображения в зависимости от размера
					// Для больших разрешений обновляем чаще, чтобы UI не зависал
					pixelCount := cfg.Width * cfg.Height
					updateInterval := minProgressUpdateInterval
					if pixelCount > 2000*2000 {
						// Очень большие разрешения - обновляем чаще (каждые 50ms)
						updateInterval = 50 * time.Millisecond
					} else if pixelCount > 1000*1000 {
						// Большие разрешения - обновляем каждые 75ms
						updateInterval = 75 * time.Millisecond
					}

					// Обновляем изображение с адаптивной частотой
					now := time.Now()
					if now.Sub(lastProgressUpdate) >= updateInterval || currentSample == totalSamples {
						imgCanvas.Refresh()
						lastProgressUpdate = now
					}
				}
			} else {
				// Даже если live update выключен, обновляем прогресс
				progress = func(currentSample, totalSamples int) {
					// Проверяем отмену
					select {
					case <-stopCh:
						// Устанавливаем флаг отмены для GPU рендерера
						if engine.GetBackend() == engine.BackendGPU {
							gpu.SetRenderCancel(true)
						}
						return
					default:
					}

					if totalSamples < 1 {
						totalSamples = 1
					}
					if currentSample > totalSamples {
						currentSample = totalSamples
					}
					progressValue := float64(currentSample) / float64(totalSamples)
					renderProgressBar.SetValue(progressValue)

					elapsed := time.Since(startTime).Seconds()
					renderTimeLabel.SetText(fmt.Sprintf("Render time: %.2fs", elapsed))

					// Для больших разрешений обновляем изображение даже если live update выключен
					// чтобы предотвратить зависание UI
					pixelCount := cfg.Width * cfg.Height
					if pixelCount > 2000*2000 {
						// Очень большие разрешения - обновляем каждые 200ms даже без live update
						now := time.Now()
						if now.Sub(lastProgressUpdate) >= 200*time.Millisecond || currentSample == totalSamples {
							imgCanvas.Refresh()
							lastProgressUpdate = now
						}
					}
				}
			}

			// Перед запуском рендера обновляем настройки GPU-денойзинга и сглаживания,
			// чтобы изменения из UI сразу применялись.
			if engine.GetBackend() == engine.BackendGPU {
				gpu.SetDenoiseConfigFromUI(denoiseEnabled, denoiseSigmaS, denoiseSigmaR)
				gpu.SetSmoothConfigFromUI(smoothEnabled, smoothRadius, smoothStrength)
				// Сбрасываем флаг отмены перед началом нового рендеринга
				gpu.SetRenderCancel(false)
			}

			engine.RenderInto(sc, cfg, img, progress)

			select {
			case <-stopCh:
				return
			default:
			}

			if !liveUpdate.Checked {
				imgCanvas.Refresh()
			}

			// Если это финальный рендер, сохраняем копию изображения для быстрого сохранения
			if final {
				mu.Lock()
				// Создаём копию изображения
				bounds := img.Bounds()
				lastFinalImage = image.NewRGBA(bounds)
				for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
					for x := bounds.Min.X; x < bounds.Max.X; x++ {
						lastFinalImage.(*image.RGBA).Set(x, y, img.At(x, y))
					}
				}
				lastFinalConfig = cfg
				mu.Unlock()
			}

			elapsed := time.Since(startTime).Seconds()
			renderProgressBar.SetValue(1.0)
			renderTimeLabel.SetText(fmt.Sprintf("Render time: %.2fs", elapsed))
			status.SetText("Done")
			log.Println("render finished")
		}()
	}

	// Очистка сохранённого финального изображения при изменении сцены
	clearFinalImage := func() {
		mu.Lock()
		lastFinalImage = nil
		mu.Unlock()
	}

	// Обёртка startRender с debounce для preview рендеринга
	startRender = func(final bool) {
		mu.Lock()
		// отменяем предыдущий таймер, если он есть (debounce)
		if renderTimer != nil {
			renderTimer.Stop()
			renderTimer = nil
		}
		// отменяем текущий рендер, если он идёт
		if stopCh != nil {
			close(stopCh)
		}
		stopCh = make(chan struct{})
		mu.Unlock()

		// Для preview рендеринга добавляем debounce (300ms), чтобы не запускать
		// рендер при каждом нажатии клавиши, а только после паузы.
		if !final {
			mu.Lock()
			renderTimer = time.AfterFunc(300*time.Millisecond, func() {
				mu.Lock()
				renderTimer = nil
				mu.Unlock()
				doRender(false)
			})
			mu.Unlock()
			return
		}

		// Для final рендеринга запускаем сразу
		doRender(true)
	}

	// Backend slider: 0 = CPU, 1 = GPU
	backendSlider := widget.NewSlider(0, 1)
	backendSlider.Step = 1
	backendSlider.Value = 0 // default: CPU
	backendLabel := widget.NewLabel("Backend: CPU")
	backendSlider.OnChanged = func(v float64) {
		if v >= 0.5 {
			backendLabel.SetText("Backend: GPU")
			engine.SetBackend(engine.BackendGPU)
			// При переключении на GPU сразу применяем текущие настройки денойзера.
			gpu.SetDenoiseConfigFromUI(denoiseEnabled, denoiseSigmaS, denoiseSigmaR)
			gpu.SetSmoothConfigFromUI(smoothEnabled, smoothRadius, smoothStrength)
		} else {
			backendLabel.SetText("Backend: CPU")
			engine.SetBackend(engine.BackendCPU)
		}
		startRender(false)
	}

	// --- Управление камерой ---
	cam := sc.Camera
	camPosX := widget.NewEntry()
	camPosY := widget.NewEntry()
	camPosZ := widget.NewEntry()
	camLookX := widget.NewEntry()
	camLookY := widget.NewEntry()
	camLookZ := widget.NewEntry()
	camFOV := widget.NewEntry()
	camAperture := widget.NewEntry()
	camFocusDist := widget.NewEntry()

	camPosX.SetText(fmt.Sprintf("%.2f", cam.Position.X))
	camPosY.SetText(fmt.Sprintf("%.2f", cam.Position.Y))
	camPosZ.SetText(fmt.Sprintf("%.2f", cam.Position.Z))
	camLookX.SetText(fmt.Sprintf("%.2f", cam.Target.X))
	camLookY.SetText(fmt.Sprintf("%.2f", cam.Target.Y))
	camLookZ.SetText(fmt.Sprintf("%.2f", cam.Target.Z))
	camFOV.SetText(fmt.Sprintf("%.1f", cam.FOV))
	camAperture.SetText(fmt.Sprintf("%.3f", cam.Aperture))
	if cam.FocusDist == 0 {
		// Если focus_dist не задан, вычисляем автоматически
		dist := math.Sqrt(
			math.Pow(cam.Position.X-cam.Target.X, 2) +
				math.Pow(cam.Position.Y-cam.Target.Y, 2) +
				math.Pow(cam.Position.Z-cam.Target.Z, 2))
		camFocusDist.SetText(fmt.Sprintf("%.2f", dist))
	} else {
		camFocusDist.SetText(fmt.Sprintf("%.2f", cam.FocusDist))
	}

	applyCamera := widget.NewButton("Apply camera", func() {
		parseF := func(e *widget.Entry, def float64) float64 {
			v, err := strconv.ParseFloat(e.Text, 64)
			if err != nil {
				return def
			}
			return v
		}
		cam.Position.X = parseF(camPosX, cam.Position.X)
		cam.Position.Y = parseF(camPosY, cam.Position.Y)
		cam.Position.Z = parseF(camPosZ, cam.Position.Z)
		cam.Target.X = parseF(camLookX, cam.Target.X)
		cam.Target.Y = parseF(camLookY, cam.Target.Y)
		cam.Target.Z = parseF(camLookZ, cam.Target.Z)
		cam.FOV = parseF(camFOV, cam.FOV)
		cam.Aperture = parseF(camAperture, cam.Aperture)
		if cam.Aperture < 0 {
			cam.Aperture = 0 // Апертура не может быть отрицательной
		}
		cam.FocusDist = parseF(camFocusDist, cam.FocusDist)
		if cam.FocusDist < 0 {
			cam.FocusDist = 0 // 0 означает автоматическое вычисление
		}
		sc.Camera = cam
		clearFinalImage() // очищаем сохранённое финальное изображение при изменении камеры
		status.SetText("Camera updated")
		startRender(false)
	})

	cameraBox := container.NewVBox(
		widget.NewLabel("Camera"),
		container.NewGridWithColumns(2,
			widget.NewLabel("Pos X"), camPosX,
			widget.NewLabel("Pos Y"), camPosY,
			widget.NewLabel("Pos Z"), camPosZ,
			widget.NewLabel("Look X"), camLookX,
			widget.NewLabel("Look Y"), camLookY,
			widget.NewLabel("Look Z"), camLookZ,
			widget.NewLabel("FOV"), camFOV,
			widget.NewLabel("Aperture"), camAperture,
			widget.NewLabel("Focus Dist"), camFocusDist,
		),
		widget.NewLabel("Aperture: 0 = DOF off, >0 = depth of field"),
		widget.NewLabel("Focus Dist: 0 = auto (distance to target)"),
		applyCamera,
	)

	// --- Управление видео ---
	// Начальная позиция камеры
	startPosX := widget.NewEntry()
	startPosY := widget.NewEntry()
	startPosZ := widget.NewEntry()
	startPosX.SetText(fmt.Sprintf("%.2f", cam.Position.X))
	startPosY.SetText(fmt.Sprintf("%.2f", cam.Position.Y))
	startPosZ.SetText(fmt.Sprintf("%.2f", cam.Position.Z))

	// Конечная позиция камеры
	endPosX := widget.NewEntry()
	endPosY := widget.NewEntry()
	endPosZ := widget.NewEntry()
	endPosX.SetText(fmt.Sprintf("%.2f", cam.Position.X))
	endPosY.SetText(fmt.Sprintf("%.2f", cam.Position.Y))
	endPosZ.SetText(fmt.Sprintf("%.2f", cam.Position.Z))

	// Начальный Target
	startTargetX := widget.NewEntry()
	startTargetY := widget.NewEntry()
	startTargetZ := widget.NewEntry()
	startTargetX.SetText(fmt.Sprintf("%.2f", cam.Target.X))
	startTargetY.SetText(fmt.Sprintf("%.2f", cam.Target.Y))
	startTargetZ.SetText(fmt.Sprintf("%.2f", cam.Target.Z))

	// Конечный Target
	endTargetX := widget.NewEntry()
	endTargetY := widget.NewEntry()
	endTargetZ := widget.NewEntry()
	endTargetX.SetText(fmt.Sprintf("%.2f", cam.Target.X))
	endTargetY.SetText(fmt.Sprintf("%.2f", cam.Target.Y))
	endTargetZ.SetText(fmt.Sprintf("%.2f", cam.Target.Z))

	// Время перемещения и FPS
	durationEntry := widget.NewEntry()
	durationEntry.SetText("5.0") // По умолчанию 5 секунд
	fpsEntry := widget.NewEntry()
	fpsEntry.SetText("30") // По умолчанию 30 FPS

	// Путь сохранения видео
	videoOutputPath := widget.NewEntry()
	videoOutputPath.SetText("output_video.avi")

	renderVideoBtn := widget.NewButton("Render Video", func() {
		// Parse values from UI
		parseF := func(e *widget.Entry, def float64) float64 {
			v, err := strconv.ParseFloat(e.Text, 64)
			if err != nil {
				return def
			}
			return v
		}
		parseI := func(e *widget.Entry, def int) int {
			v, err := strconv.ParseInt(e.Text, 10, 64)
			if err != nil {
				return def
			}
			return int(v)
		}

		// Create start and end cameras
		startCam := scene.Camera{
			Position: scene.Vec3{
				X: parseF(startPosX, cam.Position.X),
				Y: parseF(startPosY, cam.Position.Y),
				Z: parseF(startPosZ, cam.Position.Z),
			},
			Target: scene.Vec3{
				X: parseF(startTargetX, cam.Target.X),
				Y: parseF(startTargetY, cam.Target.Y),
				Z: parseF(startTargetZ, cam.Target.Z),
			},
			Up:          cam.Up,
			FOV:         cam.FOV,
			Aperture:    cam.Aperture,
			FocusDist:   cam.FocusDist,
			AspectRatio: cam.AspectRatio,
		}

		endCam := scene.Camera{
			Position: scene.Vec3{
				X: parseF(endPosX, cam.Position.X),
				Y: parseF(endPosY, cam.Position.Y),
				Z: parseF(endPosZ, cam.Position.Z),
			},
			Target: scene.Vec3{
				X: parseF(endTargetX, cam.Target.X),
				Y: parseF(endTargetY, cam.Target.Y),
				Z: parseF(endTargetZ, cam.Target.Z),
			},
			Up:          cam.Up,
			FOV:         cam.FOV,
			Aperture:    cam.Aperture,
			FocusDist:   cam.FocusDist,
			AspectRatio: cam.AspectRatio,
		}

		duration := parseF(durationEntry, 5.0)
		fps := parseI(fpsEntry, 30)
		outputPath := videoOutputPath.Text
		if outputPath == "" {
			outputPath = "output_video.avi"
		}

		// Get render settings (use final settings for video)
		cfg := engine.RenderConfig{
			Width:        finalSettings.Width,
			Height:       finalSettings.Height,
			SamplesPerPx: finalSettings.SamplesPerPx,
			MaxDepth:     finalSettings.MaxDepth,
			MaxRayDist:   float32(finalSettings.MaxRayDist),
		}

		// Start video generation in goroutine
		go func() {
			mu.Lock()
			// Save reference to imgCanvas for preview updates
			previewCanvas := imgCanvas
			previewImg := img
			mu.Unlock()

			// Update status
			status.SetText("Rendering video frames...")
			renderProgressBar.SetValue(0)
			renderProgressBar.Show()

			// Progress function with preview updates
			lastPreviewUpdate := time.Now()
			const minPreviewUpdateInterval = 100 * time.Millisecond

			progress := func(currentFrame, totalFrames int, frameImg image.Image) {
				// Update progress bar
				progressValue := float64(currentFrame) / float64(totalFrames)
				renderProgressBar.SetValue(progressValue)
				renderTimeLabel.SetText(fmt.Sprintf("Frame %d/%d", currentFrame, totalFrames))

				// Update preview window with current frame (throttled)
				now := time.Now()
				if now.Sub(lastPreviewUpdate) >= minPreviewUpdateInterval || currentFrame == totalFrames {
					mu.Lock()
					// Resize previewImg if needed
					bounds := frameImg.Bounds()
					if previewImg.Bounds().Dx() != bounds.Dx() || previewImg.Bounds().Dy() != bounds.Dy() {
						previewImg = image.NewRGBA(bounds)
					}

					// Copy frame to preview image
					for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
						for x := bounds.Min.X; x < bounds.Max.X; x++ {
							previewImg.Set(x, y, frameImg.At(x, y))
						}
					}

					previewCanvas.Image = previewImg
					mu.Unlock()

					// Update canvas - Refresh() is thread-safe in Fyne v2
					previewCanvas.Refresh()

					lastPreviewUpdate = now
				}
			}

			// Render video sequence
			frames, err := engine.RenderVideoSequence(
				sc, startCam, endCam, duration, fps, cfg, progress,
			)

			if err != nil {
				status.SetText(fmt.Sprintf("Video render error: %v", err))
				return
			}

			// Update status
			status.SetText("Creating video file...")

			// Progress function for video creation
			videoProgress := func(currentFrame, totalFrames int) {
				progressValue := float64(currentFrame) / float64(totalFrames)
				renderProgressBar.SetValue(progressValue)
				renderTimeLabel.SetText(fmt.Sprintf("Encoding frame %d/%d", currentFrame, totalFrames))
			}

			// Create video
			err = engine.CreateVideoFromFrames(frames, outputPath, float64(fps), videoProgress)

			if err != nil {
				status.SetText(fmt.Sprintf("Video creation error: %v", err))
				return
			}

			renderProgressBar.SetValue(1.0)
			status.SetText(fmt.Sprintf("Video saved to %s", outputPath))
		}()
	})

	videoBox := container.NewVBox(
		widget.NewLabel("Video Generation"),
		widget.NewLabel("Start Camera"),
		container.NewGridWithColumns(2,
			widget.NewLabel("Start Pos X"), startPosX,
			widget.NewLabel("Start Pos Y"), startPosY,
			widget.NewLabel("Start Pos Z"), startPosZ,
			widget.NewLabel("Start Target X"), startTargetX,
			widget.NewLabel("Start Target Y"), startTargetY,
			widget.NewLabel("Start Target Z"), startTargetZ,
		),
		widget.NewLabel("End Camera"),
		container.NewGridWithColumns(2,
			widget.NewLabel("End Pos X"), endPosX,
			widget.NewLabel("End Pos Y"), endPosY,
			widget.NewLabel("End Pos Z"), endPosZ,
			widget.NewLabel("End Target X"), endTargetX,
			widget.NewLabel("End Target Y"), endTargetY,
			widget.NewLabel("End Target Z"), endTargetZ,
		),
		container.NewGridWithColumns(2,
			widget.NewLabel("Duration (s)"), durationEntry,
			widget.NewLabel("FPS"), fpsEntry,
			widget.NewLabel("Output Path"), videoOutputPath,
		),
		renderVideoBtn,
	)

	// --- Управление материалами (цвет и интенсивность света, цвет/шероховатость и т.п.) ---
	materialIDs := make([]string, 0, len(sc.Materials))
	for _, m := range sc.Materials {
		materialIDs = append(materialIDs, m.ID)
	}

	var selectedMat int = -1

	matList := widget.NewList(
		func() int { return len(sc.Materials) },
		func() fyne.CanvasObject { return widget.NewLabel("") },
		func(i widget.ListItemID, o fyne.CanvasObject) {
			if i < 0 || i >= len(sc.Materials) {
				return
			}
			m := sc.Materials[i]
			o.(*widget.Label).SetText(fmt.Sprintf("%s (%s)", m.ID, m.Type))
		},
	)

	matTypeSelect := widget.NewSelect(
		[]string{
			string(scene.MaterialLambert),
			string(scene.MaterialMetal),
			string(scene.MaterialDielectric),
			string(scene.MaterialEmissive),
		},
		nil,
	)
	albR := widget.NewEntry()
	albG := widget.NewEntry()
	albB := widget.NewEntry()
	emitR := widget.NewEntry()
	emitG := widget.NewEntry()
	emitB := widget.NewEntry()
	powerEntry := widget.NewEntry()

	roughEntry := widget.NewEntry()
	iorEntry := widget.NewEntry()

	// Новые параметры для металлов
	smoothnessEntry := widget.NewEntry()
	reflectivityEntry := widget.NewEntry()

	// Новые параметры для стекла
	tintREntry := widget.NewEntry()
	tintGEntry := widget.NewEntry()
	tintBEntry := widget.NewEntry()
	absorptionScaleEntry := widget.NewEntry()

	setMaterialFormEnabled := func(enabled bool) {
		if enabled {
			matTypeSelect.Enable()
			albR.Enable()
			albG.Enable()
			albB.Enable()
			emitR.Enable()
			emitG.Enable()
			emitB.Enable()
			powerEntry.Enable()
			roughEntry.Enable()
			iorEntry.Enable()
			smoothnessEntry.Enable()
			reflectivityEntry.Enable()
			tintREntry.Enable()
			tintGEntry.Enable()
			tintBEntry.Enable()
			absorptionScaleEntry.Enable()
		} else {
			matTypeSelect.Disable()
			albR.Disable()
			albG.Disable()
			albB.Disable()
			emitR.Disable()
			emitG.Disable()
			emitB.Disable()
			powerEntry.Disable()
			roughEntry.Disable()
			iorEntry.Disable()
			smoothnessEntry.Disable()
			reflectivityEntry.Disable()
			tintREntry.Disable()
			tintGEntry.Disable()
			tintBEntry.Disable()
			absorptionScaleEntry.Disable()
		}
	}
	setMaterialFormEnabled(false)

	// Функция для показа/скрытия полей в зависимости от типа материала
	updateMaterialFormVisibility := func() {
		if selectedMat < 0 || selectedMat >= len(sc.Materials) {
			return
		}
		m := sc.Materials[selectedMat]
		isMetal := m.Type == scene.MaterialMetal || m.Type == scene.MaterialMirror
		isDielectric := m.Type == scene.MaterialDielectric

		// Показываем/скрываем поля для металлов
		if isMetal {
			smoothnessEntry.Show()
			reflectivityEntry.Show()
		} else {
			smoothnessEntry.Hide()
			reflectivityEntry.Hide()
		}

		// Показываем/скрываем поля для стекла
		if isDielectric {
			tintREntry.Show()
			tintGEntry.Show()
			tintBEntry.Show()
			absorptionScaleEntry.Show()
		} else {
			tintREntry.Hide()
			tintGEntry.Hide()
			tintBEntry.Hide()
			absorptionScaleEntry.Hide()
		}
	}

	loadMaterialToForm := func(idx int) {
		if idx < 0 || idx >= len(sc.Materials) {
			setMaterialFormEnabled(false)
			return
		}
		m := sc.Materials[idx]
		setMaterialFormEnabled(true)
		matTypeSelect.SetSelected(string(m.Type))
		albR.SetText(fmt.Sprintf("%.2f", m.Albedo.R))
		albG.SetText(fmt.Sprintf("%.2f", m.Albedo.G))
		albB.SetText(fmt.Sprintf("%.2f", m.Albedo.B))
		emitR.SetText(fmt.Sprintf("%.2f", m.Emit.R))
		emitG.SetText(fmt.Sprintf("%.2f", m.Emit.G))
		emitB.SetText(fmt.Sprintf("%.2f", m.Emit.B))
		powerEntry.SetText(fmt.Sprintf("%.2f", m.Power))
		roughEntry.SetText(fmt.Sprintf("%.2f", m.Rough))
		iorEntry.SetText(fmt.Sprintf("%.2f", m.IOR))

		// Загружаем новые параметры для металлов
		smoothness := m.Smoothness
		if smoothness == 0 && (m.Type == scene.MaterialMetal || m.Type == scene.MaterialMirror) {
			// Вычисляем из rough для обратной совместимости
			smoothness = 1.0 - m.Rough
		}
		smoothnessEntry.SetText(fmt.Sprintf("%.2f", smoothness))

		reflectivity := m.Reflectivity
		if reflectivity == 0 && (m.Type == scene.MaterialMetal || m.Type == scene.MaterialMirror) {
			reflectivity = 1.0
		}
		reflectivityEntry.SetText(fmt.Sprintf("%.2f", reflectivity))

		// Загружаем новые параметры для стекла
		tintR := m.Tint.R
		tintG := m.Tint.G
		tintB := m.Tint.B
		if tintR == 0 && tintG == 0 && tintB == 0 && m.Type == scene.MaterialDielectric {
			tintR = 1.0
			tintG = 1.0
			tintB = 1.0
		}
		tintREntry.SetText(fmt.Sprintf("%.2f", tintR))
		tintGEntry.SetText(fmt.Sprintf("%.2f", tintG))
		tintBEntry.SetText(fmt.Sprintf("%.2f", tintB))

		absorptionScale := m.AbsorptionScale
		if absorptionScale == 0 && m.Type == scene.MaterialDielectric {
			absorptionScale = 0.01 // По умолчанию 0.01 для см
		}
		absorptionScaleEntry.SetText(fmt.Sprintf("%.4f", absorptionScale))

		updateMaterialFormVisibility()
	}

	matList.OnSelected = func(id widget.ListItemID) {
		selectedMat = int(id)
		loadMaterialToForm(selectedMat)
	}

	// Обновляем видимость полей при изменении типа материала
	matTypeSelect.OnChanged = func(selected string) {
		updateMaterialFormVisibility()
	}

	applyMaterial := widget.NewButton("Apply material", func() {
		if selectedMat < 0 || selectedMat >= len(sc.Materials) {
			status.SetText("No material selected")
			return
		}
		parseF := func(e *widget.Entry, def float64) float64 {
			v, err := strconv.ParseFloat(e.Text, 64)
			if err != nil {
				return def
			}
			return v
		}
		m := sc.Materials[selectedMat]
		if matTypeSelect.Selected != "" {
			m.Type = scene.MaterialType(matTypeSelect.Selected)
		}
		m.Albedo.R = parseF(albR, m.Albedo.R)
		m.Albedo.G = parseF(albG, m.Albedo.G)
		m.Albedo.B = parseF(albB, m.Albedo.B)
		m.Emit.R = parseF(emitR, m.Emit.R)
		m.Emit.G = parseF(emitG, m.Emit.G)
		m.Emit.B = parseF(emitB, m.Emit.B)
		m.Power = parseF(powerEntry, m.Power)
		m.Rough = parseF(roughEntry, m.Rough)
		m.IOR = parseF(iorEntry, m.IOR)

		// Сохраняем новые параметры для металлов
		if m.Type == scene.MaterialMetal || m.Type == scene.MaterialMirror {
			m.Smoothness = parseF(smoothnessEntry, m.Smoothness)
			if m.Smoothness < 0 {
				m.Smoothness = 0
			}
			if m.Smoothness > 1 {
				m.Smoothness = 1
			}
			m.Reflectivity = parseF(reflectivityEntry, m.Reflectivity)
			if m.Reflectivity < 0 {
				m.Reflectivity = 0
			}
			if m.Reflectivity > 1 {
				m.Reflectivity = 1
			}
		}

		// Сохраняем новые параметры для стекла
		if m.Type == scene.MaterialDielectric {
			m.Tint.R = parseF(tintREntry, m.Tint.R)
			m.Tint.G = parseF(tintGEntry, m.Tint.G)
			m.Tint.B = parseF(tintBEntry, m.Tint.B)
		}

		sc.Materials[selectedMat] = m
		matList.Refresh()
		clearFinalImage() // очищаем сохранённое финальное изображение при изменении материала
		status.SetText("Material updated")
		startRender(false)
	})

	// Оборачиваем список в Scroll контейнер с минимальной высотой (200px)
	// чтобы было видно больше элементов и удобнее скроллить
	// widget.NewList уже имеет встроенный скроллер, но мы оборачиваем для установки минимальной высоты
	matListScroll := container.NewScroll(matList)
	matListScroll.SetMinSize(fyne.NewSize(0, 200))
	materialsBox := container.NewBorder(
		widget.NewLabel("Materials"),
		nil, nil, nil,
		matListScroll,
	)

	materialForm := container.NewVBox(
		widget.NewLabel("Selected material"),
		container.NewGridWithColumns(2,
			widget.NewLabel("Type"), matTypeSelect,
			widget.NewLabel("Albedo R"), albR,
			widget.NewLabel("Albedo G"), albG,
			widget.NewLabel("Albedo B"), albB,
			widget.NewLabel("Emit R"), emitR,
			widget.NewLabel("Emit G"), emitG,
			widget.NewLabel("Emit B"), emitB,
			widget.NewLabel("Power"), powerEntry,
			widget.NewLabel("Rough"), roughEntry,
			widget.NewLabel("IOR"), iorEntry,
			widget.NewLabel("Smoothness (Metal)"), smoothnessEntry,
			widget.NewLabel("Reflectivity (Metal)"), reflectivityEntry,
			widget.NewLabel("Tint R (Glass)"), tintREntry,
			widget.NewLabel("Tint G (Glass)"), tintGEntry,
			widget.NewLabel("Tint B (Glass)"), tintBEntry,
			widget.NewLabel("Absorption Scale (Glass)"), absorptionScaleEntry,
		),
		applyMaterial,
	)
	// --- Управление объектами сцены ---
	var selectedObj int = -1

	objList := widget.NewList(
		func() int { return len(sc.Objects) },
		func() fyne.CanvasObject { return widget.NewLabel("") },
		func(i widget.ListItemID, o fyne.CanvasObject) {
			if i < 0 || i >= len(sc.Objects) {
				return
			}
			obj := sc.Objects[i]
			o.(*widget.Label).SetText(fmt.Sprintf("%s (%s)", obj.ID, obj.Type))
		},
	)

	objTypeSelect := widget.NewSelect(
		[]string{
			string(scene.ObjectSphere),
			string(scene.ObjectPlane),
			string(scene.ObjectBox),
		},
		nil,
	)
	objMatSelect := widget.NewSelect(materialIDs, nil)
	objPosX := widget.NewEntry()
	objPosY := widget.NewEntry()
	objPosZ := widget.NewEntry()
	objSizeX := widget.NewEntry()
	objSizeY := widget.NewEntry()
	objSizeZ := widget.NewEntry()

	setObjectFormEnabled := func(enabled bool) {
		if enabled {
			objTypeSelect.Enable()
			objMatSelect.Enable()
			objPosX.Enable()
			objPosY.Enable()
			objPosZ.Enable()
			objSizeX.Enable()
			objSizeY.Enable()
			objSizeZ.Enable()
		} else {
			objTypeSelect.Disable()
			objMatSelect.Disable()
			objPosX.Disable()
			objPosY.Disable()
			objPosZ.Disable()
			objSizeX.Disable()
			objSizeY.Disable()
			objSizeZ.Disable()
		}
	}

	// по умолчанию форма неактивна до выбора объекта
	setObjectFormEnabled(false)

	loadObjectToForm := func(idx int) {
		if idx < 0 || idx >= len(sc.Objects) {
			setObjectFormEnabled(false)
			return
		}
		o := sc.Objects[idx]
		setObjectFormEnabled(true)
		objTypeSelect.SetSelected(string(o.Type))
		objMatSelect.SetSelected(o.MaterialID)
		objPosX.SetText(fmt.Sprintf("%.2f", o.Position.X))
		objPosY.SetText(fmt.Sprintf("%.2f", o.Position.Y))
		objPosZ.SetText(fmt.Sprintf("%.2f", o.Position.Z))
		objSizeX.SetText(fmt.Sprintf("%.2f", o.Size.X))
		objSizeY.SetText(fmt.Sprintf("%.2f", o.Size.Y))
		objSizeZ.SetText(fmt.Sprintf("%.2f", o.Size.Z))
	}

	objList.OnSelected = func(id widget.ListItemID) {
		selectedObj = int(id)
		loadObjectToForm(selectedObj)
	}

	applyObject := widget.NewButton("Apply object", func() {
		if selectedObj < 0 || selectedObj >= len(sc.Objects) {
			status.SetText("No object selected")
			return
		}
		parseF := func(e *widget.Entry, def float64) float64 {
			v, err := strconv.ParseFloat(e.Text, 64)
			if err != nil {
				return def
			}
			return v
		}
		o := sc.Objects[selectedObj]
		if objTypeSelect.Selected != "" {
			o.Type = scene.ObjectType(objTypeSelect.Selected)
		}
		if objMatSelect.Selected != "" {
			o.MaterialID = objMatSelect.Selected
		}
		o.Position.X = parseF(objPosX, o.Position.X)
		o.Position.Y = parseF(objPosY, o.Position.Y)
		o.Position.Z = parseF(objPosZ, o.Position.Z)
		o.Size.X = parseF(objSizeX, o.Size.X)
		o.Size.Y = parseF(objSizeY, o.Size.Y)
		o.Size.Z = parseF(objSizeZ, o.Size.Z)
		sc.Objects[selectedObj] = o
		objList.Refresh()
		clearFinalImage() // очищаем сохранённое финальное изображение при изменении объекта
		status.SetText("Object updated")
		startRender(false)
	})

	addSphere := widget.NewButton("Add sphere", func() {
		o := scene.Object{
			ID:   fmt.Sprintf("sphere-%d", len(sc.Objects)+1),
			Type: scene.ObjectSphere,
			Position: scene.Vec3{
				X: 0, Y: 1, Z: 0,
			},
			Size: scene.Vec3{
				X: 1, Y: 0, Z: 0,
			},
		}
		if len(materialIDs) > 0 {
			o.MaterialID = materialIDs[0]
		}
		sc.Objects = append(sc.Objects, o)
		objList.Refresh()
		selectedObj = len(sc.Objects) - 1
		objList.Select(widget.ListItemID(selectedObj))
		clearFinalImage() // очищаем сохранённое финальное изображение при добавлении объекта
		status.SetText("Sphere added")
		startRender(false)
	})

	addBox := widget.NewButton("Add box", func() {
		o := scene.Object{
			ID:   fmt.Sprintf("box-%d", len(sc.Objects)+1),
			Type: scene.ObjectBox,
			Position: scene.Vec3{
				X: 0, Y: 0.5, Z: 0,
			},
			Size: scene.Vec3{
				X: 1, Y: 1, Z: 1,
			},
		}
		if len(materialIDs) > 0 {
			o.MaterialID = materialIDs[0]
		}
		sc.Objects = append(sc.Objects, o)
		objList.Refresh()
		selectedObj = len(sc.Objects) - 1
		objList.Select(widget.ListItemID(selectedObj))
		clearFinalImage() // очищаем сохранённое финальное изображение при добавлении объекта
		status.SetText("Box added")
		startRender(false)
	})

	removeObj := widget.NewButton("Remove selected", func() {
		if selectedObj < 0 || selectedObj >= len(sc.Objects) {
			status.SetText("No object selected")
			return
		}
		sc.Objects = append(sc.Objects[:selectedObj], sc.Objects[selectedObj+1:]...)
		selectedObj = -1
		objList.Refresh()
		clearFinalImage() // очищаем сохранённое финальное изображение при удалении объекта
		status.SetText("Object removed")
		startRender(false)
	})

	objectForm := container.NewVBox(
		widget.NewLabel("Selected object"),
		container.NewGridWithColumns(2,
			widget.NewLabel("Type"), objTypeSelect,
			widget.NewLabel("Material"), objMatSelect,
		),
		container.NewGridWithColumns(2,
			widget.NewLabel("Pos X"), objPosX,
			widget.NewLabel("Pos Y"), objPosY,
			widget.NewLabel("Pos Z"), objPosZ,
			widget.NewLabel("Size X"), objSizeX,
			widget.NewLabel("Size Y"), objSizeY,
			widget.NewLabel("Size Z"), objSizeZ,
		),
		applyObject,
		container.NewHBox(addSphere, addBox, removeObj),
	)

	// Оборачиваем список в Scroll контейнер с минимальной высотой (200px)
	// чтобы было видно больше элементов и удобнее скроллить
	// widget.NewList уже имеет встроенный скроллер, но мы оборачиваем для установки минимальной высоты
	objListScroll := container.NewScroll(objList)
	objListScroll.SetMinSize(fyne.NewSize(0, 200))
	objectsBox := container.NewBorder(
		widget.NewLabel("Objects"),
		nil, nil, nil,
		objListScroll,
	)

	// --- Управление настройками рендера ---
	prevW := widget.NewEntry()
	prevH := widget.NewEntry()
	prevSpp := widget.NewEntry()
	prevDepth := widget.NewEntry()
	dispW := widget.NewEntry()
	dispH := widget.NewEntry()
	finalW := widget.NewEntry()
	finalH := widget.NewEntry()
	finalSpp := widget.NewEntry()
	finalDepth := widget.NewEntry()
	prevMaxRayDist := widget.NewEntry()
	finalMaxRayDist := widget.NewEntry()

	// --- Настройки тумана (Fog) для сцены ---
	var fogDensityEntry *widget.Entry
	var fogColorREntry *widget.Entry
	var fogColorGEntry *widget.Entry
	var fogColorBEntry *widget.Entry
	var fogScatterEntry *widget.Entry
	var fogAffectSkyCheck *widget.Check
	var fogEnabledCheck *widget.Check
	var fogSigmaSEntry *widget.Entry
	var fogSigmaAEntry *widget.Entry
	var fogGEntry *widget.Entry
	var fogHeteroStrengthEntry *widget.Entry
	var fogNoiseScaleEntry *widget.Entry
	var fogNoiseOctavesEntry *widget.Entry
	var fogGpuVolumetricCheck *widget.Check

	initFogControls := func() {
		fogDensityEntry = widget.NewEntry()
		fogColorREntry = widget.NewEntry()
		fogColorGEntry = widget.NewEntry()
		fogColorBEntry = widget.NewEntry()
		fogScatterEntry = widget.NewEntry()
		fogSigmaSEntry = widget.NewEntry()
		fogSigmaAEntry = widget.NewEntry()
		fogGEntry = widget.NewEntry()
		fogHeteroStrengthEntry = widget.NewEntry()
		fogNoiseScaleEntry = widget.NewEntry()
		fogNoiseOctavesEntry = widget.NewEntry()
		fogAffectSkyCheck = widget.NewCheck("Affect sky", func(b bool) {})
		fogEnabledCheck = widget.NewCheck("Enable fog (GPU only)", func(b bool) {})
		fogGpuVolumetricCheck = widget.NewCheck("Volumetric scattering (GPU only)", func(b bool) {})

		if sc.Fog != nil {
			// Включаем флаг, если есть плотность или физические коэффициенты.
			if sc.Fog.Density > 0 || sc.Fog.SigmaS > 0 || sc.Fog.SigmaA > 0 {
				fogEnabledCheck.SetChecked(true)
			} else {
				fogEnabledCheck.SetChecked(false)
			}

			fogDensityEntry.SetText(fmt.Sprintf("%.3f", sc.Fog.Density))
			fogColorREntry.SetText(fmt.Sprintf("%.2f", sc.Fog.Color.R))
			fogColorGEntry.SetText(fmt.Sprintf("%.2f", sc.Fog.Color.G))
			fogColorBEntry.SetText(fmt.Sprintf("%.2f", sc.Fog.Color.B))
			if sc.Fog.Scatter > 0 {
				fogScatterEntry.SetText(fmt.Sprintf("%.2f", sc.Fog.Scatter))
			} else {
				fogScatterEntry.SetText("1.0")
			}

			// Физические параметры объёмного тумана.
			if sc.Fog.SigmaS > 0 {
				fogSigmaSEntry.SetText(fmt.Sprintf("%.3f", sc.Fog.SigmaS))
			} else {
				fogSigmaSEntry.SetText("0.5")
			}
			if sc.Fog.SigmaA > 0 {
				fogSigmaAEntry.SetText(fmt.Sprintf("%.3f", sc.Fog.SigmaA))
			} else {
				fogSigmaAEntry.SetText("0.1")
			}
			fogGEntry.SetText(fmt.Sprintf("%.2f", sc.Fog.G))

			if sc.Fog.HeteroStrength > 0 {
				fogHeteroStrengthEntry.SetText(fmt.Sprintf("%.2f", sc.Fog.HeteroStrength))
			} else {
				fogHeteroStrengthEntry.SetText("0.0")
			}
			if sc.Fog.NoiseScale > 0 {
				fogNoiseScaleEntry.SetText(fmt.Sprintf("%.2f", sc.Fog.NoiseScale))
			} else {
				fogNoiseScaleEntry.SetText("3.0")
			}
			if sc.Fog.NoiseOctaves > 0 {
				fogNoiseOctavesEntry.SetText(strconv.Itoa(sc.Fog.NoiseOctaves))
			} else {
				fogNoiseOctavesEntry.SetText("3")
			}

			fogAffectSkyCheck.SetChecked(sc.Fog.AffectSky)
			fogGpuVolumetricCheck.SetChecked(sc.Fog.GPUVolumetric)
		} else {
			fogEnabledCheck.SetChecked(false)
			fogDensityEntry.SetText("0.0")
			fogColorREntry.SetText("0.8")
			fogColorGEntry.SetText("0.8")
			fogColorBEntry.SetText("0.8")
			fogScatterEntry.SetText("1.0")
			fogSigmaSEntry.SetText("0.5")
			fogSigmaAEntry.SetText("0.1")
			fogGEntry.SetText("0.6")
			fogHeteroStrengthEntry.SetText("0.0")
			fogNoiseScaleEntry.SetText("3.0")
			fogNoiseOctavesEntry.SetText("3")
			fogAffectSkyCheck.SetChecked(false)
			fogGpuVolumetricCheck.SetChecked(true)
		}
	}
	initFogControls()

	prevW.SetText(strconv.Itoa(previewSettings.Width))
	prevH.SetText(strconv.Itoa(previewSettings.Height))
	prevSpp.SetText(strconv.Itoa(previewSettings.SamplesPerPx))
	prevDepth.SetText(strconv.Itoa(previewSettings.MaxDepth))
	prevMaxRayDist.SetText(strconv.FormatFloat(previewSettings.MaxRayDist, 'f', 1, 64))
	dispW.SetText(strconv.Itoa(int(maxDisplayW)))
	dispH.SetText(strconv.Itoa(int(maxDisplayH)))
	finalW.SetText(strconv.Itoa(finalSettings.Width))
	finalH.SetText(strconv.Itoa(finalSettings.Height))
	finalSpp.SetText(strconv.Itoa(finalSettings.SamplesPerPx))
	finalDepth.SetText(strconv.Itoa(finalSettings.MaxDepth))
	finalMaxRayDist.SetText(strconv.FormatFloat(finalSettings.MaxRayDist, 'f', 1, 64))

	applySettings := widget.NewButton("Apply render settings", func() {
		parseI := func(e *widget.Entry, def int) int {
			v, err := strconv.Atoi(e.Text)
			if err != nil || v <= 0 {
				return def
			}
			return v
		}
		// Логическое разрешение и качество предпросмотра задаются отдельно
		newPrevW := parseI(prevW, previewSettings.Width)
		newPrevH := parseI(prevH, previewSettings.Height)

		previewSettings.Width = newPrevW
		previewSettings.Height = newPrevH
		previewSettings.SamplesPerPx = parseI(prevSpp, previewSettings.SamplesPerPx)
		previewSettings.MaxDepth = parseI(prevDepth, previewSettings.MaxDepth)

		// Логическое разрешение и качество финального рендера независимы
		oldFinalW := finalSettings.Width
		oldFinalH := finalSettings.Height
		oldFinalSpp := finalSettings.SamplesPerPx
		oldFinalDepth := finalSettings.MaxDepth
		oldFinalMaxRayDist := finalSettings.MaxRayDist

		finalSettings.Width = parseI(finalW, finalSettings.Width)
		finalSettings.Height = parseI(finalH, finalSettings.Height)
		finalSettings.SamplesPerPx = parseI(finalSpp, finalSettings.SamplesPerPx)
		finalSettings.MaxDepth = parseI(finalDepth, finalSettings.MaxDepth)

		// Если параметры финального рендера изменились, очищаем сохранённое изображение
		if oldFinalW != finalSettings.Width || oldFinalH != finalSettings.Height ||
			oldFinalSpp != finalSettings.SamplesPerPx || oldFinalDepth != finalSettings.MaxDepth ||
			oldFinalMaxRayDist != finalSettings.MaxRayDist {
			mu.Lock()
			lastFinalImage = nil
			mu.Unlock()
		}

		// Обновляем параметры тумана сцены (используется GPU path tracer).
		parseF := func(e *widget.Entry, def float64) float64 {
			v, err := strconv.ParseFloat(e.Text, 64)
			if err != nil || v <= 0 {
				return def
			}
			return v
		}
		
		// Парсим максимальную длину луча
		previewSettings.MaxRayDist = parseF(prevMaxRayDist, previewSettings.MaxRayDist)
		finalSettings.MaxRayDist = parseF(finalMaxRayDist, finalSettings.MaxRayDist)
		if fogEnabledCheck.Checked {
			density := parseF(fogDensityEntry, 0.0)
			if density < 0 {
				density = 0
			}
			colR := parseF(fogColorREntry, 0.8)
			colG := parseF(fogColorGEntry, 0.8)
			colB := parseF(fogColorBEntry, 0.8)
			scatter := parseF(fogScatterEntry, 1.0)
			if scatter < 0 {
				scatter = 0
			}
			if scatter > 1 {
				scatter = 1
			}

			sigmaS := parseF(fogSigmaSEntry, 0.0)
			if sigmaS < 0 {
				sigmaS = 0
			}
			sigmaA := parseF(fogSigmaAEntry, 0.0)
			if sigmaA < 0 {
				sigmaA = 0
			}
			gVal := parseF(fogGEntry, 0.0)
			if gVal < -0.9 {
				gVal = -0.9
			}
			if gVal > 0.9 {
				gVal = 0.9
			}
			hetero := parseF(fogHeteroStrengthEntry, 0.0)
			if hetero < 0 {
				hetero = 0
			}
			if hetero > 1 {
				hetero = 1
			}
			noiseScale := parseF(fogNoiseScaleEntry, 3.0)
			if noiseScale <= 0 {
				noiseScale = 3.0
			}
			noiseOct := parseI(fogNoiseOctavesEntry, 3)
			if noiseOct < 1 {
				noiseOct = 1
			}
			if noiseOct > 5 {
				noiseOct = 5
			}

			if sc.Fog == nil {
				sc.Fog = &scene.Fog{}
			}
			sc.Fog.Density = density
			sc.Fog.Color = scene.Color{R: colR, G: colG, B: colB}
			sc.Fog.Scatter = scatter
			sc.Fog.AffectSky = fogAffectSkyCheck.Checked
			sc.Fog.SigmaS = sigmaS
			sc.Fog.SigmaA = sigmaA
			sc.Fog.G = gVal
			sc.Fog.HeteroStrength = hetero
			sc.Fog.NoiseScale = noiseScale
			sc.Fog.NoiseOctaves = noiseOct
			sc.Fog.GPUVolumetric = fogGpuVolumetricCheck.Checked
		} else {
			sc.Fog = nil
		}

		// настраиваем отображаемый размер предпросмотра (в пикселях окна)
		newDispW := parseI(dispW, int(maxDisplayW))
		newDispH := parseI(dispH, int(maxDisplayH))
		maxDisplayW = float32(newDispW)
		maxDisplayH = float32(newDispH)

		// переинициализируем img под новое логическое разрешение ПРЕДПРОСМОТРА
		// (финальный рендер сохраняется отдельно в Save image и использует свои размеры)
		mu.Lock()
		if stopCh != nil {
			close(stopCh)
			stopCh = nil
		}
		img = image.NewRGBA(image.Rect(0, 0, previewSettings.Width, previewSettings.Height))
		for y := 0; y < previewSettings.Height; y++ {
			for x := 0; x < previewSettings.Width; x++ {
				img.Set(x, y, color.RGBA{0, 0, 0, 255})
			}
		}
		imgCanvas.Image = img
		mu.Unlock()

		// обновляем только визуальный размер canvas, логическое разрешение остаётся большим
		setCanvasSize()
		status.SetText("Render settings updated")
		startRender(false)
	})

	settingsBox := container.NewVBox(
		widget.NewLabel("Render settings"),
		widget.NewLabel("Preview"),
		container.NewGridWithColumns(2,
			widget.NewLabel("Width"), prevW,
			widget.NewLabel("Height"), prevH,
			widget.NewLabel("Samples"), prevSpp,
			widget.NewLabel("Max reflections"), prevDepth,
			widget.NewLabel("Max ray distance"), prevMaxRayDist,
		),
		widget.NewLabel("Preview display (on screen)"),
		container.NewGridWithColumns(2,
			widget.NewLabel("Disp W"), dispW,
			widget.NewLabel("Disp H"), dispH,
		),
		widget.NewLabel("Final"),
		container.NewGridWithColumns(2,
			widget.NewLabel("Width"), finalW,
			widget.NewLabel("Height"), finalH,
			widget.NewLabel("Samples"), finalSpp,
			widget.NewLabel("Max reflections"), finalDepth,
			widget.NewLabel("Max ray distance"), finalMaxRayDist,
		),
		widget.NewLabel("Fog (GPU)"),
		fogEnabledCheck,
		container.NewGridWithColumns(2,
			widget.NewLabel("Density"), fogDensityEntry,
			widget.NewLabel("Scatter"), fogScatterEntry,
			widget.NewLabel("Color R"), fogColorREntry,
			widget.NewLabel("Color G"), fogColorGEntry,
			widget.NewLabel("Color B"), fogColorBEntry,
		),
		container.NewGridWithColumns(2,
			widget.NewLabel("Sigma S"), fogSigmaSEntry,
			widget.NewLabel("Sigma A"), fogSigmaAEntry,
			widget.NewLabel("g (anisotropy)"), fogGEntry,
			widget.NewLabel("Hetero strength"), fogHeteroStrengthEntry,
			widget.NewLabel("Noise scale"), fogNoiseScaleEntry,
			widget.NewLabel("Noise octaves"), fogNoiseOctavesEntry,
		),
		fogAffectSkyCheck,
		fogGpuVolumetricCheck,
		applySettings,
	)

	previewBtn := widget.NewButton("Preview render", func() { startRender(false) })
	finalBtn := widget.NewButton("Final render", func() { startRender(true) })

	outputPath := widget.NewEntry()
	outputPath.SetText("ui_render.png")

	saveBtn := widget.NewButton("Save scene", func() {
		if err := scene.Save(scenePath, sc); err != nil {
			status.SetText(fmt.Sprintf("Save error: %v", err))
		} else {
			status.SetText("Scene saved")
		}
	})

	saveImageBtn := widget.NewButton("Save image (PNG)", func() {
		path := outputPath.Text
		if path == "" {
			path = "ui_render.png"
		}

		mu.Lock()
		savedImg := lastFinalImage
		savedCfg := lastFinalConfig
		mu.Unlock()

		if savedImg == nil {
			status.SetText("No final render available. Please render final image first.")
			return
		}

		// Сохраняем уже отрендеренное изображение без перерендеринга
		status.SetText("Saving image...")
		go func() {
			if err := engine.SavePNG(path, savedImg); err != nil {
				status.SetText(fmt.Sprintf("Save image error: %v", err))
			} else {
				status.SetText(fmt.Sprintf("Image saved to %s (%dx%d, %d samples)",
					path, savedCfg.Width, savedCfg.Height, savedCfg.SamplesPerPx))
			}
		}()
	})

	// --- GPU denoise controls (видны и полезны в режиме GPU) ---
	denoiseCheck := widget.NewCheck("GPU denoise (bilateral 3x3)", func(b bool) {
		denoiseEnabled = b
		if engine.GetBackend() == engine.BackendGPU {
			gpu.SetDenoiseConfigFromUI(denoiseEnabled, denoiseSigmaS, denoiseSigmaR)
			startRender(false)
		}
	})
	denoiseCheck.SetChecked(denoiseEnabled)

	denoiseSigmaSEntry := widget.NewEntry()
	denoiseSigmaSEntry.SetText(fmt.Sprintf("%.2f", denoiseSigmaS))
	denoiseSigmaREntry := widget.NewEntry()
	denoiseSigmaREntry.SetText(fmt.Sprintf("%.2f", denoiseSigmaR))

	applyDenoiseBtn := widget.NewButton("Apply GPU denoise", func() {
		parseF := func(e *widget.Entry, def float64) float64 {
			v, err := strconv.ParseFloat(e.Text, 64)
			if err != nil || v <= 0 {
				return def
			}
			return v
		}
		denoiseSigmaS = parseF(denoiseSigmaSEntry, denoiseSigmaS)
		denoiseSigmaR = parseF(denoiseSigmaREntry, denoiseSigmaR)
		if engine.GetBackend() == engine.BackendGPU {
			gpu.SetDenoiseConfigFromUI(denoiseEnabled, denoiseSigmaS, denoiseSigmaR)
			startRender(false)
		}
	})

	denoiseBox := container.NewVBox(
		widget.NewLabel("GPU denoise"),
		denoiseCheck,
		container.NewGridWithColumns(2,
			widget.NewLabel("Sigma S (space)"), denoiseSigmaSEntry,
			widget.NewLabel("Sigma R (color)"), denoiseSigmaREntry,
		),
		applyDenoiseBtn,
	)

	// Дополнительный фильтр сглаживания (сильный blur).
	smoothCheck := widget.NewCheck("GPU extra smoothing (strong blur)", func(b bool) {
		smoothEnabled = b
		if engine.GetBackend() == engine.BackendGPU {
			gpu.SetSmoothConfigFromUI(smoothEnabled, smoothRadius, smoothStrength)
			startRender(false)
		}
	})
	smoothCheck.SetChecked(smoothEnabled)

	smoothRadiusEntry := widget.NewEntry()
	smoothRadiusEntry.SetText(strconv.Itoa(smoothRadius))
	smoothStrengthEntry := widget.NewEntry()
	smoothStrengthEntry.SetText(fmt.Sprintf("%.2f", smoothStrength))

	applySmoothBtn := widget.NewButton("Apply smoothing", func() {
		parseI := func(e *widget.Entry, def int) int {
			v, err := strconv.Atoi(e.Text)
			if err != nil {
				return def
			}
			return v
		}
		parseF := func(e *widget.Entry, def float64) float64 {
			v, err := strconv.ParseFloat(e.Text, 64)
			if err != nil {
				return def
			}
			return v
		}
		smoothRadius = parseI(smoothRadiusEntry, smoothRadius)
		if smoothRadius < 1 {
			smoothRadius = 1
		}
		if smoothRadius > 5 {
			smoothRadius = 5
		}
		smoothStrength = parseF(smoothStrengthEntry, smoothStrength)
		if smoothStrength < 0 {
			smoothStrength = 0
		}
		if smoothStrength > 1 {
			smoothStrength = 1
		}

		if engine.GetBackend() == engine.BackendGPU {
			gpu.SetSmoothConfigFromUI(smoothEnabled, smoothRadius, smoothStrength)
			startRender(false)
		}
	})

	smoothBox := container.NewVBox(
		widget.NewLabel("GPU extra smoothing"),
		smoothCheck,
		container.NewGridWithColumns(2,
			widget.NewLabel("Radius (1-5)"), smoothRadiusEntry,
			widget.NewLabel("Strength (0-1)"), smoothStrengthEntry,
		),
		applySmoothBtn,
	)

	// Левая панель: объекты и материалы
	leftPanelContent := container.NewVBox(
		widget.NewLabel("Scene"),
		objectsBox,
		objectForm,
		materialsBox,
		materialForm,
	)
	leftPanel := container.NewVScroll(leftPanelContent)
	leftPanel.SetMinSize(fyne.NewSize(250, 0)) // Минимальная ширина для панели

	// Настройка FPS для wireframe стримминга
	wireframeFPSLabel := widget.NewLabel(fmt.Sprintf("Wireframe FPS: %.0f", wireframeFPS))
	wireframeFPSSlider := widget.NewSlider(10, 120)
	wireframeFPSSlider.Value = wireframeFPS
	wireframeFPSSlider.Step = 1
	wireframeFPSSlider.OnChanged = func(v float64) {
		mu.Lock()
		wireframeFPS = v
		mu.Unlock()
		wireframeFPSLabel.SetText(fmt.Sprintf("Wireframe FPS: %.0f", wireframeFPS))
	}

	// Режим отображения: Normal или Wireframe
	renderModeSelect := widget.NewRadioGroup([]string{"Normal", "Wireframe"}, func(selected string) {
		// Останавливаем текущий рендеринг
		mu.Lock()
		if stopCh != nil {
			close(stopCh)
			stopCh = nil
		}
		mu.Unlock()

		mode := 0
		if selected == "Wireframe" {
			mode = 1
		}
		if engine.GetBackend() == engine.BackendGPU {
			gpu.SetRenderModeFromUI(mode)
		} else {
			engine.SetCPURenderMode(mode)
		}
		startRender(false)
	})
	renderModeSelect.SetSelected("Normal")

	// Правая панель: настройки рендеринга, камера, денойзинг
	rightPanelContent := container.NewVBox(
		widget.NewLabel("Render Settings"),
		container.NewVBox(
			widget.NewLabel("Compute backend"),
			backendLabel,
			backendSlider,
		),
		widget.NewLabel("Display Mode"),
		renderModeSelect,
		wireframeFPSLabel,
		wireframeFPSSlider,
		liveUpdate,
		camControlCheck,
		container.NewHBox(previewBtn, finalBtn),
		settingsBox,
		cameraBox,
		videoBox,
		denoiseBox,
		smoothBox,
		container.NewGridWithColumns(2,
			widget.NewLabel("Scene / Image path"), outputPath,
		),
		container.NewHBox(saveBtn, saveImageBtn),
	)
	rightPanel := container.NewVScroll(rightPanelContent)
	rightPanel.SetMinSize(fyne.NewSize(300, 0)) // Минимальная ширина для панели

	// Нижняя панель: статус, прогресс, timeline (для видео)
	bottomPanel := container.NewVBox(
		status,
		renderProgressBar,
		renderTimeLabel,
		// Timeline будет добавлен позже для видео
		widget.NewLabel("Timeline (coming soon)"),
	)

	// Основная структура: левая панель | центральный viewport | правая панель
	// Используем Border для создания трехколоночного layout
	centerWithRight := container.NewHSplit(
		container.NewMax(pickableCanvas), // Центральный viewport с поддержкой кликов
		rightPanel,
	)
	centerWithRight.SetOffset(0.70) // Правая панель занимает 30% (увеличено для удобства)

	mainContent := container.NewHSplit(
		leftPanel,
		centerWithRight,
	)
	mainContent.SetOffset(0.2) // Левая панель занимает 20%

	// Вертикальный split: основной контент | нижняя панель
	content := container.NewVSplit(
		mainContent,
		bottomPanel,
	)
	content.SetOffset(0.9) // Нижняя панель занимает 10%

	w.SetContent(content)
	// Стартовый размер окна фиксированный, а не зависит напрямую от разрешения рендера.
	w.Resize(fyne.NewSize(1280, 800))
	// Автоматический предпросмотр при старте, чтобы сразу было видно картинку.
	go startRender(false)

	// Плавное управление камерой по WASDQE/стрелкам в режиме предпросмотра (как в креативе майнкрафта)
	// Состояние нажатых клавиш (защищено мьютексом для конкурентного доступа)
	var keysMutex sync.RWMutex
	keysPressed := make(map[fyne.KeyName]bool)
	keysLastPressed := make(map[fyne.KeyName]time.Time) // Время последнего нажатия
	cameraMoveSpeed := 2.0                              // единиц в секунду
	cameraRotSpeed := 2.0                               // радиан в секунду
	keyTimeout := 50 * time.Millisecond                 // Если клавиша не была нажата в течение этого времени, считаем её отпущенной

	// Обработка нажатия клавиш
	w.Canvas().SetOnTypedKey(func(ev *fyne.KeyEvent) {
		if !camControlActive {
			return
		}
		keysMutex.Lock()
		keysPressed[ev.Name] = true
		keysLastPressed[ev.Name] = time.Now()
		keysMutex.Unlock()
	})

	// Цикл плавного движения камеры (без инерции - мгновенная остановка)
	cameraUpdateTicker := time.NewTicker(16 * time.Millisecond) // ~60 FPS для плавного движения
	go func() {
		defer cameraUpdateTicker.Stop()
		for {
			select {
			case <-cameraUpdateTicker.C:
				if !camControlActive {
					continue
				}

				// Проверяем, какие клавиши всё ещё нажаты (автоматически отпускаем, если не нажаты в течение keyTimeout)
				now := time.Now()
				keysMutex.Lock()
				for key, lastTime := range keysLastPressed {
					if now.Sub(lastTime) > keyTimeout {
						keysPressed[key] = false
					}
				}
				keysMutex.Unlock()

				mu.Lock()
				anyKeyPressed := false
				changed := false
				rotated := false

				// Вычисляем направление камеры
				dirX := cam.Target.X - cam.Position.X
				dirY := cam.Target.Y - cam.Position.Y
				dirZ := cam.Target.Z - cam.Position.Z
				dirLen := math.Sqrt(dirX*dirX + dirY*dirY + dirZ*dirZ)
				if dirLen < 1e-6 {
					mu.Unlock()
					continue
				}

				// Нормализуем направление
				dirX /= dirLen
				dirY /= dirLen
				dirZ /= dirLen

				// Вычисляем базис камеры вручную
				upX, upY, upZ := cam.Up.X, cam.Up.Y, cam.Up.Z
				// u = normalize(cross(up, w))
				uX := upY*dirZ - upZ*dirY
				uY := upZ*dirX - upX*dirZ
				uZ := upX*dirY - upY*dirX
				uLen := math.Sqrt(uX*uX + uY*uY + uZ*uZ)
				if uLen > 1e-6 {
					uX /= uLen
					uY /= uLen
					uZ /= uLen
				}

				// Вычисляем углы для поворота
				yaw := math.Atan2(dirZ, dirX)
				pitch := math.Atan2(dirY, math.Hypot(dirX, dirZ))

				// Движение вперед/назад (W/S)
				dt := 16.0 / 1000.0 // 16ms в секундах
				moveStep := cameraMoveSpeed * dt

				// Читаем состояние всех клавиш под блокировкой
				keysMutex.RLock()
				keyW := keysPressed[fyne.KeyW]
				keyS := keysPressed[fyne.KeyS]
				keyA := keysPressed[fyne.KeyA]
				keyD := keysPressed[fyne.KeyD]
				keyQ := keysPressed[fyne.KeyQ]
				keyE := keysPressed[fyne.KeyE]
				keyLeft := keysPressed[fyne.KeyLeft]
				keyRight := keysPressed[fyne.KeyRight]
				keyUp := keysPressed[fyne.KeyUp]
				keyDown := keysPressed[fyne.KeyDown]
				keysMutex.RUnlock()

				if keyW {
					// Движение вперед по направлению камеры
					cam.Position.X += dirX * moveStep
					cam.Position.Y += dirY * moveStep
					cam.Position.Z += dirZ * moveStep
					cam.Target.X += dirX * moveStep
					cam.Target.Y += dirY * moveStep
					cam.Target.Z += dirZ * moveStep
					changed = true
					anyKeyPressed = true
				}
				if keyS {
					// Движение назад
					cam.Position.X -= dirX * moveStep
					cam.Position.Y -= dirY * moveStep
					cam.Position.Z -= dirZ * moveStep
					cam.Target.X -= dirX * moveStep
					cam.Target.Y -= dirY * moveStep
					cam.Target.Z -= dirZ * moveStep
					changed = true
					anyKeyPressed = true
				}

				// Движение влево/вправо (A/D)
				if keyA {
					// Движение влево (перпендикулярно направлению)
					cam.Position.X -= uX * moveStep
					cam.Position.Y -= uY * moveStep
					cam.Position.Z -= uZ * moveStep
					cam.Target.X -= uX * moveStep
					cam.Target.Y -= uY * moveStep
					cam.Target.Z -= uZ * moveStep
					changed = true
					anyKeyPressed = true
				}
				if keyD {
					// Движение вправо
					cam.Position.X += uX * moveStep
					cam.Position.Y += uY * moveStep
					cam.Position.Z += uZ * moveStep
					cam.Target.X += uX * moveStep
					cam.Target.Y += uY * moveStep
					cam.Target.Z += uZ * moveStep
					changed = true
					anyKeyPressed = true
				}

				// Движение вверх/вниз (Q/E)
				if keyQ {
					// Движение вниз
					cam.Position.Y -= moveStep
					cam.Target.Y -= moveStep
					changed = true
					anyKeyPressed = true
				}
				if keyE {
					// Движение вверх
					cam.Position.Y += moveStep
					cam.Target.Y += moveStep
					changed = true
					anyKeyPressed = true
				}

				// Поворот камеры (стрелки)
				rotStep := cameraRotSpeed * dt
				if keyLeft {
					yaw -= rotStep
					rotated = true
					anyKeyPressed = true
				}
				if keyRight {
					yaw += rotStep
					rotated = true
					anyKeyPressed = true
				}
				if keyUp {
					pitch -= rotStep // Инвертировано: Up = вверх = уменьшаем pitch
					if pitch < -math.Pi/2+0.1 {
						pitch = -math.Pi/2 + 0.1
					}
					rotated = true
					anyKeyPressed = true
				}
				if keyDown {
					pitch += rotStep // Инвертировано: Down = вниз = увеличиваем pitch
					if pitch > math.Pi/2-0.1 {
						pitch = math.Pi/2 - 0.1
					}
					rotated = true
					anyKeyPressed = true
				}

				// Применяем поворот камеры
				if rotated {
					r := dirLen
					newDirX := r * math.Cos(pitch) * math.Cos(yaw)
					newDirY := r * math.Sin(pitch)
					newDirZ := r * math.Cos(pitch) * math.Sin(yaw)
					cam.Target.X = cam.Position.X + newDirX
					cam.Target.Y = cam.Position.Y + newDirY
					cam.Target.Z = cam.Position.Z + newDirZ
				}

				mu.Unlock()

				// Обновляем камеру и UI только если что-то изменилось
				if anyKeyPressed && (changed || rotated) {
					mu.Lock()
					sc.Camera = cam
					camPosX.SetText(fmt.Sprintf("%.2f", cam.Position.X))
					camPosY.SetText(fmt.Sprintf("%.2f", cam.Position.Y))
					camPosZ.SetText(fmt.Sprintf("%.2f", cam.Position.Z))
					camLookX.SetText(fmt.Sprintf("%.2f", cam.Target.X))
					camLookY.SetText(fmt.Sprintf("%.2f", cam.Target.Y))
					camLookZ.SetText(fmt.Sprintf("%.2f", cam.Target.Z))
					mu.Unlock()

					// Запускаем рендер с debounce только один раз после движения
					// startRender уже имеет debounce, поэтому не нужно вызывать его каждый кадр
					// Вызываем только если это не wireframe режим (wireframe стримит сам)
					mu.Lock()
					isWireframe := false
					if engine.GetBackend() == engine.BackendGPU {
						isWireframe = (gpu.GetRenderMode() == 1)
					} else {
						isWireframe = (engine.GetCPURenderMode() == 1)
					}
					mu.Unlock()

					// Для normal режима запускаем рендер с debounce
					if !isWireframe {
						startRender(false)
					}
				}
			}
		}
	}()

	w.ShowAndRun()
	return nil
}

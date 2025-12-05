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
	}

	previewSettings := baseSettings
	finalSettings := baseSettings
	finalSettings.SamplesPerPx *= 4
	finalSettings.MaxDepth *= 2

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
	fpsLabel := widget.NewLabel("FPS: -")

	var mu sync.Mutex
	var stopCh chan struct{}
	var renderTimer *time.Timer             // для debounce при быстрых изменениях
	var lastFinalImage image.Image          // последнее отрендеренное финальное изображение
	var lastFinalConfig engine.RenderConfig // параметры последнего финального рендера

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
			status.SetText("Rendering...")
			startTime := time.Now()
			var cfg engine.RenderConfig
			if final {
				cfg = engine.RenderConfig{
					Width:        finalSettings.Width,
					Height:       finalSettings.Height,
					SamplesPerPx: finalSettings.SamplesPerPx,
					MaxDepth:     finalSettings.MaxDepth,
				}
			} else {
				cfg = engine.RenderConfig{
					Width:        previewSettings.Width,
					Height:       previewSettings.Height,
					SamplesPerPx: previewSettings.SamplesPerPx,
					MaxDepth:     previewSettings.MaxDepth,
				}
			}

			// переинициализируем буфер, если логическое разрешение изменилось
			mu.Lock()
			if img.Bounds().Dx() != cfg.Width || img.Bounds().Dy() != cfg.Height {
				img = image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
				imgCanvas.Image = img
			}
			// очистить изображение перед новым рендером
			for y := 0; y < cfg.Height; y++ {
				for x := 0; x < cfg.Width; x++ {
					img.Set(x, y, color.RGBA{0, 0, 0, 255})
				}
			}
			mu.Unlock()

			var progress func()
			if liveUpdate.Checked {
				progress = func() {
					// проверяем, не был ли рендер отменён
					select {
					case <-stopCh:
						return
					default:
					}
					imgCanvas.Refresh()
				}
			}

			// Перед запуском рендера обновляем настройки GPU-денойзинга и сглаживания,
			// чтобы изменения из UI сразу применялись.
			if engine.GetBackend() == engine.BackendGPU {
				gpu.SetDenoiseConfigFromUI(denoiseEnabled, denoiseSigmaS, denoiseSigmaR)
				gpu.SetSmoothConfigFromUI(smoothEnabled, smoothRadius, smoothStrength)
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
			if elapsed > 0 {
				fpsLabel.SetText(fmt.Sprintf("FPS: %.2f", 1.0/elapsed))
			}
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
	startRender := func(final bool) {
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

		// Для preview рендеринга добавляем debounce (200ms), чтобы не запускать
		// рендер при каждом нажатии клавиши, а только после паузы.
		if !final {
			mu.Lock()
			renderTimer = time.AfterFunc(200*time.Millisecond, func() {
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

	camPosX.SetText(fmt.Sprintf("%.2f", cam.Position.X))
	camPosY.SetText(fmt.Sprintf("%.2f", cam.Position.Y))
	camPosZ.SetText(fmt.Sprintf("%.2f", cam.Position.Z))
	camLookX.SetText(fmt.Sprintf("%.2f", cam.Target.X))
	camLookY.SetText(fmt.Sprintf("%.2f", cam.Target.Y))
	camLookZ.SetText(fmt.Sprintf("%.2f", cam.Target.Z))
	camFOV.SetText(fmt.Sprintf("%.1f", cam.FOV))

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
		),
		applyCamera,
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
	dispW.SetText(strconv.Itoa(int(maxDisplayW)))
	dispH.SetText(strconv.Itoa(int(maxDisplayH)))
	finalW.SetText(strconv.Itoa(finalSettings.Width))
	finalH.SetText(strconv.Itoa(finalSettings.Height))
	finalSpp.SetText(strconv.Itoa(finalSettings.SamplesPerPx))
	finalDepth.SetText(strconv.Itoa(finalSettings.MaxDepth))

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

		finalSettings.Width = parseI(finalW, finalSettings.Width)
		finalSettings.Height = parseI(finalH, finalSettings.Height)
		finalSettings.SamplesPerPx = parseI(finalSpp, finalSettings.SamplesPerPx)
		finalSettings.MaxDepth = parseI(finalDepth, finalSettings.MaxDepth)

		// Если параметры финального рендера изменились, очищаем сохранённое изображение
		if oldFinalW != finalSettings.Width || oldFinalH != finalSettings.Height ||
			oldFinalSpp != finalSettings.SamplesPerPx || oldFinalDepth != finalSettings.MaxDepth {
			mu.Lock()
			lastFinalImage = nil
			mu.Unlock()
		}

		// Обновляем параметры тумана сцены (используется GPU path tracer).
		parseF := func(e *widget.Entry, def float64) float64 {
			v, err := strconv.ParseFloat(e.Text, 64)
			if err != nil {
				return def
			}
			return v
		}
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
			widget.NewLabel("Max depth"), prevDepth,
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
			widget.NewLabel("Max depth"), finalDepth,
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

	controls := container.NewVBox(
		widget.NewLabel("Controls"),
		liveUpdate,
		container.NewVBox(
			widget.NewLabel("Compute backend"),
			backendLabel,
			backendSlider,
		),
		denoiseBox,
		smoothBox,
		camControlCheck,
		container.NewHBox(previewBtn, finalBtn),
		container.NewGridWithColumns(2,
			widget.NewLabel("Scene / Image path"), outputPath,
		),
		container.NewHBox(saveBtn, saveImageBtn),
		status,
		fpsLabel,
		cameraBox,
		settingsBox,
		materialsBox,
		materialForm,
		objectsBox,
		objectForm,
	)

	// Версия с вертикальным скроллом панели контролов (как раньше),
	// чтобы при большом числе объектов/настроек всё оставалось доступным.
	content := container.NewHSplit(
		container.NewVScroll(controls),
		container.NewMax(imgCanvas),
	)
	// слегка увеличиваем ширину левой панели, чтобы зона со списком объектов была комфортнее.
	content.SetOffset(0.4)

	w.SetContent(content)
	// Стартовый размер окна фиксированный, а не зависит напрямую от разрешения рендера.
	w.Resize(fyne.NewSize(1280, 800))
	// Автоматический предпросмотр при старте, чтобы сразу было видно картинку.
	go startRender(false)

	// Глобальное управление камерой по WASDQE/стрелкам в режиме предпросмотра.
	step := 0.5
	rotStep := 0.05 // радианы для поворота камеры
	w.Canvas().SetOnTypedKey(func(ev *fyne.KeyEvent) {
		if !camControlActive {
			return
		}
		changed := false
		rotated := false
		switch ev.Name {
		case fyne.KeyW:
			cam.Position.Z -= step
			cam.Target.Z -= step
			changed = true
		case fyne.KeyS:
			cam.Position.Z += step
			cam.Target.Z += step
			changed = true
		case fyne.KeyA:
			cam.Position.X -= step
			cam.Target.X -= step
			changed = true
		case fyne.KeyD:
			cam.Position.X += step
			cam.Target.X += step
			changed = true
		case fyne.KeyQ:
			cam.Position.Y -= step
			cam.Target.Y -= step
			changed = true
		case fyne.KeyE:
			cam.Position.Y += step
			cam.Target.Y += step
			changed = true
		case fyne.KeyLeft:
			// поворот вокруг оси Y (yaw - влево)
			dirX := cam.Target.X - cam.Position.X
			dirY := cam.Target.Y - cam.Position.Y
			dirZ := cam.Target.Z - cam.Position.Z
			yaw := math.Atan2(dirZ, dirX)
			pitch := math.Atan2(dirY, math.Hypot(dirX, dirZ))
			yaw -= rotStep
			r := math.Sqrt(dirX*dirX + dirY*dirY + dirZ*dirZ)
			newDirX := r * math.Cos(pitch) * math.Cos(yaw)
			newDirY := r * math.Sin(pitch)
			newDirZ := r * math.Cos(pitch) * math.Sin(yaw)
			cam.Target.X = cam.Position.X + newDirX
			cam.Target.Y = cam.Position.Y + newDirY
			cam.Target.Z = cam.Position.Z + newDirZ
			rotated = true
		case fyne.KeyRight:
			// поворот вокруг оси Y (yaw - вправо)
			dirX := cam.Target.X - cam.Position.X
			dirY := cam.Target.Y - cam.Position.Y
			dirZ := cam.Target.Z - cam.Position.Z
			yaw := math.Atan2(dirZ, dirX)
			pitch := math.Atan2(dirY, math.Hypot(dirX, dirZ))
			yaw += rotStep
			r := math.Sqrt(dirX*dirX + dirY*dirY + dirZ*dirZ)
			newDirX := r * math.Cos(pitch) * math.Cos(yaw)
			newDirY := r * math.Sin(pitch)
			newDirZ := r * math.Cos(pitch) * math.Sin(yaw)
			cam.Target.X = cam.Position.X + newDirX
			cam.Target.Y = cam.Position.Y + newDirY
			cam.Target.Z = cam.Position.Z + newDirZ
			rotated = true
		case fyne.KeyUp:
			// наклон камеры вверх (pitch)
			dirX := cam.Target.X - cam.Position.X
			dirY := cam.Target.Y - cam.Position.Y
			dirZ := cam.Target.Z - cam.Position.Z
			yaw := math.Atan2(dirZ, dirX)
			pitch := math.Atan2(dirY, math.Hypot(dirX, dirZ))
			pitch += rotStep
			if pitch > math.Pi/2-0.1 {
				pitch = math.Pi/2 - 0.1
			}
			r := math.Sqrt(dirX*dirX + dirY*dirY + dirZ*dirZ)
			newDirX := r * math.Cos(pitch) * math.Cos(yaw)
			newDirY := r * math.Sin(pitch)
			newDirZ := r * math.Cos(pitch) * math.Sin(yaw)
			cam.Target.X = cam.Position.X + newDirX
			cam.Target.Y = cam.Position.Y + newDirY
			cam.Target.Z = cam.Position.Z + newDirZ
			rotated = true
		case fyne.KeyDown:
			// наклон камеры вниз (pitch)
			dirX := cam.Target.X - cam.Position.X
			dirY := cam.Target.Y - cam.Position.Y
			dirZ := cam.Target.Z - cam.Position.Z
			yaw := math.Atan2(dirZ, dirX)
			pitch := math.Atan2(dirY, math.Hypot(dirX, dirZ))
			pitch -= rotStep
			if pitch < -math.Pi/2+0.1 {
				pitch = -math.Pi/2 + 0.1
			}
			r := math.Sqrt(dirX*dirX + dirY*dirY + dirZ*dirZ)
			newDirX := r * math.Cos(pitch) * math.Cos(yaw)
			newDirY := r * math.Sin(pitch)
			newDirZ := r * math.Cos(pitch) * math.Sin(yaw)
			cam.Target.X = cam.Position.X + newDirX
			cam.Target.Y = cam.Position.Y + newDirY
			cam.Target.Z = cam.Position.Z + newDirZ
			rotated = true
		}
		if !changed && !rotated {
			return
		}
		// обновляем камеру и поля UI
		sc.Camera = cam
		camPosX.SetText(fmt.Sprintf("%.2f", cam.Position.X))
		camPosY.SetText(fmt.Sprintf("%.2f", cam.Position.Y))
		camPosZ.SetText(fmt.Sprintf("%.2f", cam.Position.Z))
		camLookX.SetText(fmt.Sprintf("%.2f", cam.Target.X))
		camLookY.SetText(fmt.Sprintf("%.2f", cam.Target.Y))
		camLookZ.SetText(fmt.Sprintf("%.2f", cam.Target.Z))
		if rotated {
			status.SetText("Camera rotated (arrows)")
		} else {
			status.SetText("Camera moved (WASDQE)")
		}
		go startRender(false)
	})

	w.ShowAndRun()
	return nil
}

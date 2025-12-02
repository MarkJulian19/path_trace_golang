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
		}
	}
	setMaterialFormEnabled(false)

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
	}

	matList.OnSelected = func(id widget.ListItemID) {
		selectedMat = int(id)
		loadMaterialToForm(selectedMat)
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

	controls := container.NewVBox(
		widget.NewLabel("Controls"),
		liveUpdate,
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

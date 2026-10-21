package ui

import (
	"io"
	"log"
	"os"
	"strings"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"

	engine "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
	scenecase "github.com/MarkJulian19/path_trace_golang/internal/usecases/scene"
)

// logFilter фильтрует некритичные ошибки GLFW из логов.
type logFilter struct {
	original io.Writer
}

// isWireframeMode сообщает, активен ли режим каркасного рендера.
func isWireframeMode(renderService *engine.Service) bool {
	if renderService.GetBackend() == engine.BackendGPU {
		return renderService.GetRenderMode() == 1
	}
	return renderService.GetCPURenderMode() == 1
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

// Run запускает интерактивный интерфейс с указанной сценой и сценарием рендера.
func Run(scenePath, mode string, scenes *scenecase.Scene, renderService *engine.Service) error {
	log.Printf("UI: starting with scene %q, mode=%s\n", scenePath, mode)

	restoreLogging := configureUILogging()
	defer restoreLogging()
	w := newUIWindow()
	uiDo := runOnUI

	sc, err := scenes.Load(scenePath)
	if err != nil {
		return err
	}

	runtime := newUIRuntime(mode, sc, uiDo)
	session := newUISession(sc, scenes, scenePath, renderService, runtime, uiDo)

	w.SetContent(session.content())
	w.SetMainMenu(session.mainMenu(func() { w.Close() }))
	// Стартовый размер окна фиксированный, а не зависит напрямую от разрешения рендера.
	w.Resize(fyne.NewSize(uiWindowWidth, uiWindowHeight))
	// Автоматический предпросмотр при старте, чтобы сразу было видно картинку.
	go session.startRender(false)

	stopCamera := startSessionCamera(w, session)
	defer stopCamera()
	defer session.controller.stop()
	w.SetOnClosed(func() {
		stopCamera()
		session.controller.stop()
	})

	w.ShowAndRun()
	return nil
}

func configureUILogging() func() {
	originalLogWriter := log.Writer()
	log.SetOutput(&logFilter{original: originalLogWriter})
	if value := os.Getenv("LANG"); value == "" || value == "C" {
		_ = os.Setenv("LANG", "en_US.UTF-8")
	}
	return func() { log.SetOutput(originalLogWriter) }
}

func newUIWindow() fyne.Window {
	application := app.NewWithID("com.markjulian19.pathtracer")
	application.Settings().SetTheme(newBlenderDarkTheme())
	return application.NewWindow("Go Path Tracer")
}

func runOnUI(fn func()) {
	fyne.Do(fn)
}

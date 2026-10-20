package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/MarkJulian19/path_trace_golang/internal/service"
	"github.com/MarkJulian19/path_trace_golang/internal/ui"
	"github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
	scenecase "github.com/MarkJulian19/path_trace_golang/internal/usecases/scene"
)

func main() {
	log.Println("pathtracer: starting main()")

	scenePath := flag.String("scene", "scenes/example_simple.json", "path to scene JSON file")
	mode := flag.String("mode", "preview", "render mode: preview or final")
	useGPU := flag.Bool("gpu", false, "use GPU backend for rendering (if available)")
	headless := flag.Bool("headless", false, "render without UI and save PNG")
	output := flag.String("out", "output.png", "output PNG file for headless render")

	flag.Parse()
	log.Printf("flags: scene=%s mode=%s headless=%v out=%s\n", *scenePath, *mode, *headless, *output)
	application := service.New()

	if *useGPU {
		application.Render.SetBackend(render.BackendGPU)
	} else {
		application.Render.SetBackend(render.BackendCPU)
	}

	if *headless {
		err := renderHeadless(*scenePath, *mode, *output, application.Scenes, application.Render)
		if err != nil {
			log.Println("headless render error:", err)
			os.Exit(1)
		}
		return
	}

	err := ui.Run(*scenePath, *mode, application.Scenes, application.Render)
	if err != nil {
		log.Println("ui error:", err)
		os.Exit(1)
	}
}

func renderHeadless(
	scenePath, mode, outPath string,
	scenes *scenecase.Scene,
	renderer *render.Service,
) error {
	sc, err := scenes.Load(scenePath)
	if err != nil {
		return fmt.Errorf("load scene: %w", err)
	}

	settings := render.SettingsForMode(mode)

	img, err := renderer.Scene(sc, settings)
	if err != nil {
		return fmt.Errorf("render scene: %w", err)
	}

	err = render.SavePNG(outPath, img)
	if err != nil {
		return fmt.Errorf("save png: %w", err)
	}
	return nil
}

package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/user/pathtracer/internal/engine"
	"github.com/user/pathtracer/internal/scene"
	"github.com/user/pathtracer/internal/ui"
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

	if *useGPU {
		engine.SetBackend(engine.BackendGPU)
	} else {
		engine.SetBackend(engine.BackendCPU)
	}

	if *headless {
		if err := renderHeadless(*scenePath, *mode, *output); err != nil {
			log.Println("headless render error:", err)
			os.Exit(1)
		}
		return
	}

	if err := ui.Run(*scenePath, *mode); err != nil {
		log.Println("ui error:", err)
		os.Exit(1)
	}
}

func renderHeadless(scenePath, mode, outPath string) error {
	sc, err := scene.Load(scenePath)
	if err != nil {
		return fmt.Errorf("load scene: %w", err)
	}

	settings := engine.RenderSettingsForMode(mode)

	img, err := engine.RenderScene(sc, settings)
	if err != nil {
		return fmt.Errorf("render scene: %w", err)
	}

	if err := engine.SavePNG(outPath, img); err != nil {
		return fmt.Errorf("save png: %w", err)
	}
	return nil
}

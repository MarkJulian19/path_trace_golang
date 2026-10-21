package main

import (
	"errors"
	"flag"
	"fmt"
	"image"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"time"

	"github.com/MarkJulian19/path_trace_golang/internal/service"
	"github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

const (
	backendGPU          = "gpu"
	validationWidth     = 320
	validationHeight    = 180
	validationDepth     = 24
	outputDirectoryMode = 0o750
)

var errInvalidValidation = errors.New("invalid validation render")

type options struct {
	scenePath, output, backend, profile, reference string
	cfg                                            render.Config
	denoiseSpace, denoiseColor                     float64
	denoise                                        bool
}

func main() {
	runtime.LockOSThread()
	err := run(parseOptions())
	runtime.UnlockOSThread()
	if err != nil {
		log.Print(err)
		os.Exit(1)
	}
}

func parseOptions() options {
	var o options
	flag.StringVar(&o.scenePath, "scene", "scenes/validation/glass_grid.json", "scene JSON")
	flag.StringVar(&o.output, "out", "bin/validation/render.png", "output PNG")
	flag.StringVar(&o.backend, "backend", "cpu", "cpu or gpu; GPU fallback is an error")
	flag.StringVar(&o.profile, "cpuprofile", "", "optional CPU profile")
	flag.StringVar(&o.reference, "reference", "", "optional PNG reference for MSE/PSNR")
	flag.IntVar(&o.cfg.Width, "width", validationWidth, "image width")
	flag.IntVar(&o.cfg.Height, "height", validationHeight, "image height")
	flag.IntVar(&o.cfg.SamplesPerPx, "spp", 64, "samples per pixel")
	flag.IntVar(&o.cfg.MaxDepth, "depth", validationDepth, "maximum path depth")
	flag.Uint64Var(&o.cfg.Seed, "seed", 1, "deterministic seed; 0 selects a random seed")
	flag.BoolVar(&o.denoise, "denoise", false, "enable display denoising")
	flag.Float64Var(&o.denoiseSpace, "denoise-space", 1, "denoise spatial sigma")
	flag.Float64Var(&o.denoiseColor, "denoise-color", 0, "denoise color sigma; 0 uses the renderer default")
	flag.Parse()
	o.cfg.MaxRayDist = 100
	return o
}

func run(o options) error {
	if !validOptions(o) {
		return errInvalidValidation
	}
	app := service.New()
	sc, err := app.Scenes.Load(o.scenePath)
	if err != nil {
		return fmt.Errorf("load validation scene: %w", err)
	}
	if o.backend == backendGPU {
		app.Render.SetBackend(render.BackendGPU)
		app.Render.PrepareGPU()
	}
	app.Render.SetDenoiseConfigFromUI(o.denoise, o.denoiseSpace, o.denoiseColor)
	app.Render.SetSmoothConfigFromUI(false, 1, 0)
	app.Render.SetLivePreview(false)
	finish, err := beginProfile(o.profile)
	if err != nil {
		return err
	}
	img := image.NewRGBA(image.Rect(0, 0, o.cfg.Width, o.cfg.Height))
	start := time.Now()
	actual := app.Render.IntoWithCancel(sc, o.cfg, img, nil, nil)
	elapsed := time.Since(start)
	err = finish()
	if err != nil {
		return err
	}
	if o.backend == backendGPU && actual != render.BackendGPU {
		return fmt.Errorf("%w: requested GPU renderer fell back to CPU", errInvalidValidation)
	}
	return saveResult(o, img, elapsed)
}

func saveResult(o options, img image.Image, elapsed time.Duration) error {
	err := os.MkdirAll(filepath.Dir(o.output), outputDirectoryMode)
	if err != nil {
		return fmt.Errorf("create validation directory: %w", err)
	}
	err = render.SavePNG(o.output, img)
	if err != nil {
		return fmt.Errorf("save validation image: %w", err)
	}
	log.Printf("render complete backend=%s size=%dx%d spp=%d seed=%d denoise=%v elapsed=%s output=%s",
		o.backend, o.cfg.Width, o.cfg.Height, o.cfg.SamplesPerPx, o.cfg.Seed, o.denoise, elapsed, o.output)
	if o.reference != "" {
		return compareReference(o.reference, img)
	}
	return nil
}

func beginProfile(path string) (func() error, error) {
	if path == "" {
		return func() error { return nil }, nil
	}
	file, err := os.Create(filepath.Clean(path))
	if err != nil {
		return nil, fmt.Errorf("create CPU profile: %w", err)
	}
	err = pprof.StartCPUProfile(file)
	if err != nil {
		return nil, errors.Join(err, file.Close())
	}
	return func() error {
		pprof.StopCPUProfile()
		return file.Close()
	}, nil
}

func validOptions(o options) bool {
	return o.cfg.Width > 0 && o.cfg.Height > 0 && o.cfg.SamplesPerPx > 0 && o.cfg.MaxDepth > 0 &&
		(o.backend == "cpu" || o.backend == backendGPU)
}

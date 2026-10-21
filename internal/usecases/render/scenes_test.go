package render_test

import (
	"image"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
	"github.com/MarkJulian19/path_trace_golang/internal/repository/scenejson"
	render "github.com/MarkJulian19/path_trace_golang/internal/usecases/render"
)

func TestBundledScenesRenderAndRoundTrip(t *testing.T) {
	t.Parallel()
	paths, err := filepath.Glob("../../../scenes/*.json")
	validationPaths, validationErr := filepath.Glob("../../../scenes/validation/*.json")
	if validationErr != nil {
		t.Fatal(validationErr)
	}
	paths = append(paths, validationPaths...)
	galleryPaths, galleryErr := filepath.Glob("../../../scenes/gallery/*.json")
	if galleryErr != nil {
		t.Fatal(galleryErr)
	}
	paths = append(paths, galleryPaths...)
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) == 0 {
		t.Fatal("no bundled scenes found")
	}
	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			t.Parallel()
			repository := scenejson.New()
			sc, loadErr := repository.Load(path)
			if loadErr != nil {
				t.Fatal(loadErr)
			}
			output := filepath.Join(t.TempDir(), "scene.json")
			saveErr := repository.Save(output, sc)
			if saveErr != nil {
				t.Fatal(saveErr)
			}
			reloaded, reloadErr := repository.Load(output)
			if reloadErr != nil {
				t.Fatal(reloadErr)
			}
			if !reflect.DeepEqual(sc, reloaded) {
				t.Fatal("saving and reloading changed scene data")
			}
			service := render.New(nil, nil)
			cfg := render.Config{Width: 32, Height: 18, SamplesPerPx: 2, MaxDepth: 4, MaxRayDist: 1000}
			for _, mode := range []int{0, 1} {
				service.SetCPURenderMode(mode)
				img := image.NewRGBA(image.Rect(0, 0, cfg.Width, cfg.Height))
				completed := false
				service.Into(sc, cfg, img, func(current, total int) { completed = current == total })
				if !completed {
					t.Fatalf("mode=%d: render did not report completion", mode)
				}
				assertRenderedPixels(t, img, mode)
			}
		})
	}
}

func assertRenderedPixels(t *testing.T, img *image.RGBA, mode int) {
	t.Helper()
	visible := false
	for i := 0; i < len(img.Pix); i += 4 {
		if img.Pix[i+3] != 255 {
			t.Fatalf("mode=%d: pixel %d was not rendered", mode, i/4)
		}
		visible = visible || img.Pix[i] != 0 || img.Pix[i+1] != 0 || img.Pix[i+2] != 0
	}
	if !visible {
		t.Fatalf("mode=%d: scene is entirely black", mode)
	}
}

func TestCPUSaturatedEmissiveColorsStayBright(t *testing.T) {
	t.Parallel()
	sc := &scene.Scene{
		Camera: scene.Camera{Position: scene.Vec3{Z: 4}, Up: scene.Vec3{Y: 1}, FOV: 45},
		Materials: []scene.Material{{
			ID: "light", Type: scene.MaterialEmissive, Emit: scene.Color{R: 1, G: 1, B: 1}, Power: 100,
		}},
		Objects: []scene.Object{{Type: scene.ObjectSphere, MaterialID: "light", Size: scene.Vec3{X: 1}}},
	}
	for _, mode := range []int{0, 1} {
		service := render.New(nil, nil)
		service.SetCPURenderMode(mode)
		cfg := render.Config{Width: 32, Height: 18, SamplesPerPx: 1, MaxDepth: 2}
		img := service.Render(sc, cfg)
		r, g, b, a := img.At(16, 9).RGBA()
		if r != 65535 || g != 65535 || b != 65535 || a != 65535 {
			t.Fatalf("mode=%d: saturated white wrapped to (%d,%d,%d,%d)", mode, r, g, b, a)
		}
	}
}

func TestCPUMaterialsReflectOrTransmitSky(t *testing.T) {
	t.Parallel()
	for _, typ := range []scene.MaterialType{
		scene.MaterialLambert, scene.MaterialMetal, scene.MaterialMirror, scene.MaterialDielectric,
	} {
		t.Run(string(typ), func(t *testing.T) {
			t.Parallel()
			sc := &scene.Scene{
				Camera: scene.Camera{Position: scene.Vec3{Z: 4}, Up: scene.Vec3{Y: 1}, FOV: 45},
				Materials: []scene.Material{{
					ID: "surface", Type: typ, Albedo: scene.Color{R: 1, G: 1, B: 1}, Rough: 0.3, IOR: 1.5,
				}},
				Objects: []scene.Object{{Type: scene.ObjectSphere, MaterialID: "surface", Size: scene.Vec3{X: 1}}},
				Sky:     &scene.Sky{Type: "solid", Color: scene.Color{B: 0.8}},
			}
			cfg := render.Config{Width: 33, Height: 33, SamplesPerPx: 8, MaxDepth: 8}
			img := render.New(nil, nil).Render(sc, cfg)
			r, _, b, _ := img.At(16, 16).RGBA()
			if b == 0 || b <= r {
				t.Fatalf("surface lost blue sky: red=%d blue=%d", r, b)
			}
		})
	}
}

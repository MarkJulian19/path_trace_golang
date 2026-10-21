# Gallery image sources

[Project overview](../../README.md) · [Русское описание](../../README.ru.md)

The eight render images were produced by this application on **Intel Arc 140T**, using native Windows OpenGL compute (driver `32.0.101.6554`). All use **2560×1440, 1000 samples per pixel, depth 24, seed 42**. There is no CPU fallback, external retouching, or additional blur. Images are stored as original PNG files.

The display denoiser is enabled with **Space = 0.05, Color = 0**. In the renderer, Color zero selects the default color threshold (`0.15`); it does not disable the filter. Space `0.05` makes the weight of a pixel one pixel away approximately `exp(-200)`, so neighboring pixels contribute negligibly. Fine detail and residual path-tracing noise remain visible. Extra GPU smoothing is disabled.

| Image | Scene | What it shows |
| --- | --- | --- |
| [Glass studio](glass-studio.png) | [glass_studio.json](../../scenes/gallery/glass_studio.json) | Clear, amber and blue absorbing glass under large area lights; polished metal. |
| [Warm fog](fog-shafts.png) | [fog_shafts.json](../../scenes/gallery/fog_shafts.json) | Warm area lighting scattered through a participating medium. |
| [Without fog](fog-clear.png) | [fog_clear.json](../../scenes/gallery/fog_clear.json) | Identical camera, objects and lights to the warm fog scene; fog disabled. |
| [Midnight](midnight.png) | [midnight.json](../../scenes/gallery/midnight.json) | Dark surroundings, colored emissive sources and reflections. |
| [Prism Atrium](prism-atrium.png) | [engine_showcase.json](../../scenes/engine_showcase.json) | The bundled atrium scene with fog and its original camera aperture. |
| [Clear atrium](prism-clear.png) | [prism_clear.json](../../scenes/gallery/prism_clear.json) | Atrium without fog, with aperture zero for sharp detail. |
| [Metals and glass](metal-and-glass.png) | [rtx_showcase.json](../../scenes/rtx_showcase.json) | Colored lights reflected and refracted by glass and metal. |
| [Glass refraction](glass-refraction.png) | [glass_grid.json](../../scenes/validation/glass_grid.json) | Grid distortion through a sphere and displacement through a parallel glass slab. |

The studio and warm fog scenes preserve the lights, camera and object layout of the original approved `gallery-studio-preview.png` and `gallery-fog-preview.png`. Their JSON sources are committed so the compositions can be reproduced.

## Reproduce the renders

Build the CLI for the host with a working hardware OpenGL 4.3+ driver:

```bash
go build -o bin/validate-render ./cmd/validate-render
./bin/validate-render -backend gpu \
  -scene scenes/gallery/glass_studio.json \
  -width 2560 -height 1440 -spp 1000 -depth 24 -seed 42 \
  -denoise -denoise-space 0.05 -denoise-color 0 \
  -out output/renders/glass-studio.png
```

Use the scene path from the table for the other images. On Windows, build `bin/validate-render.exe` and run `.\bin\validate-render.exe` with the same flags. Under WSL, use a native Windows build if the Linux driver reports `llvmpipe`; software OpenGL executes on CPU. The diagnostic CLI rejects GPU fallback and logs the actual GL vendor and renderer.

[gallery.json](gallery.json) records image hashes, scenes, renderer and measured wall times. Different drivers can produce slightly different floating-point results and sampling noise, even with the same seed.

## Desktop interface

[interface.png](interface.png) captures the running application window. It is a separate UI render, so window scaling and sampling noise can differ from the standalone gallery images. The viewport render uses the glass studio scene on GPU at 2560×1440 and 1000 SPP, with the same weak denoise settings.

To open that scene in the editor:

```bash
make run SCENE=gallery/glass_studio GPU=1
```

The screenshot shows real controls and completed rendering, without a fabricated interface or composited viewport.

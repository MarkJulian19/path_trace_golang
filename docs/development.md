# Development and troubleshooting

[Project overview](../README.md) · [Русское описание](../README.ru.md) · [Validation scenes](../scenes/validation/README.md)

## Building

The module is `github.com/MarkJulian19/path_trace_golang`. Go modules allow you to clone it outside `GOPATH`.

Build requirements:

- Go 1.27.2+ and `CGO_ENABLED=1`.
- A C compiler: GCC/Clang, or MinGW-w64 for Windows builds.
- OpenGL and window-system development libraries for Fyne/GLFW.
- A desktop session for the editor; OpenGL 4.3+ for the compute backend.

See the root README for Ubuntu packages and native Windows commands.

```bash
make                 # native -> bin/pathtracer
make linux           # Linux binary when building on Linux
make windows         # MinGW-w64 cross-build when available on Linux
make darwin          # native macOS build, or configured cross-build
make help
```

`make` defaults to building the application. Formatting is an explicit `make fmt` action. `make run` starts the editor with `output/renders/` as its working directory and an absolute scene path, so default PNG/video filenames are saved there instead of the repository root. Direct executable launches use the caller's working directory.

Run these as separate commands. `make linux | make windows` is a shell pipeline, not a way to build both targets.

On Linux, the Makefile uses `-tags x11`. For an environment configured for both X11 and Wayland, use `make TAGS=`. Cross-compilation requires a suitable C toolchain; some targets can also use an installed `fyne-cross` and Docker. See the actual [Makefile](../Makefile) for the chosen route on each host. Windows cross-builds do not validate runtime behavior on Windows.

Avoid `go mod vendor`: GLFW's Wayland build relies on files in the module cache that vendoring may omit. The Makefile uses `-mod=mod`.

## Command-line tools

Run from the repository root so relative scene paths resolve.

### Desktop application and preset image rendering

```bash
go run ./cmd/render -scene scenes/example_simple.json
./bin/pathtracer -scene scenes/engine_showcase.json -gpu
./bin/pathtracer -headless -scene scenes/example_simple.json -out output/renders/example.png
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `-scene` | `scenes/example_simple.json` | Scene JSON path. |
| `-mode` | `preview` | Application preset: `preview` or `final`. |
| `-gpu` | `false` | Request GPU; the application can fall back to CPU. |
| `-headless` | `false` | Render an image without opening the editor. |
| `-out` | `output.png` | PNG destination for headless mode. |

The editor reads valid scene render settings, and exposes separate preview/final controls. The `cmd/render -headless` path instead uses `SettingsForMode`: its preset dimensions and sample count do not come from the scene's `settings` block. The final headless preset is 1920×1080 at 1000 SPP and may take a long time. Use the explicit tool below for predictable batch jobs.

### Explicit render parameters

The Makefile exposes the high-quality GPU still-image settings:

```bash
make render-image SCENE=gallery/glass_studio
make render-image SCENE=gallery/glass_studio BACKEND=cpu WIDTH=1280 HEIGHT=720 SPP=128
```

The default output is `output/renders/<scene-name>.png`. Use `OUT=...` to choose another path. `make scenes` lists bundled scenes recursively, including `gallery/` and `validation/`.


```bash
go run ./cmd/validate-render \
  -scene scenes/validation/glass_grid.json \
  -backend cpu -width 640 -height 360 -spp 256 -depth 24 \
  -seed 42 -denoise -out output/renders/glass.png
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `-scene` | `scenes/validation/glass_grid.json` | Scene JSON path. |
| `-backend` | `cpu` | `cpu` or `gpu`; GPU fallback is an error. |
| `-width`, `-height` | `320`, `180` | Output dimensions. |
| `-spp` | `64` | Samples per pixel. |
| `-depth` | `24` | Maximum path depth. |
| `-seed` | `1` | Fixed seed; `0` selects random initialization. |
| `-denoise` | `false` | Apply the common display denoiser. |
| `-denoise-space` | `1` | Spatial sigma in pixels; used with `-denoise`. |
| `-denoise-color` | `0` | Color sigma; `0` selects the default threshold of `0.15`. |
| `-out` | `bin/validation/render.png` | PNG output; parent directories are created. |
| `-reference` | empty | PNG reference for display-space MSE/PSNR. |
| `-cpuprofile` | empty | Write a Go CPU profile. |

The tool currently uses a ray-distance limit of 100 scene units. CPU and GPU use different random generators, so the same seed provides repeatability within a backend, not identical CPU/GPU paths. Image-space MSE is useful for noise comparisons; it does not establish physical correctness of linear radiance.

## GPU troubleshooting

The compute backend needs OpenGL 4.3+. It does not use CUDA, Vulkan ray tracing, DXR, or dedicated RT cores. A scene filename containing `gpu` or `rtx` selects a scene, not a backend.

If the UI reports **CPU fallback**, inspect stderr for initialization or shader errors. Renderer names such as `llvmpipe`, `softpipe`, or `SwiftShader` indicate software OpenGL: compute is executed on CPU. In that case the application normally chooses its CPU renderer. In WSL, use a correctly configured hardware driver/passthrough or a native build to exercise hardware acceleration.

To deliberately run GLSL on a software driver for testing:

```bash
PATHTRACER_GPU_ALLOW_SOFTWARE=1 go run ./cmd/validate-render \
  -backend gpu -scene scenes/validation/glass_grid.json -out output/renders/software-gl.png
```

### Scheduling and diagnostics

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `PATHTRACER_WORKERS` | Available logical CPUs | Override the CPU worker count. |
| `PATHTRACER_GPU_DEBUG` | Off | GPU diagnostics and timing logs. |
| `PATHTRACER_GPU_LIVE_READBACK` | On | Intermediate GPU image transfers; `0` keeps final readback. |
| `PATHTRACER_GPU_TILE_SIZE` | `256` | Adaptive dispatch size limit, clamped to 32..512 and aligned to 32. |
| `PATHTRACER_GPU_SAMPLES_PER_PASS` | `4` | Sample batch, clamped to 1..16. |
| `PATHTRACER_GPU_TEMPORAL` | On for hardware GL | Temporal-target setting; the exact accumulated HDR mean is preserved. |

GPU dispatch starts at 64×64 and adapts toward an 8 ms budget. One tile is submitted at a time; a dedicated GL thread waits for completion between submissions. This bounds queued work and provides cancellation points. Driver compilation or an unusually slow tile can exceed the target budget.

The editor controls live updates and filtering, which can override environment defaults. Denoising for both backends runs on CPU after tone mapping, does not modify the linear HDR estimator, and may soften fine details. It is not a replacement for sampling. Eight-bit output uses deterministic dithering to reduce gradient banding.

## Validation

Ordinary checks:

```bash
go test ./...
go test -race ./...
go test -tags benchinternal ./...
go vet ./...
go build ./...
golangci-lint run
```

`golangci-lint fmt` applies the repository formatters. Run `go generate ./...` when changing mocked interfaces, and review the generated diff.

Actual GLSL integration tests require a display and an OpenGL 4.3+ context:

```bash
PATHTRACER_GPU_INTEGRATION=1 PATHTRACER_GPU_TEMPORAL=1 \
  PATHTRACER_GPU_TILE_SIZE=33 go test -race ./internal/transport/gpu \
  -run '^TestGPU(PhysicsAudit|Integration)$' -count=1 -v -timeout=180s
```

These tests cover materials, optical references, accumulation, readback, resizing, cancellation, adaptive tiles, and Normal/Wireframe rendering of bundled scenes. Surface regressions also verify that a reflected ray cannot immediately re-intersect its own convex sphere or box, that a box uses the actual intersected slab normal near an edge, and that a thin glass slab retains both interfaces. CPU and GPU reconstruct hit points on their geometric surfaces before applying the secondary-ray offset; GPU offset components round away from zero. They can execute through Mesa software GL; passing on Mesa is not a benchmark or runtime guarantee for every hardware driver.

For reproducible visual tests and reference comparisons, follow the [validation-scene guide](../scenes/validation/README.md). Keep denoising disabled when checking the underlying transport numerically. The transport model uses RGB; glass is an ideal specular dielectric, and metals use the implemented GGX/Smith/Schlick model. Complex specular caustics can converge slowly.

## Repository map

```text
cmd/
  render/             Desktop entry point and preset headless rendering
  validate-render/    Explicit render settings, seed, metrics, profiling
internal/
  entities/           Shared scene/render types and errors
  repository/         JSON scene persistence
  usecases/render/    CPU integrator, materials, optics, image/video output
  usecases/scene/     Scene loading and saving use cases
  transport/gpu/      OpenGL resources, dispatch, GLSL, integration tests
  renderdisplay/      Shared eight-bit output quantization
  renderfilter/       Shared display denoising
  service/            Application wiring
  ui/                 Fyne editor, render lifecycle, camera and controls
scenes/               Example scenes
  gallery/            Reproducible gallery compositions
  validation/         Focused optical/noise regression scenes
docs/                 User/developer documentation
  images/             Published gallery PNGs and their provenance
  math/               Mathematical notes, published PDF and LaTeX source
  media/              Seven-second MP4/GIF preview and provenance
bin/                  Local executables (ignored)
output/               Local renders, profiles and archived images (ignored)
```

`make bench-profile` writes profiles and its test executable under `output/profiles/`. `make video-preview` encodes the rendered frame sequence as MP4 and GIF; see [video notes](media/README.md). `make clean` removes build directories and preserves `output/`, including archived images.

`bin/`, `output/`, and `.codegraph/` are ignored build/workspace artifacts. Documentation images are stored under `docs/images/` so they can be committed and displayed on GitHub. Old root-level `ui_render*.png` images are preserved locally under `output/archive/legacy-renders/`; they are excluded from Git.

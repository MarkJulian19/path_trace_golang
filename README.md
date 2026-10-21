# path_trace_golang

CPU/GPU path tracer with a Fyne UI.

## Module

```text
github.com/MarkJulian19/path_trace_golang
```

Clone anywhere — Go modules resolve packages from this path, not from your filesystem location.

## Requirements

- Go 1.27.2+
- CGO enabled (`CGO_ENABLED=1`)
- C compiler (`gcc` / `clang`)
- OpenGL + X11 (+ Wayland for Fyne 2.8 / GLFW 3.4) development libraries

Ubuntu / Debian / WSL:

```bash
sudo apt-get install -y gcc pkg-config libgl1-mesa-dev xorg-dev \
  libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev libxxf86vm-dev \
  libwayland-dev libxkbcommon-dev wayland-protocols
```

## Build

```bash
git clone https://github.com/MarkJulian19/path_trace_golang.git
cd path_trace_golang
make            # -> bin/pathtracer (Linux uses -tags x11)
```

Do not use `go mod vendor` here: GLFW 3.4 Wayland headers are skipped by vendoring. The Makefile builds with `-mod=mod`.

Cross-platform (needs [fyne-cross](https://github.com/fyne-io/fyne-cross) or mingw):

```bash
go install github.com/fyne-io/fyne-cross@latest
make linux | make windows | make darwin | make all
```

## Run

```bash
make scenes                 # list scenes/
make run                    # UI, scenes/example_simple.json
make run-showcase             # Prism Atrium — full engine demo (GPU)
make run SCENE=engine_showcase GPU=1 MODE=final

make headless               # PNG -> bin/example_simple.png
make headless-rtx OUT=rtx.png
```

## GPU debug

```bash
PATHTRACER_GPU_DEBUG=1 make run-gpu
PATHTRACER_GPU_LIVE_READBACK=0 make run-gpu   # faster preview (no mid-frame PCIe)
PATHTRACER_GPU_TILE_SIZE=128 make run-gpu     # smaller GPU batches for Windows/TDR-safe preview
PATHTRACER_GPU_SAMPLES_PER_PASS=2 make run-gpu # lower GPU load / higher preview responsiveness
```

Also: `PATHTRACER_GPU_DENOISE`, `PATHTRACER_GPU_SMOOTH`, `PATHTRACER_GPU_TEMPORAL=0`.
Live GPU readback is throttled to every few passes; use `PATHTRACER_GPU_LIVE_READBACK=0`
when only the final frame is needed.
Tile size is clamped to `32..512`; the default is `256`.
Tile size is rounded down to a multiple of 32 so adjacent dispatches do not overlap.

CPU and GPU share the Normal / Wireframe selection. Wireframe is a continuous,
colored surface preview; switching to Normal stops the stream and starts a new render.
The status displays `CPU fallback` when the selected GPU backend cannot run.

**WSL note:** if logs show `llvmpipe` / `softpipe` / `SwiftShader`, the “GPU” backend is **software OpenGL on CPU** — the discrete GPU stays idle. Use a native Windows/Linux build with a real GL 4.3+ driver, or enable GPU passthrough.

Software OpenGL uses the CPU renderer by default. To explicitly exercise compute
shaders on Mesa, set `PATHTRACER_GPU_ALLOW_SOFTWARE=1` (this still runs on the CPU).

## Validation

```bash
make generate fmt test test-race lint build
go vet ./...
go build ./...
PATHTRACER_GPU_INTEGRATION=1 PATHTRACER_GPU_TEMPORAL=1 \
  PATHTRACER_GPU_TILE_SIZE=33 go test ./internal/transport/gpu \
  -run '^TestGPUIntegration$' -count=1 -v -timeout=90s
```

The OpenGL integration test requires a display and GL 4.3+. It executes the actual
shaders, including mode changes, resizing, temporal accumulation and live/final
readback. The other tests cover UI completion and cancellation, CPU rendering,
and loading, saving and rendering every bundled scene.

# Cross-platform build for pathtracer (Fyne + CGO).
#
#   make / make build          - native host binary -> bin/pathtracer
#   make run                   - UI with SCENE (default: example_simple)
#   make run-gpu / run-metal … - shortcuts for scenes/
#   make headless              - PNG render without UI
#   make scenes                - list available scenes
#   make linux|windows|darwin|all
#
# Examples:
#   make run SCENE=gpu_showcase GPU=1
#   make run-rtx MODE=final
#   make headless SCENE=metal_glass_room
#   make render-image SCENE=gallery/glass_studio
#
# Cross-compiles: go install github.com/fyne-io/fyne-cross@latest

.DEFAULT_GOAL := build

APP       := pathtracer
PKG       := ./cmd/render
BIN_DIR   := bin
SCENES_DIR := scenes
OUTPUT_DIR ?= output
RENDER_DIR := $(OUTPUT_DIR)/renders
PROFILE_DIR := $(OUTPUT_DIR)/profiles
VIDEO_FRAME_DIR ?= $(OUTPUT_DIR)/video-1080p500/frames
VIDEO_FPS ?= 24
VIDEO_FRAMES ?= 168
GIF_WIDTH ?= 560
MEDIA_DIR := docs/media
FFMPEG ?= $(if $(wildcard $(BIN_DIR)/tools/ffmpeg),$(BIN_DIR)/tools/ffmpeg,ffmpeg)
VERSION   ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)

GO        ?= go
# -mod=mod: glfw v3.4 Wayland headers live in module cache; go mod vendor skips them
GOFLAGS   ?= -trimpath -mod=mod
LDFLAGS   ?= -s -w -X main.version=$(VERSION)
CGO       := CGO_ENABLED=1

# Linux: X11 only by default (WSL/desktop without full Wayland protocol codegen).
# Full X11+Wayland: make TAGS=
TAGS      ?= $(if $(filter linux,$(shell $(GO) env GOOS)),x11,)

HOST_OS   := $(shell $(GO) env GOOS)
HOST_ARCH := $(shell $(GO) env GOARCH)

CC_WINDOWS ?= x86_64-w64-mingw32-gcc
CC_DARWIN  ?=

BUILD_TAGS := $(if $(TAGS),-tags '$(TAGS)',)

# --- run / scene options ---
# SCENE: basename without .json, or path under scenes/, or absolute/relative path
SCENE     ?= example_simple
MODE      ?= preview
GPU       ?= 0
OUT       ?= $(RENDER_DIR)/$(notdir $(basename $(SCENE))).png
BACKEND   ?= gpu
WIDTH     ?= 2560
HEIGHT    ?= 1440
SPP       ?= 1000
DEPTH     ?= 24
SEED      ?= 42

SCENE_PATH := $(if $(wildcard $(SCENE)),$(SCENE),$(if $(wildcard $(SCENES_DIR)/$(SCENE).json),$(SCENES_DIR)/$(SCENE).json,$(if $(wildcard $(SCENES_DIR)/$(SCENE)),$(SCENES_DIR)/$(SCENE),$(SCENES_DIR)/$(SCENE).json)))

RUN_FLAGS := -scene $(SCENE_PATH) -mode $(MODE)
ifeq ($(GPU),1)
RUN_FLAGS += -gpu
endif

BIN := ./$(BIN_DIR)/$(APP)

.PHONY: help build native linux windows darwin all clean deps deps-windows \
	fmt lint generate test test-race bench bench-profile \
	run headless scenes validate-render render-image video-preview \
	run-simple run-gpu run-metal run-rtx run-showcase run-test run-comprehensive \
	headless-simple headless-gpu headless-metal headless-rtx headless-showcase headless-test headless-comprehensive

fmt:
	golangci-lint fmt

lint:
	golangci-lint run

generate:
	$(GO) generate ./...

test:
	$(GO) test ./...

test-race:
	$(GO) test -race ./...

bench:
	$(GO) test ./internal/usecases/render ./internal/repository/scenejson ./internal/usecases/scene \
		-run '^$$' -bench '^(BenchmarkRender|BenchmarkInto|BenchmarkPickObject|BenchmarkScene$$|BenchmarkSavePNG|BenchmarkInterpolateCamera|BenchmarkVideoSequence|BenchmarkServiceAPI/(Render|Into|Scene)|BenchmarkServiceVideoSequence|BenchmarkCreateVideoFromFrames|BenchmarkRepository(Load|Save))' -benchmem -count=5
	$(GO) test ./internal/usecases/render ./internal/repository/scenejson ./internal/usecases/scene ./internal/transport/gpu ./internal/service \
		-run '^$$' -bench '^(BenchmarkSettingsForMode|BenchmarkDefaultVideoEncodingConfig|BenchmarkRepositoryNew|BenchmarkSceneUsecase(New|Load|Save)|BenchmarkServiceAPI/(New|SetBackend|GetBackend|SetCPURenderMode|GetCPURenderMode)|BenchmarkServiceGPUControls|BenchmarkGPU|BenchmarkContainerNew)' -benchmem -benchtime=100ms -count=3
	$(GO) test -tags benchinternal ./internal/usecases/render ./internal/repository/scenejson ./internal/transport/gpu \
		-run '^$$' -bench '^BenchmarkInternal' -benchmem -benchtime=100ms -count=3

bench-profile:
	mkdir -p "$(PROFILE_DIR)"
	$(GO) test ./internal/usecases/render -run '^$$' -bench '^BenchmarkRender/medium$$' \
		-benchmem -cpuprofile="$(PROFILE_DIR)/render.cpu.pprof" -memprofile="$(PROFILE_DIR)/render.mem.pprof" \
		-o "$(PROFILE_DIR)/render.test"

help:
	@echo "Build:"
	@echo "  make / make build     native $(HOST_OS)/$(HOST_ARCH) -> $(BIN_DIR)/$(APP)"
	@echo "  make linux|windows|darwin|all"
	@echo ""
	@echo "Run (UI):"
	@echo "  make run              SCENE=$(SCENE) MODE=$(MODE) GPU=$(GPU)"
	@echo "  make run-simple|run-gpu|run-metal|run-rtx|run-showcase|run-test|run-comprehensive"
	@echo "  make run SCENE=rtx_showcase GPU=1 MODE=final"
	@echo ""
	@echo "Headless PNG:"
	@echo "  make headless         OUT=$(OUT)"
	@echo "  make headless-simple|headless-gpu|…  (same scene shortcuts)"
	@echo "  make render-image SCENE=gallery/glass_studio BACKEND=gpu SPP=1000"
	@echo "  renders -> $(RENDER_DIR)/; profiles -> $(PROFILE_DIR)/"
	@echo "  make video-preview    encode $(VIDEO_FRAMES) frames at $(VIDEO_FPS) FPS as MP4 + README GIF"
	@echo ""
	@echo "Other: make scenes | make clean | make deps"
	@echo "Linux default TAGS=x11 (TAGS= for X11+Wayland)."

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

build: native

native: $(BIN_DIR)
	$(CGO) $(GO) build $(GOFLAGS) $(BUILD_TAGS) -ldflags '$(LDFLAGS)' \
		-o $(BIN_DIR)/$(APP) $(PKG)
	@echo "=> $(BIN_DIR)/$(APP) ($(HOST_OS)/$(HOST_ARCH) tags=$(TAGS))"

# --- scenes ---

scenes:
	@echo "Available scenes in $(SCENES_DIR)/:"
	@find "$(SCENES_DIR)" -type f -name '*.json' | sort | sed 's|^$(SCENES_DIR)/||;s|\.json$$||;s/^/  /'
	@echo ""
	@echo "Usage: make run SCENE=<name>   or   make run-gpu / run-metal / …"

run: native
	@test -f "$(SCENE_PATH)" || { echo "scene not found: $(SCENE_PATH)"; echo "try: make scenes"; exit 1; }
	@mkdir -p "$(RENDER_DIR)"
	@echo "=> $(BIN) $(RUN_FLAGS) (working directory: $(RENDER_DIR))"
	cd "$(RENDER_DIR)" && "$(abspath $(BIN))" -scene "$(abspath $(SCENE_PATH))" -mode "$(MODE)" $(if $(filter 1,$(GPU)),-gpu,)

headless: native
	@test -f "$(SCENE_PATH)" || { echo "scene not found: $(SCENE_PATH)"; echo "try: make scenes"; exit 1; }
	@mkdir -p "$$(dirname "$(OUT)")"
	@echo "=> $(BIN) -headless $(RUN_FLAGS) -out $(OUT)"
	$(BIN) -headless $(RUN_FLAGS) -out "$(OUT)"
	@echo "=> wrote $(OUT)"

validate-render: $(BIN_DIR)
	$(CGO) $(GO) build $(GOFLAGS) $(BUILD_TAGS) -o "$(BIN_DIR)/validate-render" ./cmd/validate-render

render-image: validate-render
	@test -f "$(SCENE_PATH)" || { echo "scene not found: $(SCENE_PATH)"; echo "try: make scenes"; exit 1; }
	"./$(BIN_DIR)/validate-render" -scene "$(SCENE_PATH)" -backend "$(BACKEND)" \
		-width $(WIDTH) -height $(HEIGHT) -spp $(SPP) -depth $(DEPTH) -seed $(SEED) \
		-denoise -denoise-space 0.05 -denoise-color 0 -out "$(OUT)"

video-preview:
	@test -f "$(VIDEO_FRAME_DIR)/0000.png" -a -f "$(VIDEO_FRAME_DIR)/$$(printf '%04d' $$(($(VIDEO_FRAMES) - 1))).png" || { echo "$(VIDEO_FRAMES) rendered frames required in $(VIDEO_FRAME_DIR)/"; exit 1; }
	@mkdir -p "$(MEDIA_DIR)"
	"$(FFMPEG)" -hide_banner -loglevel warning -y -framerate $(VIDEO_FPS) -start_number 0 \
		-i "$(VIDEO_FRAME_DIR)/%04d.png" -frames:v $(VIDEO_FRAMES) -an -c:v libx264 -preset medium \
		-crf 18 -pix_fmt yuv420p -movflags +faststart "$(MEDIA_DIR)/studio-preview.mp4"
	"$(FFMPEG)" -hide_banner -loglevel warning -y -i "$(MEDIA_DIR)/studio-preview.mp4" \
		-filter_complex "fps=$(VIDEO_FPS),scale=$(GIF_WIDTH):-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=sierra2_4a" \
		-loop 0 "$(MEDIA_DIR)/studio-preview.gif"

# Named scene shortcuts (UI)
run-simple:        ; @$(MAKE) run SCENE=example_simple
run-gpu:           ; @$(MAKE) run SCENE=gpu_showcase GPU=1
run-metal:         ; @$(MAKE) run SCENE=metal_glass_room
run-rtx:           ; @$(MAKE) run SCENE=rtx_showcase GPU=1
run-showcase:      ; @$(MAKE) run SCENE=engine_showcase GPU=1
run-test:          ; @$(MAKE) run SCENE=test_scene
run-comprehensive: ; @$(MAKE) run SCENE=test_comprehensive

# Named scene shortcuts (headless PNG -> output/renders/<scene>.png)
headless-simple:        ; @$(MAKE) headless SCENE=example_simple
headless-gpu:           ; @$(MAKE) headless SCENE=gpu_showcase GPU=1
headless-metal:         ; @$(MAKE) headless SCENE=metal_glass_room
headless-rtx:           ; @$(MAKE) headless SCENE=rtx_showcase GPU=1
headless-showcase:      ; @$(MAKE) headless SCENE=engine_showcase GPU=1 MODE=final
headless-test:          ; @$(MAKE) headless SCENE=test_scene
headless-comprehensive: ; @$(MAKE) headless SCENE=test_comprehensive

# --- cross-platform ---

HAS_FYNE_CROSS := $(shell command -v fyne-cross 2>/dev/null)

define build_native_like
	@mkdir -p $(BIN_DIR)
	$(CGO) GOOS=$(1) GOARCH=$(2) $(GO) build $(GOFLAGS) $(BUILD_TAGS) -ldflags '$(LDFLAGS)' \
		-o $(BIN_DIR)/$(APP)-$(1)-$(2)$(3) $(PKG)
	@echo "=> $(BIN_DIR)/$(APP)-$(1)-$(2)$(3)"
endef

define build_fyne_cross
	@mkdir -p $(BIN_DIR)
	fyne-cross $(1) -arch=$(2) -name=$(APP) -app-id=com.markjulian19.pathtracer ./cmd/render
	@if [ -d fyne-cross/dist/$(1)-$(2) ]; then \
		cp -f fyne-cross/dist/$(1)-$(2)/$(APP)* $(BIN_DIR)/ 2>/dev/null || true; \
		cp -f fyne-cross/dist/$(1)-$(2)/*.tar.xz $(BIN_DIR)/ 2>/dev/null || true; \
		cp -f fyne-cross/dist/$(1)-$(2)/*.zip $(BIN_DIR)/ 2>/dev/null || true; \
	fi
	@echo "=> fyne-cross $(1)/$(2)"
endef

linux: $(BIN_DIR)
ifeq ($(HOST_OS),linux)
	$(call build_native_like,linux,$(HOST_ARCH),)
else ifneq ($(HAS_FYNE_CROSS),)
	$(call build_fyne_cross,linux,amd64)
else
	$(error Cross-build to linux: install fyne-cross or build on Linux)
endif

windows: $(BIN_DIR)
ifeq ($(HOST_OS),windows)
	$(call build_native_like,windows,$(HOST_ARCH),.exe)
else ifneq ($(HAS_FYNE_CROSS),)
	$(call build_fyne_cross,windows,amd64)
else ifneq ($(shell command -v $(CC_WINDOWS) 2>/dev/null),)
	$(CGO) CC=$(CC_WINDOWS) GOOS=windows GOARCH=amd64 $(GO) build $(GOFLAGS) \
		-ldflags '$(LDFLAGS) -H windowsgui' -o $(BIN_DIR)/$(APP)-windows-amd64.exe $(PKG)
	@echo "=> $(BIN_DIR)/$(APP)-windows-amd64.exe (mingw)"
	@echo "note: for a real NVIDIA/AMD GPU run this .exe on Windows (not under WSL llvmpipe)"
else
	$(info Cross-build to windows needs a C cross-compiler (CGO/Fyne).)
	$(info )
	$(info Option A — MinGW:)
	$(info   sudo apt-get install -y gcc-mingw-w64-x86-64)
	$(info   make windows)
	$(info )
	$(info Option B — fyne-cross + Docker:)
	$(info   go install github.com/fyne-io/fyne-cross@latest)
	$(info   make windows)
	$(info )
	$(info Option C — best for real GPU: build natively on Windows)
	$(info   go build -o pathtracer.exe ./cmd/render)
	$(error missing $(CC_WINDOWS) and fyne-cross — or run: make deps-windows)
endif

.PHONY: deps-windows
deps-windows:
	@echo "Installing MinGW-w64 for Windows cross-compile..."
	sudo apt-get install -y gcc-mingw-w64-x86-64
	@echo "OK: $$(command -v x86_64-w64-mingw32-gcc)"
	@echo "Now run: make windows"

darwin: $(BIN_DIR)
ifeq ($(HOST_OS),darwin)
	$(call build_native_like,darwin,$(HOST_ARCH),)
else ifneq ($(HAS_FYNE_CROSS),)
	$(call build_fyne_cross,darwin,amd64)
	-$(call build_fyne_cross,darwin,arm64)
else ifneq ($(CC_DARWIN),)
	$(CGO) CC=$(CC_DARWIN) GOOS=darwin GOARCH=amd64 $(GO) build $(GOFLAGS) \
		-ldflags '$(LDFLAGS)' -o $(BIN_DIR)/$(APP)-darwin-amd64 $(PKG)
	$(CGO) CC=$(CC_DARWIN) GOOS=darwin GOARCH=arm64 $(GO) build $(GOFLAGS) \
		-ldflags '$(LDFLAGS)' -o $(BIN_DIR)/$(APP)-darwin-arm64 $(PKG)
else
	$(error Cross-build to darwin: install fyne-cross or set CC_DARWIN=...)
endif

all:
	@$(MAKE) linux; $(MAKE) windows; $(MAKE) darwin; true

deps:
	$(GO) mod download
	$(GO) mod tidy

clean:
	rm -rf $(BIN_DIR) fyne-cross
	@echo "cleaned $(BIN_DIR)/ and fyne-cross/"

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
#   make headless SCENE=metal_glass_room OUT=out.png
#
# Cross-compiles: go install github.com/fyne-io/fyne-cross@latest

APP       := pathtracer
PKG       := ./cmd/render
BIN_DIR   := bin
SCENES_DIR := scenes
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
OUT       ?= $(BIN_DIR)/$(SCENE).png

SCENE_PATH := $(if $(wildcard $(SCENE)),$(SCENE),$(if $(wildcard $(SCENES_DIR)/$(SCENE).json),$(SCENES_DIR)/$(SCENE).json,$(if $(wildcard $(SCENES_DIR)/$(SCENE)),$(SCENES_DIR)/$(SCENE),$(SCENES_DIR)/$(SCENE).json)))

RUN_FLAGS := -scene $(SCENE_PATH) -mode $(MODE)
ifeq ($(GPU),1)
RUN_FLAGS += -gpu
endif

BIN := ./$(BIN_DIR)/$(APP)

.PHONY: help build native linux windows darwin all clean deps deps-windows \
	run headless scenes \
	run-simple run-gpu run-metal run-rtx run-showcase run-test run-comprehensive \
	headless-simple headless-gpu headless-metal headless-rtx headless-showcase headless-test headless-comprehensive

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
	@ls -1 $(SCENES_DIR)/*.json 2>/dev/null | sed 's|.*/||;s|\.json$$||' | sed 's/^/  /'
	@echo ""
	@echo "Usage: make run SCENE=<name>   or   make run-gpu / run-metal / …"

run: native
	@test -f "$(SCENE_PATH)" || { echo "scene not found: $(SCENE_PATH)"; echo "try: make scenes"; exit 1; }
	@echo "=> $(BIN) $(RUN_FLAGS)"
	$(BIN) $(RUN_FLAGS)

headless: native
	@test -f "$(SCENE_PATH)" || { echo "scene not found: $(SCENE_PATH)"; echo "try: make scenes"; exit 1; }
	@mkdir -p $(dir $(OUT))
	@echo "=> $(BIN) -headless $(RUN_FLAGS) -out $(OUT)"
	$(BIN) -headless $(RUN_FLAGS) -out $(OUT)
	@echo "=> wrote $(OUT)"

# Named scene shortcuts (UI)
run-simple:        ; @$(MAKE) run SCENE=example_simple
run-gpu:           ; @$(MAKE) run SCENE=gpu_showcase GPU=1
run-metal:         ; @$(MAKE) run SCENE=metal_glass_room
run-rtx:           ; @$(MAKE) run SCENE=rtx_showcase GPU=1
run-showcase:      ; @$(MAKE) run SCENE=engine_showcase GPU=1
run-test:          ; @$(MAKE) run SCENE=test_scene
run-comprehensive: ; @$(MAKE) run SCENE=test_comprehensive

# Named scene shortcuts (headless PNG -> bin/<scene>.png)
headless-simple:        ; @$(MAKE) headless SCENE=example_simple OUT=$(BIN_DIR)/example_simple.png
headless-gpu:           ; @$(MAKE) headless SCENE=gpu_showcase GPU=1 OUT=$(BIN_DIR)/gpu_showcase.png
headless-metal:         ; @$(MAKE) headless SCENE=metal_glass_room OUT=$(BIN_DIR)/metal_glass_room.png
headless-rtx:           ; @$(MAKE) headless SCENE=rtx_showcase GPU=1 OUT=$(BIN_DIR)/rtx_showcase.png
headless-showcase:      ; @$(MAKE) headless SCENE=engine_showcase GPU=1 MODE=final OUT=$(BIN_DIR)/engine_showcase.png
headless-test:          ; @$(MAKE) headless SCENE=test_scene OUT=$(BIN_DIR)/test_scene.png
headless-comprehensive: ; @$(MAKE) headless SCENE=test_comprehensive OUT=$(BIN_DIR)/test_comprehensive.png

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

# Contributing

[Project overview](README.md) · [Development guide](docs/development.md)

For a rendering bug, include the scene JSON or a minimal scene, the CPU/GPU backend, output dimensions, sample count, path depth, and seed. Attach the resulting image and describe the expected result. For GPU issues, include the operating system and GL vendor/renderer/version from the logs, and mention whether software OpenGL or CPU fallback was used.

For image comparisons, use `cmd/validate-render` with fixed settings. Disable denoising when investigating transport or numerical differences. Keep a raw image alongside any filtered result.

Before submitting code changes:

```bash
golangci-lint fmt
go test ./...
go test -race ./...
go vet ./...
go build ./...
golangci-lint run
```

Changes to materials, sampling, intersections, or light transport should include a focused numerical regression or a reproducible validation scene. GPU changes should also be checked with the [OpenGL integration tests](docs/development.md#validation) when a suitable context is available. State which driver you tested and which checks you could not run.

Keep unrelated refactors out of fixes. Explain the visible behavior being changed, any intentional image differences, and the tests or comparison renders used to validate it. For documentation-only changes, verify relative links, image paths, and runnable examples.

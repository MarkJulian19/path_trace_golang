# Gallery scenes

These scenes accompany the [project gallery](../../docs/images/README.md). Each sets a 2560×1440 viewport and 1000 samples per pixel. The explicit render CLI can override those scene settings.

- `glass_studio.json`: the original studio preview composition, with clear, amber and blue glass, a silver sphere and three large emissive box lights.
- `fog_shafts.json`: the original warm fog preview composition, with glass and metal spheres, a window-sized emissive box and dark slats.
- `fog_clear.json`: the same camera, materials, objects and lighting as `fog_shafts.json`, with only fog removed.
- `midnight.json`: `rtx_showcase.json` with a dark solid background, no fog, zero aperture and emissive power multiplied by 0.35.
- `prism_clear.json`: `engine_showcase.json` with fog disabled and aperture set to zero.

Render with a native hardware GPU driver:

```bash
go run ./cmd/validate-render -backend gpu \
  -scene scenes/gallery/glass_studio.json \
  -width 2560 -height 1440 -spp 1000 -depth 24 -seed 42 \
  -denoise -denoise-space 0.05 -denoise-color 0 \
  -out output/renders/glass-studio.png
```

These are demonstration scenes. Analytical optical checks live in [scenes/validation](../validation/README.md).

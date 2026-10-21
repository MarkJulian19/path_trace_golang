# Render gallery

[Project overview](../README.md) · [Русское описание](../README.ru.md) · [Image settings](images/README.md)

![Glass studio: clear, amber and blue glass under large area lights](images/glass-studio.png)

[Studio scene](../scenes/gallery/glass_studio.json)

All eight render images use **GPU · 2560×1440 · 1000 SPP · depth 24**. Display denoising is set to **Space 0.05, Color 0**, with extra smoothing disabled. These settings preserve fine detail and visible sampling noise. See [sources, settings and reproduction commands](images/README.md).

### Warm volumetric fog

![Warm light spreading through fog around glass and metal spheres](images/fog-shafts.png)

[Full size](images/fog-shafts.png) · [Scene](../scenes/gallery/fog_shafts.json). Warm area lighting and volumetric scattering; the studio scene above and this composition retain the original gallery preview lighting.

### The same room without fog

![The same objects and lighting with fog disabled, revealing the room and shadows](images/fog-clear.png)

[Full size](images/fog-clear.png) · [Scene](../scenes/gallery/fog_clear.json). Camera, objects and lights match the fog image; only the participating medium is disabled.

### Low light and reflections

![Glass and polished metal lit by small colored light sources in a dark room](images/midnight.png)

[Full size](images/midnight.png) · [Scene](../scenes/gallery/midnight.json).

### Prism Atrium with fog

![Glass, metal and colored area lights in a room with volumetric fog](images/prism-atrium.png)

[Full size](images/prism-atrium.png) · [Scene](../scenes/engine_showcase.json).

### Prism Atrium without fog

![The atrium's glass and metal objects under colored lighting without fog](images/prism-clear.png)

[Full size](images/prism-clear.png) · [Scene](../scenes/gallery/prism_clear.json). Fog is disabled and the aperture is zero for a sharp comparison.

### Metals, mirrors, and colored lights

![Colored illumination reflected by metal, mirror, and glass objects](images/metal-and-glass.png)

[Full size](images/metal-and-glass.png) · [Scene](../scenes/rtx_showcase.json).

### Refraction through glass

![A grid distorted by a glass sphere and displaced behind a parallel glass slab](images/glass-refraction.png)

[Full size](images/glass-refraction.png) · [Scene](../scenes/validation/glass_grid.json). A sphere bends the grid; a parallel slab displaces it. See the [optical validation scenes](../scenes/validation/README.md).


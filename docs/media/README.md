# Seven-second GPU preview

[Project overview](../../README.md) · [MP4](studio-preview.mp4) · [Animated GIF](studio-preview.gif)

Both project READMEs include an HTML MP4 player for compatible local Markdown previews and a separate animated GIF for GitHub. If a local preview blocks HTML video, open `studio-preview.mp4` directly in a browser or media player.

The preview contains 168 independently rendered frames with a moving camera, rather than a pan over a still image. It uses the [approved glass studio composition](../../scenes/gallery/glass_studio.json), preserving all lights, materials and objects.

- MP4: 1920×1080, 24 FPS, exactly 7 seconds, H.264 with `yuv420p` and fast-start metadata.
- README GIF: 560×315, 24 FPS, 7 seconds, looping.
- Rendering: hardware GPU, 500 samples per pixel, depth 24, ray limit 100, seed 42 for every frame.
- Display denoise: Space `0.05`, Color `0`; extra smoothing disabled. Residual sampling noise remains visible.
- Still gallery photographs retain their separate 2560×1440 / 1000 SPP settings.

For frame index `i = 0..167`, the camera moves on a small closed arc:

```text
phase = 2*pi*i/168
angle = atan2(6,8) + radians(5)*sin(phase)
position = (10*sin(angle), 2.8 + 0.075*(1-cos(phase)), 10*cos(angle))
target = (0,1,0)
```

All other camera settings come from the studio scene. The periodic path provides a continuous loop. Intermediate frame scenes, PNGs and logs are local artifacts in `output/video-1080p500/`, excluded from Git. [preview.json](preview.json) records the renderer, settings and output hashes.

Frames use the application’s existing rendering service and GPU adapter in one native Windows process, reusing its OpenGL context for the sequence. A local batch helper is kept under the ignored `output/_render-batch/` directory. Hardware acceleration uses Intel Arc. The existing diagnostic CLI can reproduce an individual frame with the same settings:

```powershell
.\bin\validate-render-windows-amd64.exe -backend gpu `
  -scene output/video-1080p500/scenes/0000.json `
  -width 1920 -height 1080 -spp 500 -depth 24 -seed 42 `
  -denoise -denoise-space 0.05 -denoise-color 0 `
  -out output/video-1080p500/frames/0000.png
```

With all 168 frames present and FFmpeg available, regenerate the published files:

```bash
make video-preview
# Or select an installed FFmpeg explicitly:
make video-preview FFMPEG=/path/to/ffmpeg
```

The application renders every frame through its existing rendering service; FFmpeg encodes the PNG sequence into MP4 and GIF.

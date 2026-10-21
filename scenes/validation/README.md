# Контрольные сцены оптики и шума

Все размеры заданы в единицах сцены. Апертура выключена, чтобы глубина резкости
не мешала сравнению. Файлы открываются обычным выбором сцены в UI.

| Сцена | Что анализировать |
|---|---|
| `glass_grid.json` | Искажение сетки сферой и боковой сдвиг за параллельной пластиной, IOR=1.5 |
| `nested_glass.json` | Стеклянная сфера с воздушной полостью, четыре последовательные границы |
| `thin_glass.json` | Пластина толщиной 0.00002: выходная поверхность не должна исчезать |
| `glass_absorption.json` | Одинаковый IOR и толщины 0.2 / 0.6 / 1.2: изолированная проверка Beer–Lambert |
| `lighting_noise.json` | Маленькая сферическая лампа, диффузные стены, GGX, зеркало, стекло и каустика |
| `lighting_gradient.json` | Радиальное освещение плоскости без fog: проверка концентрических полос при 2560×1440 |
| `white_furnace.json` | Единичное окружение: Lambert, зеркало, стекло; отдельный GGX со стандартными потерями однократного рассеяния |

В `glass_grid` луч после выхода из параллельной пластины должен быть параллелен
падающему; меняется положение, а не итоговый угол. Криволинейная сфера меняет
направление. У `thin_glass` боковой сдвиг почти незаметен, но Fresnel остаётся.
В `glass_absorption` IOR=1 намеренно исключает преломление и поверхностное отражение.

Из корня проекта:

```sh
go run ./cmd/validate-render -scene scenes/validation/glass_grid.json \
  -backend cpu -width 320 -height 180 -spp 64 -seed 1 \
  -out bin/validation/glass-grid-cpu.png

go run ./cmd/validate-render -scene scenes/validation/glass_grid.json \
  -backend gpu -width 320 -height 180 -spp 64 -seed 1 \
  -out bin/validation/glass-grid-gpu.png
```

Для GPU нужен GL 4.3+. Команда считает CPU fallback ошибкой, поэтому сравнение
не сможет незаметно измерить CPU вместо GPU. Для намеренной проверки GLSL через
программный Mesa перед командой задайте `PATHTRACER_GPU_ALLOW_SOFTWARE=1`.

Сравнение зернистости с эталоном:

```sh
go run ./cmd/validate-render -scene scenes/validation/lighting_noise.json \
  -width 160 -height 90 -spp 8192 -seed 771 -out bin/validation/reference.png

go run ./cmd/validate-render -scene scenes/validation/lighting_noise.json \
  -width 160 -height 90 -spp 32 -seed 1 -denoise=false \
  -out bin/validation/raw.png -reference bin/validation/reference.png

go run ./cmd/validate-render -scene scenes/validation/lighting_noise.json \
  -width 160 -height 90 -spp 32 -seed 1 -denoise \
  -out bin/validation/denoised.png -reference bin/validation/reference.png
```

Для GPU добавьте `-backend gpu`. Ненулевой seed воспроизводит изображение в рамках
одного бэкенда и версии ядра; CPU/GPU используют разные генераторы, поэтому
сравнивать их попиксельно как одинаковую случайную последовательность нельзя.
`-reference` выводит MSE и PSNR в RGB изображения после ACES/гаммы. Это метрика
изображения, а не проверка физической точности линейного радианса.
`-cpuprofile /tmp/render.pprof` включает профиль CPU; изучить его можно командой
`go tool pprof -top /tmp/render.pprof`.

Проверка полос на гладком градиенте без denoise:

```sh
go run ./cmd/validate-render -scene scenes/validation/lighting_gradient.json \
  -backend cpu -width 2560 -height 1440 -spp 32 -depth 1 -seed 1 \
  -denoise=false -out bin/validation/gradient-2k.png
```

Для GPU замените `-backend cpu` на `-backend gpu`. CPU/GPU используют общий
дизеринг при выводе восьмибитного цвета: он подавляет полосы, сохраняя средний цвет
и линейный HDR. Порог стабилен между кадрами и не использует RNG трассировки.

Денойзер по умолчанию выключен в этой диагностической команде. В UI флажок
`CPU/GPU denoise` управляет обоими бэкендами. Фильтр работает после HDR-преобразования,
сохраняет сильные границы и может сглаживать мелкие детали; для численной проверки
оптики его следует выключать. Он не устраняет полностью сложную каустику при малом SPP.

Оба режима всех этих сцен включены в обычные CPU-регрессии и опциональный GPU-тест:

```sh
go test ./internal/usecases/render
PATHTRACER_GPU_INTEGRATION=1 go test ./internal/transport/gpu \
  -run '^TestGPU(PhysicsAudit|Integration)$' -count=1 -v -timeout=180s
```

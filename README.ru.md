# Go Path Tracer

**Интерактивный 3D-пастрейсер на Go: рендер на CPU и GPU через OpenGL, редактор сцен и экспорт изображений.**

Программа позволяет исследовать преломление света в стекле, отражения в металлах и зеркалах, непрямое освещение и объёмный туман. Меняйте материалы и положение объектов, двигайте камеру, сравнивайте CPU/GPU и сохраняйте результат в PNG или анимацию камеры.

[English](README.md) · [Быстрый запуск](#быстрый-запуск) · [Документация](#документация)

![Бесцветное, янтарное и голубое стекло при студийном освещении](docs/images/glass-studio.png)

*Стеклянная студия — настоящий GPU-рендер: **2560×1440, 1000 сэмплов на пиксель**. [Полный размер](docs/images/glass-studio.png) · [Сцена](scenes/gallery/glass_studio.json) · [Параметры рендера](docs/images/README.md).*

## Семисекундное GPU-превью

Ниже встроен MP4-плеер для локального предпросмотра Markdown с поддержкой HTML-видео. Для GitHub отдельно показано анимированное GIF-превью.

<video src="docs/media/studio-preview.mp4" controls loop muted playsinline preload="metadata" width="960" poster="docs/images/glass-studio.png">
  <a href="docs/media/studio-preview.mp4">Смотреть семисекундный GPU-рендер</a>
</video>

**Анимированное GIF-превью**

[![Плавное движение камеры вокруг бесцветного, янтарного и голубого стекла](docs/media/studio-preview.gif)](docs/media/studio-preview.mp4)

[Смотреть или скачать MP4](docs/media/studio-preview.mp4) · [Параметры и исходная сцена](docs/media/README.md). Анимация воспроизводится прямо в README. MP4: **1920×1080, 24 кадра/с, 500 SPP, 7 секунд**, рендер на GPU.

## Интерфейс

![Работающее приложение: редактор сцены, GPU-рендер и настройки изображения](docs/images/interface.png)

[Открыть снимок интерфейса в полном размере](docs/images/interface.png). В редакторе можно менять объекты и материалы, двигать камеру, выбирать CPU/GPU и сохранять готовое изображение.

## Возможности

| Возможность | Что доступно |
| --- | --- |
| CPU и GPU | Пул работников Go или вычислительные шейдеры GLSL с OpenGL 4.3+. |
| Геометрия | Сферы, боксы вдоль координатных осей и горизонтальные плоскости в JSON-сценах. |
| Материалы | Диффузные поверхности, шероховатые металлы, идеальные зеркала, стекло с поглощением и светящиеся объекты. |
| Освещение | Многократные переотражения, источники-сферы и светящиеся боксы, MIS, объёмный туман. |
| Редактор | Компоновка outliner / viewport / inspector, выбор объектов мышью, перемещение объектов стрелками и камеры WASDQE. |
| Предпросмотр | Промежуточные кадры, отмена рендера, цветной геометрический режим **Wireframe**. |
| Экспорт | PNG, общий денойзер CPU/GPU и отдельные настройки preview/final. |
| Анимация | Интерполяция движения камеры и экспорт MJPEG/AVI. |
| Проверки | Сцены JSON, фиксированный seed, CLI с явными параметрами и контрольные сцены оптики. |

Проект предназначен для изучения рендеринга и визуальных экспериментов. При малом числе сэмплов изображение шумит; сложная каустика стекла может потребовать длительного расчёта. GPU-бэкенд использует **OpenGL compute**. Название сцены `rtx_showcase` не означает обязательное наличие RTX или использование аппаратного API трассировки лучей.

## Галерея

Все восемь фотографий, включая студию в начале страницы: **GPU · 2560×1440 · 1000 SPP · глубина 24**. Денойз: **Space 0.05, Color 0**, дополнительное размытие выключено. [Параметры и команды повторения](docs/images/README.md).

### Тёплый объёмный туман

![Тёплый объёмный туман](docs/images/fog-shafts.png)

[Полный размер](docs/images/fog-shafts.png) · [Исходная сцена](scenes/gallery/fog_shafts.json)

### Та же комната без тумана

![Та же комната без тумана](docs/images/fog-clear.png)

[Полный размер](docs/images/fog-clear.png) · [Исходная сцена](scenes/gallery/fog_clear.json)

### Темнота и отражения

![Темнота и отражения](docs/images/midnight.png)

[Полный размер](docs/images/midnight.png) · [Исходная сцена](scenes/gallery/midnight.json)

### Prism Atrium с туманом

![Prism Atrium с туманом](docs/images/prism-atrium.png)

[Полный размер](docs/images/prism-atrium.png) · [Исходная сцена](scenes/engine_showcase.json)

### Prism Atrium без тумана

![Prism Atrium без тумана](docs/images/prism-clear.png)

[Полный размер](docs/images/prism-clear.png) · [Исходная сцена](scenes/gallery/prism_clear.json)

### Металлы, зеркала и цветное освещение

![Металлы, зеркала и цветное освещение](docs/images/metal-and-glass.png)

[Полный размер](docs/images/metal-and-glass.png) · [Исходная сцена](scenes/rtx_showcase.json)

### Преломление света

![Преломление света](docs/images/glass-refraction.png)

[Полный размер](docs/images/glass-refraction.png) · [Исходная сцена](scenes/validation/glass_grid.json)

## Быстрый запуск

### Linux / Ubuntu

Нужны **Go 1.27.2+**, C-компилятор и библиотеки для сборки графического приложения:

```bash
sudo apt-get install -y gcc pkg-config libgl1-mesa-dev xorg-dev \
  libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev libxxf86vm-dev \
  libwayland-dev libxkbcommon-dev wayland-protocols

git clone https://github.com/MarkJulian19/path_trace_golang.git
cd path_trace_golang
make
make run SCENE=example_simple
```

Исполняемый файл — `bin/pathtracer`. Последняя команда открывает редактор с CPU-бэкендом. Чтобы открыть сцену из начала README:

```bash
make run SCENE=gallery/glass_studio GPU=1
```

### Windows

Установите Go и C-компилятор **MinGW-w64**, доступный в `PATH`. Из корня репозитория в PowerShell:

```powershell
$env:CGO_ENABLED = "1"
$env:CC = "gcc"
go build -o bin/pathtracer.exe ./cmd/render
.\bin\pathtracer.exe -scene scenes/example_simple.json
```

При запуске рядом должны быть доступны сцены из `scenes/`. [Другие варианты сборки](docs/development.md#building).

### Включение GPU

Выберите GPU в редакторе или запустите:

```bash
./bin/pathtracer -scene scenes/engine_showcase.json -gpu
```

Нужен драйвер с **OpenGL 4.3+**. Если GPU-бэкенд не запустится, приложение переключится на CPU и сообщит об этом. Программный OpenGL (`llvmpipe`, например в некоторых конфигурациях WSL) выполняет вычисления на CPU и не задействует аппаратное ускорение. [Диагностика GPU](docs/development.md#gpu-troubleshooting).

### Первый рендер

1. Откройте `example_simple` на CPU и выберите объект в списке сцены.
2. Измените материал или настройки камеры и нажмите **Preview render**. В верхней панели выберите **Move**, затем перетаскивайте выбранный объект за оси X/Y/Z или двигайте его стрелками; **Page Up/Page Down** меняют высоту. Для перемещения камеры включите **WASDQE camera control**.
3. Задайте итоговое разрешение и число сэмплов, затем нажмите **Final render**.
4. После завершения укажите путь и нажмите **Save image (PNG)**. Сохраняется законченный финальный рендер. **Save scene** записывает изменения обратно в открытый JSON-файл.

## Рендер без редактора

Для явного выбора разрешения, числа сэмплов и seed:

```bash
go run ./cmd/validate-render \
  -scene scenes/gallery/glass_studio.json -backend gpu \
  -width 2560 -height 1440 -spp 1000 -depth 24 -seed 42 \
  -denoise -denoise-space 0.05 -denoise-color 0 -out output/renders/render.png
```

`-backend gpu` запускает GLSL; переход на CPU в этой диагностической команде считается ошибкой. Чем больше сэмплов на пиксель (`-spp`), тем обычно меньше шум и дольше расчёт.

Быстрый запуск с настройками пресета приложения:

```bash
make headless SCENE=example_simple OUT=output/renders/example.png
```

[Полный список флагов и особенности пресетов](docs/development.md#command-line-tools).

## С чего начать

| Сцена | Для чего подходит |
| --- | --- |
| [example_simple](scenes/example_simple.json) | Первое знакомство с материалами. |
| [engine_showcase](scenes/engine_showcase.json) | Prism Atrium: цветной свет, стекло, металлы и туман. |
| [rtx_showcase](scenes/rtx_showcase.json) | Серебряные и медные поверхности под цветными источниками. |
| [metal_glass_room](scenes/metal_glass_room.json) | Простая комната со стеклом и металлом. |
| [gpu_showcase](scenes/gpu_showcase.json) | Разные типы объектов, материалов и источников. |
| [test_comprehensive](scenes/test_comprehensive.json) | Большая сцена для проверки геометрии и материалов. |
| [Контрольные сцены](scenes/validation/README.md) | Преломление, поглощение, шум, градиенты и white furnace. |

`make scenes` выводит список основных сцен. JSON хранит камеру, объекты, материалы, источники, окружение и настройки расчёта; его можно редактировать вручную.

## Документация

- [Разработка, CLI и диагностика](docs/development.md)
- [Источники изображений и команды для повторения](docs/images/README.md)
- [Контрольные сцены оптики и шума](scenes/validation/README.md)
- [Математика рендеринга — PDF](docs/math/gpu_pathtracer_math.pdf) и [LaTeX](docs/math/gpu_pathtracer_math.tex): справочный материал; актуальное поведение определяется кодом и тестами.
- [Участие в разработке](CONTRIBUTING.md)

В [`internal/`](internal/) находятся структуры сцен, CPU-ядро, GPU-адаптер, JSON-хранилище и интерфейс Fyne. [Карта репозитория](docs/development.md#repository-map).

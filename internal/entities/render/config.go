package render

import "image"

// Config задаёт параметры построения одного кадра.
type Config struct {
	Width        int
	Height       int
	SamplesPerPx int
	MaxDepth     int
	MaxRayDist   float32
}

// Progress сообщает о завершении очередного этапа построения кадра.
type Progress func(currentSample, totalSamples int)

// FrameProgress сообщает о завершении построения кадра видеопоследовательности.
type FrameProgress func(currentFrame, totalFrames int, frame image.Image)

// VideoProgress сообщает о записи кадра в видеофайл.
type VideoProgress func(currentFrame, totalFrames int)

package render

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

var (
	errNoFrames       = errors.New("no frames to encode")
	errUint32Overflow = errors.New("value does not fit uint32")
	errInt16Overflow  = errors.New("value does not fit int16")
	errInt32Overflow  = errors.New("value does not fit int32")
)

// InterpolateCamera performs linear interpolation between two camera positions.
// T should be in range [0, 1] where 0 = startCam, 1 = endCam.
func InterpolateCamera(startCam, endCam scene.Camera, t float64) scene.Camera {
	// Clamp t to [0, 1]
	if t < 0 {
		t = 0
	}
	if t > 1 {
		t = 1
	}

	// Linear interpolation of Position and Target
	posX := startCam.Position.X + (endCam.Position.X-startCam.Position.X)*t
	posY := startCam.Position.Y + (endCam.Position.Y-startCam.Position.Y)*t
	posZ := startCam.Position.Z + (endCam.Position.Z-startCam.Position.Z)*t

	targetX := startCam.Target.X + (endCam.Target.X-startCam.Target.X)*t
	targetY := startCam.Target.Y + (endCam.Target.Y-startCam.Target.Y)*t
	targetZ := startCam.Target.Z + (endCam.Target.Z-startCam.Target.Z)*t

	// Create interpolated camera with same other parameters
	result := startCam
	result.Position = scene.Vec3{X: posX, Y: posY, Z: posZ}
	result.Target = scene.Vec3{X: targetX, Y: targetY, Z: targetZ}
	// FOV, Up, Aperture, FocusDist, AspectRatio remain unchanged

	return result
}

// VideoSequence строит последовательность кадров CPU-рендером.
func VideoSequence(
	sc *scene.Scene,
	startCam, endCam scene.Camera,
	duration float64,
	fps int,
	cfg Config,
	progress func(currentFrame, totalFrames int, frameImg image.Image),
) ([]image.Image, error) {
	return videoSequence(nil, sc, startCam, endCam, duration, fps, cfg, progress)
}

// VideoSequence строит последовательность кадров выбранным средством рендера.
func (s *Service) VideoSequence(
	sc *scene.Scene,
	startCam, endCam scene.Camera,
	duration float64,
	fps int,
	cfg Config,
	progress func(currentFrame, totalFrames int, frameImg image.Image),
) ([]image.Image, error) {
	return videoSequence(s, sc, startCam, endCam, duration, fps, cfg, progress)
}

func videoSequence(
	renderer *Service,
	sc *scene.Scene,
	startCam, endCam scene.Camera,
	duration float64,
	fps int,
	cfg Config,
	progress func(currentFrame, totalFrames int, frameImg image.Image),
) ([]image.Image, error) {
	// Calculate total number of frames
	totalFrames := max(int(duration*float64(fps)), 1)

	// Save original camera
	originalCam := sc.Camera

	// Prepare frames array
	frames := make([]image.Image, 0, totalFrames)

	// Render each frame
	for frame := range totalFrames {
		frameImg, err := renderVideoFrame(renderer, sc, startCam, endCam, cfg, frame, totalFrames)
		if err != nil {
			// Restore original camera on error
			sc.Camera = originalCam
			return nil, fmt.Errorf("render frame %d: %w", frame, err)
		}

		// Add frame to array
		frames = append(frames, frameImg)

		// Call progress callback
		if progress != nil {
			progress(frame+1, totalFrames, frameImg)
		}
	}

	// Restore original camera
	sc.Camera = originalCam

	return frames, nil
}

func renderVideoFrame(
	renderer *Service,
	sc *scene.Scene,
	startCam, endCam scene.Camera,
	cfg Config,
	frame, totalFrames int,
) (image.Image, error) {
	t := 0.0
	if totalFrames > 1 {
		t = float64(frame) / float64(totalFrames-1)
	}
	sc.Camera = InterpolateCamera(startCam, endCam, t)
	settings := scene.RenderSettings{
		Width:        cfg.Width,
		Height:       cfg.Height,
		SamplesPerPx: cfg.SamplesPerPx,
		MaxDepth:     cfg.MaxDepth,
		MaxRayDist:   float64(cfg.MaxRayDist),
	}
	return renderScene(renderer, sc, settings)
}

func renderScene(renderer *Service, sc *scene.Scene, settings scene.RenderSettings) (image.Image, error) {
	if renderer != nil {
		return renderer.Scene(sc, settings)
	}
	return Scene(sc, settings)
}

// VideoEncodingConfig holds configuration for video encoding.
type VideoEncodingConfig struct {
	// Video bitrate in kbps (default: 2000)
	// JPEG quality 1-100, higher is better (default: 90)
	PixelFormat string
	Bitrate     int
	Quality     int
	// Pixel format: "yuv420p", "yuv444p" (default: "yuv420p")
}

// DefaultVideoEncodingConfig returns default encoding configuration.
func DefaultVideoEncodingConfig() VideoEncodingConfig {
	return VideoEncodingConfig{
		Bitrate:     videoDefaultBitrate,
		Quality:     videoDefaultQuality, // Reduced quality for lower bitrate and better playback performance
		PixelFormat: "yuv420p",
	}
}

// CreateVideoFromFrames creates a video file from a sequence of frames.
// Creates AVI format with Motion JPEG for better Windows Media Player compatibility.
// Progress callback receives (currentFrame, totalFrames) for UI updates.
func CreateVideoFromFrames(
	frames []image.Image,
	outputPath string,
	fps float64,
	progress func(currentFrame, totalFrames int),
) error {
	return CreateVideoFromFramesWithConfig(frames, outputPath, fps, DefaultVideoEncodingConfig(), progress)
}

// CreateVideoFromFramesWithConfig creates an AVI video with custom encoding settings.
// This implementation creates an AVI file with Motion JPEG codec.
// AVI format has better compatibility with Windows Media Player.
func CreateVideoFromFramesWithConfig(
	frames []image.Image,
	outputPath string,
	fps float64,
	config VideoEncodingConfig,
	progress func(currentFrame, totalFrames int),
) (err error) {
	if len(frames) == 0 {
		return errNoFrames
	}

	// Get frame dimensions from first frame
	firstFrame := frames[0]
	bounds := firstFrame.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	aviPath := aviPathFor(outputPath)

	// Create output file
	file, err := os.Create(filepath.Clean(aviPath))
	if err != nil {
		return fmt.Errorf("create output file: %w", err)
	}
	defer func() {
		closeErr := file.Close()
		if closeErr != nil && err == nil {
			err = fmt.Errorf("close output file: %w", closeErr)
		}
	}()

	jpegFrames, err := encodeVideoFrames(frames, config.Quality, progress)
	if err != nil {
		return err
	}

	// Create AVI file with Motion JPEG
	err = createAVIWithMotionJPEG(file, jpegFrames, width, height, fps)
	if err != nil {
		return fmt.Errorf("create AVI: %w", err)
	}

	return nil
}

func aviPathFor(outputPath string) string {
	if len(outputPath) >= videoExtensionLength && outputPath[len(outputPath)-videoExtensionLength:] == ".avi" {
		return outputPath
	}
	if len(outputPath) > videoExtensionLength {
		ext := outputPath[len(outputPath)-videoExtensionLength:]
		if ext == ".mov" || ext == ".mp4" {
			return outputPath[:len(outputPath)-videoExtensionLength] + ".avi"
		}
	}
	return outputPath + ".avi"
}

func encodeVideoFrames(
	frames []image.Image,
	quality int,
	progress func(currentFrame, totalFrames int),
) ([][]byte, error) {
	jpegFrames := make([][]byte, 0, len(frames))
	for i, frame := range frames {
		jpegData, err := encodeFrameToJPEG(frame, quality)
		if err != nil {
			return nil, fmt.Errorf("encode frame %d: %w", i, err)
		}
		jpegFrames = append(jpegFrames, jpegData)
		if progress != nil {
			progress(i+1, len(frames))
		}
	}
	return jpegFrames, nil
}

// encodeFrameToJPEG encodes an image frame to JPEG format.
func encodeFrameToJPEG(img image.Image, quality int) ([]byte, error) {
	var buf bytes.Buffer

	// Ensure quality is in valid range
	if quality < 1 {
		quality = 1
	}
	if quality > 100 {
		quality = 100
	}

	// Encode as JPEG
	err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: quality})
	if err != nil {
		return nil, fmt.Errorf("jpeg encode: %w", err)
	}

	return buf.Bytes(), nil
}

type aviWriter struct {
	err  error
	file *os.File
}

func (w *aviWriter) string(value string) {
	if w.err == nil {
		w.err = writeString(w.file, value)
	}
}

func (w *aviWriter) uint32(value uint32) {
	if w.err == nil {
		w.err = writeUint32LE(w.file, value)
	}
}

func (w *aviWriter) uint16(value uint16) {
	if w.err == nil {
		w.err = writeUint16LE(w.file, value)
	}
}

func (w *aviWriter) int16(value int16) {
	if w.err == nil {
		w.err = writeInt16LE(w.file, value)
	}
}

func (w *aviWriter) int32(value int32) {
	if w.err == nil {
		w.err = writeInt32LE(w.file, value)
	}
}

func (w *aviWriter) seek() int64 {
	if w.err != nil {
		return 0
	}
	var position int64
	position, w.err = w.file.Seek(0, io.SeekCurrent)
	return position
}

func (w *aviWriter) done() error {
	return w.err
}

type aviHeader struct {
	riffSizePos, hdrlSizePos int64
	suggestedBufferSize      uint32
}

func writeAVIHeader(
	file *os.File,
	totalFrames, totalDataSize uint32,
	width, height int,
	fps float64,
	microSecPerFrame uint32,
) (aviHeader, error) {
	writer := &aviWriter{file: file}
	header := aviHeader{}
	writer.string("RIFF")
	header.riffSizePos = writer.seek()
	writer.uint32(0)
	writer.string("AVI ")
	writer.string("LIST")
	header.hdrlSizePos = writer.seek()
	writer.uint32(0)
	writer.string("hdrl")
	err := writer.done()
	if err != nil {
		return aviHeader{}, err
	}

	width32, height32, err := checkedAVIFrameDimensions(width, height)
	if err != nil {
		return aviHeader{}, err
	}
	header.suggestedBufferSize = max(totalDataSize/totalFrames, videoMinimumBufferSize)
	writeAVIHChunk(
		writer,
		totalFrames,
		totalDataSize,
		fps,
		microSecPerFrame,
		header.suggestedBufferSize,
		width32,
		height32,
	)
	return header, writer.done()
}

func checkedAVIFrameDimensions(width, height int) (uint32, uint32, error) {
	width32, err := uint32Checked(int64(width))
	if err != nil {
		return 0, 0, err
	}
	height32, err := uint32Checked(int64(height))
	if err != nil {
		return 0, 0, err
	}
	return width32, height32, nil
}

func writeAVIHChunk(
	writer *aviWriter,
	totalFrames, totalDataSize uint32,
	fps float64,
	microSecPerFrame, suggestedBufferSize, width, height uint32,
) {
	maxBytesPerSec := uint32(float64(totalDataSize) * fps / float64(totalFrames))
	if maxBytesPerSec == 0 {
		maxBytesPerSec = videoMicrosecondsPerSecond
	}
	writer.string("avih")
	writer.uint32(videoChunkHeaderSize)
	writer.uint32(microSecPerFrame)
	writer.uint32(maxBytesPerSec)
	writer.uint32(0)
	writer.uint32(uint32(videoKeyframeFlag))
	writer.uint32(totalFrames)
	writer.uint32(0)
	writer.uint32(1)
	writer.uint32(suggestedBufferSize)
	writer.uint32(width)
	writer.uint32(height)
	for range 4 {
		writer.uint32(0)
	}
}

func writeAVIStreamHeader(
	file *os.File,
	header aviHeader,
	totalFrames, totalDataSize uint32,
	width, height int,
	fps float64,
) error {
	writer := &aviWriter{file: file}
	writer.string("LIST")
	strlSizePos := writer.seek()
	writer.uint32(0)
	writer.string("strl")
	err := writeAVIStreamChunk(
		writer,
		totalFrames,
		header.suggestedBufferSize,
		fps,
		width,
		height,
		totalDataSize,
	)
	if err != nil {
		return err
	}
	strlEnd := writer.seek()
	err = writer.done()
	if err != nil {
		return err
	}
	err = updateAVIListSize(file, strlSizePos)
	if err != nil {
		return err
	}
	_, err = file.Seek(strlEnd, io.SeekStart)
	if err != nil {
		return err
	}
	return finalizeAVIHeader(file, header.hdrlSizePos, strlEnd)
}

func finalizeAVIHeader(file *os.File, hdrlSizePos, streamEnd int64) error {
	hdrlEnd, err := file.Seek(streamEnd, io.SeekStart)
	if err != nil {
		return err
	}
	err = updateAVIListSizeAt(file, hdrlSizePos, hdrlEnd)
	if err != nil {
		return err
	}
	_, err = file.Seek(hdrlEnd, io.SeekStart)
	return err
}

func writeAVIStreamChunk(
	writer *aviWriter,
	totalFrames, suggestedBufferSize uint32,
	fps float64,
	width, height int,
	totalDataSize uint32,
) error {
	width16, err := int16Checked(int64(width))
	if err != nil {
		return err
	}
	height16, err := int16Checked(int64(height))
	if err != nil {
		return err
	}
	width32, err := int32Checked(int64(width))
	if err != nil {
		return err
	}
	height32, err := int32Checked(int64(height))
	if err != nil {
		return err
	}
	writeAVIStrh(writer, totalFrames, suggestedBufferSize, fps, width16, height16)
	writeAVIStrf(writer, totalDataSize, totalFrames, width32, height32)
	return writer.done()
}

func writeAVIStrh(writer *aviWriter, totalFrames, suggestedBufferSize uint32, fps float64, width, height int16) {
	writer.string("strh")
	writer.uint32(videoChunkHeaderSize)
	writer.string("vids")
	writer.string("MJPG")
	writer.uint32(0)
	writer.uint16(0)
	writer.uint16(0)
	writer.uint32(0)
	writer.uint32(1)
	writer.uint32(max(uint32(fps), 1))
	writer.uint32(0)
	writer.uint32(totalFrames)
	writer.uint32(suggestedBufferSize)
	writer.uint32(0)
	writer.uint32(0)
	writer.int16(0)
	writer.int16(0)
	writer.int16(width)
	writer.int16(height)
}

func writeAVIStrf(writer *aviWriter, totalDataSize, totalFrames uint32, width, height int32) {
	writer.string("strf")
	bitmapInfoHeaderSize := uint32(videoBitmapInfoHeaderSize)
	writer.uint32(bitmapInfoHeaderSize)
	writer.uint32(bitmapInfoHeaderSize)
	writer.int32(width)
	writer.int32(height)
	writer.uint16(1)
	writer.uint16(videoStreamHeaderSize)
	writer.string("MJPG")
	avgFrameSize := totalDataSize / totalFrames
	if avgFrameSize == 0 {
		avgFrameSize = videoMinimumBufferSize
	}
	writer.uint32(avgFrameSize)
	writer.int32(0)
	writer.int32(0)
	writer.uint32(0)
	writer.uint32(0)
}

func updateAVIListSizeAt(file *os.File, sizePos, listEnd int64) error {
	size, err := uint32Checked(listEnd - sizePos - int64(videoExtensionLength))
	if err != nil {
		return err
	}
	_, err = file.Seek(sizePos, io.SeekStart)
	if err != nil {
		return err
	}
	err = writeUint32LE(file, size)
	if err != nil {
		return err
	}
	return nil
}

// createAVIWithMotionJPEG creates an AVI file with Motion JPEG codec
// AVI with Motion JPEG has excellent compatibility with Windows Media Player.
func createAVIWithMotionJPEG(file *os.File, jpegFrames [][]byte, width, height int, fps float64) error {
	// AVI uses little-endian byte order
	// Calculate frame rate and duration
	// For AVI: Scale = 1, Rate = fps (not fps * 100)
	// microSecPerFrame is in microseconds (1,000,000 microseconds = 1 second)
	microSecPerFrame := max(
		// Microseconds per frame (1,000,000 = 1 second)
		uint32(videoMicrosecondsPerSecond/fps), 1)
	totalFrames, err := uint32Checked(int64(len(jpegFrames)))
	if err != nil {
		return err
	}

	// Calculate total data size
	totalDataSize := uint32(0)
	for _, frame := range jpegFrames {
		frameSize, err := uint32Checked(int64(len(frame)))
		if err != nil {
			return err
		}
		totalDataSize += frameSize
		// Add chunk header size (8 bytes: 4 for fourcc, 4 for size)
		totalDataSize += 8
	}

	header, err := writeAVIHeader(file, totalFrames, totalDataSize, width, height, fps, microSecPerFrame)
	if err != nil {
		return err
	}

	err = writeAVIStreamHeader(file, header, totalFrames, totalDataSize, width, height, fps)
	if err != nil {
		return err
	}

	return writeAVIMovie(file, jpegFrames, header.riffSizePos)
}

func writeAVIMovie(file *os.File, jpegFrames [][]byte, riffSizePos int64) error {
	moviStart, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	err = writeString(file, "LIST")
	if err != nil {
		return err
	}
	moviSizePos, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	err = writeUint32LE(file, 0)
	if err != nil {
		return err
	}
	err = writeString(file, "movi")
	if err != nil {
		return err
	}
	moviDataStart := moviStart + int64(videoListHeaderSize)
	indexEntries, err := writeAVIFrames(file, jpegFrames, moviDataStart)
	if err != nil {
		return err
	}
	err = updateAVIListSize(file, moviSizePos)
	if err != nil {
		return err
	}
	err = writeAVIIndex(file, indexEntries)
	if err != nil {
		return err
	}
	return updateAVIRiffSize(file, riffSizePos)
}

func writeAVIFrames(file *os.File, frames [][]byte, moviDataStart int64) ([]indexEntry, error) {
	entries := make([]indexEntry, 0, len(frames))
	for i, frame := range frames {
		entry, err := writeAVIFrame(file, frame, i, moviDataStart)
		if err != nil {
			return nil, err
		}
		entries = append(entries, entry)
	}
	return entries, nil
}

func writeAVIFrame(file *os.File, frame []byte, index int, moviDataStart int64) (indexEntry, error) {
	chunkPos, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return indexEntry{}, err
	}
	err = writeString(file, "00dc")
	if err != nil {
		return indexEntry{}, err
	}
	frameSize, err := uint32Checked(int64(len(frame)))
	if err != nil {
		return indexEntry{}, err
	}
	err = writeUint32LE(file, frameSize)
	if err != nil {
		return indexEntry{}, err
	}
	_, err = file.Write(frame)
	if err != nil {
		return indexEntry{}, fmt.Errorf("write frame %d: %w", index, err)
	}
	chunkOffset, err := uint32Checked(chunkPos - moviDataStart)
	if err != nil {
		return indexEntry{}, err
	}
	if frameSize%2 != 0 {
		err = writeUint8LE(file, 0)
		if err != nil {
			return indexEntry{}, err
		}
	}
	return indexEntry{
		ckid:   [4]byte{'0', '0', 'd', 'c'},
		flags:  uint32(videoKeyframeFlag),
		chunk:  chunkOffset,
		length: frameSize,
	}, nil
}

func updateAVIListSize(file *os.File, sizePos int64) error {
	listEnd, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	size, err := uint32Checked(listEnd - sizePos - int64(videoExtensionLength))
	if err != nil {
		return err
	}
	_, err = file.Seek(sizePos, io.SeekStart)
	if err != nil {
		return err
	}
	err = writeUint32LE(file, size)
	if err != nil {
		return err
	}
	_, err = file.Seek(listEnd, io.SeekStart)
	return err
}

func writeAVIIndex(file *os.File, entries []indexEntry) error {
	err := writeString(file, "idx1")
	if err != nil {
		return err
	}
	indexSize, err := uint32Checked(int64(len(entries) * 16))
	if err != nil {
		return err
	}
	err = writeUint32LE(file, indexSize)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		_, err = file.Write(entry.ckid[:])
		if err != nil {
			return err
		}
		err = writeUint32LE(file, entry.flags)
		if err != nil {
			return err
		}
		err = writeUint32LE(file, entry.chunk)
		if err != nil {
			return err
		}
		err = writeUint32LE(file, entry.length)
		if err != nil {
			return err
		}
	}
	return nil
}

func updateAVIRiffSize(file *os.File, riffSizePos int64) error {
	fileEnd, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	riffSize, err := uint32Checked(fileEnd - riffSizePos - int64(videoExtensionLength))
	if err != nil {
		return err
	}
	_, err = file.Seek(riffSizePos, io.SeekStart)
	if err != nil {
		return err
	}
	return writeUint32LE(file, riffSize)
}

// indexEntry represents an AVI index entry.
type indexEntry struct {
	ckid   [4]byte // Chunk ID
	flags  uint32  // Flags
	chunk  uint32  // Chunk offset (relative to movi list)
	length uint32  // Chunk length
}

// Helper functions for writing AVI (little-endian)

func writeUint16LE(w io.Writer, v uint16) error {
	return binary.Write(w, binary.LittleEndian, v)
}

func writeUint32LE(w io.Writer, v uint32) error {
	return binary.Write(w, binary.LittleEndian, v)
}

func writeInt16LE(w io.Writer, v int16) error {
	return binary.Write(w, binary.LittleEndian, v)
}

func writeInt32LE(w io.Writer, v int32) error {
	return binary.Write(w, binary.LittleEndian, v)
}

func writeUint8LE(w io.Writer, v uint8) error {
	return binary.Write(w, binary.LittleEndian, v)
}

func writeString(w io.Writer, s string) error {
	_, err := w.Write([]byte(s))
	return err
}

func uint32Checked(value int64) (uint32, error) {
	if value < 0 || value > math.MaxUint32 {
		return 0, fmt.Errorf("%w: %d", errUint32Overflow, value)
	}
	return uint32(value), nil
}

func int16Checked(value int64) (int16, error) {
	if value < math.MinInt16 || value > math.MaxInt16 {
		return 0, fmt.Errorf("%w: %d", errInt16Overflow, value)
	}
	return int16(value), nil
}

func int32Checked(value int64) (int32, error) {
	if value < math.MinInt32 || value > math.MaxInt32 {
		return 0, fmt.Errorf("%w: %d", errInt32Overflow, value)
	}
	return int32(value), nil
}

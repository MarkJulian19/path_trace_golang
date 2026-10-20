package render

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"os"
	"path/filepath"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

var errNoFrames = errors.New("no frames to encode")

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
	for frame := 0; frame < totalFrames; frame++ {
		// Calculate interpolation parameter t in [0, 1]
		var t float64
		if totalFrames > 1 {
			t = float64(frame) / float64(totalFrames-1)
		} else {
			t = 0
		}

		// Interpolate camera
		cam := InterpolateCamera(startCam, endCam, t)
		sc.Camera = cam

		// Render frame
		settings := scene.RenderSettings{
			Width:        cfg.Width,
			Height:       cfg.Height,
			SamplesPerPx: cfg.SamplesPerPx,
			MaxDepth:     cfg.MaxDepth,
			MaxRayDist:   float64(cfg.MaxRayDist),
		}

		frameImg, err := renderScene(renderer, sc, settings)
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

func renderScene(renderer *Service, sc *scene.Scene, settings scene.RenderSettings) (image.Image, error) {
	if renderer != nil {
		return renderer.Scene(sc, settings)
	}
	return Scene(sc, settings)
}

// VideoEncodingConfig holds configuration for video encoding.
type VideoEncodingConfig struct {
	Bitrate     int    // Video bitrate in kbps (default: 2000)
	Quality     int    // JPEG quality 1-100, higher is better (default: 90)
	PixelFormat string // Pixel format: "yuv420p", "yuv444p" (default: "yuv420p")
}

// DefaultVideoEncodingConfig returns default encoding configuration.
func DefaultVideoEncodingConfig() VideoEncodingConfig {
	return VideoEncodingConfig{
		Bitrate:     2000,
		Quality:     70, // Reduced quality for lower bitrate and better playback performance
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

	// Ensure .avi extension
	aviPath := outputPath
	if len(outputPath) < 4 || outputPath[len(outputPath)-4:] != ".avi" {
		// Remove .mov or .mp4 if present
		if len(outputPath) > 4 {
			ext := outputPath[len(outputPath)-4:]
			if ext == ".mov" || ext == ".mp4" {
				aviPath = outputPath[:len(outputPath)-4] + ".avi"
			} else {
				aviPath = outputPath + ".avi"
			}
		} else {
			aviPath = outputPath + ".avi"
		}
	}

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

	// Encode frames to JPEG
	jpegFrames := make([][]byte, 0, len(frames))
	for i, frame := range frames {
		jpegData, err := encodeFrameToJPEG(frame, config.Quality)
		if err != nil {
			return fmt.Errorf("encode frame %d: %w", i, err)
		}
		jpegFrames = append(jpegFrames, jpegData)

		if progress != nil {
			progress(i+1, len(frames))
		}
	}

	// Create AVI file with Motion JPEG
	err = createAVIWithMotionJPEG(file, jpegFrames, width, height, fps)
	if err != nil {
		return fmt.Errorf("create AVI: %w", err)
	}

	return nil
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

// createAVIWithMotionJPEG creates an AVI file with Motion JPEG codec
// AVI with Motion JPEG has excellent compatibility with Windows Media Player.
func createAVIWithMotionJPEG(file *os.File, jpegFrames [][]byte, width, height int, fps float64) error {
	// AVI uses little-endian byte order
	// Calculate frame rate and duration
	// For AVI: Scale = 1, Rate = fps (not fps * 100)
	// microSecPerFrame is in microseconds (1,000,000 microseconds = 1 second)
	microSecPerFrame := max(
		// Microseconds per frame (1,000,000 = 1 second)
		uint32(1000000/fps), 1)
	totalFrames := uint32(len(jpegFrames))

	// Calculate total data size
	totalDataSize := uint32(0)
	for _, frame := range jpegFrames {
		totalDataSize += uint32(len(frame))
		// Add chunk header size (8 bytes: 4 for fourcc, 4 for size)
		totalDataSize += 8
	}

	// Write RIFF header
	err := writeString(file, "RIFF")
	if err != nil {
		return err
	}
	// RIFF size will be written later, placeholder
	riffSizePos, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	err = writeUint32LE(file, 0)
	if err != nil { // Placeholder
		return err
	}
	err = writeString(file, "AVI ")
	if err != nil {
		return err
	}

	// Write hdrl list (header list)
	err = writeString(file, "LIST")
	if err != nil {
		return err
	}
	hdrlSizePos, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	err = writeUint32LE(file, 0)
	if err != nil { // Placeholder
		return err
	}
	err = writeString(file, "hdrl")
	if err != nil {
		return err
	}

	// Write avih (AVI header)
	err = writeString(file, "avih")
	if err != nil {
		return err
	}
	err = writeUint32LE(file, 56)
	if err != nil { // Size of avih chunk (56 bytes)
		return err
	}
	// Microseconds per frame
	err = writeUint32LE(file, microSecPerFrame)
	if err != nil {
		return err
	}
	// Max bytes per second (approximate)
	// Calculate based on fps
	maxBytesPerSec := uint32(float64(totalDataSize) * fps / float64(totalFrames))
	if maxBytesPerSec == 0 {
		maxBytesPerSec = 1000000 // Default 1MB/s
	}
	err = writeUint32LE(file, maxBytesPerSec)
	if err != nil {
		return err
	}
	// Padding granularity
	err = writeUint32LE(file, 0)
	if err != nil {
		return err
	}
	// Flags (AVIF_HASINDEX = 0x10, AVIF_MUSTUSEINDEX = 0x20)
	flags := uint32(0x10)
	err = writeUint32LE(file, flags)
	if err != nil {
		return err
	}
	// Total frames
	err = writeUint32LE(file, totalFrames)
	if err != nil {
		return err
	}
	// Initial frames (0 for non-interleaved)
	err = writeUint32LE(file, 0)
	if err != nil {
		return err
	}
	// Number of streams (1 video stream)
	err = writeUint32LE(file, 1)
	if err != nil {
		return err
	}
	// Suggested buffer size
	suggestedBufferSize := max(totalDataSize/totalFrames, 1024)
	err = writeUint32LE(file, suggestedBufferSize)
	if err != nil {
		return err
	}
	// Width
	err = writeUint32LE(file, uint32(width))
	if err != nil {
		return err
	}
	// Height
	err = writeUint32LE(file, uint32(height))
	if err != nil {
		return err
	}
	// Reserved (4 DWORDs)
	for range 4 {
		err = writeUint32LE(file, 0)
		if err != nil {
			return err
		}
	}

	// Write strl list (stream list)
	err = writeString(file, "LIST")
	if err != nil {
		return err
	}
	strlSizePos, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	err = writeUint32LE(file, 0)
	if err != nil { // Placeholder
		return err
	}
	err = writeString(file, "strl")
	if err != nil {
		return err
	}

	// Write strh (stream header)
	err = writeString(file, "strh")
	if err != nil {
		return err
	}
	err = writeUint32LE(file, 56)
	if err != nil { // Size of strh chunk (56 bytes)
		return err
	}
	// fccType: "vids" for video stream
	err = writeString(file, "vids")
	if err != nil {
		return err
	}
	// fccHandler: "MJPG" for Motion JPEG
	err = writeString(file, "MJPG")
	if err != nil {
		return err
	}
	// Flags
	err = writeUint32LE(file, 0)
	if err != nil {
		return err
	}
	// Priority and language
	err = writeUint16LE(file, 0)
	if err != nil {
		return err
	}
	err = writeUint16LE(file, 0)
	if err != nil {
		return err
	}
	// Initial frames
	err = writeUint32LE(file, 0)
	if err != nil {
		return err
	}
	// Scale (1 for video - means Rate is in frames per second)
	err = writeUint32LE(file, 1)
	if err != nil {
		return err
	}
	// Rate (frame rate in frames per second, since Scale = 1)
	rate := max(uint32(fps), 1)
	err = writeUint32LE(file, rate)
	if err != nil {
		return err
	}
	// Start time
	err = writeUint32LE(file, 0)
	if err != nil {
		return err
	}
	// Length (total frames)
	err = writeUint32LE(file, totalFrames)
	if err != nil {
		return err
	}
	// Suggested buffer size
	err = writeUint32LE(file, suggestedBufferSize)
	if err != nil {
		return err
	}
	// Quality
	err = writeUint32LE(file, 0)
	if err != nil {
		return err
	}
	// Sample size (0 for variable)
	err = writeUint32LE(file, 0)
	if err != nil {
		return err
	}
	// Frame rectangle (left, top, right, bottom)
	err = writeInt16LE(file, 0)
	if err != nil { // left
		return err
	}
	err = writeInt16LE(file, 0)
	if err != nil { // top
		return err
	}
	err = writeInt16LE(file, int16(width))
	if err != nil { // right
		return err
	}
	err = writeInt16LE(file, int16(height))
	if err != nil { // bottom
		return err
	}

	// Write strf (stream format) - BITMAPINFOHEADER for Motion JPEG
	err = writeString(file, "strf")
	if err != nil {
		return err
	}
	bitmapInfoHeaderSize := uint32(40)
	err = writeUint32LE(file, bitmapInfoHeaderSize)
	if err != nil {
		return err
	}
	// biSize (size of BITMAPINFOHEADER)
	err = writeUint32LE(file, bitmapInfoHeaderSize)
	if err != nil {
		return err
	}
	// biWidth
	err = writeInt32LE(file, int32(width))
	if err != nil {
		return err
	}
	// biHeight (positive for bottom-up, which is standard for AVI)
	err = writeInt32LE(file, int32(height))
	if err != nil {
		return err
	}
	// biPlanes
	err = writeUint16LE(file, 1)
	if err != nil {
		return err
	}
	// biBitCount (24 bits per pixel)
	err = writeUint16LE(file, 24)
	if err != nil {
		return err
	}
	// biCompression: "MJPG" fourcc
	err = writeString(file, "MJPG")
	if err != nil {
		return err
	}
	// biSizeImage (average frame size for better compatibility)
	avgFrameSize := totalDataSize / totalFrames
	if avgFrameSize == 0 {
		avgFrameSize = 1024
	}
	err = writeUint32LE(file, avgFrameSize)
	if err != nil {
		return err
	}
	// biXPelsPerMeter (0 = unspecified)
	err = writeInt32LE(file, 0)
	if err != nil {
		return err
	}
	// biYPelsPerMeter (0 = unspecified)
	err = writeInt32LE(file, 0)
	if err != nil {
		return err
	}
	// biClrUsed (0 for JPEG)
	err = writeUint32LE(file, 0)
	if err != nil {
		return err
	}
	// biClrImportant (0)
	err = writeUint32LE(file, 0)
	if err != nil {
		return err
	}

	// Update strl size
	// LIST size = size of data after size field, including "strl" fourcc (4 bytes) and all content
	strlEnd, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	strlSize := uint32(strlEnd - strlSizePos - 4) // Size after size field (includes "strl" + content)
	_, err = file.Seek(strlSizePos, io.SeekStart)
	if err != nil {
		return err
	}
	err = writeUint32LE(file, strlSize)
	if err != nil {
		return err
	}
	_, err = file.Seek(strlEnd, io.SeekStart)
	if err != nil {
		return err
	}

	// Update hdrl size
	// LIST size = size of data after size field, including "hdrl" fourcc (4 bytes) and all content
	hdrlEnd, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	hdrlSize := uint32(hdrlEnd - hdrlSizePos - 4) // Size after size field (includes "hdrl" + content)
	_, err = file.Seek(hdrlSizePos, io.SeekStart)
	if err != nil {
		return err
	}
	err = writeUint32LE(file, hdrlSize)
	if err != nil {
		return err
	}
	_, err = file.Seek(hdrlEnd, io.SeekStart)
	if err != nil {
		return err
	}

	// Write movi list (movie data)
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
	if err != nil { // Placeholder
		return err
	}
	err = writeString(file, "movi")
	if err != nil {
		return err
	}

	// Write video frames as 00dc chunks (compressed video data)
	// movi data starts after "LIST" (4) + size (4) + "movi" (4) = 12 bytes from moviStart
	moviDataStart := moviStart + 12
	indexEntries := make([]indexEntry, 0, len(jpegFrames))
	for i, frame := range jpegFrames {
		// Get position before writing chunk (this is where chunk starts relative to movi data)
		chunkPos, err := file.Seek(0, io.SeekCurrent)
		if err != nil {
			return err
		}

		// Write chunk fourcc: "00dc" (stream 0, compressed video)
		err = writeString(file, "00dc")
		if err != nil {
			return err
		}
		// Write chunk size (data size only, no padding)
		frameSize := uint32(len(frame))
		err = writeUint32LE(file, frameSize)
		if err != nil {
			return err
		}
		// Write frame data
		_, err = file.Write(frame)
		if err != nil {
			return fmt.Errorf("write frame %d: %w", i, err)
		}

		// Calculate offset relative to movi data start (chunk starts at chunkPos)
		chunkOffset := uint32(chunkPos - moviDataStart)

		// Pad to 2-byte boundary if needed (padding is NOT included in chunk size or index)
		if frameSize%2 != 0 {
			err = writeUint8LE(file, 0)
			if err != nil {
				return err
			}
		}

		// Record index entry - offset is position of chunk start relative to movi data
		indexEntries = append(indexEntries, indexEntry{
			ckid:   [4]byte{'0', '0', 'd', 'c'},
			flags:  0x10, // AVIIF_KEYFRAME
			chunk:  chunkOffset,
			length: frameSize, // Size without padding
		})
	}

	// Update movi size
	// LIST size = size of data after size field, including "movi" fourcc (4 bytes) and all content
	moviEnd, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	moviSize := uint32(moviEnd - moviSizePos - 4) // Size after size field (includes "movi" + content)
	_, err = file.Seek(moviSizePos, io.SeekStart)
	if err != nil {
		return err
	}
	err = writeUint32LE(file, moviSize)
	if err != nil {
		return err
	}
	_, err = file.Seek(moviEnd, io.SeekStart)
	if err != nil {
		return err
	}

	// Write idx1 (index)
	err = writeString(file, "idx1")
	if err != nil {
		return err
	}
	idx1Size := uint32(len(indexEntries) * 16) // 16 bytes per entry
	err = writeUint32LE(file, idx1Size)
	if err != nil {
		return err
	}
	for _, entry := range indexEntries {
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

	// Update RIFF size
	fileEnd, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	riffSize := uint32(fileEnd - riffSizePos - 4)
	_, err = file.Seek(riffSizePos, io.SeekStart)
	if err != nil {
		return err
	}
	err = writeUint32LE(file, riffSize)
	if err != nil {
		return err
	}

	return nil
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

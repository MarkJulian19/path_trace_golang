package engine

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"os"
	"time"

	"github.com/user/pathtracer/internal/scene"
)

// InterpolateCamera performs linear interpolation between two camera positions.
// t should be in range [0, 1] where 0 = startCam, 1 = endCam.
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

// RenderVideoSequence renders a sequence of frames by interpolating the camera
// between startCam and endCam over the specified duration and FPS.
// progress callback receives (currentFrame, totalFrames, frameImg) for UI updates.
func RenderVideoSequence(
	sc *scene.Scene,
	startCam, endCam scene.Camera,
	duration float64,
	fps int,
	cfg RenderConfig,
	progress func(currentFrame, totalFrames int, frameImg image.Image),
) ([]image.Image, error) {
	// Calculate total number of frames
	totalFrames := int(duration * float64(fps))
	if totalFrames < 1 {
		totalFrames = 1
	}

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

		frameImg, err := RenderScene(sc, settings)
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

// VideoEncodingConfig holds configuration for video encoding
type VideoEncodingConfig struct {
	Bitrate     int    // Video bitrate in kbps (default: 2000)
	Quality     int    // JPEG quality 1-100, higher is better (default: 90)
	PixelFormat string // Pixel format: "yuv420p", "yuv444p" (default: "yuv420p")
}

// DefaultVideoEncodingConfig returns default encoding configuration
func DefaultVideoEncodingConfig() VideoEncodingConfig {
	return VideoEncodingConfig{
		Bitrate:     2000,
		Quality:     70, // Reduced quality for lower bitrate and better playback performance
		PixelFormat: "yuv420p",
	}
}

// CreateVideoFromFrames creates a video file from a sequence of frames.
// Creates AVI format with Motion JPEG for better Windows Media Player compatibility.
// progress callback receives (currentFrame, totalFrames) for UI updates.
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
) error {
	if len(frames) == 0 {
		return fmt.Errorf("no frames to encode")
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
	file, err := os.Create(aviPath)
	if err != nil {
		return fmt.Errorf("create output file: %w", err)
	}
	defer file.Close()

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
	if err := createAVIWithMotionJPEG(file, jpegFrames, width, height, fps); err != nil {
		return fmt.Errorf("create AVI: %w", err)
	}

	return nil
}

// encodeFrameToJPEG encodes an image frame to JPEG format
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
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: quality}); err != nil {
		return nil, fmt.Errorf("jpeg encode: %w", err)
	}

	return buf.Bytes(), nil
}

// createMOVWithMotionJPEG creates a QuickTime MOV file with Motion JPEG codec
// MOV format has better compression and playback performance than AVI
func createMOVWithMotionJPEG(file *os.File, jpegFrames [][]byte, width, height int, fps float64) error {
	// Validate input parameters
	if len(jpegFrames) == 0 {
		return fmt.Errorf("no frames to encode")
	}
	if width <= 0 || height <= 0 {
		return fmt.Errorf("invalid dimensions: %dx%d", width, height)
	}
	if fps <= 0 {
		return fmt.Errorf("invalid fps: %f", fps)
	}

	// QuickTime MOV uses big-endian byte order (like MP4)
	// Calculate timescale and frame duration
	timescale := uint32(600) // Use 600 as timescale for better precision
	frameDuration := uint32(float64(timescale) / fps)
	if frameDuration < 1 {
		frameDuration = 1
	}

	// Write ftyp box (file type)
	// Use qt (QuickTime) as major brand for better Windows Media Player compatibility with Motion JPEG
	if err := writeMP4Box(file, "ftyp", func(w io.Writer) error {
		// Major brand: qt (QuickTime) - better for Motion JPEG in Windows Media Player
		if err := writeString(w, "qt  "); err != nil {
			return err
		}
		// Minor version (0 for QuickTime)
		if err := writeUint32BE(w, 0); err != nil {
			return err
		}
		// Compatible brands
		if err := writeString(w, "qt  "); err != nil { // QuickTime
			return err
		}
		if err := writeString(w, "isom"); err != nil { // ISO Base Media compatibility
			return err
		}
		if err := writeString(w, "iso2"); err != nil { // ISO Base Media v2
			return err
		}
		return nil
	}); err != nil {
		return fmt.Errorf("write ftyp: %w", err)
	}

	// Calculate total data size for metadata
	totalDataSize := uint64(0)
	for _, frame := range jpegFrames {
		totalDataSize += uint64(len(frame))
	}

	// For fast start (better compatibility with Windows Media Player), write moov before mdat
	// Strategy: Write moov to buffer first, then write to file, then write mdat, then update offsets
	moovStart, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}

	// Calculate approximate mdat start position for placeholder offsets
	// We'll estimate moov size and use that to calculate offsets
	// After writing moov, we'll know exact mdat start and can update offsets
	var moovBuf bytes.Buffer
	// Use placeholder offsets - we'll update them after writing mdat
	placeholderOffsets := make([]uint64, len(jpegFrames))
	// Estimate: moov is typically 500-2000 bytes, mdat header is 8 bytes
	estimatedMoovSize := uint64(1500)
	estimatedMdatStart := moovStart + int64(estimatedMoovSize)
	mdatDataStart := estimatedMdatStart + 8 // mdat header (4 size + 4 "mdat")

	// Calculate placeholder offsets using estimated positions
	currentOffset := mdatDataStart
	avgFrameSize := totalDataSize / uint64(len(jpegFrames))
	if avgFrameSize == 0 {
		avgFrameSize = 100000 // Default estimate
	}
	for i := range jpegFrames {
		placeholderOffsets[i] = uint64(currentOffset)
		currentOffset += int64(avgFrameSize)
	}

	// Write moov to buffer with placeholder offsets
	if err := writeMOVMoovBox(&moovBuf, jpegFrames, placeholderOffsets, width, height, fps, timescale, frameDuration, totalDataSize); err != nil {
		return fmt.Errorf("write moov to buffer: %w", err)
	}

	// Write moov from buffer to file
	moovBytes := moovBuf.Bytes()
	if _, err := file.Write(moovBytes); err != nil {
		return fmt.Errorf("write moov: %w", err)
	}
	moovEnd := moovStart + int64(len(moovBytes))

	// Now write mdat box
	mdatStart, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}

	// Write mdat header
	mdatSizePos, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if err := writeUint32BE(file, 0); err != nil { // Placeholder for size
		return err
	}
	if err := writeString(file, "mdat"); err != nil {
		return err
	}

	// Write all JPEG frames to mdat and track actual offsets
	// Chunk offsets must be absolute (relative to file start), pointing to the start of each frame's data
	chunkOffsets := make([]uint64, len(jpegFrames))
	for i, frame := range jpegFrames {
		// Get position before writing frame (this is where frame data starts)
		// This position is absolute (relative to file start), which is what we need for stco
		currentPos, err := file.Seek(0, io.SeekCurrent)
		if err != nil {
			return fmt.Errorf("get position for frame %d: %w", i, err)
		}

		// Validate position is reasonable
		if currentPos < mdatStart {
			return fmt.Errorf("invalid chunk offset %d for frame %d: before mdat start %d", currentPos, i, mdatStart)
		}

		chunkOffsets[i] = uint64(currentPos)

		// Write frame data
		if _, err := file.Write(frame); err != nil {
			return fmt.Errorf("write frame %d: %w", i, err)
		}
	}

	// Validate all chunk offsets are absolute, in correct order, and point to valid locations
	for i := 1; i < len(chunkOffsets); i++ {
		if chunkOffsets[i] <= chunkOffsets[i-1] {
			return fmt.Errorf("chunk offsets not in ascending order: frame %d offset %d <= frame %d offset %d", i, chunkOffsets[i], i-1, chunkOffsets[i-1])
		}
		// Verify each offset is after mdat header (mdatStart + 8)
		if chunkOffsets[i] < uint64(mdatStart)+8 {
			return fmt.Errorf("chunk offset %d for frame %d is before mdat data start", chunkOffsets[i], i)
		}
	}

	// Update mdat size
	mdatEnd, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	mdatSize := uint32(mdatEnd - mdatStart)
	if _, err := file.Seek(mdatSizePos, io.SeekStart); err != nil {
		return err
	}
	if err := writeUint32BE(file, mdatSize); err != nil {
		return err
	}
	if _, err := file.Seek(mdatEnd, io.SeekStart); err != nil {
		return err
	}

	// Validate chunk offsets before updating
	fileEnd, err := file.Seek(0, io.SeekEnd)
	if err != nil {
		return fmt.Errorf("get file end: %w", err)
	}
	// Reset to end position after validation
	if _, err := file.Seek(fileEnd, io.SeekStart); err != nil {
		return fmt.Errorf("seek to file end: %w", err)
	}
	for i, offset := range chunkOffsets {
		if offset >= uint64(fileEnd) {
			return fmt.Errorf("chunk offset %d for frame %d exceeds file size %d", offset, i, fileEnd)
		}
		if offset == 0 {
			return fmt.Errorf("invalid chunk offset 0 for frame %d", i)
		}
	}

	// Update chunk offsets in moov box (now we know exact offsets)
	// moovEnd was already calculated above, reuse it
	if err := updateChunkOffsetsInMoov(file, moovStart, moovEnd, chunkOffsets); err != nil {
		return fmt.Errorf("update chunk offsets: %w", err)
	}

	// Validate file structure after updates
	// Verify file ends at expected position
	finalPos, err := file.Seek(0, io.SeekEnd)
	if err != nil {
		return fmt.Errorf("get final file position: %w", err)
	}
	if finalPos != mdatEnd {
		return fmt.Errorf("file structure mismatch: expected end at %d, got %d", mdatEnd, finalPos)
	}

	// Verify moov box is still valid (size hasn't changed)
	moovBoxSize := moovEnd - moovStart
	if moovBoxSize != int64(len(moovBytes)) {
		return fmt.Errorf("moov box size mismatch: expected %d, got %d", len(moovBytes), moovBoxSize)
	}

	return nil
}

// writeMOVMoovBox writes the moov box for QuickTime MOV format
func writeMOVMoovBox(w io.Writer, jpegFrames [][]byte, chunkOffsets []uint64, width, height int, fps float64, timescale, frameDuration uint32, totalDataSize uint64) error {
	return writeMP4Box(w, "moov", func(w io.Writer) error {
		// Write mvhd box (movie header)
		// mvhd version 0 must be exactly 100 bytes: 1 version + 3 flags + 4 creation + 4 modification + 4 timescale + 4 duration + 4 rate + 2 volume + 2 reserved + 8 reserved + 36 matrix + 24 pre-defined + 4 next track ID
		if err := writeMP4Box(w, "mvhd", func(w io.Writer) error {
			version := uint8(0)
			flags := uint32(0)
			if err := writeUint8(w, version); err != nil {
				return err
			}
			if err := writeUint24BE(w, flags); err != nil {
				return err
			}
			// Creation time, modification time (seconds since 1904-01-01)
			// Use current time for better metadata compatibility
			now := uint32(time.Now().Unix() + 2082844800) // Offset from 1904-01-01 to 1970-01-01
			if err := writeUint32BE(w, now); err != nil {
				return err
			}
			if err := writeUint32BE(w, now); err != nil {
				return err
			}
			// Timescale
			if err := writeUint32BE(w, timescale); err != nil {
				return err
			}
			// Duration
			duration := uint64(len(jpegFrames)) * uint64(frameDuration)
			if err := writeUint32BE(w, uint32(duration)); err != nil {
				return err
			}
			// Rate (1.0 = 0x00010000)
			if err := writeUint32BE(w, 0x00010000); err != nil {
				return err
			}
			// Volume (1.0 = 0x0100)
			if err := writeUint16BE(w, 0x0100); err != nil {
				return err
			}
			// Reserved
			if err := writeUint16BE(w, 0); err != nil {
				return err
			}
			if err := writeUint32BE(w, 0); err != nil {
				return err
			}
			if err := writeUint32BE(w, 0); err != nil {
				return err
			}
			// Matrix (identity matrix)
			matrix := [9]int32{0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000}
			for _, m := range matrix {
				if err := writeInt32BE(w, m); err != nil {
					return err
				}
			}
			// Pre-defined (6 DWORDs)
			for i := 0; i < 6; i++ {
				if err := writeUint32BE(w, 0); err != nil {
					return err
				}
			}
			// Next track ID
			if err := writeUint32BE(w, 2); err != nil {
				return err
			}
			return nil
		}); err != nil {
			return fmt.Errorf("write mvhd: %w", err)
		}

		// Write trak box (track)
		if err := writeMOVTrakBox(w, jpegFrames, chunkOffsets, width, height, fps, timescale, frameDuration, totalDataSize); err != nil {
			return fmt.Errorf("write trak: %w", err)
		}

		return nil
	})
}

// writeMOVTrakBox writes the trak box for QuickTime MOV
func writeMOVTrakBox(w io.Writer, jpegFrames [][]byte, chunkOffsets []uint64, width, height int, fps float64, timescale, frameDuration uint32, totalDataSize uint64) error {
	return writeMP4Box(w, "trak", func(w io.Writer) error {
		// Write tkhd box (track header)
		// tkhd version 0 must be exactly 84 bytes: 1 version + 3 flags + 4 creation + 4 modification + 4 track ID + 4 reserved + 4 duration + 8 reserved + 2 layer + 2 alternate group + 2 volume + 2 reserved + 36 matrix + 4 width + 4 height
		if err := writeMP4Box(w, "tkhd", func(w io.Writer) error {
			version := uint8(0)
			flags := uint32(0x000007) // Track enabled, in movie, in preview
			if err := writeUint8(w, version); err != nil {
				return err
			}
			if err := writeUint24BE(w, flags); err != nil {
				return err
			}
			// Creation time, modification time (seconds since 1904-01-01)
			now := uint32(time.Now().Unix() + 2082844800)
			if err := writeUint32BE(w, now); err != nil {
				return err
			}
			if err := writeUint32BE(w, now); err != nil {
				return err
			}
			// Track ID
			if err := writeUint32BE(w, 1); err != nil {
				return err
			}
			// Reserved
			if err := writeUint32BE(w, 0); err != nil {
				return err
			}
			// Duration
			duration := uint64(len(jpegFrames)) * uint64(frameDuration)
			if err := writeUint32BE(w, uint32(duration)); err != nil {
				return err
			}
			// Reserved
			if err := writeUint32BE(w, 0); err != nil {
				return err
			}
			if err := writeUint32BE(w, 0); err != nil {
				return err
			}
			// Layer, alternate group
			if err := writeUint16BE(w, 0); err != nil {
				return err
			}
			if err := writeUint16BE(w, 0); err != nil {
				return err
			}
			// Volume
			if err := writeUint16BE(w, 0); err != nil {
				return err
			}
			// Reserved
			if err := writeUint16BE(w, 0); err != nil {
				return err
			}
			// Matrix (identity matrix)
			matrix := [9]int32{0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000}
			for _, m := range matrix {
				if err := writeInt32BE(w, m); err != nil {
					return err
				}
			}
			// Width, height (fixed point 16.16)
			if err := writeUint32BE(w, uint32(width<<16)); err != nil {
				return err
			}
			if err := writeUint32BE(w, uint32(height<<16)); err != nil {
				return err
			}
			return nil
		}); err != nil {
			return fmt.Errorf("write tkhd: %w", err)
		}

		// Write mdia box (media)
		if err := writeMOVMdiaBox(w, jpegFrames, chunkOffsets, width, height, fps, timescale, frameDuration, totalDataSize); err != nil {
			return fmt.Errorf("write mdia: %w", err)
		}

		return nil
	})
}

// writeMOVMdiaBox writes the mdia box for QuickTime MOV
func writeMOVMdiaBox(w io.Writer, jpegFrames [][]byte, chunkOffsets []uint64, width, height int, fps float64, timescale, frameDuration uint32, totalDataSize uint64) error {
	return writeMP4Box(w, "mdia", func(w io.Writer) error {
		// Write mdhd box (media header)
		// mdhd version 0 must be exactly 32 bytes: 1 version + 3 flags + 4 creation + 4 modification + 4 timescale + 4 duration + 2 language + 2 quality
		if err := writeMP4Box(w, "mdhd", func(w io.Writer) error {
			version := uint8(0)
			flags := uint32(0)
			if err := writeUint8(w, version); err != nil {
				return err
			}
			if err := writeUint24BE(w, flags); err != nil {
				return err
			}
			// Creation time, modification time (seconds since 1904-01-01)
			now := uint32(time.Now().Unix() + 2082844800)
			if err := writeUint32BE(w, now); err != nil {
				return err
			}
			if err := writeUint32BE(w, now); err != nil {
				return err
			}
			// Timescale
			if err := writeUint32BE(w, timescale); err != nil {
				return err
			}
			// Duration
			duration := uint64(len(jpegFrames)) * uint64(frameDuration)
			if err := writeUint32BE(w, uint32(duration)); err != nil {
				return err
			}
			// Language (und = undefined, 0x55C4)
			if err := writeUint16BE(w, 0x55C4); err != nil {
				return err
			}
			// Quality
			if err := writeUint16BE(w, 0); err != nil {
				return err
			}
			return nil
		}); err != nil {
			return fmt.Errorf("write mdhd: %w", err)
		}

		// Write hdlr box (handler reference)
		if err := writeMP4Box(w, "hdlr", func(w io.Writer) error {
			version := uint8(0)
			flags := uint32(0)
			if err := writeUint8(w, version); err != nil {
				return err
			}
			if err := writeUint24BE(w, flags); err != nil {
				return err
			}
			// Handler type: vide
			if err := writeString(w, "vide"); err != nil {
				return err
			}
			// Reserved
			if err := writeUint32BE(w, 0); err != nil {
				return err
			}
			if err := writeUint32BE(w, 0); err != nil {
				return err
			}
			if err := writeUint32BE(w, 0); err != nil {
				return err
			}
			// Name (null-terminated)
			if err := writeString(w, "VideoHandler"); err != nil {
				return err
			}
			// Pad to 4-byte boundary
			if err := writeUint8(w, 0); err != nil {
				return err
			}
			return nil
		}); err != nil {
			return fmt.Errorf("write hdlr: %w", err)
		}

		// Write minf box (media information)
		if err := writeMOVMinfBox(w, jpegFrames, chunkOffsets, width, height, fps, timescale, frameDuration, totalDataSize); err != nil {
			return fmt.Errorf("write minf: %w", err)
		}

		return nil
	})
}

// writeMOVMinfBox writes the minf box for QuickTime MOV
func writeMOVMinfBox(w io.Writer, jpegFrames [][]byte, chunkOffsets []uint64, width, height int, fps float64, timescale, frameDuration uint32, totalDataSize uint64) error {
	return writeMP4Box(w, "minf", func(w io.Writer) error {
		// Write vmhd box (video media header)
		if err := writeMP4Box(w, "vmhd", func(w io.Writer) error {
			version := uint8(0)
			flags := uint32(1) // Flags
			if err := writeUint8(w, version); err != nil {
				return err
			}
			if err := writeUint24BE(w, flags); err != nil {
				return err
			}
			// Graphics mode, opcolor
			if err := writeUint16BE(w, 0); err != nil {
				return err
			}
			if err := writeUint16BE(w, 0); err != nil {
				return err
			}
			if err := writeUint16BE(w, 0); err != nil {
				return err
			}
			return nil
		}); err != nil {
			return fmt.Errorf("write vmhd: %w", err)
		}

		// Write dinf box (data information)
		if err := writeMP4Box(w, "dinf", func(w io.Writer) error {
			// Write dref box (data reference)
			if err := writeMP4Box(w, "dref", func(w io.Writer) error {
				version := uint8(0)
				flags := uint32(0)
				if err := writeUint8(w, version); err != nil {
					return err
				}
				if err := writeUint24BE(w, flags); err != nil {
					return err
				}
				// Entry count
				if err := writeUint32BE(w, 1); err != nil {
					return err
				}
				// Write url box
				if err := writeMP4Box(w, "url ", func(w io.Writer) error {
					version := uint8(0)
					flags := uint32(1) // Self-contained
					if err := writeUint8(w, version); err != nil {
						return err
					}
					if err := writeUint24BE(w, flags); err != nil {
						return err
					}
					return nil
				}); err != nil {
					return fmt.Errorf("write url: %w", err)
				}
				return nil
			}); err != nil {
				return fmt.Errorf("write dref: %w", err)
			}
			return nil
		}); err != nil {
			return fmt.Errorf("write dinf: %w", err)
		}

		// Write stbl box (sample table)
		if err := writeMOVStblBox(w, jpegFrames, chunkOffsets, width, height, fps, timescale, frameDuration, totalDataSize); err != nil {
			return fmt.Errorf("write stbl: %w", err)
		}

		return nil
	})
}

// writeMOVStblBox writes the stbl box for QuickTime MOV
func writeMOVStblBox(w io.Writer, jpegFrames [][]byte, chunkOffsets []uint64, width, height int, fps float64, timescale, frameDuration uint32, totalDataSize uint64) error {
	return writeMP4Box(w, "stbl", func(w io.Writer) error {
		// Write stsd box (sample description)
		// Motion JPEG sample entry structure (jpeg box):
		// - Box header: 4 bytes (size) + 4 bytes ("jpeg") = 8 bytes
		// - Reserved: 6 bytes
		// - Data reference index: 2 bytes
		// - Version: 2 bytes
		// - Revision level: 2 bytes
		// - Vendor: 4 bytes ("jpeg")
		// - Temporal quality: 4 bytes
		// - Spatial quality: 4 bytes
		// - Width: 2 bytes
		// - Height: 2 bytes
		// - Horizontal resolution: 4 bytes
		// - Vertical resolution: 4 bytes
		// - Reserved: 4 bytes
		// - Frame count: 2 bytes
		// - Compressor name: 32 bytes
		// - Depth: 2 bytes
		// - Color table ID: 2 bytes
		// Total content: 6+2+2+2+4+4+4+2+2+4+4+4+2+32+2+2 = 86 bytes
		// Total box size: 8 (header) + 86 (content) = 94 bytes
		jpegBoxSize := uint32(94) // 8 header + 86 content = 94 bytes total

		if err := writeMP4Box(w, "stsd", func(w io.Writer) error {
			version := uint8(0)
			flags := uint32(0)
			if err := writeUint8(w, version); err != nil {
				return err
			}
			if err := writeUint24BE(w, flags); err != nil {
				return err
			}
			// Entry count
			if err := writeUint32BE(w, 1); err != nil {
				return err
			}
			// Write Motion JPEG sample entry
			// Write jpeg box manually with explicit size for better compatibility
			if err := writeUint32BE(w, jpegBoxSize); err != nil {
				return err
			}
			if err := writeString(w, "jpeg"); err != nil {
				return err
			}
			// Reserved (6 bytes)
			for i := 0; i < 6; i++ {
				if err := writeUint8(w, 0); err != nil {
					return err
				}
			}
			// Data reference index
			if err := writeUint16BE(w, 1); err != nil {
				return err
			}
			// Video info
			if err := writeUint16BE(w, 0); err != nil { // Version
				return err
			}
			if err := writeUint16BE(w, 0); err != nil { // Revision level
				return err
			}
			if err := writeString(w, "jpeg"); err != nil { // Vendor
				return err
			}
			if err := writeUint32BE(w, 0); err != nil { // Temporal quality
				return err
			}
			if err := writeUint32BE(w, 0); err != nil { // Spatial quality
				return err
			}
			// Width, height
			if err := writeUint16BE(w, uint16(width)); err != nil {
				return err
			}
			if err := writeUint16BE(w, uint16(height)); err != nil {
				return err
			}
			// Horiz/Vert resolution (72 dpi = 0x00480000)
			if err := writeUint32BE(w, 0x00480000); err != nil {
				return err
			}
			if err := writeUint32BE(w, 0x00480000); err != nil {
				return err
			}
			// Reserved
			if err := writeUint32BE(w, 0); err != nil {
				return err
			}
			// Frame count
			if err := writeUint16BE(w, 1); err != nil {
				return err
			}
			// Compressor name (32 bytes, null-terminated)
			compressorName := make([]byte, 32)
			copy(compressorName, "Motion JPEG")
			if _, err := w.Write(compressorName); err != nil {
				return err
			}
			// Depth, color table ID
			if err := writeUint16BE(w, 24); err != nil { // 24-bit
				return err
			}
			if err := writeUint16BE(w, 0xFFFF); err != nil { // No color table
				return err
			}
			return nil
		}); err != nil {
			return fmt.Errorf("write stsd: %w", err)
		}

		// Write stts box (time-to-sample)
		if err := writeMP4Box(w, "stts", func(w io.Writer) error {
			version := uint8(0)
			flags := uint32(0)
			if err := writeUint8(w, version); err != nil {
				return err
			}
			if err := writeUint24BE(w, flags); err != nil {
				return err
			}
			// Entry count
			if err := writeUint32BE(w, 1); err != nil {
				return err
			}
			// Entry
			if err := writeUint32BE(w, uint32(len(jpegFrames))); err != nil { // Sample count
				return err
			}
			if err := writeUint32BE(w, frameDuration); err != nil { // Sample delta
				return err
			}
			return nil
		}); err != nil {
			return fmt.Errorf("write stts: %w", err)
		}

		// Write stsc box (sample-to-chunk)
		// For Motion JPEG, each chunk contains exactly 1 sample (1 frame)
		// Entry: first_chunk=1, samples_per_chunk=1 means all chunks have 1 sample each
		if err := writeMP4Box(w, "stsc", func(w io.Writer) error {
			version := uint8(0)
			flags := uint32(0)
			if err := writeUint8(w, version); err != nil {
				return err
			}
			if err := writeUint24BE(w, flags); err != nil {
				return err
			}
			// Entry count: 1 entry means all chunks from chunk 1 onwards have the same structure
			if err := writeUint32BE(w, 1); err != nil {
				return err
			}
			// Entry: first chunk = 1, samples per chunk = 1, sample description index = 1
			if err := writeUint32BE(w, 1); err != nil { // First chunk (1-based)
				return err
			}
			if err := writeUint32BE(w, 1); err != nil { // Samples per chunk (1 frame per chunk)
				return err
			}
			if err := writeUint32BE(w, 1); err != nil { // Sample description index (points to stsd entry 1)
				return err
			}
			return nil
		}); err != nil {
			return fmt.Errorf("write stsc: %w", err)
		}

		// Write stsz box (sample size)
		if err := writeMP4Box(w, "stsz", func(w io.Writer) error {
			version := uint8(0)
			flags := uint32(0)
			if err := writeUint8(w, version); err != nil {
				return err
			}
			if err := writeUint24BE(w, flags); err != nil {
				return err
			}
			// Sample size (0 = variable)
			if err := writeUint32BE(w, 0); err != nil {
				return err
			}
			// Sample count
			if err := writeUint32BE(w, uint32(len(jpegFrames))); err != nil {
				return err
			}
			// Sample sizes
			for _, frame := range jpegFrames {
				if err := writeUint32BE(w, uint32(len(frame))); err != nil {
					return err
				}
			}
			return nil
		}); err != nil {
			return fmt.Errorf("write stsz: %w", err)
		}

		// Write stco box (chunk offset) - 32-bit version
		if err := writeMP4Box(w, "stco", func(w io.Writer) error {
			version := uint8(0)
			flags := uint32(0)
			if err := writeUint8(w, version); err != nil {
				return err
			}
			if err := writeUint24BE(w, flags); err != nil {
				return err
			}
			// Entry count
			if err := writeUint32BE(w, uint32(len(chunkOffsets))); err != nil {
				return err
			}
			// Chunk offsets (32-bit)
			for _, offset := range chunkOffsets {
				if offset > 0xFFFFFFFF {
					// Need to use co64 instead
					return fmt.Errorf("chunk offset too large for stco, need co64")
				}
				if err := writeUint32BE(w, uint32(offset)); err != nil {
					return err
				}
			}
			return nil
		}); err != nil {
			return fmt.Errorf("write stco: %w", err)
		}

		return nil
	})
}

// findBoxInData recursively finds a box with the given type in the data
// Returns the position of the box content (after size and type) and the box size, or error if not found
func findBoxInData(data []byte, boxType string, startPos int) (contentPos int, boxSize uint32, err error) {
	pos := startPos
	for pos < len(data)-8 {
		if pos+8 > len(data) {
			break
		}
		boxSize := uint32(data[pos])<<24 | uint32(data[pos+1])<<16 | uint32(data[pos+2])<<8 | uint32(data[pos+3])
		currentBoxType := string(data[pos+4 : pos+8])

		if boxSize == 0 || boxSize == 1 {
			break // Invalid size or extended size (not supported)
		}
		if int(boxSize) > len(data)-pos {
			break // Box extends beyond data
		}

		contentPos := pos + 8 // Position after size and type

		if currentBoxType == boxType {
			return contentPos, boxSize, nil
		}

		// Recursively search in container boxes (boxes that can contain other boxes)
		containerBoxes := map[string]bool{
			"moov": true, "trak": true, "mdia": true, "minf": true, "stbl": true,
			"edts": true, "dinf": true, "udta": true,
		}
		if containerBoxes[currentBoxType] {
			// Search recursively inside this container box
			if foundPos, foundSize, err := findBoxInData(data, boxType, contentPos); err == nil {
				return foundPos, foundSize, nil
			}
		}

		pos += int(boxSize)
	}

	return 0, 0, fmt.Errorf("box %s not found", boxType)
}

// updateChunkOffsetsInMoov updates chunk offsets in stco box within moov
func updateChunkOffsetsInMoov(file *os.File, moovStart, moovEnd int64, chunkOffsets []uint64) error {
	// Read moov box
	moovSize := moovEnd - moovStart
	moovData := make([]byte, moovSize)
	if _, err := file.ReadAt(moovData, moovStart); err != nil {
		return fmt.Errorf("read moov: %w", err)
	}

	// Find stco box recursively in nested structure: moov -> trak -> mdia -> minf -> stbl -> stco
	// Start from position 8 (skip moov header: size + type)
	stcoContentPos, _, err := findBoxInData(moovData, "stco", 8)
	if err != nil {
		return fmt.Errorf("find stco box: %w", err)
	}

	// stco box structure: version (1) + flags (3) + entry count (4) + offsets (4 * count)
	// Content starts after size (4) + type (4) = 8 bytes
	// So offsetPos is stcoContentPos + 8 (version + flags + entry count)
	offsetPos := stcoContentPos + 8 // After version (1) + flags (3) + entry count (4)

	// Verify we have enough space
	if offsetPos+len(chunkOffsets)*4 > len(moovData) {
		return fmt.Errorf("stco box too small for offsets: need %d bytes, have %d", offsetPos+len(chunkOffsets)*4, len(moovData))
	}

	// Verify entry count matches
	entryCount := uint32(moovData[stcoContentPos+4])<<24 | uint32(moovData[stcoContentPos+5])<<16 | uint32(moovData[stcoContentPos+6])<<8 | uint32(moovData[stcoContentPos+7])
	if entryCount != uint32(len(chunkOffsets)) {
		return fmt.Errorf("stco entry count mismatch: expected %d, found %d", len(chunkOffsets), entryCount)
	}

	// Write new offsets
	for i, offset := range chunkOffsets {
		if offset > 0xFFFFFFFF {
			return fmt.Errorf("chunk offset too large for stco: %d", offset)
		}
		offsetBytes := make([]byte, 4)
		offsetBytes[0] = byte(offset >> 24)
		offsetBytes[1] = byte(offset >> 16)
		offsetBytes[2] = byte(offset >> 8)
		offsetBytes[3] = byte(offset)
		copy(moovData[offsetPos+i*4:], offsetBytes)
	}

	// Write updated moov back to file
	if _, err := file.WriteAt(moovData, moovStart); err != nil {
		return fmt.Errorf("write updated moov: %w", err)
	}
	return nil
}

// Helper functions for writing MP4/MOV boxes (big-endian)

// writeMP4Box writes an MP4/MOV box with the given type and content
func writeMP4Box(w io.Writer, boxType string, writeContent func(io.Writer) error) error {
	// Create buffer for content
	var contentBuf bytes.Buffer
	if err := writeContent(&contentBuf); err != nil {
		return err
	}

	// Write box size (4 bytes, big-endian)
	// Size includes: 4 bytes (size field) + 4 bytes (type) + content
	size := uint32(8 + contentBuf.Len())
	// Handle large boxes (size > 2^32-1, use size=1 and extended size)
	if size > 0xFFFFFFFF-8 {
		// For very large boxes, we'd need extended size, but this is unlikely for our use case
		return fmt.Errorf("box size too large: %d", size)
	}
	if err := writeUint32BE(w, size); err != nil {
		return err
	}

	// Write box type
	if err := writeString(w, boxType); err != nil {
		return err
	}

	// Write content
	if _, err := w.Write(contentBuf.Bytes()); err != nil {
		return err
	}

	return nil
}

func writeUint8(w io.Writer, v uint8) error {
	return binary.Write(w, binary.BigEndian, v)
}

func writeUint16BE(w io.Writer, v uint16) error {
	return binary.Write(w, binary.BigEndian, v)
}

func writeUint24BE(w io.Writer, v uint32) error {
	buf := make([]byte, 3)
	buf[0] = byte(v >> 16)
	buf[1] = byte(v >> 8)
	buf[2] = byte(v)
	_, err := w.Write(buf)
	return err
}

func writeUint32BE(w io.Writer, v uint32) error {
	return binary.Write(w, binary.BigEndian, v)
}

func writeInt32BE(w io.Writer, v int32) error {
	return binary.Write(w, binary.BigEndian, v)
}

// createAVIWithMotionJPEG creates an AVI file with Motion JPEG codec
// AVI with Motion JPEG has excellent compatibility with Windows Media Player
func createAVIWithMotionJPEG(file *os.File, jpegFrames [][]byte, width, height int, fps float64) error {
	// AVI uses little-endian byte order
	// Calculate frame rate and duration
	// For AVI: Scale = 1, Rate = fps (not fps * 100)
	// microSecPerFrame is in microseconds (1,000,000 microseconds = 1 second)
	microSecPerFrame := uint32(1000000 / fps) // Microseconds per frame (1,000,000 = 1 second)
	if microSecPerFrame < 1 {
		microSecPerFrame = 1
	}
	totalFrames := uint32(len(jpegFrames))

	// Calculate total data size
	totalDataSize := uint32(0)
	for _, frame := range jpegFrames {
		totalDataSize += uint32(len(frame))
		// Add chunk header size (8 bytes: 4 for fourcc, 4 for size)
		totalDataSize += 8
	}

	// Write RIFF header
	if err := writeString(file, "RIFF"); err != nil {
		return err
	}
	// RIFF size will be written later, placeholder
	riffSizePos, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if err := writeUint32LE(file, 0); err != nil { // Placeholder
		return err
	}
	if err := writeString(file, "AVI "); err != nil {
		return err
	}

	// Write hdrl list (header list)
	if err := writeString(file, "LIST"); err != nil {
		return err
	}
	hdrlSizePos, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if err := writeUint32LE(file, 0); err != nil { // Placeholder
		return err
	}
	if err := writeString(file, "hdrl"); err != nil {
		return err
	}

	// Write avih (AVI header)
	if err := writeString(file, "avih"); err != nil {
		return err
	}
	if err := writeUint32LE(file, 56); err != nil { // Size of avih chunk (56 bytes)
		return err
	}
	// Microseconds per frame
	if err := writeUint32LE(file, microSecPerFrame); err != nil {
		return err
	}
	// Max bytes per second (approximate)
	// Calculate based on fps
	maxBytesPerSec := uint32(float64(totalDataSize) * fps / float64(totalFrames))
	if maxBytesPerSec == 0 {
		maxBytesPerSec = 1000000 // Default 1MB/s
	}
	if err := writeUint32LE(file, maxBytesPerSec); err != nil {
		return err
	}
	// Padding granularity
	if err := writeUint32LE(file, 0); err != nil {
		return err
	}
	// Flags (AVIF_HASINDEX = 0x10, AVIF_MUSTUSEINDEX = 0x20)
	flags := uint32(0x10)
	if err := writeUint32LE(file, flags); err != nil {
		return err
	}
	// Total frames
	if err := writeUint32LE(file, totalFrames); err != nil {
		return err
	}
	// Initial frames (0 for non-interleaved)
	if err := writeUint32LE(file, 0); err != nil {
		return err
	}
	// Number of streams (1 video stream)
	if err := writeUint32LE(file, 1); err != nil {
		return err
	}
	// Suggested buffer size
	suggestedBufferSize := totalDataSize / totalFrames
	if suggestedBufferSize < 1024 {
		suggestedBufferSize = 1024
	}
	if err := writeUint32LE(file, suggestedBufferSize); err != nil {
		return err
	}
	// Width
	if err := writeUint32LE(file, uint32(width)); err != nil {
		return err
	}
	// Height
	if err := writeUint32LE(file, uint32(height)); err != nil {
		return err
	}
	// Reserved (4 DWORDs)
	for i := 0; i < 4; i++ {
		if err := writeUint32LE(file, 0); err != nil {
			return err
		}
	}

	// Write strl list (stream list)
	if err := writeString(file, "LIST"); err != nil {
		return err
	}
	strlSizePos, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if err := writeUint32LE(file, 0); err != nil { // Placeholder
		return err
	}
	if err := writeString(file, "strl"); err != nil {
		return err
	}

	// Write strh (stream header)
	if err := writeString(file, "strh"); err != nil {
		return err
	}
	if err := writeUint32LE(file, 56); err != nil { // Size of strh chunk (56 bytes)
		return err
	}
	// fccType: "vids" for video stream
	if err := writeString(file, "vids"); err != nil {
		return err
	}
	// fccHandler: "MJPG" for Motion JPEG
	if err := writeString(file, "MJPG"); err != nil {
		return err
	}
	// Flags
	if err := writeUint32LE(file, 0); err != nil {
		return err
	}
	// Priority and language
	if err := writeUint16LE(file, 0); err != nil {
		return err
	}
	if err := writeUint16LE(file, 0); err != nil {
		return err
	}
	// Initial frames
	if err := writeUint32LE(file, 0); err != nil {
		return err
	}
	// Scale (1 for video - means Rate is in frames per second)
	if err := writeUint32LE(file, 1); err != nil {
		return err
	}
	// Rate (frame rate in frames per second, since Scale = 1)
	rate := uint32(fps)
	if rate < 1 {
		rate = 1
	}
	if err := writeUint32LE(file, rate); err != nil {
		return err
	}
	// Start time
	if err := writeUint32LE(file, 0); err != nil {
		return err
	}
	// Length (total frames)
	if err := writeUint32LE(file, totalFrames); err != nil {
		return err
	}
	// Suggested buffer size
	if err := writeUint32LE(file, suggestedBufferSize); err != nil {
		return err
	}
	// Quality
	if err := writeUint32LE(file, 0); err != nil {
		return err
	}
	// Sample size (0 for variable)
	if err := writeUint32LE(file, 0); err != nil {
		return err
	}
	// Frame rectangle (left, top, right, bottom)
	if err := writeInt16LE(file, 0); err != nil { // left
		return err
	}
	if err := writeInt16LE(file, 0); err != nil { // top
		return err
	}
	if err := writeInt16LE(file, int16(width)); err != nil { // right
		return err
	}
	if err := writeInt16LE(file, int16(height)); err != nil { // bottom
		return err
	}

	// Write strf (stream format) - BITMAPINFOHEADER for Motion JPEG
	if err := writeString(file, "strf"); err != nil {
		return err
	}
	bitmapInfoHeaderSize := uint32(40)
	if err := writeUint32LE(file, bitmapInfoHeaderSize); err != nil {
		return err
	}
	// biSize (size of BITMAPINFOHEADER)
	if err := writeUint32LE(file, bitmapInfoHeaderSize); err != nil {
		return err
	}
	// biWidth
	if err := writeInt32LE(file, int32(width)); err != nil {
		return err
	}
	// biHeight (positive for bottom-up, which is standard for AVI)
	if err := writeInt32LE(file, int32(height)); err != nil {
		return err
	}
	// biPlanes
	if err := writeUint16LE(file, 1); err != nil {
		return err
	}
	// biBitCount (24 bits per pixel)
	if err := writeUint16LE(file, 24); err != nil {
		return err
	}
	// biCompression: "MJPG" fourcc
	if err := writeString(file, "MJPG"); err != nil {
		return err
	}
	// biSizeImage (average frame size for better compatibility)
	avgFrameSize := totalDataSize / totalFrames
	if avgFrameSize == 0 {
		avgFrameSize = 1024
	}
	if err := writeUint32LE(file, avgFrameSize); err != nil {
		return err
	}
	// biXPelsPerMeter (0 = unspecified)
	if err := writeInt32LE(file, 0); err != nil {
		return err
	}
	// biYPelsPerMeter (0 = unspecified)
	if err := writeInt32LE(file, 0); err != nil {
		return err
	}
	// biClrUsed (0 for JPEG)
	if err := writeUint32LE(file, 0); err != nil {
		return err
	}
	// biClrImportant (0)
	if err := writeUint32LE(file, 0); err != nil {
		return err
	}

	// Update strl size
	// LIST size = size of data after size field, including "strl" fourcc (4 bytes) and all content
	strlEnd, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	strlSize := uint32(strlEnd - strlSizePos - 4) // Size after size field (includes "strl" + content)
	if _, err := file.Seek(strlSizePos, io.SeekStart); err != nil {
		return err
	}
	if err := writeUint32LE(file, strlSize); err != nil {
		return err
	}
	if _, err := file.Seek(strlEnd, io.SeekStart); err != nil {
		return err
	}

	// Update hdrl size
	// LIST size = size of data after size field, including "hdrl" fourcc (4 bytes) and all content
	hdrlEnd, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	hdrlSize := uint32(hdrlEnd - hdrlSizePos - 4) // Size after size field (includes "hdrl" + content)
	if _, err := file.Seek(hdrlSizePos, io.SeekStart); err != nil {
		return err
	}
	if err := writeUint32LE(file, hdrlSize); err != nil {
		return err
	}
	if _, err := file.Seek(hdrlEnd, io.SeekStart); err != nil {
		return err
	}

	// Write movi list (movie data)
	moviStart, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if err := writeString(file, "LIST"); err != nil {
		return err
	}
	moviSizePos, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	if err := writeUint32LE(file, 0); err != nil { // Placeholder
		return err
	}
	if err := writeString(file, "movi"); err != nil {
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
		if err := writeString(file, "00dc"); err != nil {
			return err
		}
		// Write chunk size (data size only, no padding)
		frameSize := uint32(len(frame))
		if err := writeUint32LE(file, frameSize); err != nil {
			return err
		}
		// Write frame data
		if _, err := file.Write(frame); err != nil {
			return fmt.Errorf("write frame %d: %w", i, err)
		}

		// Calculate offset relative to movi data start (chunk starts at chunkPos)
		chunkOffset := uint32(chunkPos - moviDataStart)

		// Pad to 2-byte boundary if needed (padding is NOT included in chunk size or index)
		if frameSize%2 != 0 {
			if err := writeUint8LE(file, 0); err != nil {
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
	if _, err := file.Seek(moviSizePos, io.SeekStart); err != nil {
		return err
	}
	if err := writeUint32LE(file, moviSize); err != nil {
		return err
	}
	if _, err := file.Seek(moviEnd, io.SeekStart); err != nil {
		return err
	}

	// Write idx1 (index)
	if err := writeString(file, "idx1"); err != nil {
		return err
	}
	idx1Size := uint32(len(indexEntries) * 16) // 16 bytes per entry
	if err := writeUint32LE(file, idx1Size); err != nil {
		return err
	}
	for _, entry := range indexEntries {
		if _, err := file.Write(entry.ckid[:]); err != nil {
			return err
		}
		if err := writeUint32LE(file, entry.flags); err != nil {
			return err
		}
		if err := writeUint32LE(file, entry.chunk); err != nil {
			return err
		}
		if err := writeUint32LE(file, entry.length); err != nil {
			return err
		}
	}

	// Update RIFF size
	fileEnd, err := file.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	riffSize := uint32(fileEnd - riffSizePos - 4)
	if _, err := file.Seek(riffSizePos, io.SeekStart); err != nil {
		return err
	}
	if err := writeUint32LE(file, riffSize); err != nil {
		return err
	}

	return nil
}

// indexEntry represents an AVI index entry
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

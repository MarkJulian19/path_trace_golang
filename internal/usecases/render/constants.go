package render

const (
	renderDegreesPerHalfTurn     = 180.0
	renderViewportDiameter       = 2.0
	renderHalf                   = 0.5
	renderUnitValue              = 1.0
	renderRoughnessThreshold     = 1e-6
	renderRoughnessScale         = 0.1
	renderMetalRoughThreshold    = 1e-4
	renderThroughputCutoff       = 0.01
	renderDefaultIOR             = 1.5
	renderScatterLengthThreshold = 1e-8
	renderReflectanceExponent    = 5
	renderRandomShiftA           = 13
	renderRandomShiftB           = 7
	renderRandomShiftC           = 17
	renderRandomMantissaShift    = 11
	renderRandomMantissaScale    = 9007199254740992.0
	renderRGBAComponents         = 4
	renderMaxWorkers             = 128
	renderColorMax               = 255.0
	renderEdgeThreshold          = 0.95
	renderGrayWireframe          = 0.8
	renderMinRayDistance         = 1e-9

	defaultWidth           = 1920
	defaultHeight          = 1080
	defaultMaxDepth        = 80
	defaultMaxRayDistance  = 1000.0
	previewWidth           = 400
	previewHeight          = 225
	previewSamplesPerPixel = 20
	previewMaxDepth        = 20

	videoDefaultBitrate        = 2000
	videoDefaultQuality        = 70
	videoExtensionLength       = 4
	videoMicrosecondsPerSecond = 1000000
	videoChunkHeaderSize       = 56
	videoKeyframeFlag          = 0x10
	videoMinimumBufferSize     = 1024
	videoBitmapInfoHeaderSize  = 40
	videoStreamHeaderSize      = 24
	videoListHeaderSize        = 12
)

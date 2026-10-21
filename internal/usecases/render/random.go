package render

import (
	cryptorand "crypto/rand"
	"encoding/binary"
	"time"
)

// randSource хранит локальное состояние xorshift64 без общего доступа между работниками.
type randSource struct {
	state uint64
}

func newRandSource() *randSource {
	var seedBytes [8]byte
	seed := uint64(time.Now().UnixNano())
	_, err := cryptorand.Read(seedBytes[:])
	if err == nil {
		seed = binary.LittleEndian.Uint64(seedBytes[:])
	}
	if seed == 0 {
		seed = 1
	}
	return &randSource{state: seed}
}

// Float64 возвращает псевдослучайное число в диапазоне [0, 1).
func (rs *randSource) Float64() float64 {
	if rs == nil {
		// Это не должно происходить в нормальной работе
		return renderHalf
	}

	// Быстрый генератор используется после криптографической инициализации состояния.
	rs.state ^= rs.state << renderRandomShiftA
	rs.state ^= rs.state >> renderRandomShiftB
	rs.state ^= rs.state << renderRandomShiftC
	return float64(rs.state>>renderRandomMantissaShift) * (1.0 / renderRandomMantissaScale)
}

func (rs *randSource) startPixelSample(seed uint64, pixel, sample int) {
	if seed == 0 {
		return
	}
	const (
		shiftA    = 30
		shiftB    = 27
		shiftC    = 31
		finishMix = 0x94d049bb133111eb
	)
	const pixelMix = 0x9e3779b97f4a7c15
	const sampleMix = 0xbf58476d1ce4e5b9
	x := seed + uint64(max(pixel, 0))*pixelMix + uint64(max(sample, 0))*sampleMix
	x = (x ^ (x >> shiftA)) * sampleMix
	x = (x ^ (x >> shiftB)) * finishMix
	rs.state = (x ^ (x >> shiftC)) | 1
}

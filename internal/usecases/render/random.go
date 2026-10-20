package render

import (
	cryptorand "crypto/rand"
	"encoding/binary"
	"time"
)

// randSource is a lightweight wrapper around math/rand.Rand.
// It is not safe for concurrent use, so each goroutine must have its own instance.
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

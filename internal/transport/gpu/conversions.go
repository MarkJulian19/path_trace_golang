package gpu

import "math"

func glInt32(value int) int32 {
	if value > math.MaxInt32 {
		return math.MaxInt32
	}
	if value < math.MinInt32 {
		return math.MinInt32
	}
	return int32(value)
}

func glUint32(value int) uint32 {
	if value <= 0 {
		return 0
	}
	if uint64(value) > math.MaxUint32 {
		return math.MaxUint32
	}
	return uint32(value)
}

func glSeed(value int64) uint32 {
	return uint32(value & math.MaxUint32)
}

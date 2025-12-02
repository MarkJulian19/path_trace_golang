package engine

import (
	"math/rand"
	"time"
)

// randSource is a lightweight wrapper around math/rand.Rand.
// It is not safe for concurrent use, so each goroutine must have its own instance.
type randSource struct {
	r *rand.Rand
}

func newRandSource() *randSource {
	return &randSource{
		r: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (rs *randSource) Float64() float64 {
	return rs.r.Float64()
}



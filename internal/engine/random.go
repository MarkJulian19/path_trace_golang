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
	source := rand.NewSource(time.Now().UnixNano())
	r := rand.New(source)
	if r == nil {
		// Fallback: используем глобальный генератор, если New вернул nil
		// Это не должно происходить, но защищает от паники
		r = rand.New(rand.NewSource(time.Now().UnixNano() + 1))
	}
	return &randSource{
		r: r,
	}
}

func (rs *randSource) Float64() float64 {
	if rs == nil || rs.r == nil {
		// Fallback: возвращаем случайное значение без использования rng
		// Это не должно происходить в нормальной работе
		return 0.5
	}
	return rs.r.Float64()
}

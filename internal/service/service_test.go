package service_test

import (
	"testing"

	appservice "github.com/MarkJulian19/path_trace_golang/internal/service"
)

func BenchmarkContainerNew(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		benchmarkContainer = appservice.New()
	}
}

var benchmarkContainer *appservice.Container

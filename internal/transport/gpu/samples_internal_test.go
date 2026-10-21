package gpu

import (
	"testing"
	"time"
)

func TestAdaptiveDispatchBudget(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name                   string
		current, pixels, limit int
		elapsed                time.Duration
		want                   int
	}{
		{"slow tile shrinks", 256, 256 * 256, 256, 32 * time.Millisecond, 128},
		{"fast tile grows", 64, 64 * 64, 256, time.Millisecond, 128},
		{"configured limit", 256, 256 * 256, 256, time.Millisecond, 256},
		{"minimum workgroup", 32, 32 * 32, 256, time.Second, 32},
		{"partial edge does not inflate tile", 128, 128 * 5, 256, time.Millisecond, 128},
	} {
		if got := nextDispatchTileSize(tc.current, tc.pixels, tc.limit, tc.elapsed); got != tc.want {
			t.Errorf("%s: got=%d want=%d", tc.name, got, tc.want)
		}
	}
}

func TestDispatchGroups(t *testing.T) {
	t.Parallel()

	tests := []struct {
		size, workgroup, want int
	}{
		{size: 1, workgroup: 32, want: 1},
		{size: 32, workgroup: 32, want: 1},
		{size: 33, workgroup: 32, want: 2},
		{size: 256, workgroup: 16, want: 16},
		{size: 257, workgroup: 16, want: 17},
	}

	for _, tt := range tests {
		if got := dispatchGroups(tt.size, tt.workgroup); got != tt.want {
			t.Errorf("dispatchGroups(%d, %d) = %d, want %d", tt.size, tt.workgroup, got, tt.want)
		}
	}
}

func TestSampleUpdateIntervalLivePreview(t *testing.T) {
	t.Parallel()

	if got := sampleUpdateInterval(1920*1080, true); got != gpuSampleIntervalLive {
		t.Fatalf("live preview interval = %d, want %d", got, gpuSampleIntervalLive)
	}
}

func TestSampleUpdateIntervalWithoutLivePreview(t *testing.T) {
	t.Parallel()

	tests := []struct {
		pixels, want int
	}{
		{pixels: 3000 * 3000, want: 1},
		{pixels: 1200 * 1200, want: 2},
		{pixels: 640 * 480, want: gpuSampleIntervalMedium},
		{pixels: 320 * 240, want: gpuSampleIntervalDefault},
	}

	for _, tt := range tests {
		if got := sampleUpdateInterval(tt.pixels, false); got != tt.want {
			t.Errorf("sampleUpdateInterval(%d, false) = %d, want %d", tt.pixels, got, tt.want)
		}
	}
}

func TestBeginSampleRenderClearsStaleCancelForInteractiveRequest(t *testing.T) {
	t.Parallel()

	SetRenderCancel(true)
	defer SetRenderCancel(false)

	if !beginSampleRender(gpuSampleRequest{cancelled: func() bool { return false }}) {
		t.Fatal("interactive request was cancelled by a stale global flag")
	}
	if IsRenderCancelled() {
		t.Fatal("global cancellation flag was not cleared for a new interactive request")
	}

	if beginSampleRender(gpuSampleRequest{cancelled: func() bool { return true }}) {
		t.Fatal("cancelled interactive request was started")
	}
	if !IsRenderCancelled() {
		t.Fatal("cancelled interactive request did not preserve the global flag")
	}
}

package render

// Backend задаёт место выполнения тяжёлых вычислений рендера.
type Backend int

const (
	// BackendCPU выбирает CPU-рендерер.
	BackendCPU Backend = iota
	// BackendGPU выбирает GPU-рендерер с откатом на CPU.
	BackendGPU
)

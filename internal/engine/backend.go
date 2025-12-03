package engine

// Backend defines where heavy rendering computations are executed.
// Right now we support CPU and a placeholder for GPU.
type Backend int

const (
	BackendCPU Backend = iota
	BackendGPU
)

var currentBackend = BackendCPU

// SetBackend selects active render backend (CPU or GPU).
// If an unknown value is passed, CPU backend will be used.
func SetBackend(b Backend) {
	switch b {
	case BackendCPU, BackendGPU:
		currentBackend = b
	default:
		currentBackend = BackendCPU
	}
}

// GetBackend returns currently selected render backend.
func GetBackend() Backend {
	return currentBackend
}



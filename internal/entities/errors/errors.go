package errors

import "errors"

// ErrGPUOperation обозначает отказ команды OpenGL при построении изображения.
var ErrGPUOperation = errors.New("OpenGL operation failed")

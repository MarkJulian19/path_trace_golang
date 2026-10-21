package main

import (
	"fmt"
	"image"
	"image/png"
	"log"
	"math"
	"os"
	"path/filepath"
)

func compareReference(path string, actual image.Image) (err error) {
	file, err := os.Open(filepath.Clean(path))
	if err != nil {
		return fmt.Errorf("open reference: %w", err)
	}
	defer func() {
		closeErr := file.Close()
		if closeErr != nil && err == nil {
			err = fmt.Errorf("close reference: %w", closeErr)
		}
	}()
	reference, err := png.Decode(file)
	if err != nil {
		return fmt.Errorf("decode reference: %w", err)
	}
	if reference.Bounds().Size() != actual.Bounds().Size() {
		return fmt.Errorf("%w: reference dimensions differ", errInvalidValidation)
	}
	mse := imageMSE(actual, reference)
	psnr := math.Inf(1)
	if mse > 0 {
		const decibels = 10
		psnr = -decibels * math.Log10(mse)
	}
	log.Printf("reference comparison mse=%.9g rmse=%.9g psnr=%.4f reference=%s", mse, math.Sqrt(mse), psnr, path)
	return nil
}

func imageMSE(actual, reference image.Image) float64 {
	const channels = 3
	const colorMaximum = 65535
	var sum float64
	bounds := actual.Bounds()
	for y := range bounds.Dy() {
		for x := range bounds.Dx() {
			r, g, b, _ := actual.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			rr, rg, rb, _ := reference.At(reference.Bounds().Min.X+x, reference.Bounds().Min.Y+y).RGBA()
			dr, dg, db := float64(r)-float64(rr), float64(g)-float64(rg), float64(b)-float64(rb)
			sum += dr*dr + dg*dg + db*db
		}
	}
	return sum / (float64(bounds.Dx()*bounds.Dy()) * channels * colorMaximum * colorMaximum)
}

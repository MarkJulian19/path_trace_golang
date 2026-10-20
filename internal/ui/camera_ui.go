package ui

import (
	"fmt"
	"strconv"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/widget"

	"github.com/MarkJulian19/path_trace_golang/internal/entities/scene"
)

func newCameraUI(
	cam *scene.Camera,
	sc *scene.Scene,
	status *widget.Label,
	clearFinalImage func(),
	startRender func(bool),
) ([6]*widget.Entry, fyne.CanvasObject) {
	entries, fields := newCameraFields(*cam)
	apply := newCameraApply(sc, status, clearFinalImage, startRender, cam, fields)
	return entries, newCameraBox(fields, apply)
}

type cameraFields struct {
	posX, posY, posZ     *widget.Entry
	lookX, lookY, lookZ  *widget.Entry
	fov, aperture, focus *widget.Entry
}

func newCameraFields(cam scene.Camera) ([6]*widget.Entry, cameraFields) {
	fields := cameraFields{
		posX: widget.NewEntry(), posY: widget.NewEntry(), posZ: widget.NewEntry(),
		lookX: widget.NewEntry(), lookY: widget.NewEntry(), lookZ: widget.NewEntry(),
		fov: widget.NewEntry(), aperture: widget.NewEntry(), focus: widget.NewEntry(),
	}
	entries := [6]*widget.Entry{fields.posX, fields.posY, fields.posZ, fields.lookX, fields.lookY, fields.lookZ}
	setCameraEntries(entries, cam)
	fields.fov.SetText(fmt.Sprintf("%.1f", cam.FOV))
	fields.aperture.SetText(fmt.Sprintf("%.3f", cam.Aperture))
	fields.focus.SetText(fmt.Sprintf("%.2f", cameraFocusDistance(&scene.Scene{Camera: cam})))
	if cam.FocusDist != 0 {
		fields.focus.SetText(fmt.Sprintf("%.2f", cam.FocusDist))
	}
	return entries, fields
}

func newCameraApply(
	sc *scene.Scene,
	status *widget.Label,
	clearFinalImage func(),
	startRender func(bool),
	cam *scene.Camera,
	fields cameraFields,
) *widget.Button {
	return widget.NewButton("Apply camera", func() {
		cam.Position.X = parseCameraFloat(fields.posX, cam.Position.X)
		cam.Position.Y = parseCameraFloat(fields.posY, cam.Position.Y)
		cam.Position.Z = parseCameraFloat(fields.posZ, cam.Position.Z)
		cam.Target.X = parseCameraFloat(fields.lookX, cam.Target.X)
		cam.Target.Y = parseCameraFloat(fields.lookY, cam.Target.Y)
		cam.Target.Z = parseCameraFloat(fields.lookZ, cam.Target.Z)
		cam.FOV = parseCameraFloat(fields.fov, cam.FOV)
		cam.Aperture = max(parseCameraFloat(fields.aperture, cam.Aperture), 0)
		cam.FocusDist = max(parseCameraFloat(fields.focus, cam.FocusDist), 0)
		sc.Camera = *cam
		clearFinalImage()
		status.SetText("Camera updated")
		startRender(false)
	})
}

func newCameraBox(fields cameraFields, apply *widget.Button) fyne.CanvasObject {
	return container.NewVBox(
		widget.NewLabel("Camera"),
		container.NewGridWithColumns(
			2,
			widget.NewLabel(
				"Pos X",
			),
			fields.posX,
			widget.NewLabel("Pos Y"),
			fields.posY,
			widget.NewLabel("Pos Z"),
			fields.posZ,
			widget.NewLabel(
				"Look X",
			),
			fields.lookX,
			widget.NewLabel("Look Y"),
			fields.lookY,
			widget.NewLabel("Look Z"),
			fields.lookZ,
			widget.NewLabel(
				"FOV",
			),
			fields.fov,
			widget.NewLabel("Aperture"),
			fields.aperture,
			widget.NewLabel("Focus Dist"),
			fields.focus,
		),
		widget.NewLabel("Aperture: 0 = DOF off, >0 = depth of field"),
		widget.NewLabel("Focus Dist: 0 = auto (distance to target)"),
		apply,
	)
}

func setCameraEntries(entries [6]*widget.Entry, cam scene.Camera) {
	values := [6]float64{cam.Position.X, cam.Position.Y, cam.Position.Z, cam.Target.X, cam.Target.Y, cam.Target.Z}
	for index, value := range values {
		entries[index].SetText(fmt.Sprintf("%.2f", value))
	}
}

func parseCameraFloat(entry *widget.Entry, fallback float64) float64 {
	value, err := strconv.ParseFloat(entry.Text, 64)
	if err != nil {
		return fallback
	}
	return value
}

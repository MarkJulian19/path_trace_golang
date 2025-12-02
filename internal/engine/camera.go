package engine

import (
	"math"

	"github.com/user/pathtracer/internal/scene"
)

type camera struct {
	origin          vec3
	lowerLeftCorner vec3
	horizontal      vec3
	vertical        vec3
	u, v, w         vec3
	lensRadius      float64
	rng             *randSource
}

func newCamera(scCam scene.Camera, cfg RenderConfig, rng *randSource) camera {
	aspect := float64(cfg.Width) / float64(cfg.Height)
	if scCam.AspectRatio != 0 {
		aspect = scCam.AspectRatio
	}

	theta := scCam.FOV * math.Pi / 180
	h := math.Tan(theta / 2)
	viewportHeight := 2.0 * h
	viewportWidth := aspect * viewportHeight

	origin := v(scCam.Position.X, scCam.Position.Y, scCam.Position.Z)
	target := v(scCam.Target.X, scCam.Target.Y, scCam.Target.Z)
	up := v(scCam.Up.X, scCam.Up.Y, scCam.Up.Z)

	w := origin.sub(target).unit()
	u := up.cross(w).unit()
	vVec := w.cross(u)

	focusDist := scCam.FocusDist
	if focusDist == 0 {
		focusDist = origin.sub(target).length()
	}

	horizontal := u.mul(viewportWidth * focusDist)
	vertical := vVec.mul(viewportHeight * focusDist)
	lowerLeftCorner := origin.sub(horizontal.div(2)).sub(vertical.div(2)).sub(w.mul(focusDist))

	return camera{
		origin:          origin,
		lowerLeftCorner: lowerLeftCorner,
		horizontal:      horizontal,
		vertical:        vertical,
		u:               u,
		v:               vVec,
		w:               w,
		lensRadius:      scCam.Aperture / 2,
		rng:             rng,
	}
}

func (c camera) getRay(s, t float64) ray {
	if c.lensRadius > 0 {
		rd := randomInUnitSphere(c.rng).mul(c.lensRadius)
		offset := c.u.mul(rd.x).add(c.v.mul(rd.y))
		return ray{
			orig: c.origin.add(offset),
			dir:  c.lowerLeftCorner.add(c.horizontal.mul(s)).add(c.vertical.mul(t)).sub(c.origin).sub(offset),
		}
	}

	return ray{
		orig: c.origin,
		dir:  c.lowerLeftCorner.add(c.horizontal.mul(s)).add(c.vertical.mul(t)).sub(c.origin),
	}
}



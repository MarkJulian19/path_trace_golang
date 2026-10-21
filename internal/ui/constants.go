package ui

import "time"

const (
	uiMaxDisplayWidth        = 1024.0
	uiMaxDisplayHeight       = 768.0
	uiDegreesPerHalfTurn     = 180.0
	uiViewportDiameter       = 2.0
	uiPreviewIntervalSlow    = 50
	uiPreviewIntervalNormal  = 75
	uiRenderDebounce         = 300 * time.Millisecond
	uiInteractiveRenderRate  = 120 * time.Millisecond
	uiHalf                   = 0.5
	uiDefaultVideoDuration   = 5.0
	uiDefaultVideoFPS        = 30
	uiPanelHeight            = 140
	uiPanelWidth             = 240
	uiWireframeFPSMax        = 120
	uiWireframeFPSDefault    = 60.0
	uiDenoiseSigmaRDefault   = 0.15
	uiRightPanelOffset       = 0.70
	uiLeftPanelOffset        = 0.2
	uiNineTenths             = 0.9
	uiWindowWidth            = 1440
	uiWindowHeight           = 900
	uiEditorBodyOffset       = 0.88
	uiObjectMoveStep         = 0.05
	uiTransformStepWidth     = 64
	uiTransformStepHeight    = 28
	uiCompactButtonSize      = 28
	uiViewportMinWidth       = 320
	uiViewportMinHeight      = 220
	uiInspectorMinWidth      = 260
	uiStatusMinHeight        = 86
	uiGizmoAxisLength        = 0.22
	uiGizmoHandleSize        = 18
	uiGizmoLineWidth         = 3
	uiGizmoLabelSize         = 16
	uiGizmoLabelOffset       = 5
	uiGizmoRedR              = 0xf2
	uiGizmoRedG              = 0x5f
	uiGizmoRedB              = 0x5c
	uiGizmoGreenR            = 0x62
	uiGizmoGreenG            = 0xd8
	uiGizmoBlueR             = 0x5f
	uiGizmoBlueG             = 0x9d
	uiKeyTimeoutMillis       = 50
	uiFloatEpsilon           = 1e-6
	uiFrameMillis            = 16.0
	uiMillisPerSecond        = 1000.0
	uiPitchMargin            = 0.1
	uiFogColorDefault        = 0.8
	uiAbsorptionScaleDefault = 0.01
	uiFogNoiseScaleDefault   = 3.0
	uiFogNoiseOctavesDefault = 3
	uiMaxSmoothRadius        = 5
	uiProgressUpdateInterval = 100 * time.Millisecond
	uiUnitValue              = 1.0

	uiThemeBackground  = 0x1e
	uiThemeButton      = 0x3d
	uiThemeDisabled    = 0x2a
	uiThemeForeground  = 0xcc
	uiThemeTextMuted   = 0x66
	uiThemePlaceholder = 0x88
	uiThemePrimary     = 0x37
	uiThemeHover       = 0x4a
	uiThemeAlpha       = 0xff
	uiThemeGreen       = 0x6f
	uiThemeBlue        = 0xa5
)

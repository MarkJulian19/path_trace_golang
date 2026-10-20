package ui

import (
	"image/color"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/theme"
)

// blenderDarkTheme реализует темную тему в стиле Blender.
type blenderDarkTheme struct {
	variant fyne.ThemeVariant
}

var blenderThemeColors = map[fyne.ThemeColorName]color.Color{
	theme.ColorNameBackground: color.RGBA{
		R: uiThemeBackground,
		G: uiThemeBackground,
		B: uiThemeBackground,
		A: uiThemeAlpha,
	},
	theme.ColorNameButton: color.RGBA{R: uiThemeButton, G: uiThemeButton, B: uiThemeButton, A: uiThemeAlpha},
	theme.ColorNameDisabledButton: color.RGBA{
		R: uiThemeDisabled,
		G: uiThemeDisabled,
		B: uiThemeDisabled,
		A: uiThemeAlpha,
	},
	theme.ColorNameForeground: color.RGBA{
		R: uiThemeForeground,
		G: uiThemeForeground,
		B: uiThemeForeground,
		A: uiThemeAlpha,
	},
	theme.ColorNameDisabled: color.RGBA{
		R: uiThemeTextMuted,
		G: uiThemeTextMuted,
		B: uiThemeTextMuted,
		A: uiThemeAlpha,
	},
	theme.ColorNamePlaceHolder: color.RGBA{
		R: uiThemePlaceholder,
		G: uiThemePlaceholder,
		B: uiThemePlaceholder,
		A: uiThemeAlpha,
	},
	theme.ColorNamePrimary:   color.RGBA{R: uiThemePrimary, G: uiThemeGreen, B: uiThemeBlue, A: uiThemeAlpha},
	theme.ColorNameHover:     color.RGBA{R: uiThemeHover, G: uiThemeHover, B: uiThemeHover, A: uiThemeAlpha},
	theme.ColorNameFocus:     color.RGBA{R: uiThemePrimary, G: uiThemeGreen, B: uiThemeBlue, A: uiThemeAlpha},
	theme.ColorNameScrollBar: color.RGBA{R: uiThemeButton, G: uiThemeButton, B: uiThemeButton, A: uiThemeAlpha},
	theme.ColorNameShadow:    color.RGBA{R: 0, G: 0, B: 0, A: uiThemePlaceholder},
	theme.ColorNameSeparator: color.RGBA{
		R: uiThemeDisabled,
		G: uiThemeDisabled,
		B: uiThemeDisabled,
		A: uiThemeAlpha,
	},
	theme.ColorNameInputBackground: color.RGBA{
		R: uiThemeDisabled,
		G: uiThemeDisabled,
		B: uiThemeDisabled,
		A: uiThemeAlpha,
	},
	theme.ColorNameInputBorder: color.RGBA{R: uiThemeButton, G: uiThemeButton, B: uiThemeButton, A: uiThemeAlpha},
}

func newBlenderDarkTheme() fyne.Theme {
	return &blenderDarkTheme{variant: theme.VariantDark}
}

// Color возвращает цвет темы для указанного элемента интерфейса.
func (t *blenderDarkTheme) Color(name fyne.ThemeColorName, variant fyne.ThemeVariant) color.Color {
	if value, ok := blenderThemeColors[name]; ok {
		return value
	}
	return theme.DefaultTheme().Color(name, variant)
}

// Font возвращает шрифт темы.
func (t *blenderDarkTheme) Font(style fyne.TextStyle) fyne.Resource {
	return theme.DefaultTheme().Font(style)
}

// Icon возвращает иконку темы.
func (t *blenderDarkTheme) Icon(name fyne.ThemeIconName) fyne.Resource {
	return theme.DefaultTheme().Icon(name)
}

// Size возвращает размер элемента темы.
func (t *blenderDarkTheme) Size(name fyne.ThemeSizeName) float32 {
	return theme.DefaultTheme().Size(name)
}

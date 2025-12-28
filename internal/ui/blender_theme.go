package ui

import (
	"image/color"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/theme"
)

// blenderDarkTheme реализует темную тему в стиле Blender
type blenderDarkTheme struct {
	variant fyne.ThemeVariant
}

func newBlenderDarkTheme() fyne.Theme {
	return &blenderDarkTheme{variant: theme.VariantDark}
}

func (t *blenderDarkTheme) Color(name fyne.ThemeColorName, variant fyne.ThemeVariant) color.Color {
	switch name {
	case theme.ColorNameBackground:
		return color.RGBA{R: 0x1e, G: 0x1e, B: 0x1e, A: 0xff} // Темный фон как в Blender
	case theme.ColorNameButton:
		return color.RGBA{R: 0x3d, G: 0x3d, B: 0x3d, A: 0xff} // Кнопки
	case theme.ColorNameDisabledButton:
		return color.RGBA{R: 0x2a, G: 0x2a, B: 0x2a, A: 0xff} // Отключенные кнопки
	case theme.ColorNameForeground:
		return color.RGBA{R: 0xcc, G: 0xcc, B: 0xcc, A: 0xff} // Основной текст
	case theme.ColorNameDisabled:
		return color.RGBA{R: 0x66, G: 0x66, B: 0x66, A: 0xff} // Отключенный текст
	case theme.ColorNamePlaceHolder:
		return color.RGBA{R: 0x88, G: 0x88, B: 0x88, A: 0xff} // Плейсхолдер
	case theme.ColorNamePrimary:
		return color.RGBA{R: 0x37, G: 0x6f, B: 0xa5, A: 0xff} // Основной цвет (синий как в Blender)
	case theme.ColorNameHover:
		return color.RGBA{R: 0x4a, G: 0x4a, B: 0x4a, A: 0xff} // Hover
	case theme.ColorNameFocus:
		return color.RGBA{R: 0x37, G: 0x6f, B: 0xa5, A: 0xff} // Фокус
	case theme.ColorNameScrollBar:
		return color.RGBA{R: 0x3d, G: 0x3d, B: 0x3d, A: 0xff} // Скроллбар
	case theme.ColorNameShadow:
		return color.RGBA{R: 0x00, G: 0x00, B: 0x00, A: 0x88} // Тень
	case theme.ColorNameSeparator:
		return color.RGBA{R: 0x2a, G: 0x2a, B: 0x2a, A: 0xff} // Разделитель
	case theme.ColorNameInputBackground:
		return color.RGBA{R: 0x2a, G: 0x2a, B: 0x2a, A: 0xff} // Фон ввода
	case theme.ColorNameInputBorder:
		return color.RGBA{R: 0x3d, G: 0x3d, B: 0x3d, A: 0xff} // Граница ввода
	default:
		return theme.DefaultTheme().Color(name, variant)
	}
}

func (t *blenderDarkTheme) Font(style fyne.TextStyle) fyne.Resource {
	return theme.DefaultTheme().Font(style)
}

func (t *blenderDarkTheme) Icon(name fyne.ThemeIconName) fyne.Resource {
	return theme.DefaultTheme().Icon(name)
}

func (t *blenderDarkTheme) Size(name fyne.ThemeSizeName) float32 {
	return theme.DefaultTheme().Size(name)
}


// Package renderdisplay преобразует непрерывный цвет в отображаемые восьмибитные каналы.
package renderdisplay

const (
	byteMax       = 255
	ditherShiftA  = 30
	ditherShiftB  = 27
	ditherShiftC  = 31
	mantissaShift = 11
	mantissaBits  = 53
	mantissaScale = 1.0 / (1 << mantissaBits)
	ditherOffset  = 0x9e3779b97f4a7c15
	ditherMixA    = 0xbf58476d1ce4e5b9
	ditherMixB    = 0x94d049bb133111eb
)

// Dither возвращает неизменный порог [0, 1) для пикселя, независимый от RNG трассировки.
// Один порог для RGB сохраняет нейтральные цвета без цветного зерна и мерцания.
func Dither(pixel int) float64 {
	x := uint64(max(pixel, 0)) + ditherOffset
	x = (x ^ (x >> ditherShiftA)) * ditherMixA
	x = (x ^ (x >> ditherShiftB)) * ditherMixB
	x ^= x >> ditherShiftC
	return float64(x>>mantissaShift) * mantissaScale
}

// Byte квантует канал [0, 255] с несмещённым округлением вместо ступенчатых градиентов.
// Ошибка меньше одного уровня канала; линейная энергия и HDR не изменяются.
func Byte(value, threshold float64) byte {
	if !(value > 0) {
		return 0
	}
	if value >= byteMax {
		return byteMax
	}
	return byte(value + threshold)
}

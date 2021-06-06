package gonlpclassifier

func Bound(number, min, max float64) float64 {
	if number > max {
		return max
	} else if number < min {
		return min
	} else {
		return number
	}
}

func SumFloat(d ...float64) float64 {
	if len(d) == 0 {
		return .0
	}

	initial := d[0]

	for _, accum := range d[1:] {
		initial += accum
	}

	return initial
}

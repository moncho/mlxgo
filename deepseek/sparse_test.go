package deepseek

import (
	"math"
	"slices"
	"testing"
)

func TestSparsePlan(t *testing.T) {
	ids := []int32{3, 1, 0, 3, 1, 0}
	groups, inverse, err := planExperts(ids, 3, 2, 5)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(groups[0].rows, []int32{1, 2}) || !slices.Equal(groups[1].slots, []int32{1, 4}) || len(groups[2].rows) != 0 || len(groups[4].rows) != 0 {
		t.Fatal(groups)
	}
	var joined []int32
	for _, g := range groups {
		for _, slot := range g.slots {
			joined = append(joined, slot)
		}
	}
	for slot, position := range inverse {
		if joined[position] != int32(slot) {
			t.Fatal(inverse)
		}
	}
	for _, bad := range []struct {
		ids                []int32
		tokens, k, experts int
	}{
		{nil, 0, 1, 2}, {nil, math.MaxInt32, 2, 2}, {[]int32{0}, 1, 2, 2},
		{[]int32{-1}, 1, 1, 2}, {[]int32{2}, 1, 1, 2}, {[]int32{1, 1}, 1, 2, 2},
	} {
		if _, _, err := planExperts(bad.ids, bad.tokens, bad.k, bad.experts); err == nil {
			t.Fatal("accepted", bad)
		}
	}
}

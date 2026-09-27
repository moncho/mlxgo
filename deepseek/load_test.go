package deepseek

import "testing"

func TestLoadSelections(t *testing.T) {
	c := readModelFixture(t).Config
	for _, o := range []LoadOptions{
		{MaxBytes: -1}, {FP4Experts: []ExpertID{{Layer: -1}}},
		{FP4Experts: []ExpertID{{Index: -2}}}, {FP4Experts: []ExpertID{{Layer: c.Layers}}},
		{FP8AttentionKV: []int{-1}}, {FP8AttentionQB: []int{c.Layers}},
		{FP4Experts: []ExpertID{{Layer: 0, Index: 0}}}, // fixture dimensions are not aligned
	} {
		if _, _, err := loadLayout(c, o); err == nil {
			t.Fatal("invalid selection accepted", o)
		}
	}
	c.Dim = 32
	c.InterDim = 32
	c.QRank = 32
	c.HeadDim = 32
	c.ORank = 32
	for _, o := range []LoadOptions{{FP4Experts: []ExpertID{{}, {}}}, {FP8AttentionKV: []int{0, 0}}} {
		if _, _, err := loadLayout(c, o); err == nil {
			t.Fatal("duplicate selection accepted")
		}
	}
}

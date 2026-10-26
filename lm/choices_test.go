package lm

import "testing"

func TestChoicesValidation(t *testing.T) {
	for _, sequences := range [][][]int32{nil, {{}}, {{-1}}, {{1}, {1}}, {{1}, {1, 2}}, {{1, 2}, {1}}} {
		if _, err := NewChoices(sequences); err == nil {
			t.Fatalf("accepted invalid choices %v", sequences)
		}
	}
	seq := [][]int32{{1, 0}, {1, 2, 0}}
	c, err := NewChoices(seq)
	if err != nil {
		t.Fatal(err)
	}
	seq[0][0] = 3
	if c.root.edges[1] == nil || c.root.edges[3] != nil {
		t.Fatal("choices retained mutable caller input")
	}
	s := &errorSession{}
	if _, err := c.Greedy(s, []int32{1}, 2); err == nil || s.calls != 0 {
		t.Fatal("short cap touched session")
	}
	if _, err := c.Greedy(s, nil, 3); err == nil || s.calls != 0 {
		t.Fatal("empty prompt touched session")
	}
	if _, err := c.Greedy(s, []int32{1}, 3); err == nil || s.calls != 1 {
		t.Fatal("step error not propagated")
	}
	var empty Choices
	if _, err := empty.Greedy(s, []int32{1}, 3); err == nil {
		t.Fatal("accepted uninitialized choices")
	}
}

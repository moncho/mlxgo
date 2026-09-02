//go:build mlx

package mlx

import "testing"

func TestMLXBuildValidatesClosureInputsBeforeCallingNativeCode(t *testing.T) {
	if _, err := NewClosure(nil); err == nil {
		t.Fatal("expected NewClosure with nil function to fail")
	}
	if _, err := NewValueAndGrad(nil); err == nil {
		t.Fatal("expected NewValueAndGrad with nil function to fail")
	}
	if _, err := NewValueAndGrad(func([]Array) ([]Array, error) {
		return nil, nil
	}, -1); err == nil {
		t.Fatal("expected NewValueAndGrad with negative argnum to fail")
	}
}

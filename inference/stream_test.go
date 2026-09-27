package inference

import (
	"errors"
	"strings"
	"testing"
)

func TestTextStream(t *testing.T) {
	var s textStream
	var out strings.Builder
	emit := func(v string) error { out.WriteString(v); return nil }
	for _, tc := range []struct{ decoded, want string }{
		{"a\xe2", "a"}, {"a\xe2\x82", "a"}, {"a\xe2\x82\xac", "a\xe2\x82\xac"},
		{"a\xe2\x82\xac\ufffd", "a\xe2\x82\xac"},
		{"a\xe2\x82\xac\xf0\x9f", "a\xe2\x82\xac"},
		{"a\xe2\x82\xac\xf0\x9f\x98\x80", "a\xe2\x82\xac\xf0\x9f\x98\x80"},
	} {
		if err := s.update(tc.decoded, false, emit); err != nil {
			t.Fatal(err)
		}
		if out.String() != tc.want {
			t.Fatalf("got %q, want %q", out.String(), tc.want)
		}
	}
	final := out.String() + "\xe2"
	if err := s.update(final, true, emit); err != nil || out.String() != final {
		t.Fatalf("flush %q %v", out.String(), err)
	}
	if err := s.update("changed", false, emit); err == nil {
		t.Fatal("accepted changed prefix")
	}
	var failed textStream
	want := errors.New("stop")
	if err := failed.update("x", false, func(string) error { return want }); !errors.Is(err, want) || failed.emitted != "" {
		t.Fatalf("callback error %v", err)
	}
}

package quant

import (
	"errors"
	"fmt"
	"io"
)

var ErrMemoryBudget = errors.New("quant: matrix exceeds memory budget")

// ReadMatrix decodes raw row-major weight and scale streams into a float32
// matrix. maxBytes caps the returned numeric buffer plus this function's input
// scratch buffers, checked before allocating or reading. It does not cap reader
// internals, allocator overhead, previously returned matrices, or MLX memory.
//
// Only 32 rows of encoded data/scales are buffered at a time; the full encoded
// matrix is never retained. Each reader must end immediately after its matrix.
// On malformed, nonfinite, truncated or trailing data it returns nil, error.
// This reads raw tensors, not safetensors headers; callers supply validated
// dimensions and own integrity checks (for example, io.TeeReader with SHA-256).
func ReadMatrix(data, scales io.Reader, rows, cols int, format Format, rounding Rounding, maxBytes int64) ([]float32, error) {
	maxInt := int(^uint(0) >> 1)
	if data == nil || scales == nil || rows <= 0 || cols <= 0 || rows > maxInt/4/cols {
		return nil, fmt.Errorf("quant: invalid readers or overflowing dimensions")
	}
	if rounding != Float32 && rounding != BFloat16 {
		return nil, fmt.Errorf("quant: invalid rounding")
	}
	chunkRows, blocks := min(32, rows), (cols-1)/32+1
	dataBytes, scaleBytes := chunkRows*cols, blocks
	switch format {
	case FP8Block32:
	case FP8Row32, FP4Row32:
		if cols%32 != 0 {
			return nil, fmt.Errorf("quant: row-scaled columns must be divisible by 32")
		}
		scaleBytes *= chunkRows
		if format == FP4Row32 {
			dataBytes /= 2
		}
	default:
		return nil, fmt.Errorf("quant: invalid format")
	}
	// Subtract instead of summing to avoid overflow even for adversarial sizes.
	outputBytes := int64(rows*cols) * 4
	if maxBytes < outputBytes || maxBytes-outputBytes < int64(dataBytes)+int64(scaleBytes) {
		return nil, ErrMemoryBudget
	}
	out := make([]float32, rows*cols)
	db, sb := make([]byte, dataBytes), make([]byte, scaleBytes)
	for row := 0; row < rows; {
		nr := min(32, rows-row)
		nd, ns := nr*cols, blocks
		if format != FP8Block32 {
			ns *= nr
		}
		if format == FP4Row32 {
			nd /= 2
		}
		if _, err := io.ReadFull(data, db[:nd]); err != nil {
			return nil, fmt.Errorf("quant: weight row %d: %w", row, err)
		}
		if _, err := io.ReadFull(scales, sb[:ns]); err != nil {
			return nil, fmt.Errorf("quant: scale row %d: %w", row, err)
		}
		if err := Decode(out[row*cols:(row+nr)*cols], db[:nd], sb[:ns], nr, cols, format, rounding); err != nil {
			return nil, fmt.Errorf("quant: row chunk %d: %w", row, err)
		}
		row += nr
	}
	for _, r := range []io.Reader{data, scales} {
		var extra [1]byte
		if _, err := io.ReadFull(r, extra[:]); err != io.EOF {
			return nil, fmt.Errorf("quant: expected end of tensor stream, got %v", err)
		}
	}
	return out, nil
}

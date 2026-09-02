package mlx

// DType identifies an MLX array element type.
type DType int

const (
	Bool DType = iota
	UInt8
	UInt16
	UInt32
	UInt64
	Int8
	Int16
	Int32
	Int64
	Float16
	Float32
	Float64
	BFloat16
	Complex64
)

func (d DType) String() string {
	switch d {
	case Bool:
		return "bool"
	case UInt8:
		return "uint8"
	case UInt16:
		return "uint16"
	case UInt32:
		return "uint32"
	case UInt64:
		return "uint64"
	case Int8:
		return "int8"
	case Int16:
		return "int16"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	case Float16:
		return "float16"
	case Float32:
		return "float32"
	case Float64:
		return "float64"
	case BFloat16:
		return "bfloat16"
	case Complex64:
		return "complex64"
	default:
		return "unknown"
	}
}

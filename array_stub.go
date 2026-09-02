//go:build !mlx

package mlx

import "errors"

var errBuiltWithoutMLX = errors.New("mlxgo: build with -tags mlx and install mlx-c to enable MLX bindings")

// Array is an MLX array handle. The non-MLX build keeps the API visible for
// editors and ordinary Go tooling, but all operations return an explanatory
// error.
type Array struct{}

// SafeTensors is a lazily loaded safetensors handle in the MLX build.
type SafeTensors struct{}

// DeviceType identifies an MLX device class in the native build.
type DeviceType int

const (
	// DeviceGPU selects an Apple Silicon GPU device.
	DeviceGPU DeviceType = iota
	// DeviceCPU selects a CPU device.
	DeviceCPU
)

func NewFloat32(_ []float32, _ []int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func NewFloat64(_ []float64, _ []int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func NewInt32(_ []int32, _ []int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func NewInt64(_ []int64, _ []int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func NewScalarFloat32(_ float32) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func NewScalarFloat64(_ float64) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func NewScalarInt(_ int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Zeros(_ []int, _ DType) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Ones(_ []int, _ DType) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Full(_ []int, _ float64, _ DType) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func ZerosLike(_ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func OnesLike(_ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Add(_, _ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func AddMM(_, _, _ Array, _, _ float32) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Subtract(_, _ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Multiply(_, _ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Divide(_, _ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Matmul(_, _ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Maximum(_, _ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Minimum(_, _ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Power(_, _ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Clip(_, _, _ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Arange(_, _, _ float64) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func ArangeDType(_, _, _ float64, _ DType) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func AsType(_ Array, _ DType) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Reshape(_ Array, _ []int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Transpose(_ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func TransposeAxes(_ Array, _ []int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func BroadcastTo(_ Array, _ []int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func ExpandDims(_ Array, _ int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func ExpandDimsAxes(_ Array, _ []int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Squeeze(_ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func SqueezeAxis(_ Array, _ int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func SqueezeAxes(_ Array, _ []int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Flatten(_ Array, _, _ int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Abs(_ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Exp(_ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Log(_ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Negative(_ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Square(_ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Sqrt(_ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Sigmoid(_ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Tanh(_ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Sin(_ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Cos(_ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func ReLU(_ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func StopGradient(_ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Contiguous(_ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Sum(_ Array, _ bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func SumAxis(_ Array, _ int, _ bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func SumAxes(_ Array, _ []int, _ bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Mean(_ Array, _ bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func MeanAxis(_ Array, _ int, _ bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func MeanAxes(_ Array, _ []int, _ bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func LogSumExp(_ Array, _ bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func LogSumExpAxis(_ Array, _ int, _ bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func LogSumExpAxes(_ Array, _ []int, _ bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Softmax(_ Array, _ bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func SoftmaxAxis(_ Array, _ int, _ bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func SoftmaxAxes(_ Array, _ []int, _ bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Argmax(_ Array, _ bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func ArgmaxAxis(_ Array, _ int, _ bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Argmin(_ Array, _ bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func ArgminAxis(_ Array, _ int, _ bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Equal(_, _ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Greater(_, _ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func GreaterEqual(_, _ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Less(_, _ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func LessEqual(_, _ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Where(_, _, _ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Take(_, _ Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func TakeAxis(_, _ Array, _ int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func TakeAlongAxis(_, _ Array, _ int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Gather(_, _ Array, _ int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func GatherSlices(_, _ Array, _ int, _ []int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Concatenate(_ []Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func ConcatenateAxis(_ []Array, _ int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Stack(_ []Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func StackAxis(_ []Array, _ int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Load(_ string) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func Save(_ string, _ Array) error {
	return errBuiltWithoutMLX
}

func LoadSafetensors(_ string) (*SafeTensors, error) {
	return nil, errBuiltWithoutMLX
}

func SaveSafetensors(_ string, _ map[string]Array, _ map[string]string) error {
	return errBuiltWithoutMLX
}

func RandomSeed(_ uint64) error {
	return errBuiltWithoutMLX
}

func RandomKey(_ uint64) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func RandomNormal(_ []int, _ DType, _, _ float32) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func RandomUniform(_ []int, _ DType, _, _ float32) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func RandomRandint(_ []int, _ DType, _, _ int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func RandomBernoulli(_ []int, _ float32) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func RandomCategorical(_ Array, _ int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func (*SafeTensors) Close() error {
	return nil
}

func (*SafeTensors) Get(_ string) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

func (*SafeTensors) Metadata(_ string) (string, bool, error) {
	return "", false, errBuiltWithoutMLX
}

func SetDefaultCPU() error {
	return errBuiltWithoutMLX
}

func SetDefaultGPU() error {
	return errBuiltWithoutMLX
}

func SetDefaultDevice(_ DeviceType, _ int) error {
	return errBuiltWithoutMLX
}

func (d DeviceType) String() string {
	switch d {
	case DeviceGPU:
		return "gpu"
	case DeviceCPU:
		return "cpu"
	default:
		return "unknown"
	}
}

func (*Array) Close() error {
	return nil
}

func (Array) Eval() error {
	return errBuiltWithoutMLX
}

func (Array) Shape() []int {
	return nil
}

func (Array) DType() (DType, error) {
	return Float32, errBuiltWithoutMLX
}

func (Array) Size() int {
	return 0
}

func (Array) Float32Data() ([]float32, error) {
	return nil, errBuiltWithoutMLX
}

func (Array) Float64Data() ([]float64, error) {
	return nil, errBuiltWithoutMLX
}

func (Array) Int32Data() ([]int32, error) {
	return nil, errBuiltWithoutMLX
}

func (Array) Int64Data() ([]int64, error) {
	return nil, errBuiltWithoutMLX
}

func (Array) UInt32Data() ([]uint32, error) {
	return nil, errBuiltWithoutMLX
}

func (Array) UInt64Data() ([]uint64, error) {
	return nil, errBuiltWithoutMLX
}

func (Array) BoolData() ([]bool, error) {
	return nil, errBuiltWithoutMLX
}

func (Array) String() string {
	return errBuiltWithoutMLX.Error()
}

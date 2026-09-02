//go:build mlx && mlxruntime

package mlx

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func TestRuntimeArrayOps(t *testing.T) {
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}

	a := mustNewFloat32(t, []float32{1, 2, 3, 4}, []int{2, 2})
	defer a.Close()
	b := mustNewFloat32(t, []float32{10, 20, 30, 40}, []int{2, 2})
	defer b.Close()

	add := mustAdd(t, a, b)
	defer add.Close()
	assertFloat32Data(t, add, []float32{11, 22, 33, 44})

	sub := mustSubtract(t, b, a)
	defer sub.Close()
	assertFloat32Data(t, sub, []float32{9, 18, 27, 36})

	mul := mustMultiply(t, a, b)
	defer mul.Close()
	assertFloat32Data(t, mul, []float32{10, 40, 90, 160})

	div := mustDivide(t, b, a)
	defer div.Close()
	assertFloat32Data(t, div, []float32{10, 10, 10, 10})
}

func TestRuntimeMatmulReshapeAndReductions(t *testing.T) {
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}

	x := mustNewFloat32(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	defer x.Close()
	w := mustNewFloat32(t, []float32{1, 2, 3, 4, 5, 6}, []int{3, 2})
	defer w.Close()

	product := mustMatmul(t, x, w)
	defer product.Close()
	assertShape(t, product, []int{2, 2})
	assertFloat32Data(t, product, []float32{22, 28, 49, 64})

	reshaped := mustReshape(t, product, []int{4})
	defer reshaped.Close()
	assertShape(t, reshaped, []int{4})
	assertFloat32Data(t, reshaped, []float32{22, 28, 49, 64})

	sum := mustSum(t, reshaped, false)
	defer sum.Close()
	assertFloat32Data(t, sum, []float32{163})

	mean := mustMean(t, reshaped, false)
	defer mean.Close()
	assertFloat32Close(t, mean, []float32{40.75})
}

func TestRuntimeDTypesAndCasting(t *testing.T) {
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}

	ints := mustNewInt32(t, []int32{1, 2, 3, 4}, []int{2, 2})
	defer ints.Close()
	assertInt32Data(t, ints, []int32{1, 2, 3, 4})

	casted := mustAsType(t, ints, Float32)
	defer casted.Close()
	assertFloat32Data(t, casted, []float32{1, 2, 3, 4})
}

func TestRuntimeArrayCopySharesCloseState(t *testing.T) {
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}

	a := mustNewFloat32(t, []float32{1, 2}, []int{2})
	copied := a
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := copied.Float32Data(); err == nil {
		t.Fatal("expected copied array to observe closed state")
	}
	if err := copied.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRuntimeCreationElementwiseAndShapeOps(t *testing.T) {
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}

	zeros := mustZeros(t, []int{2, 2}, Float32)
	defer zeros.Close()
	assertFloat32Data(t, zeros, []float32{0, 0, 0, 0})

	ones := mustOnes(t, []int{2, 2}, Float32)
	defer ones.Close()
	assertFloat32Data(t, ones, []float32{1, 1, 1, 1})

	full := mustFull(t, []int{2, 2}, 3, Float32)
	defer full.Close()
	assertFloat32Data(t, full, []float32{3, 3, 3, 3})

	squared := mustSquare(t, full)
	defer squared.Close()
	assertFloat32Data(t, squared, []float32{9, 9, 9, 9})

	root := mustSqrt(t, squared)
	defer root.Close()
	assertFloat32Data(t, root, []float32{3, 3, 3, 3})

	neg := mustNegative(t, root)
	defer neg.Close()
	assertFloat32Data(t, neg, []float32{-3, -3, -3, -3})

	maxed := mustMaximum(t, neg, zeros)
	defer maxed.Close()
	assertFloat32Data(t, maxed, []float32{0, 0, 0, 0})

	minned := mustMinimum(t, full, ones)
	defer minned.Close()
	assertFloat32Data(t, minned, []float32{1, 1, 1, 1})

	matrix := mustNewFloat32(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	defer matrix.Close()
	transposed := mustTranspose(t, matrix)
	defer transposed.Close()
	assertShape(t, transposed, []int{3, 2})
	assertFloat32Data(t, transposed, []float32{1, 4, 2, 5, 3, 6})

	expanded := mustExpandDims(t, full, 0)
	defer expanded.Close()
	assertShape(t, expanded, []int{1, 2, 2})

	squeezed := mustSqueezeAxis(t, expanded, 0)
	defer squeezed.Close()
	assertShape(t, squeezed, []int{2, 2})

	flattened := mustFlatten(t, expanded, 0, -1)
	defer flattened.Close()
	assertShape(t, flattened, []int{4})

	scalar := mustNewFloat32(t, []float32{7}, []int{1})
	defer scalar.Close()
	broadcasted := mustBroadcastTo(t, scalar, []int{2, 2})
	defer broadcasted.Close()
	assertFloat32Data(t, broadcasted, []float32{7, 7, 7, 7})

	sigmoid := mustSigmoid(t, zeros)
	defer sigmoid.Close()
	assertFloat32Data(t, sigmoid, []float32{0.5, 0.5, 0.5, 0.5})

	relu := mustReLU(t, neg)
	defer relu.Close()
	assertFloat32Data(t, relu, []float32{0, 0, 0, 0})
}

func TestRuntimeScalarAndEmptyShapes(t *testing.T) {
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}

	scalar := mustNewFloat32(t, []float32{7}, nil)
	defer scalar.Close()
	assertShape(t, scalar, []int{})
	assertFloat32Data(t, scalar, []float32{7})

	zeroDim := mustNewFloat32(t, nil, []int{0})
	defer zeroDim.Close()
	assertShape(t, zeroDim, []int{0})
	assertFloat32Data(t, zeroDim, []float32{})

	zeroScalar := mustZeros(t, nil, Float32)
	defer zeroScalar.Close()
	assertShape(t, zeroScalar, []int{})
	assertFloat32Data(t, zeroScalar, []float32{0})
}

func TestRuntimeStridedDataReadsUseContiguousCopy(t *testing.T) {
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}

	matrix := mustNewFloat32(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	defer matrix.Close()
	transposed := mustTranspose(t, matrix)
	defer transposed.Close()
	assertShape(t, transposed, []int{3, 2})
	assertFloat32Data(t, transposed, []float32{1, 4, 2, 5, 3, 6})

	row := mustNewFloat32(t, []float32{1, 2}, []int{2})
	defer row.Close()
	broadcasted := mustBroadcastTo(t, row, []int{4, 2})
	defer broadcasted.Close()
	assertFloat32Data(t, broadcasted, []float32{1, 2, 1, 2, 1, 2, 1, 2})
}

func TestRuntimeInvalidNativeOperationReturnsError(t *testing.T) {
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}

	a := mustNewFloat32(t, []float32{1, 2, 3}, []int{3})
	defer a.Close()
	b := mustNewFloat32(t, []float32{1, 2, 3, 4, 5}, []int{5})
	defer b.Close()

	_, err := Add(a, b)
	if err == nil {
		t.Fatal("expected shape mismatch to return an error")
	}
	if !strings.Contains(err.Error(), "broadcast") {
		t.Fatalf("expected MLX error details, got %v", err)
	}
}

func TestRuntimeConcurrentBuildAndEval(t *testing.T) {
	tests := []struct {
		name string
		set  func() error
	}{
		{name: "cpu", set: SetDefaultCPU},
		{name: "gpu", set: SetDefaultGPU},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.set(); err != nil {
				t.Fatal(err)
			}

			const workers = 8
			const iterations = 500

			var wg sync.WaitGroup
			errs := make(chan error, workers)
			for worker := 0; worker < workers; worker++ {
				wg.Add(1)
				go func(worker int) {
					defer wg.Done()
					for i := 0; i < iterations; i++ {
						if err := buildAndEvalOnce(worker, i); err != nil {
							errs <- err
							return
						}
					}
				}(worker)
			}
			wg.Wait()
			close(errs)

			for err := range errs {
				t.Fatal(err)
			}
		})
	}
}

func TestRuntimeConcurrentErrorMessagesAreThreadLocal(t *testing.T) {
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}

	const workers = 8
	const iterations = 400

	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				if err := failWithDistinctMLXError(worker, i); err != nil {
					errs <- err
					return
				}
			}
		}(worker)
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		t.Fatal(err)
	}
}

func TestRuntimeModelOps(t *testing.T) {
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}

	logits := mustNewFloat32(t, []float32{1, 2, 3, 1, 3, 2}, []int{2, 3})
	defer logits.Close()

	probs := mustSoftmaxAxis(t, logits, 1, true)
	defer probs.Close()
	assertShape(t, probs, []int{2, 3})
	probData, err := probs.Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, probData[0]+probData[1]+probData[2], 1)
	assertClose(t, probData[3]+probData[4]+probData[5], 1)

	prediction := mustArgmaxAxis(t, probs, 1, false)
	defer prediction.Close()
	predictionInt := mustAsType(t, prediction, Int32)
	defer predictionInt.Close()
	assertInt32Data(t, predictionInt, []int32{2, 1})

	values := mustNewFloat32(t, []float32{10, 20, 30, 40}, []int{4})
	defer values.Close()
	indices := mustNewInt32(t, []int32{3, 1, 0}, []int{3})
	defer indices.Close()

	taken := mustTake(t, values, indices)
	defer taken.Close()
	assertFloat32Data(t, taken, []float32{40, 20, 10})

	a := mustNewFloat32(t, []float32{1, 2}, []int{1, 2})
	defer a.Close()
	b := mustNewFloat32(t, []float32{3, 4}, []int{1, 2})
	defer b.Close()

	concat := mustConcatenateAxis(t, []Array{a, b}, 0)
	defer concat.Close()
	assertShape(t, concat, []int{2, 2})
	assertFloat32Data(t, concat, []float32{1, 2, 3, 4})

	stack := mustStackAxis(t, []Array{a, b}, 0)
	defer stack.Close()
	assertShape(t, stack, []int{2, 1, 2})
	assertFloat32Data(t, stack, []float32{1, 2, 3, 4})

	threshold := mustFull(t, []int{4}, 25, Float32)
	defer threshold.Close()
	mask := mustGreater(t, values, threshold)
	defer mask.Close()
	assertBoolData(t, mask, []bool{false, false, true, true})

	fallback := mustZeros(t, []int{4}, Float32)
	defer fallback.Close()
	selected := mustWhere(t, mask, values, fallback)
	defer selected.Close()
	assertFloat32Data(t, selected, []float32{0, 0, 30, 40})
}

func TestRuntimeNNAndOptimizerHelpers(t *testing.T) {
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}

	x := mustNewFloat32(t, []float32{1, 2, 3, 4}, []int{2, 2})
	defer x.Close()
	weight := mustNewFloat32(t, []float32{1, 2}, []int{2, 1})
	defer weight.Close()
	bias := mustNewFloat32(t, []float32{0.5}, []int{1, 1})
	defer bias.Close()
	target := mustNewFloat32(t, []float32{6, 10}, []int{2, 1})
	defer target.Close()

	predictions, err := Linear(x, weight, bias)
	if err != nil {
		t.Fatal(err)
	}
	defer predictions.Close()
	assertFloat32Data(t, predictions, []float32{5.5, 11.5})

	mse, err := MSELoss(predictions, target)
	if err != nil {
		t.Fatal(err)
	}
	defer mse.Close()
	assertFloat32Close(t, mse, []float32{1.25})

	logits := mustNewFloat32(t, []float32{1, 2, 3, 1, 3, 2}, []int{2, 3})
	defer logits.Close()
	classes := mustNewInt32(t, []int32{2, 1}, []int{2})
	defer classes.Close()
	oneHot := mustNewFloat32(t, []float32{0, 0, 1, 0, 1, 0}, []int{2, 3})
	defer oneHot.Close()

	ce, err := CrossEntropyAxis(logits, classes, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer ce.Close()
	assertFloat32Close(t, ce, []float32{0.40760595})

	oneHotCE, err := SoftmaxCrossEntropyAxis(logits, oneHot, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer oneHotCE.Close()
	assertFloat32Close(t, oneHotCE, []float32{0.40760595})

	params := mustNewFloat32(t, []float32{1, 2}, []int{2})
	defer params.Close()
	grads := mustNewFloat32(t, []float32{0.5, -1}, []int{2})
	defer grads.Close()
	next, err := SGD([]Array{params}, []Array{grads}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	defer CloseArrays(next)
	if len(next) != 1 {
		t.Fatalf("next length = %d, want 1", len(next))
	}
	assertFloat32Close(t, next[0], []float32{0.95, 2.1})
}

func TestRuntimeRandomOps(t *testing.T) {
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	if err := RandomSeed(7); err != nil {
		t.Fatal(err)
	}

	key, err := RandomKey(11)
	if err != nil {
		t.Fatal(err)
	}
	defer key.Close()

	normal, err := RandomNormal([]int{2, 3}, Float32, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer normal.Close()
	assertShape(t, normal, []int{2, 3})
	dtype, err := normal.DType()
	if err != nil {
		t.Fatal(err)
	}
	if dtype != Float32 {
		t.Fatalf("normal dtype = %s, want %s", dtype, Float32)
	}

	randint, err := RandomRandint([]int{2, 2}, Int32, 0, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer randint.Close()
	assertShape(t, randint, []int{2, 2})
	if _, err := randint.Int32Data(); err != nil {
		t.Fatal(err)
	}
}

func TestRuntimeSaveLoad(t *testing.T) {
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	file := filepath.Join(dir, "weights.npy")

	a := mustNewFloat32(t, []float32{1, 2, 3, 4}, []int{2, 2})
	defer a.Close()
	if err := Save(file, a); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(file); err != nil {
		t.Fatal(err)
	}

	loaded, err := Load(file)
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	assertShape(t, loaded, []int{2, 2})
	assertFloat32Data(t, loaded, []float32{1, 2, 3, 4})
}

func TestRuntimeClosureApply(t *testing.T) {
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}

	closure, err := NewClosure(func(inputs []Array) ([]Array, error) {
		out, err := Multiply(inputs[0], inputs[0])
		if err != nil {
			return nil, err
		}
		return []Array{out}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer closure.Close()

	input := mustNewFloat32(t, []float32{2, 3, 4}, []int{3})
	defer input.Close()

	outputs, err := closure.Apply(input)
	if err != nil {
		t.Fatal(err)
	}
	defer closeArrays(outputs)

	if len(outputs) != 1 {
		t.Fatalf("outputs length = %d, want 1", len(outputs))
	}
	assertFloat32Data(t, outputs[0], []float32{4, 9, 16})
}

func TestRuntimeValueAndGrad(t *testing.T) {
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}

	vg, err := NewValueAndGrad(func(inputs []Array) ([]Array, error) {
		squared, err := Square(inputs[0])
		if err != nil {
			return nil, err
		}
		loss, err := Mean(squared, false)
		if err != nil {
			_ = squared.Close()
			return nil, err
		}
		_ = squared.Close()
		return []Array{loss}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer vg.Close()

	x := mustNewFloat32(t, []float32{1, 2, 3}, []int{3})
	defer x.Close()

	values, grads, err := vg.Apply(x)
	if err != nil {
		t.Fatal(err)
	}
	defer closeArrays(values)
	defer closeArrays(grads)

	if len(values) != 1 {
		t.Fatalf("values length = %d, want 1", len(values))
	}
	if len(grads) != 1 {
		t.Fatalf("grads length = %d, want 1", len(grads))
	}
	assertFloat32Close(t, values[0], []float32{14.0 / 3.0})
	assertFloat32Close(t, grads[0], []float32{2.0 / 3.0, 4.0 / 3.0, 2})
}

func buildAndEvalOnce(worker, iteration int) error {
	left, err := NewFloat32([]float32{float32(worker), float32(iteration)}, []int{2})
	if err != nil {
		return fmt.Errorf("worker %d iteration %d new left: %w", worker, iteration, err)
	}
	defer left.Close()

	right, err := NewFloat32([]float32{1, 2}, []int{2})
	if err != nil {
		return fmt.Errorf("worker %d iteration %d new right: %w", worker, iteration, err)
	}
	defer right.Close()

	sum, err := Add(left, right)
	if err != nil {
		return fmt.Errorf("worker %d iteration %d add: %w", worker, iteration, err)
	}
	defer sum.Close()

	if err := sum.Eval(); err != nil {
		return fmt.Errorf("worker %d iteration %d eval: %w", worker, iteration, err)
	}

	data, err := sum.Float32Data()
	if err != nil {
		return fmt.Errorf("worker %d iteration %d data: %w", worker, iteration, err)
	}
	expected := []float32{float32(worker) + 1, float32(iteration) + 2}
	if !reflect.DeepEqual(data, expected) {
		return fmt.Errorf("worker %d iteration %d data = %v, want %v", worker, iteration, data, expected)
	}
	return nil
}

func failWithDistinctMLXError(worker, iteration int) error {
	if iteration%2 == 0 {
		left, err := NewFloat32([]float32{1, 2, 3}, []int{3})
		if err != nil {
			return fmt.Errorf("worker %d iteration %d new add left: %w", worker, iteration, err)
		}
		defer left.Close()

		right, err := NewFloat32([]float32{1, 2, 3, 4, 5}, []int{5})
		if err != nil {
			return fmt.Errorf("worker %d iteration %d new add right: %w", worker, iteration, err)
		}
		defer right.Close()

		_, err = Add(left, right)
		if err == nil {
			return fmt.Errorf("worker %d iteration %d add succeeded unexpectedly", worker, iteration)
		}
		text := err.Error()
		if !strings.Contains(text, "broadcast") || strings.Contains(text, "reshape") {
			return fmt.Errorf("worker %d iteration %d add error = %q", worker, iteration, text)
		}
		return nil
	}

	input, err := NewFloat32([]float32{1, 2, 3}, []int{3})
	if err != nil {
		return fmt.Errorf("worker %d iteration %d new reshape input: %w", worker, iteration, err)
	}
	defer input.Close()

	_, err = Reshape(input, []int{4, 4})
	if err == nil {
		return fmt.Errorf("worker %d iteration %d reshape succeeded unexpectedly", worker, iteration)
	}
	text := err.Error()
	if !strings.Contains(text, "reshape") || strings.Contains(text, "broadcast") {
		return fmt.Errorf("worker %d iteration %d reshape error = %q", worker, iteration, text)
	}
	return nil
}

func mustNewFloat32(t *testing.T, data []float32, shape []int) Array {
	t.Helper()
	arr, err := NewFloat32(data, shape)
	return mustOp(t, arr, err)
}

func mustNewInt32(t *testing.T, data []int32, shape []int) Array {
	t.Helper()
	arr, err := NewInt32(data, shape)
	return mustOp(t, arr, err)
}

func mustZeros(t *testing.T, shape []int, dtype DType) Array {
	t.Helper()
	arr, err := Zeros(shape, dtype)
	return mustOp(t, arr, err)
}

func mustOnes(t *testing.T, shape []int, dtype DType) Array {
	t.Helper()
	arr, err := Ones(shape, dtype)
	return mustOp(t, arr, err)
}

func mustFull(t *testing.T, shape []int, value float64, dtype DType) Array {
	t.Helper()
	arr, err := Full(shape, value, dtype)
	return mustOp(t, arr, err)
}

func mustAdd(t *testing.T, a, b Array) Array {
	t.Helper()
	arr, err := Add(a, b)
	return mustOp(t, arr, err)
}

func mustSubtract(t *testing.T, a, b Array) Array {
	t.Helper()
	arr, err := Subtract(a, b)
	return mustOp(t, arr, err)
}

func mustMultiply(t *testing.T, a, b Array) Array {
	t.Helper()
	arr, err := Multiply(a, b)
	return mustOp(t, arr, err)
}

func mustDivide(t *testing.T, a, b Array) Array {
	t.Helper()
	arr, err := Divide(a, b)
	return mustOp(t, arr, err)
}

func mustMaximum(t *testing.T, a, b Array) Array {
	t.Helper()
	arr, err := Maximum(a, b)
	return mustOp(t, arr, err)
}

func mustMinimum(t *testing.T, a, b Array) Array {
	t.Helper()
	arr, err := Minimum(a, b)
	return mustOp(t, arr, err)
}

func mustGreater(t *testing.T, a, b Array) Array {
	t.Helper()
	arr, err := Greater(a, b)
	return mustOp(t, arr, err)
}

func mustWhere(t *testing.T, condition, x, y Array) Array {
	t.Helper()
	arr, err := Where(condition, x, y)
	return mustOp(t, arr, err)
}

func mustMatmul(t *testing.T, a, b Array) Array {
	t.Helper()
	arr, err := Matmul(a, b)
	return mustOp(t, arr, err)
}

func mustReshape(t *testing.T, a Array, shape []int) Array {
	t.Helper()
	arr, err := Reshape(a, shape)
	return mustOp(t, arr, err)
}

func mustTranspose(t *testing.T, a Array) Array {
	t.Helper()
	arr, err := Transpose(a)
	return mustOp(t, arr, err)
}

func mustBroadcastTo(t *testing.T, a Array, shape []int) Array {
	t.Helper()
	arr, err := BroadcastTo(a, shape)
	return mustOp(t, arr, err)
}

func mustExpandDims(t *testing.T, a Array, axis int) Array {
	t.Helper()
	arr, err := ExpandDims(a, axis)
	return mustOp(t, arr, err)
}

func mustSqueezeAxis(t *testing.T, a Array, axis int) Array {
	t.Helper()
	arr, err := SqueezeAxis(a, axis)
	return mustOp(t, arr, err)
}

func mustFlatten(t *testing.T, a Array, startAxis, endAxis int) Array {
	t.Helper()
	arr, err := Flatten(a, startAxis, endAxis)
	return mustOp(t, arr, err)
}

func mustNegative(t *testing.T, a Array) Array {
	t.Helper()
	arr, err := Negative(a)
	return mustOp(t, arr, err)
}

func mustSquare(t *testing.T, a Array) Array {
	t.Helper()
	arr, err := Square(a)
	return mustOp(t, arr, err)
}

func mustSqrt(t *testing.T, a Array) Array {
	t.Helper()
	arr, err := Sqrt(a)
	return mustOp(t, arr, err)
}

func mustSigmoid(t *testing.T, a Array) Array {
	t.Helper()
	arr, err := Sigmoid(a)
	return mustOp(t, arr, err)
}

func mustReLU(t *testing.T, a Array) Array {
	t.Helper()
	arr, err := ReLU(a)
	return mustOp(t, arr, err)
}

func mustSum(t *testing.T, a Array, keepdims bool) Array {
	t.Helper()
	arr, err := Sum(a, keepdims)
	return mustOp(t, arr, err)
}

func mustMean(t *testing.T, a Array, keepdims bool) Array {
	t.Helper()
	arr, err := Mean(a, keepdims)
	return mustOp(t, arr, err)
}

func mustSoftmaxAxis(t *testing.T, a Array, axis int, precise bool) Array {
	t.Helper()
	arr, err := SoftmaxAxis(a, axis, precise)
	return mustOp(t, arr, err)
}

func mustArgmaxAxis(t *testing.T, a Array, axis int, keepdims bool) Array {
	t.Helper()
	arr, err := ArgmaxAxis(a, axis, keepdims)
	return mustOp(t, arr, err)
}

func mustTake(t *testing.T, a, indices Array) Array {
	t.Helper()
	arr, err := Take(a, indices)
	return mustOp(t, arr, err)
}

func mustConcatenateAxis(t *testing.T, arrays []Array, axis int) Array {
	t.Helper()
	arr, err := ConcatenateAxis(arrays, axis)
	return mustOp(t, arr, err)
}

func mustStackAxis(t *testing.T, arrays []Array, axis int) Array {
	t.Helper()
	arr, err := StackAxis(arrays, axis)
	return mustOp(t, arr, err)
}

func mustAsType(t *testing.T, a Array, dtype DType) Array {
	t.Helper()
	arr, err := AsType(a, dtype)
	return mustOp(t, arr, err)
}

func mustOp(t *testing.T, arr Array, err error) Array {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	return arr
}

func assertShape(t *testing.T, arr Array, want []int) {
	t.Helper()
	if got := arr.Shape(); !reflect.DeepEqual(got, want) {
		t.Fatalf("shape = %v, want %v", got, want)
	}
}

func assertFloat32Data(t *testing.T, arr Array, want []float32) {
	t.Helper()
	got, err := arr.Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("data = %v, want %v", got, want)
	}
}

func assertFloat32Close(t *testing.T, arr Array, want []float32) {
	t.Helper()
	got, err := arr.Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("data length = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("data[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func assertClose(t *testing.T, got, want float32) {
	t.Helper()
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Fatalf("value = %v, want %v", got, want)
	}
}

func assertInt32Data(t *testing.T, arr Array, want []int32) {
	t.Helper()
	got, err := arr.Int32Data()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("data = %v, want %v", got, want)
	}
}

func assertBoolData(t *testing.T, arr Array, want []bool) {
	t.Helper()
	got, err := arr.BoolData()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("data = %v, want %v", got, want)
	}
}

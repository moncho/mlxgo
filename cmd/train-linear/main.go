//go:build mlx

package main

import (
	"fmt"
	"log"

	"github.com/moncho/mlxgo"
)

func main() {
	x := must(mlx.NewFloat32([]float32{
		0, 0,
		1, 0,
		0, 1,
		2, 3,
	}, []int{4, 2}))
	defer x.Close()

	y := must(mlx.NewFloat32([]float32{0.5, 2.5, -0.5, 1.5}, []int{4, 1}))
	defer y.Close()

	xT := must(mlx.Transpose(x))
	defer xT.Close()

	w := must(mlx.Zeros([]int{2, 1}, mlx.Float32))
	b := must(mlx.Zeros([]int{1, 1}, mlx.Float32))

	twoOverN := must(mlx.NewScalarFloat32(0.5))
	defer twoOverN.Close()
	learningRate := must(mlx.NewScalarFloat32(0.1))
	defer learningRate.Close()

	for step := 0; step < 80; step++ {
		pred := must(mlx.Linear(x, w, b))
		err := must(mlx.Subtract(pred, y))
		loss := must(mlx.MSELoss(pred, y))

		gradWBase := must(mlx.Matmul(xT, err))
		gradW := must(mlx.Multiply(gradWBase, twoOverN))
		gradBBase := must(mlx.Sum(err, false))
		gradB := must(mlx.Multiply(gradBBase, twoOverN))

		nextParams := mustSlice(mlx.SGDWithLearningRate([]mlx.Array{w, b}, []mlx.Array{gradW, gradB}, learningRate))
		nextW := nextParams[0]
		nextB := nextParams[1]

		if err := mlx.Eval(nextW, nextB); err != nil {
			log.Fatal(err)
		}

		if step%20 == 0 || step == 79 {
			lossData, err := loss.Float32Data()
			if err != nil {
				log.Fatal(err)
			}
			fmt.Printf("step=%02d loss=%.6f\n", step, lossData[0])
		}

		_ = w.Close()
		_ = b.Close()
		w = nextW
		b = nextB

		_ = mlx.CloseArrays([]mlx.Array{pred, err, loss, gradWBase, gradW, gradBBase, gradB})
	}
	defer w.Close()
	defer b.Close()

	weights, err := w.Float32Data()
	if err != nil {
		log.Fatal(err)
	}
	bias, err := b.Float32Data()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("weights=%v bias=%.6f\n", weights, bias[0])
}

func must(arr mlx.Array, err error) mlx.Array {
	if err != nil {
		log.Fatal(err)
	}
	return arr
}

func mustSlice(arrays []mlx.Array, err error) []mlx.Array {
	if err != nil {
		log.Fatal(err)
	}
	return arrays
}

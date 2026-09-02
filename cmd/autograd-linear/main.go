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

	lossAndGrad, err := mlx.NewValueAndGrad(func(params []mlx.Array) ([]mlx.Array, error) {
		w := params[0]
		b := params[1]

		pred, err := mlx.Linear(x, w, b)
		if err != nil {
			return nil, err
		}
		loss, err := mlx.MSELoss(pred, y)
		_ = pred.Close()
		if err != nil {
			return nil, err
		}
		return []mlx.Array{loss}, nil
	}, 0, 1)
	if err != nil {
		log.Fatal(err)
	}
	defer lossAndGrad.Close()

	w := must(mlx.Zeros([]int{2, 1}, mlx.Float32))
	b := must(mlx.Zeros([]int{1, 1}, mlx.Float32))
	learningRate := must(mlx.NewScalarFloat32(0.1))
	defer learningRate.Close()

	for step := 0; step < 80; step++ {
		values, grads, err := lossAndGrad.Apply(w, b)
		if err != nil {
			log.Fatal(err)
		}

		nextParams := mustSlice(mlx.SGDWithLearningRate([]mlx.Array{w, b}, grads, learningRate))
		nextW := nextParams[0]
		nextB := nextParams[1]

		if err := mlx.Eval(nextW, nextB); err != nil {
			log.Fatal(err)
		}

		if step%20 == 0 || step == 79 {
			lossData, err := values[0].Float32Data()
			if err != nil {
				log.Fatal(err)
			}
			fmt.Printf("step=%02d loss=%.6f\n", step, lossData[0])
		}

		_ = w.Close()
		_ = b.Close()
		w = nextW
		b = nextB

		_ = mlx.CloseArrays(values)
		_ = mlx.CloseArrays(grads)
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

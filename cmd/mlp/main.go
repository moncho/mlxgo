//go:build mlx

package main

import (
	"fmt"
	"log"

	"github.com/moncho/mlxgo"
)

func main() {
	if err := mlx.SetDefaultCPU(); err != nil {
		log.Fatal(err)
	}

	x := must(mlx.NewFloat32([]float32{0.7, -1.2, 0.3, 2.0}, []int{1, 4}))
	defer x.Close()

	w1 := must(mlx.NewFloat32([]float32{
		0.8, -0.4, 0.2, 0.1, -0.6,
		-0.5, 0.9, 0.3, -0.8, 0.4,
		0.7, 0.2, -0.3, 0.5, 0.1,
		0.1, -0.7, 0.6, 0.9, -0.2,
	}, []int{4, 5}))
	defer w1.Close()

	b1 := must(mlx.NewFloat32([]float32{0.1, -0.2, 0.05, 0.15, -0.1}, []int{1, 5}))
	defer b1.Close()

	w2 := must(mlx.NewFloat32([]float32{
		0.6, -0.2, 0.1,
		-0.4, 0.7, 0.2,
		0.3, 0.1, -0.5,
		0.2, -0.6, 0.8,
		-0.1, 0.4, 0.5,
	}, []int{5, 3}))
	defer w2.Close()

	b2 := must(mlx.NewFloat32([]float32{0.05, -0.05, 0.1}, []int{1, 3}))
	defer b2.Close()

	hBiased := must(mlx.Linear(x, w1, b1))
	defer hBiased.Close()

	h := must(mlx.ReLU(hBiased))
	defer h.Close()

	logits := must(mlx.Linear(h, w2, b2))
	defer logits.Close()

	probs := must(mlx.SoftmaxAxis(logits, 1, true))
	defer probs.Close()

	prediction := must(mlx.ArgmaxAxis(probs, 1, false))
	defer prediction.Close()
	predictionInt := must(mlx.AsType(prediction, mlx.Int32))
	defer predictionInt.Close()

	probData, err := probs.Float32Data()
	if err != nil {
		log.Fatal(err)
	}
	classData, err := predictionInt.Int32Data()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("probabilities=%v\n", probData)
	fmt.Printf("predicted_class=%d\n", classData[0])
}

func must(arr mlx.Array, err error) mlx.Array {
	if err != nil {
		log.Fatal(err)
	}
	return arr
}

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

	features := must(mlx.NewFloat32([]float32{
		0, 0,
		1, 0,
		0, 1,
		2, 3,
	}, []int{4, 2}))
	defer features.Close()

	weights := must(mlx.NewFloat32([]float32{2, -1}, []int{2, 1}))
	defer weights.Close()

	bias := must(mlx.NewFloat32([]float32{0.5, 0.5, 0.5, 0.5}, []int{4, 1}))
	defer bias.Close()

	labels := must(mlx.NewFloat32([]float32{0.25, 2.25, -0.25, 1.25}, []int{4, 1}))
	defer labels.Close()

	predictions := must(mlx.Linear(features, weights, bias))
	defer predictions.Close()

	loss := must(mlx.MSELoss(predictions, labels))
	defer loss.Close()

	predictionData, err := predictions.Float32Data()
	if err != nil {
		log.Fatal(err)
	}
	lossData, err := loss.Float32Data()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("predictions=%v\n", predictionData)
	fmt.Printf("mse=%v\n", lossData[0])
}

func must(arr mlx.Array, err error) mlx.Array {
	if err != nil {
		log.Fatal(err)
	}
	return arr
}

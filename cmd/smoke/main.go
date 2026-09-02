//go:build mlx

package main

import (
	"fmt"
	"log"

	"github.com/moncho/mlxgo"
)

func main() {
	a, err := mlx.NewFloat32([]float32{1, 2, 3, 4}, []int{2, 2})
	if err != nil {
		log.Fatal(err)
	}
	defer a.Close()

	b, err := mlx.NewFloat32([]float32{10, 20, 30, 40}, []int{2, 2})
	if err != nil {
		log.Fatal(err)
	}
	defer b.Close()

	c, err := mlx.Add(a, b)
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()

	data, err := c.Float32Data()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("shape=%v data=%v\n", c.Shape(), data)
}

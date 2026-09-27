//go:build mlx

package mlx

/*
#cgo darwin,arm64 CFLAGS: -I/opt/homebrew/include
#cgo darwin,arm64 CXXFLAGS: -I/opt/homebrew/include -std=c++17
#cgo darwin,arm64 LDFLAGS: -L/opt/homebrew/lib -lmlxc
*/
import "C"

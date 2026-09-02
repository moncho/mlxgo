// Package mlx provides a small Go wrapper around Apple's MLX through the
// official mlx-c bridge.
//
// Build real MLX bindings with:
//
//	CGO_ENABLED=1 go test -tags mlx ./...
//
// The default build exposes the same public API as stubs so editors and normal
// Go tooling work without the native MLX dependency.
//
// The package includes small helpers for common model code: Linear, MSELoss,
// CrossEntropyAxis, SoftmaxCrossEntropyAxis, and SGD updates.
//
// NewValueAndGrad wraps MLX's closure-based autograd. Callback inputs are
// borrowed arrays; callback outputs are transferred to MLX.
package mlx

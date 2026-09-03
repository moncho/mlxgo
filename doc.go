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
// Operations use GPU index 0 by default. Call SetDefaultCPU for CPU execution
// or SetDefaultDevice to choose CPU or GPU index 0 explicitly.
// Native MLX calls are serialized on a dedicated OS thread so Go callers can
// build and evaluate arrays from ordinary goroutines without stream-affinity
// failures. Batch can be used to amortize dispatcher overhead across a sequence
// of MLX calls.
//
// NewValueAndGrad wraps MLX's closure-based autograd. Callback inputs are
// temporary handles managed by the wrapper; callback outputs are transferred to
// MLX. Callbacks run on the MLX worker thread, so they must not block on
// goroutines or work that needs to call MLX.
package mlx

GO ?= go
GOCACHE ?= $(CURDIR)/.gocache

.PHONY: test test-native test-runtime smoke linear mlp train-linear autograd-linear

test:
	GOCACHE="$(GOCACHE)" $(GO) test ./...

test-native:
	GOCACHE="$(GOCACHE)" $(GO) test -tags mlx ./...

test-runtime:
	GOCACHE="$(GOCACHE)" CGO_ENABLED=1 $(GO) test -tags "mlx mlxruntime" ./...

smoke:
	GOCACHE="$(GOCACHE)" CGO_ENABLED=1 $(GO) run -tags mlx ./cmd/smoke

linear:
	GOCACHE="$(GOCACHE)" CGO_ENABLED=1 $(GO) run -tags mlx ./cmd/linear

mlp:
	GOCACHE="$(GOCACHE)" CGO_ENABLED=1 $(GO) run -tags mlx ./cmd/mlp

train-linear:
	GOCACHE="$(GOCACHE)" CGO_ENABLED=1 $(GO) run -tags mlx ./cmd/train-linear

autograd-linear:
	GOCACHE="$(GOCACHE)" CGO_ENABLED=1 $(GO) run -tags mlx ./cmd/autograd-linear

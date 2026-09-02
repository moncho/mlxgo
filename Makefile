GO ?= go
GOCACHE ?= $(CURDIR)/.gocache

.PHONY: test test-native test-runtime vet vet-native test-race test-race-native smoke linear mlp train-linear autograd-linear

test:
	GOCACHE="$(GOCACHE)" $(GO) test ./...

test-native:
	GOCACHE="$(GOCACHE)" $(GO) test -tags mlx ./...

test-runtime:
	GOCACHE="$(GOCACHE)" CGO_ENABLED=1 $(GO) test -tags "mlx mlxruntime" ./...

vet:
	GOCACHE="$(GOCACHE)" $(GO) vet ./...

vet-native:
	GOCACHE="$(GOCACHE)" CGO_ENABLED=1 $(GO) vet -tags mlx ./...

test-race:
	GOCACHE="$(GOCACHE)" $(GO) test -race ./...

test-race-native:
	GOCACHE="$(GOCACHE)" CGO_ENABLED=1 $(GO) test -race -tags mlx ./...

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

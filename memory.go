package mlx

// MemoryUsage reports process-wide MLX allocator counters, not Go heap usage
// or process RSS. Evaluate pending work before sampling. Cached bytes are free
// allocations retained for reuse; peak tracks active bytes since the last reset.
type MemoryUsage struct {
	ActiveBytes uint64
	CacheBytes  uint64
	PeakBytes   uint64
}

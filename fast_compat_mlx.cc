//go:build mlx

#include "fast_compat.h"

// mlx-c added force_fused after 0.6.0. Overload resolution checks the installed
// header at compile time; no version guess or incompatible function cast.
using LegacyAttention = int (*)(mlx_array*, mlx_array, mlx_array, mlx_array,
    float, const char*, mlx_array, mlx_array, mlx_stream);
using CurrentAttention = int (*)(mlx_array*, mlx_array, mlx_array, mlx_array,
    float, const char*, mlx_array, mlx_array, bool, mlx_stream);

static int attention(LegacyAttention fn, mlx_array* out, mlx_array q,
    mlx_array k, mlx_array v, float scale, const char* mask, mlx_stream stream) {
  return fn(out, q, k, v, scale, mask, mlx_array{}, mlx_array{}, stream);
}
static int attention(CurrentAttention fn, mlx_array* out, mlx_array q,
    mlx_array k, mlx_array v, float scale, const char* mask, mlx_stream stream) {
  return fn(out, q, k, v, scale, mask, mlx_array{}, mlx_array{}, false, stream);
}
extern "C" int mlxgo_sdpa(mlx_array* out, mlx_array q, mlx_array k, mlx_array v,
    float scale, const char* mask, mlx_stream stream) {
  return attention(mlx_fast_scaled_dot_product_attention, out, q, k, v, scale, mask, stream);
}

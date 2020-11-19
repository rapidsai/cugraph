namespace cugraph {
namespace detail {

extern template __device__ float parallel_prefix_sum(int32_t, int32_t const*, float const*);
extern template __device__ double parallel_prefix_sum(int32_t, int32_t const*, double const*);
extern template __device__ float parallel_prefix_sum(int64_t, int32_t const*, float const*);
extern template __device__ double parallel_prefix_sum(int64_t, int32_t const*, double const*);
extern template __device__ float parallel_prefix_sum(int64_t, int64_t const*, float const*);
extern template __device__ double parallel_prefix_sum(int64_t, int64_t const*, double const*);

extern template void offsets_to_indices<int, int>(int const*, int, int*);
extern template void offsets_to_indices<long, int>(long const*, int, int*);
extern template void offsets_to_indices<long, long>(long const*, long, long*);

extern template __global__ void offsets_to_indices_kernel<int, int>(int const*, int, int*);
extern template __global__ void offsets_to_indices_kernel<long, int>(long const*, int, int*);
extern template __global__ void offsets_to_indices_kernel<long, long>(long const*, long, long*);

}  // namespace detail
}  // namespace cugraph

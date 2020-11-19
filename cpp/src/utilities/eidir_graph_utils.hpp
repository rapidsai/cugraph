namespace cugraph {
namespace detail {

template __device__ float parallel_prefix_sum(int32_t, int32_t const*, float const*);
template __device__ double parallel_prefix_sum(int32_t, int32_t const*, double const*);
template __device__ float parallel_prefix_sum(int64_t, int32_t const*, float const*);
template __device__ double parallel_prefix_sum(int64_t, int32_t const*, double const*);
template __device__ float parallel_prefix_sum(int64_t, int64_t const*, float const*);
template __device__ double parallel_prefix_sum(int64_t, int64_t const*, double const*);

template void offsets_to_indices<int, int>(int const*, int, int*);
template void offsets_to_indices<long, int>(long const*, int, int*);
template void offsets_to_indices<long, long>(long const*, long, long*);

template __global__ void offsets_to_indices_kernel<int, int>(int const*, int, int*);
template __global__ void offsets_to_indices_kernel<long, int>(long const*, int, int*);
template __global__ void offsets_to_indices_kernel<long, long>(long const*, long, long*);

}  // namespace detail
}  // namespace cugraph

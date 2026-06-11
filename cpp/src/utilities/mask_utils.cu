/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit instantiations for cugraph/utilities/mask_utils.cuh.
 */

#include <cugraph/export.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cstdint>

namespace cugraph {
namespace detail {

size_t count_set_bits(rmm::cuda_stream_view stream_view,
                      uint32_t const* mask_first,
                      size_t num_bits)
{
  return thrust::transform_reduce(
    rmm::exec_policy(stream_view),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(packed_bool_size(num_bits)),
    cuda::proclaim_return_type<size_t>([mask_first, num_bits] __device__(size_t i) -> size_t {
      auto word = *(mask_first + i);
      if ((i + 1) * packed_bools_per_word() > num_bits) {
        word &= packed_bool_partial_mask(num_bits % packed_bools_per_word());
      }
      return static_cast<size_t>(__popc(word));
    }),
    size_t{0},
    cuda::std::plus<size_t>{});
}

size_t count_set_bits(raft::handle_t const& handle, uint32_t const* mask_first, size_t num_bits)
{
  return count_set_bits(handle.get_stream(), mask_first, num_bits);
}

template <typename InputIterator, typename OutputIterator>
OutputIterator copy_if_mask_set_impl(raft::handle_t const& handle,
                                     InputIterator input_first,
                                     InputIterator input_last,
                                     uint32_t const* mask_first,
                                     OutputIterator output_first)
{
  return thrust::copy_if(
    handle.get_thrust_policy(),
    input_first,
    input_last,
    cuda::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                  check_bit_set_t<uint32_t const*, size_t>{mask_first, size_t{0}}),
    output_first,
    is_equal_t<bool>{true});
}

template <typename InputIterator, typename OutputIterator>
OutputIterator copy_if_mask_unset_impl(raft::handle_t const& handle,
                                       InputIterator input_first,
                                       InputIterator input_last,
                                       uint32_t const* mask_first,
                                       OutputIterator output_first)
{
  return thrust::copy_if(
    handle.get_thrust_policy(),
    input_first,
    input_last,
    cuda::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                  check_bit_set_t<uint32_t const*, size_t>{mask_first, size_t{0}}),
    output_first,
    is_equal_t<bool>{false});
}

#define CUGRAPH_COPY_IF_MASK_SCALAR_INST(T)                                                \
  template CUGRAPH_EXPORT T* copy_if_mask_set_impl<T*, T*>(raft::handle_t const& handle,   \
                                                           T* input_first,                 \
                                                           T* input_last,                  \
                                                           uint32_t const* mask_first,     \
                                                           T* output_first);               \
  template CUGRAPH_EXPORT T* copy_if_mask_unset_impl<T*, T*>(raft::handle_t const& handle, \
                                                             T* input_first,               \
                                                             T* input_last,                \
                                                             uint32_t const* mask_first,   \
                                                             T* output_first)

CUGRAPH_COPY_IF_MASK_SCALAR_INST(std::int32_t);
CUGRAPH_COPY_IF_MASK_SCALAR_INST(std::int64_t);
CUGRAPH_COPY_IF_MASK_SCALAR_INST(float);
CUGRAPH_COPY_IF_MASK_SCALAR_INST(double);

#undef CUGRAPH_COPY_IF_MASK_SCALAR_INST

}  // namespace detail
}  // namespace cugraph

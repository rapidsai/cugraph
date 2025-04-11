/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/core/handle.hpp>

#include <cuda/functional>
#include <cuda/std/iterator>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace cugraph {

namespace detail {

template <typename MaskIterator>  // should be packed bool
__device__ size_t count_set_bits(MaskIterator mask_first, size_t start_offset, size_t num_bits)
{
  static_assert(
    std::is_same_v<typename thrust::iterator_traits<MaskIterator>::value_type, uint32_t>);

  size_t ret{0};

  mask_first   = mask_first + packed_bool_offset(start_offset);
  start_offset = start_offset % packed_bools_per_word();
  if (start_offset != 0) {
    auto mask = ~packed_bool_partial_mask(start_offset);
    if (start_offset + num_bits < packed_bools_per_word()) {
      mask &= packed_bool_partial_mask(start_offset + num_bits);
    }
    ret += __popc(*mask_first & mask);
    num_bits -= __popc(mask);
    ++mask_first;
  }

  return thrust::transform_reduce(
    thrust::seq,
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(packed_bool_size(num_bits)),
    [mask_first, num_bits] __device__(size_t i) {
      auto word = *(mask_first + i);
      if ((i + 1) * packed_bools_per_word() > num_bits) {
        word &= packed_bool_partial_mask(num_bits % packed_bools_per_word());
      }
      return static_cast<size_t>(__popc(word));
    },
    ret,
    thrust::plus<size_t>{});
}

// @p n starts from 1
template <typename MaskIterator>  // should be packed bool
__device__ size_t
find_nth_set_bits(MaskIterator mask_first, size_t start_offset, size_t num_bits, size_t n)
{
  static_assert(
    std::is_same_v<typename thrust::iterator_traits<MaskIterator>::value_type, uint32_t>);
  assert(n >= 1);
  assert(n <= num_bits);

  size_t pos{0};

  mask_first   = mask_first + packed_bool_offset(start_offset);
  start_offset = start_offset % packed_bools_per_word();
  if (start_offset != 0) {
    auto mask = ~packed_bool_partial_mask(start_offset);
    if (start_offset + num_bits < packed_bools_per_word()) {
      mask &= packed_bool_partial_mask(start_offset + num_bits);
    }
    auto word         = *mask_first & mask;
    auto num_set_bits = __popc(word);
    if (n <= num_set_bits) {
      return static_cast<size_t>(__fns(word, start_offset, n)) - start_offset;
    }
    pos += __popc(mask);
    n -= num_set_bits;
    ++mask_first;
  }

  while (pos < num_bits) {
    auto mask         = ((num_bits - pos) >= packed_bools_per_word())
                          ? packed_bool_full_mask()
                          : packed_bool_partial_mask(num_bits - pos);
    auto word         = *mask_first & mask;
    auto num_set_bits = __popc(word);
    if (n <= num_set_bits) { return pos + static_cast<size_t>(__fns(word, 0, n)); }
    pos += __popc(mask);
    n -= num_set_bits;
    ++mask_first;
  }

  return std::numeric_limits<size_t>::max();
}

template <typename InputIterator,
          typename MaskIterator,  // should be packed bool
          typename OutputIterator,
          typename input_value_type =
            typename thrust::iterator_traits<InputIterator>::value_type,  // for packed bool support
          typename output_value_type = typename thrust::iterator_traits<
            OutputIterator>::value_type>  // for packed bool support
__device__ size_t copy_if_mask_set(InputIterator input_first,
                                   MaskIterator mask_first,
                                   OutputIterator output_first,
                                   size_t input_start_offset,
                                   size_t output_start_offset,
                                   size_t num_items)
{
  static_assert(
    std::is_same_v<typename thrust::iterator_traits<MaskIterator>::value_type, uint32_t>);
  static_assert(
    std::is_same_v<typename thrust::iterator_traits<InputIterator>::value_type, input_value_type> ||
    cugraph::has_packed_bool_element<InputIterator, input_value_type>());
  static_assert(std::is_same_v<typename thrust::iterator_traits<OutputIterator>::value_type,
                               output_value_type> ||
                cugraph::has_packed_bool_element<OutputIterator, output_value_type>());

  static_assert(!cugraph::has_packed_bool_element<InputIterator, input_value_type>() &&
                  !cugraph::has_packed_bool_element<OutputIterator, output_value_type>(),
                "unimplemented.");

  return static_cast<size_t>(cuda::std::distance(
    output_first + output_start_offset,
    thrust::copy_if(thrust::seq,
                    input_first + input_start_offset,
                    input_first + (input_start_offset + num_items),
                    thrust::make_transform_iterator(
                      thrust::make_counting_iterator(size_t{0}),
                      check_bit_set_t<MaskIterator, size_t>{mask_first, size_t{0}}) +
                      input_start_offset,
                    output_first + output_start_offset,
                    is_equal_t<bool>{true})));
}

template <typename MaskIterator>  // should be packed bool
size_t count_set_bits(raft::handle_t const& handle, MaskIterator mask_first, size_t num_bits)
{
  static_assert(
    std::is_same_v<typename thrust::iterator_traits<MaskIterator>::value_type, uint32_t>);

  return thrust::transform_reduce(
    handle.get_thrust_policy(),
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
    thrust::plus<size_t>{});
}

template <typename InputIterator,
          typename MaskIterator,  // should be packed bool
          typename OutputIterator>
OutputIterator copy_if_mask_set(raft::handle_t const& handle,
                                InputIterator input_first,
                                InputIterator input_last,
                                MaskIterator mask_first,
                                OutputIterator output_first)
{
  return thrust::copy_if(
    handle.get_thrust_policy(),
    input_first,
    input_last,
    thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                    check_bit_set_t<MaskIterator, size_t>{mask_first, size_t{0}}),
    output_first,
    is_equal_t<bool>{true});
}

}  // namespace detail

}  // namespace cugraph

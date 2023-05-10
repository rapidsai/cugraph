/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cugraph {

namespace detail {

template <typename MaskIterator>
size_t count_set_bits(raft::handle_t const& handle, MaskIterator mask_first, size_t num_bits)
{
  static_assert(
    std::is_same_v<typename thrust::iterator_traits<MaskIterator>::value_type, uint32_t>);

  return thrust::transform_reduce(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(packed_bool_size(num_bits)),
    [mask_first, num_bits] __device__(size_t i) {
      auto word = *(mask_first + i);
      if ((i + 1) * packed_bools_per_word() > num_bits) {
        word &= packed_bool_partial_mask(num_bits % packed_bools_per_word());
      }
      return static_cast<size_t>(__popc(word));
    },
    size_t{0},
    thrust::plus<size_t>{});
}

template <typename InputIterator, typename MaskIterator, typename OutputIterator>
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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/iterator_utils.hpp>
#include <cugraph/utilities/thrust_wrappers/scatter.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/tuple>
#include <thrust/copy.h>

#include <cstddef>
#include <type_traits>

namespace cugraph {
namespace detail {

template <typename Iterator>
inline constexpr bool is_permute_zip_iterator_v = is_thrust_zip_iterator_v<Iterator>;

template <typename ZipIterator>
inline constexpr std::size_t permute_zip_iterator_arity_v =
  cuda::std::tuple_size<typename ZipIterator::iterator_tuple>::value;

template <typename T>
void permute_in_place_impl(T* first,
                           std::size_t const* map_first,
                           std::size_t num_elements,
                           rmm::cuda_stream_view stream_view);

template <typename Iterator>
void permute_scalar_in_place(Iterator first,
                             std::size_t const* map_first,
                             std::size_t num_elements,
                             rmm::cuda_stream_view stream_view)
{
  if constexpr (scatter_supported_scalar_value_v<iterator_value_t<Iterator>>) {
    permute_in_place_impl(first, map_first, num_elements, stream_view);
  } else {
    using value_t     = iterator_value_t<Iterator>;
    auto const policy = rmm::exec_policy(stream_view);
    auto tmp_buffer   = allocate_dataframe_buffer<value_t>(num_elements, stream_view);
    auto tmp_first    = get_dataframe_buffer_begin(tmp_buffer);
    auto last         = first + num_elements;
    ::cugraph::scatter(policy, first, last, map_first, tmp_first);
    thrust::copy(policy, tmp_first, tmp_first + num_elements, first);
  }
}

template <typename ZipIterator, std::size_t I, std::size_t N>
struct permute_zip_in_place_split_impl {
  static void run(ZipIterator first,
                  ZipIterator last,
                  std::size_t const* map_first,
                  rmm::cuda_stream_view stream_view)
  {
    auto const num_elements = static_cast<std::size_t>(cuda::std::distance(first, last));
    permute_scalar_in_place(
      cuda::std::get<I>(first.get_iterator_tuple()), map_first, num_elements, stream_view);
    permute_zip_in_place_split_impl<ZipIterator, I + 1, N>::run(
      first, last, map_first, stream_view);
  }
};

template <typename ZipIterator, std::size_t I>
struct permute_zip_in_place_split_impl<ZipIterator, I, I> {
  static void run(ZipIterator, ZipIterator, std::size_t const*, rmm::cuda_stream_view) {}
};

template <typename Iterator>
std::enable_if_t<is_permute_zip_iterator_v<Iterator>> permute_in_place(
  Iterator first, Iterator last, std::size_t const* map_first, rmm::cuda_stream_view stream_view)
{
  permute_zip_in_place_split_impl<Iterator, 0, permute_zip_iterator_arity_v<Iterator>>::run(
    first, last, map_first, stream_view);
}

template <typename Iterator>
std::enable_if_t<!is_permute_zip_iterator_v<Iterator>> permute_in_place(
  Iterator first, Iterator last, std::size_t const* map_first, rmm::cuda_stream_view stream_view)
{
  auto const num_elements = static_cast<std::size_t>(cuda::std::distance(first, last));
  permute_scalar_in_place(first, map_first, num_elements, stream_view);
}

}  // namespace detail
}  // namespace cugraph

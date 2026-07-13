/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/iterator_utils.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <cstdint>
#include <type_traits>

namespace CUGRAPH_EXPORT cugraph {

namespace detail {

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
    thrust::copy_if(
      thrust::seq,
      input_first + input_start_offset,
      input_first + (input_start_offset + num_items),
      cuda::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                    check_bit_set_t<MaskIterator, size_t>{mask_first, size_t{0}}) +
        input_start_offset,
      output_first + output_start_offset,
      is_equal_t<bool>{true})));
}

/** Whether @p InputIterator / @p OutputIterator match an out-of-line @ref copy_if_mask_set_impl. */
template <typename InputIterator, typename OutputIterator>
inline constexpr bool copy_if_mask_supported_iterator_v =
  std::is_same_v<std::remove_cv_t<InputIterator>, std::remove_cv_t<OutputIterator>> &&
  std::disjunction_v<std::is_same<iterator_value_t<InputIterator>, std::int32_t>,
                     std::is_same<iterator_value_t<InputIterator>, std::int64_t>,
                     std::is_same<iterator_value_t<InputIterator>, float>,
                     std::is_same<iterator_value_t<InputIterator>, double>>;

template <typename InputIterator, typename OutputIterator>
OutputIterator copy_if_mask_set_impl(raft::handle_t const& handle,
                                     InputIterator input_first,
                                     InputIterator input_last,
                                     uint32_t const* mask_first,
                                     OutputIterator output_first);

template <typename InputIterator, typename OutputIterator>
OutputIterator copy_if_mask_unset_impl(raft::handle_t const& handle,
                                       InputIterator input_first,
                                       InputIterator input_last,
                                       uint32_t const* mask_first,
                                       OutputIterator output_first);

template <typename InputIterator,
          typename OutputIterator,
          std::enable_if_t<!is_thrust_zip_iterator_v<InputIterator>, bool> = true>
OutputIterator copy_if_mask_set(raft::handle_t const& handle,
                                InputIterator input_first,
                                InputIterator input_last,
                                uint32_t const* mask_first,
                                OutputIterator output_first)
{
  if constexpr (copy_if_mask_supported_iterator_v<InputIterator, OutputIterator>) {
    return copy_if_mask_set_impl(handle, input_first, input_last, mask_first, output_first);
  } else {
    return thrust::copy_if(handle.get_thrust_policy(),
                           input_first,
                           input_last,
                           cuda::make_transform_iterator(
                             thrust::make_counting_iterator(size_t{0}),
                             check_bit_set_t<uint32_t const*, size_t>{mask_first, size_t{0}}),
                           output_first,
                           is_equal_t<bool>{true});
  }
}

template <typename InputIterator,
          typename OutputIterator,
          std::enable_if_t<!is_thrust_zip_iterator_v<InputIterator>, bool> = true>
OutputIterator copy_if_mask_unset(raft::handle_t const& handle,
                                  InputIterator input_first,
                                  InputIterator input_last,
                                  uint32_t const* mask_first,
                                  OutputIterator output_first)
{
  if constexpr (copy_if_mask_supported_iterator_v<InputIterator, OutputIterator>) {
    return copy_if_mask_unset_impl(handle, input_first, input_last, mask_first, output_first);
  } else {
    return thrust::copy_if(handle.get_thrust_policy(),
                           input_first,
                           input_last,
                           cuda::make_transform_iterator(
                             thrust::make_counting_iterator(size_t{0}),
                             check_bit_set_t<uint32_t const*, size_t>{mask_first, size_t{0}}),
                           output_first,
                           is_equal_t<bool>{false});
  }
}

template <typename InputZipIterator, typename OutputZipIterator, size_t I, size_t N>
struct copy_if_mask_set_zip_split_impl {
  static void run(raft::handle_t const& handle,
                  InputZipIterator input_first,
                  InputZipIterator input_last,
                  uint32_t const* mask_first,
                  OutputZipIterator output_first)
  {
    auto const& input_tuple      = input_first.get_iterator_tuple();
    auto const& input_last_tuple = input_last.get_iterator_tuple();
    auto const& output_tuple     = output_first.get_iterator_tuple();

    copy_if_mask_set(handle,
                     cuda::std::get<I>(input_tuple),
                     cuda::std::get<I>(input_last_tuple),
                     mask_first,
                     cuda::std::get<I>(output_tuple));
    copy_if_mask_set_zip_split_impl<InputZipIterator, OutputZipIterator, I + 1, N>::run(
      handle, input_first, input_last, mask_first, output_first);
  }
};

template <typename InputZipIterator, typename OutputZipIterator, size_t I>
struct copy_if_mask_set_zip_split_impl<InputZipIterator, OutputZipIterator, I, I> {
  static void run(
    raft::handle_t const&, InputZipIterator, InputZipIterator, uint32_t const*, OutputZipIterator)
  {
  }
};

template <typename InputZipIterator, typename OutputZipIterator, size_t I, size_t N>
struct copy_if_mask_unset_zip_split_impl {
  static void run(raft::handle_t const& handle,
                  InputZipIterator input_first,
                  InputZipIterator input_last,
                  uint32_t const* mask_first,
                  OutputZipIterator output_first)
  {
    auto const& input_tuple      = input_first.get_iterator_tuple();
    auto const& input_last_tuple = input_last.get_iterator_tuple();
    auto const& output_tuple     = output_first.get_iterator_tuple();

    copy_if_mask_unset(handle,
                       cuda::std::get<I>(input_tuple),
                       cuda::std::get<I>(input_last_tuple),
                       mask_first,
                       cuda::std::get<I>(output_tuple));
    copy_if_mask_unset_zip_split_impl<InputZipIterator, OutputZipIterator, I + 1, N>::run(
      handle, input_first, input_last, mask_first, output_first);
  }
};

template <typename InputZipIterator, typename OutputZipIterator, size_t I>
struct copy_if_mask_unset_zip_split_impl<InputZipIterator, OutputZipIterator, I, I> {
  static void run(
    raft::handle_t const&, InputZipIterator, InputZipIterator, uint32_t const*, OutputZipIterator)
  {
  }
};

template <typename InputZipIterator, typename OutputZipIterator>
OutputZipIterator copy_if_mask_set_zip_split(raft::handle_t const& handle,
                                             InputZipIterator input_first,
                                             InputZipIterator input_last,
                                             uint32_t const* mask_first,
                                             OutputZipIterator output_first)
{
  static_assert(cuda::std::tuple_size<typename InputZipIterator::iterator_tuple>::value ==
                  cuda::std::tuple_size<typename OutputZipIterator::iterator_tuple>::value,
                "copy_if_mask_set zip overload requires matching tuple arity.");

  constexpr size_t tuple_size =
    cuda::std::tuple_size<typename InputZipIterator::iterator_tuple>::value;

  auto const& input_tuple      = input_first.get_iterator_tuple();
  auto const& input_last_tuple = input_last.get_iterator_tuple();
  auto const& output_tuple     = output_first.get_iterator_tuple();

  auto output_end = copy_if_mask_set(handle,
                                     cuda::std::get<0>(input_tuple),
                                     cuda::std::get<0>(input_last_tuple),
                                     mask_first,
                                     cuda::std::get<0>(output_tuple));
  if constexpr (tuple_size > 1) {
    copy_if_mask_set_zip_split_impl<InputZipIterator, OutputZipIterator, 1, tuple_size>::run(
      handle, input_first, input_last, mask_first, output_first);
  }
  return output_first + (output_end - cuda::std::get<0>(output_tuple));
}

template <typename InputZipIterator, typename OutputZipIterator>
OutputZipIterator copy_if_mask_unset_zip_split(raft::handle_t const& handle,
                                               InputZipIterator input_first,
                                               InputZipIterator input_last,
                                               uint32_t const* mask_first,
                                               OutputZipIterator output_first)
{
  static_assert(cuda::std::tuple_size<typename InputZipIterator::iterator_tuple>::value ==
                  cuda::std::tuple_size<typename OutputZipIterator::iterator_tuple>::value,
                "copy_if_mask_unset zip overload requires matching tuple arity.");

  constexpr size_t tuple_size =
    cuda::std::tuple_size<typename InputZipIterator::iterator_tuple>::value;

  auto const& input_tuple      = input_first.get_iterator_tuple();
  auto const& input_last_tuple = input_last.get_iterator_tuple();
  auto const& output_tuple     = output_first.get_iterator_tuple();

  auto output_end = copy_if_mask_unset(handle,
                                       cuda::std::get<0>(input_tuple),
                                       cuda::std::get<0>(input_last_tuple),
                                       mask_first,
                                       cuda::std::get<0>(output_tuple));
  if constexpr (tuple_size > 1) {
    copy_if_mask_unset_zip_split_impl<InputZipIterator, OutputZipIterator, 1, tuple_size>::run(
      handle, input_first, input_last, mask_first, output_first);
  }
  return output_first + (output_end - cuda::std::get<0>(output_tuple));
}

template <typename InputIterator,
          typename OutputIterator,
          std::enable_if_t<is_thrust_zip_iterator_v<InputIterator>, bool> = true>
OutputIterator copy_if_mask_set(raft::handle_t const& handle,
                                InputIterator input_first,
                                InputIterator input_last,
                                uint32_t const* mask_first,
                                OutputIterator output_first)
{
  static_assert(is_thrust_zip_iterator_v<OutputIterator>,
                "copy_if_mask_set zip overload requires a zip output iterator.");
  return copy_if_mask_set_zip_split(handle, input_first, input_last, mask_first, output_first);
}

template <typename InputIterator,
          typename OutputIterator,
          std::enable_if_t<is_thrust_zip_iterator_v<InputIterator>, bool> = true>
OutputIterator copy_if_mask_unset(raft::handle_t const& handle,
                                  InputIterator input_first,
                                  InputIterator input_last,
                                  uint32_t const* mask_first,
                                  OutputIterator output_first)
{
  static_assert(is_thrust_zip_iterator_v<OutputIterator>,
                "copy_if_mask_unset zip overload requires a zip output iterator.");
  return copy_if_mask_unset_zip_split(handle, input_first, input_last, mask_first, output_first);
}

// Reorder @p input_first..@p input_last in-place into [set bits | unset bits] using @p mask_first.
// Set bits are compacted to the front; unset bits are moved to the back via a temp buffer of size
// @p second_size.
template <typename InputIterator,
          std::enable_if_t<!is_thrust_zip_iterator_v<InputIterator>, bool> = true>
void partition_by_mask(InputIterator input_first,
                       InputIterator input_last,
                       uint32_t const* mask_first,
                       size_t first_size,
                       size_t second_size,
                       rmm::cuda_stream_view stream_view,
                       std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  using element_t = typename thrust::iterator_traits<InputIterator>::value_type;

  CUGRAPH_EXPECTS(!large_buffer_type || large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  auto tmp_buffer =
    large_buffer_type
      ? large_buffer_manager::allocate_memory_buffer<element_t>(second_size, stream_view)
      : allocate_dataframe_buffer<element_t>(second_size, stream_view);

  raft::handle_t handle{stream_view};

  copy_if_mask_unset(
    handle, input_first, input_last, mask_first, get_dataframe_buffer_begin(tmp_buffer));
  copy_if_mask_set(handle, input_first, input_last, mask_first, input_first);
  thrust::copy(rmm::exec_policy(stream_view),
               get_dataframe_buffer_cbegin(tmp_buffer),
               get_dataframe_buffer_cend(tmp_buffer),
               input_first + first_size);
}

template <typename ZipIterator, size_t I, size_t N>
struct partition_by_mask_zip_split_impl {
  static void run(ZipIterator input_first,
                  ZipIterator input_last,
                  uint32_t const* mask_first,
                  size_t first_size,
                  size_t second_size,
                  rmm::cuda_stream_view stream_view,
                  std::optional<large_buffer_type_t> large_buffer_type)
  {
    auto const& input_tuple      = input_first.get_iterator_tuple();
    auto const& input_last_tuple = input_last.get_iterator_tuple();

    detail::partition_by_mask(cuda::std::get<I>(input_tuple),
                              cuda::std::get<I>(input_last_tuple),
                              mask_first,
                              first_size,
                              second_size,
                              stream_view,
                              large_buffer_type);
    partition_by_mask_zip_split_impl<ZipIterator, I + 1, N>::run(
      input_first, input_last, mask_first, first_size, second_size, stream_view, large_buffer_type);
  }
};

template <typename ZipIterator, size_t I>
struct partition_by_mask_zip_split_impl<ZipIterator, I, I> {
  static void run(ZipIterator,
                  ZipIterator,
                  uint32_t const*,
                  size_t,
                  size_t,
                  rmm::cuda_stream_view,
                  std::optional<large_buffer_type_t>)
  {
  }
};

template <typename ZipIterator>
void partition_by_mask_zip_split(
  ZipIterator input_first,
  ZipIterator input_last,
  uint32_t const* mask_first,
  size_t first_size,
  size_t second_size,
  rmm::cuda_stream_view stream_view,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  constexpr size_t tuple_size = cuda::std::tuple_size<typename ZipIterator::iterator_tuple>::value;

  partition_by_mask_zip_split_impl<ZipIterator, 0, tuple_size>::run(
    input_first, input_last, mask_first, first_size, second_size, stream_view, large_buffer_type);
}

// Reorder each component of @p input_first..@p input_last in-place into [set bits | unset bits].
template <typename ZipIterator,
          std::enable_if_t<is_thrust_zip_iterator_v<ZipIterator>, bool> = true>
void partition_by_mask(ZipIterator input_first,
                       ZipIterator input_last,
                       uint32_t const* mask_first,
                       size_t first_size,
                       size_t second_size,
                       rmm::cuda_stream_view stream_view,
                       std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  partition_by_mask_zip_split(
    input_first, input_last, mask_first, first_size, second_size, stream_view, large_buffer_type);
}

}  // namespace detail

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
    cuda::std::plus<size_t>{});
}

size_t count_set_bits(rmm::exec_policy_nosync const& policy,
                      uint32_t const* mask_first,
                      size_t num_bits);

template <typename InputIterator, typename OutputIterator>
OutputIterator copy_if_mask_set(raft::handle_t const& handle,
                                InputIterator input_first,
                                InputIterator input_last,
                                uint32_t const* mask_first,
                                OutputIterator output_first)
{
  return detail::copy_if_mask_set(handle, input_first, input_last, mask_first, output_first);
}

template <typename InputIterator, typename OutputIterator>
OutputIterator copy_if_mask_unset(raft::handle_t const& handle,
                                  InputIterator input_first,
                                  InputIterator input_last,
                                  uint32_t const* mask_first,
                                  OutputIterator output_first)
{
  return detail::copy_if_mask_unset(handle, input_first, input_last, mask_first, output_first);
}

template <typename comparison_t>
std::tuple<size_t, rmm::device_uvector<uint32_t>> mark_entries(
  size_t num_entries,
  comparison_t comparison,
  rmm::cuda_stream_view stream_view,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  auto marked_entries = large_buffer_type ? large_buffer_manager::allocate_memory_buffer<uint32_t>(
                                              cugraph::packed_bool_size(num_entries), stream_view)
                                          : rmm::device_uvector<uint32_t>(
                                              cugraph::packed_bool_size(num_entries), stream_view);

  thrust::tabulate(rmm::exec_policy(stream_view),
                   marked_entries.begin(),
                   marked_entries.end(),
                   [comparison, num_entries] __device__(size_t idx) {
                     auto word          = cugraph::packed_bool_empty_mask();
                     size_t start_index = idx * cugraph::packed_bools_per_word();
                     size_t bits_in_this_word =
                       (start_index + cugraph::packed_bools_per_word() < num_entries)
                         ? cugraph::packed_bools_per_word()
                         : (num_entries - start_index);

                     for (size_t bit = 0; bit < bits_in_this_word; ++bit) {
                       if (comparison(start_index + bit)) word |= cugraph::packed_bool_mask(bit);
                     }

                     return word;
                   });

  size_t bit_count =
    count_set_bits(rmm::exec_policy_nosync(stream_view), marked_entries.begin(), num_entries);

  return std::make_tuple(bit_count, std::move(marked_entries));
}

template <typename T>
rmm::device_uvector<T> keep_marked_entries(
  raft::handle_t const& handle,
  rmm::device_uvector<T>&& vector,
  raft::device_span<uint32_t const> keep_flags,
  size_t keep_count,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  auto result = large_buffer_type
                  ? large_buffer_manager::allocate_memory_buffer<T>(keep_count, handle.get_stream())
                  : rmm::device_uvector<T>(keep_count, handle.get_stream());

  copy_if_mask_set(handle, vector.begin(), vector.end(), keep_flags.begin(), result.begin());
  vector.resize(0, handle.get_stream());
  vector.shrink_to_fit(handle.get_stream());

  return result;
}

template <typename InputIterator>
void partition_by_mask(InputIterator input_first,
                       InputIterator input_last,
                       uint32_t const* mask_first,
                       size_t first_size,
                       size_t second_size,
                       rmm::cuda_stream_view stream_view,
                       std::optional<large_buffer_type_t> large_buffer_type)
{
  detail::partition_by_mask(
    input_first, input_last, mask_first, first_size, second_size, stream_view, large_buffer_type);
}

}  // namespace CUGRAPH_EXPORT cugraph

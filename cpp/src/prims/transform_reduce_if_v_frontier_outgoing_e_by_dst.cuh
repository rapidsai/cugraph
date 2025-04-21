/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "detail/graph_partition_utils.cuh"
#include "prims/detail/extract_transform_if_v_frontier_e.cuh"
#include "prims/detail/optional_dataframe_buffer.hpp"
#include "prims/detail/prim_utils.cuh"
#include "prims/property_op_utils.cuh"
#include "prims/reduce_op.cuh"

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/integer_sequence.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace cugraph {

namespace detail {

int32_t constexpr update_v_frontier_from_outgoing_e_kernel_block_size = 512;

template <typename key_t,
          typename payload_t,
          typename vertex_t,
          typename src_value_t,
          typename dst_value_t,
          typename e_value_t,
          typename EdgeOp>
struct transform_reduce_if_v_frontier_call_e_op_t {
  EdgeOp e_op{};

  __device__ std::conditional_t<!std::is_same_v<key_t, void> && !std::is_same_v<payload_t, void>,
                                thrust::tuple<key_t, payload_t>,
                                std::conditional_t<!std::is_same_v<key_t, void>, key_t, payload_t>>
  operator()(key_t key, vertex_t dst, src_value_t sv, dst_value_t dv, e_value_t ev) const
  {
    auto reduce_by = dst;
    if constexpr (std::is_same_v<key_t, vertex_t> && std::is_same_v<payload_t, void>) {
      return reduce_by;
    } else if constexpr (std::is_same_v<key_t, vertex_t> && !std::is_same_v<payload_t, void>) {
      auto e_op_result = e_op(key, dst, sv, dv, ev);  // payload
      return thrust::make_tuple(reduce_by, e_op_result);
    } else if constexpr (!std::is_same_v<key_t, vertex_t> && std::is_same_v<payload_t, void>) {
      auto e_op_result = e_op(key, dst, sv, dv, ev);  // tag
      return thrust::make_tuple(reduce_by, e_op_result);
    } else {
      auto e_op_result = e_op(key, dst, sv, dv, ev);  // (tag, payload)
      return thrust::make_tuple(thrust::make_tuple(reduce_by, thrust::get<0>(e_op_result)),
                                thrust::get<1>(e_op_result));
    }
  }
};

template <typename InputKeyIterator, typename key_t>
struct update_keep_flag_t {
  using input_key_t =
    typename thrust::iterator_traits<InputKeyIterator>::value_type;  // uint32_t (compressed) or
                                                                     // key_t (i.e. vertex_t)

  raft::device_span<uint32_t> bitmap{};
  raft::device_span<uint32_t> keep_flags{};
  key_t v_range_first{};
  InputKeyIterator input_key_first{};
  cuda::std::optional<input_key_t> invalid_input_key{};

  __device__ void operator()(size_t i) const
  {
    auto v = *(input_key_first + i);
    if (invalid_input_key && (v == *invalid_input_key)) {
      return;  // just discard
    }
    input_key_t v_offset{};
    if constexpr ((sizeof(key_t) == 8) && std::is_same_v<input_key_t, uint32_t>) {
      v_offset = v;
    } else {
      v_offset = v - v_range_first;
    }
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> bitmap_word(
      bitmap[packed_bool_offset(v_offset)]);
    auto old = bitmap_word.fetch_or(packed_bool_mask(v_offset), cuda::std::memory_order_relaxed);
    if ((old & packed_bool_mask(v_offset)) == packed_bool_empty_mask()) {
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> keep_flag_word(
        keep_flags[packed_bool_offset(i)]);
      keep_flag_word.fetch_or(packed_bool_mask(i), cuda::std::memory_order_relaxed);
    }
  }
};

template <typename priority_t, typename vertex_t, typename payload_t>
std::tuple<rmm::device_uvector<vertex_t>, optional_dataframe_buffer_type_t<payload_t>>
filter_buffer_elements(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&&
    unique_v_buffer,  // assumes that buffer elements are locally reduced first and unique
  optional_dataframe_buffer_type_t<payload_t>&& payload_buffer,
  raft::device_span<vertex_t const> vertex_partition_range_offsets,  // size = major_comm_size + 1
  vertex_t allreduce_count_per_rank,
  int subgroup_size)
{
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_rank = major_comm.get_rank();
  auto const major_comm_size = major_comm.get_size();

  rmm::device_uvector<priority_t> priorities(allreduce_count_per_rank * major_comm_size,
                                             handle.get_stream());
  thrust::fill(handle.get_thrust_policy(),
               priorities.begin(),
               priorities.end(),
               std::numeric_limits<priority_t>::max());
  thrust::for_each(
    handle.get_thrust_policy(),
    unique_v_buffer.begin(),
    unique_v_buffer.end(),
    [offsets    = vertex_partition_range_offsets,
     priorities = raft::device_span<priority_t>(priorities.data(), priorities.size()),
     allreduce_count_per_rank,
     subgroup_size,
     major_comm_rank,
     major_comm_size] __device__(auto v) {
      auto root = cuda::std::distance(
        offsets.begin() + 1,
        thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), v));
      auto v_offset = v - offsets[root];
      if (v_offset < allreduce_count_per_rank) {
        priorities[allreduce_count_per_rank * root + v_offset] =
          rank_to_priority<vertex_t, priority_t>(
            major_comm_rank, root, subgroup_size, major_comm_size, v_offset);
      }
    });
  device_allreduce(major_comm,
                   priorities.data(),
                   priorities.data(),
                   priorities.size(),
                   raft::comms::op_t::MIN,
                   handle.get_stream());
  if constexpr (std::is_same_v<payload_t, void>) {
    unique_v_buffer.resize(
      cuda::std::distance(
        unique_v_buffer.begin(),
        thrust::remove_if(
          handle.get_thrust_policy(),
          unique_v_buffer.begin(),
          unique_v_buffer.end(),
          unique_v_buffer.begin(),
          [offsets    = vertex_partition_range_offsets,
           priorities = raft::device_span<priority_t const>(priorities.data(), priorities.size()),
           allreduce_count_per_rank,
           subgroup_size,
           major_comm_rank,
           major_comm_size] __device__(auto v) {
            auto root = cuda::std::distance(
              offsets.begin() + 1,
              thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), v));
            auto v_offset = v - offsets[root];
            if (v_offset < allreduce_count_per_rank) {
              auto selected_rank = priority_to_rank<vertex_t, priority_t>(
                priorities[allreduce_count_per_rank * root + v_offset],
                root,
                subgroup_size,
                major_comm_size,
                v_offset);
              return major_comm_rank != selected_rank;
            } else {
              return false;
            }
          })),
      handle.get_stream());
  } else {
    auto kv_pair_first = thrust::make_zip_iterator(unique_v_buffer.begin(),
                                                   get_dataframe_buffer_begin(payload_buffer));
    unique_v_buffer.resize(
      cuda::std::distance(
        kv_pair_first,
        thrust::remove_if(
          handle.get_thrust_policy(),
          kv_pair_first,
          kv_pair_first + unique_v_buffer.size(),
          unique_v_buffer.begin(),
          [offsets    = vertex_partition_range_offsets,
           priorities = raft::device_span<priority_t const>(priorities.data(), priorities.size()),
           allreduce_count_per_rank,
           subgroup_size,
           major_comm_rank,
           major_comm_size] __device__(auto v) {
            auto root = cuda::std::distance(
              offsets.begin() + 1,
              thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), v));
            auto v_offset = v - offsets[root];
            if (v_offset < allreduce_count_per_rank) {
              auto selected_rank = priority_to_rank<vertex_t, priority_t>(
                priorities[allreduce_count_per_rank * root + v_offset],
                root,
                subgroup_size,
                major_comm_size,
                v_offset);
              return major_comm_rank != selected_rank;
            } else {
              return false;
            }
          })),
      handle.get_stream());
    resize_dataframe_buffer(payload_buffer, unique_v_buffer.size(), handle.get_stream());
  }

  return std::make_tuple(std::move(unique_v_buffer), std::move(payload_buffer));
}

template <typename input_key_t /* uint32_t if 64 bit vertex IDs are compressed to 32 bit offsets,
                                  otherwise input_key_t == output_key_t */
          ,
          typename key_t,
          typename payload_t,
          typename ReduceOp>
std::tuple<dataframe_buffer_type_t<key_t>, optional_dataframe_buffer_type_t<payload_t>>
sort_and_reduce_buffer_elements(
  raft::handle_t const& handle,
  dataframe_buffer_type_t<input_key_t>&& key_buffer,
  optional_dataframe_buffer_type_t<payload_t>&& payload_buffer,
  ReduceOp reduce_op,
  std::conditional_t<std::is_integral_v<key_t>, std::tuple<key_t, key_t>, std::byte /* dummy */>
    vertex_range,
  std::optional<input_key_t> invalid_key /* drop (key, (payload)) pairs with invalid key */)
{
  constexpr bool compressed =
    std::is_integral_v<key_t> && (sizeof(key_t) == 8) &&
    std::is_same_v<input_key_t, uint32_t>;  // we currently compress only when key_t is an integral
                                            // type (i.e. vertex_t)
  static_assert(compressed || std::is_same_v<input_key_t, key_t>);

  if constexpr (std::is_integral_v<key_t> &&
                (std::is_same_v<payload_t, void> ||
                 std::is_same_v<ReduceOp,
                                reduce_op::any<typename ReduceOp::value_type>>)) {  // try to use
                                                                                    // bitmap for
                                                                                    // filtering
    key_t range_size = std::get<1>(vertex_range) - std::get<0>(vertex_range);
    if (static_cast<double>(size_dataframe_buffer(key_buffer)) >=
        static_cast<double>(range_size) *
          0.125 /* tuning parameter */) {  // use bitmap for filtering
      rmm::device_uvector<uint32_t> bitmap(packed_bool_size(range_size), handle.get_stream());
      rmm::device_uvector<uint32_t> keep_flags(packed_bool_size(size_dataframe_buffer(key_buffer)),
                                               handle.get_stream());
      thrust::fill(
        handle.get_thrust_policy(), bitmap.begin(), bitmap.end(), packed_bool_empty_mask());
      thrust::fill(
        handle.get_thrust_policy(), keep_flags.begin(), keep_flags.end(), packed_bool_empty_mask());
      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_counting_iterator(size_t{0}),
                       thrust::make_counting_iterator(size_dataframe_buffer(key_buffer)),
                       update_keep_flag_t<decltype(get_dataframe_buffer_begin(key_buffer)), key_t>{
                         raft::device_span<uint32_t>(bitmap.data(), bitmap.size()),
                         raft::device_span<uint32_t>(keep_flags.data(), keep_flags.size()),
                         std::get<0>(vertex_range),
                         get_dataframe_buffer_begin(key_buffer),
                         to_thrust_optional(invalid_key)});
      auto stencil_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<bool>(
          [keep_flags = raft::device_span<uint32_t const>(keep_flags.data(),
                                                          keep_flags.size())] __device__(size_t i) {
            return (keep_flags[packed_bool_offset(i)] & packed_bool_mask(i)) !=
                   packed_bool_empty_mask();
          }));
      if constexpr (std::is_same_v<payload_t, void>) {
        resize_dataframe_buffer(
          key_buffer,
          cuda::std::distance(get_dataframe_buffer_begin(key_buffer),
                              thrust::remove_if(handle.get_thrust_policy(),
                                                get_dataframe_buffer_begin(key_buffer),
                                                get_dataframe_buffer_end(key_buffer),
                                                stencil_first,
                                                is_not_equal_t<bool>{true})),
          handle.get_stream());
        shrink_to_fit_dataframe_buffer(key_buffer, handle.get_stream());
        thrust::sort(handle.get_thrust_policy(),
                     get_dataframe_buffer_begin(key_buffer),
                     get_dataframe_buffer_end(key_buffer));
      } else {
        static_assert(std::is_same_v<ReduceOp, reduce_op::any<typename ReduceOp::value_type>>);
        auto pair_first = thrust::make_zip_iterator(get_dataframe_buffer_begin(key_buffer),
                                                    get_dataframe_buffer_begin(payload_buffer));
        resize_dataframe_buffer(
          key_buffer,
          cuda::std::distance(pair_first,
                              thrust::remove_if(handle.get_thrust_policy(),
                                                pair_first,
                                                pair_first + size_dataframe_buffer(key_buffer),
                                                stencil_first,
                                                is_not_equal_t<bool>{true})),
          handle.get_stream());
        resize_dataframe_buffer(
          payload_buffer, size_dataframe_buffer(key_buffer), handle.get_stream());
        shrink_to_fit_dataframe_buffer(key_buffer, handle.get_stream());
        shrink_to_fit_dataframe_buffer(payload_buffer, handle.get_stream());
        thrust::sort_by_key(handle.get_thrust_policy(),
                            get_dataframe_buffer_begin(key_buffer),
                            get_dataframe_buffer_end(key_buffer),
                            get_dataframe_buffer_begin(payload_buffer));
      }

      if constexpr (compressed) {
        rmm::device_uvector<key_t> output_key_buffer(key_buffer.size(), handle.get_stream());
        thrust::transform(handle.get_thrust_policy(),
                          key_buffer.begin(),
                          key_buffer.end(),
                          output_key_buffer.begin(),
                          cuda::proclaim_return_type<key_t>(
                            [v_first = std::get<0>(vertex_range)] __device__(uint32_t v_offset) {
                              return static_cast<key_t>(v_first + v_offset);
                            }));
        return std::make_tuple(std::move(output_key_buffer), std::move(payload_buffer));
      } else {
        return std::make_tuple(std::move(key_buffer), std::move(payload_buffer));
      }
    }
  }

  if constexpr (std::is_same_v<payload_t, void>) {
    thrust::sort(handle.get_thrust_policy(),
                 get_dataframe_buffer_begin(key_buffer),
                 get_dataframe_buffer_end(key_buffer));
  } else {
    thrust::sort_by_key(handle.get_thrust_policy(),
                        get_dataframe_buffer_begin(key_buffer),
                        get_dataframe_buffer_end(key_buffer),
                        get_optional_dataframe_buffer_begin<payload_t>(payload_buffer));
  }

  auto output_key_buffer = allocate_dataframe_buffer<key_t>(0, handle.get_stream());
  if constexpr (std::is_same_v<payload_t, void>) {
    if constexpr (compressed) {
      resize_dataframe_buffer(
        output_key_buffer, size_dataframe_buffer(key_buffer), handle.get_stream());
      auto input_key_first = thrust::make_transform_iterator(
        get_dataframe_buffer_begin(key_buffer),
        cuda::proclaim_return_type<key_t>(
          [v_first = std::get<0>(vertex_range)] __device__(auto v_offset) {
            return static_cast<key_t>(v_first + v_offset);
          }));
      resize_dataframe_buffer(
        output_key_buffer,
        cuda::std::distance(
          get_dataframe_buffer_begin(output_key_buffer),
          thrust::copy_if(handle.get_thrust_policy(),
                          input_key_first,
                          input_key_first + size_dataframe_buffer(key_buffer),
                          thrust::make_counting_iterator(size_t{0}),
                          get_dataframe_buffer_begin(output_key_buffer),
                          cuda::proclaim_return_type<bool>(
                            [key_first   = get_dataframe_buffer_begin(key_buffer),
                             invalid_key = to_thrust_optional(invalid_key)] __device__(size_t i) {
                              auto key = *(key_first + i);
                              if (invalid_key && (key == *invalid_key)) {
                                return false;
                              } else if ((i != 0) && (key == *(key_first + (i - 1)))) {
                                return false;
                              } else {
                                return true;
                              }
                            }))),
        handle.get_stream());
    } else {
      resize_dataframe_buffer(
        key_buffer,
        cuda::std::distance(
          get_dataframe_buffer_begin(key_buffer),
          thrust::remove_if(handle.get_thrust_policy(),
                            get_dataframe_buffer_begin(key_buffer),
                            get_dataframe_buffer_end(key_buffer),
                            thrust::make_counting_iterator(size_t{0}),
                            cuda::proclaim_return_type<bool>(
                              [key_first   = get_dataframe_buffer_begin(key_buffer),
                               invalid_key = to_thrust_optional(invalid_key)] __device__(size_t i) {
                                auto key = *(key_first + i);
                                if (invalid_key && (key == *invalid_key)) {
                                  return true;
                                } else if ((i != 0) && (key == *(key_first + (i - 1)))) {
                                  return true;
                                } else {
                                  return false;
                                }
                              }))),
        handle.get_stream());
      output_key_buffer = std::move(key_buffer);
    }
    shrink_to_fit_dataframe_buffer(output_key_buffer, handle.get_stream());
  } else if constexpr (std::is_same_v<ReduceOp, reduce_op::any<typename ReduceOp::value_type>>) {
    if constexpr (compressed) {
      resize_dataframe_buffer(
        output_key_buffer, size_dataframe_buffer(key_buffer), handle.get_stream());
      auto input_key_first = thrust::make_transform_iterator(
        get_dataframe_buffer_begin(key_buffer),
        cuda::proclaim_return_type<key_t>(
          [v_first = std::get<0>(vertex_range)] __device__(auto v_offset) {
            return static_cast<key_t>(v_first + v_offset);
          }));
      auto tmp_payload_buffer = allocate_dataframe_buffer<payload_t>(
        size_dataframe_buffer(payload_buffer), handle.get_stream());
      auto input_pair_first =
        thrust::make_zip_iterator(input_key_first, get_dataframe_buffer_begin(payload_buffer));
      auto output_pair_first =
        thrust::make_zip_iterator(get_dataframe_buffer_begin(output_key_buffer),
                                  get_dataframe_buffer_begin(tmp_payload_buffer));
      resize_dataframe_buffer(
        output_key_buffer,
        cuda::std::distance(
          output_pair_first,
          thrust::copy_if(handle.get_thrust_policy(),
                          input_pair_first,
                          input_pair_first + size_dataframe_buffer(key_buffer),
                          thrust::make_counting_iterator(size_t{0}),
                          output_pair_first,
                          cuda::proclaim_return_type<bool>(
                            [key_first   = get_dataframe_buffer_begin(key_buffer),
                             invalid_key = to_thrust_optional(invalid_key)] __device__(size_t i) {
                              auto key = *(key_first + i);
                              if (invalid_key && (key == *invalid_key)) {
                                return false;
                              } else if ((i != 0) && (key == *(key_first + (i - 1)))) {
                                return false;
                              } else {
                                return true;
                              }
                            }))),
        handle.get_stream());
      resize_dataframe_buffer(
        tmp_payload_buffer, size_dataframe_buffer(output_key_buffer), handle.get_stream());
      payload_buffer = std::move(tmp_payload_buffer);
    } else {
      auto pair_first = thrust::make_zip_iterator(get_dataframe_buffer_begin(key_buffer),
                                                  get_dataframe_buffer_begin(payload_buffer));
      resize_dataframe_buffer(
        key_buffer,
        cuda::std::distance(
          pair_first,
          thrust::remove_if(handle.get_thrust_policy(),
                            pair_first,
                            pair_first + size_dataframe_buffer(key_buffer),
                            thrust::make_counting_iterator(size_t{0}),
                            cuda::proclaim_return_type<bool>(
                              [key_first   = get_dataframe_buffer_begin(key_buffer),
                               invalid_key = to_thrust_optional(invalid_key)] __device__(size_t i) {
                                auto key = *(key_first + i);
                                if (invalid_key && (key == *invalid_key)) {
                                  return true;
                                } else if ((i != 0) && (key == *(key_first + (i - 1)))) {
                                  return true;
                                } else {
                                  return false;
                                }
                              }))),
        handle.get_stream());
      resize_dataframe_buffer(
        payload_buffer, size_dataframe_buffer(key_buffer), handle.get_stream());
      output_key_buffer = std::move(key_buffer);
    }
    shrink_to_fit_dataframe_buffer(output_key_buffer, handle.get_stream());
    shrink_to_fit_dataframe_buffer(payload_buffer, handle.get_stream());
  } else {
    if (invalid_key) {
      auto pair_first = thrust::make_zip_iterator(get_dataframe_buffer_begin(key_buffer),
                                                  get_dataframe_buffer_begin(payload_buffer));
      resize_dataframe_buffer(
        key_buffer,
        cuda::std::distance(pair_first,
                            thrust::remove_if(handle.get_thrust_policy(),
                                              pair_first,
                                              pair_first + size_dataframe_buffer(key_buffer),
                                              cuda::proclaim_return_type<bool>(
                                                [invalid_key = *invalid_key] __device__(auto kv) {
                                                  auto key = thrust::get<0>(kv);
                                                  return key == invalid_key;
                                                }))),
        handle.get_stream());
      resize_dataframe_buffer(
        payload_buffer, size_dataframe_buffer(key_buffer), handle.get_stream());
    }
    auto num_uniques =
      thrust::count_if(handle.get_thrust_policy(),
                       thrust::make_counting_iterator(size_t{0}),
                       thrust::make_counting_iterator(size_dataframe_buffer(key_buffer)),
                       is_first_in_run_t<decltype(get_dataframe_buffer_begin(key_buffer))>{
                         get_dataframe_buffer_begin(key_buffer)});

    auto new_key_buffer = allocate_dataframe_buffer<key_t>(num_uniques, handle.get_stream());
    auto new_payload_buffer =
      allocate_dataframe_buffer<payload_t>(num_uniques, handle.get_stream());

    if constexpr (compressed) {
      auto input_key_first = thrust::make_transform_iterator(
        get_dataframe_buffer_begin(key_buffer),
        cuda::proclaim_return_type<key_t>(
          [v_first = std::get<0>(vertex_range)] __device__(auto v_offset) {
            return static_cast<key_t>(v_first + v_offset);
          }));
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            input_key_first,
                            input_key_first + size_dataframe_buffer(key_buffer),
                            get_optional_dataframe_buffer_begin<payload_t>(payload_buffer),
                            get_dataframe_buffer_begin(new_key_buffer),
                            get_dataframe_buffer_begin(new_payload_buffer),
                            thrust::equal_to<key_t>(),
                            reduce_op);
    } else {
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            get_dataframe_buffer_begin(key_buffer),
                            get_dataframe_buffer_end(key_buffer),
                            get_optional_dataframe_buffer_begin<payload_t>(payload_buffer),
                            get_dataframe_buffer_begin(new_key_buffer),
                            get_dataframe_buffer_begin(new_payload_buffer),
                            thrust::equal_to<key_t>(),
                            reduce_op);
    }

    output_key_buffer = std::move(new_key_buffer);
    payload_buffer    = std::move(new_payload_buffer);
  }

  return std::make_tuple(std::move(output_key_buffer), std::move(payload_buffer));
}

template <typename GraphViewType,
          typename KeyBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename ReduceOp,
          typename PredOp>
std::conditional_t<
  !std::is_same_v<typename ReduceOp::value_type, void>,
  std::tuple<dataframe_buffer_type_t<typename KeyBucketType::key_type>,
             detail::optional_dataframe_buffer_type_t<typename ReduceOp::value_type>>,
  dataframe_buffer_type_t<typename KeyBucketType::key_type>>
transform_reduce_if_v_frontier_outgoing_e_by_dst(raft::handle_t const& handle,
                                                 GraphViewType const& graph_view,
                                                 KeyBucketType const& frontier,
                                                 EdgeSrcValueInputWrapper edge_src_value_input,
                                                 EdgeDstValueInputWrapper edge_dst_value_input,
                                                 EdgeValueInputWrapper edge_value_input,
                                                 EdgeOp e_op,
                                                 ReduceOp reduce_op,
                                                 PredOp pred_op,
                                                 bool do_expensive_check = false)
{
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

  using vertex_t  = typename GraphViewType::vertex_type;
  using edge_t    = typename GraphViewType::edge_type;
  using key_t     = typename KeyBucketType::key_type;
  using payload_t = typename ReduceOp::value_type;

  if (do_expensive_check) {
    // currently, nothing to do
  }

  // 1. fill the buffer

  detail::transform_reduce_if_v_frontier_call_e_op_t<key_t,
                                                     payload_t,
                                                     vertex_t,
                                                     typename EdgeSrcValueInputWrapper::value_type,
                                                     typename EdgeDstValueInputWrapper::value_type,
                                                     typename EdgeValueInputWrapper::value_type,
                                                     EdgeOp>
    e_op_wrapper{e_op};

  auto [key_buffer, payload_buffer] =
    detail::extract_transform_if_v_frontier_e<false, key_t, payload_t>(handle,
                                                                       graph_view,
                                                                       frontier,
                                                                       edge_src_value_input,
                                                                       edge_dst_value_input,
                                                                       edge_value_input,
                                                                       e_op_wrapper,
                                                                       pred_op,
                                                                       do_expensive_check);
  // 2. reduce the buffer

  std::vector<vertex_t> vertex_partition_range_offsets{};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_rank = major_comm.get_rank();
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto const minor_comm_size = minor_comm.get_size();
    vertex_partition_range_offsets = std::vector<vertex_t>(major_comm_size + 1);
    for (int i = 0; i < major_comm_size; ++i) {
      auto vertex_partition_id =
        detail::compute_local_edge_partition_minor_range_vertex_partition_id_t{
          major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
      vertex_partition_range_offsets[i] =
        graph_view.vertex_partition_range_first(vertex_partition_id);
    }
    vertex_partition_range_offsets.back() = graph_view.local_edge_partition_dst_range_last();
  } else {
    vertex_partition_range_offsets =
      std::vector<vertex_t>{graph_view.local_edge_partition_dst_range_first(),
                            graph_view.local_edge_partition_dst_range_last()};
  }
  std::conditional_t<std::is_integral_v<key_t>, std::tuple<key_t, key_t>, std::byte /* dummy */>
    vertex_range{};
  if constexpr (std::is_integral_v<key_t>) {
    vertex_range = std::make_tuple(vertex_partition_range_offsets.front(),
                                   vertex_partition_range_offsets.back());
  }
  std::tie(key_buffer, payload_buffer) =
    detail::sort_and_reduce_buffer_elements<key_t, key_t, payload_t, ReduceOp>(
      handle,
      std::move(key_buffer),
      std::move(payload_buffer),
      reduce_op,
      vertex_range,
      std::nullopt);
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    if (major_comm_size > 1) {
      size_t local_key_buffer_size = size_dataframe_buffer(key_buffer);
      auto avg_key_buffer_size =
        host_scalar_allreduce(
          major_comm, local_key_buffer_size, raft::comms::op_t::SUM, handle.get_stream()) /
        major_comm_size;

      rmm::device_uvector<vertex_t> d_vertex_partition_range_offsets(
        vertex_partition_range_offsets.size(), handle.get_stream());
      raft::update_device(d_vertex_partition_range_offsets.data(),
                          vertex_partition_range_offsets.data(),
                          vertex_partition_range_offsets.size(),
                          handle.get_stream());

      constexpr bool try_compression = (sizeof(vertex_t) == 8) && std::is_same_v<key_t, vertex_t>;
      std::conditional_t<try_compression, vertex_t, std::byte /* dummy */>
        max_vertex_partition_size{};
      if constexpr (try_compression) {
        for (int i = 0; i < major_comm_size; ++i) {
          max_vertex_partition_size =
            std::max(vertex_partition_range_offsets[i + 1] - vertex_partition_range_offsets[i],
                     max_vertex_partition_size);
        }
      }

      if constexpr (std::is_same_v<key_t, vertex_t> &&
                    std::is_same_v<ReduceOp, reduce_op::any<typename ReduceOp::value_type>>) {
        vertex_t min_vertex_partition_size = std::numeric_limits<vertex_t>::max();
        for (int i = 0; i < major_comm_size; ++i) {
          min_vertex_partition_size =
            std::min(vertex_partition_range_offsets[i + 1] - vertex_partition_range_offsets[i],
                     min_vertex_partition_size);
        }

        auto segment_offsets = graph_view.local_vertex_partition_segment_offsets();
        auto& comm           = handle.get_comms();
        auto const comm_size = comm.get_size();
        if (segment_offsets &&
            (static_cast<double>(avg_key_buffer_size) >
             static_cast<double>(graph_view.number_of_vertices() / comm_size) *
               double{0.2})) {  // duplicates expected for high in-degree vertices (and we assume
                                // correlation between in-degrees & out-degrees)  // FIXME: we need
                                // a better criterion
          size_t key_size{0};
          size_t payload_size{0};
          if constexpr (try_compression) {
            if (max_vertex_partition_size <= std::numeric_limits<uint32_t>::max()) {
              key_size = sizeof(uint32_t);
            } else {
              key_size = sizeof(key_t);
            }
          } else {
            if constexpr (std::is_arithmetic_v<key_t>) {
              key_size = sizeof(key_t);
            } else {
              key_size = sum_thrust_tuple_element_sizes<key_t>();
            }
          }
          if constexpr (!std::is_same_v<payload_t, void>) {
            if constexpr (std::is_arithmetic_v<payload_t>) {
              payload_size = sizeof(payload_t);
            } else {
              payload_size = sum_thrust_tuple_element_sizes<payload_t>();
            }
          }

          int subgroup_size{};
          int num_gpus_per_node{};
          RAFT_CUDA_TRY(cudaGetDeviceCount(&num_gpus_per_node));
          if (comm_size <= num_gpus_per_node) {
            subgroup_size = major_comm_size;
          } else {
            auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
            auto const minor_comm_size = minor_comm.get_size();
            subgroup_size              = partition_manager::map_major_comm_to_gpu_row_comm
                                           ? std::min(major_comm_size, num_gpus_per_node)
                                           : std::max(num_gpus_per_node / minor_comm_size, int{1});
          }

          auto p2p_size_per_rank       = avg_key_buffer_size * (key_size + payload_size);
          auto p2p_size_per_node       = p2p_size_per_rank * std::min(num_gpus_per_node, comm_size);
          auto allreduce_size_per_node = p2p_size_per_node / 16 /* tuning parameter */;
          auto allreduce_size_per_rank =
            allreduce_size_per_node / (major_comm_size * (num_gpus_per_node / subgroup_size));

          if (major_comm_size <= std::numeric_limits<uint8_t>::max()) {  // priority = uint8_t
            std::tie(key_buffer, payload_buffer) =
              filter_buffer_elements<uint8_t, key_t, payload_t>(
                handle,
                std::move(key_buffer),
                std::move(payload_buffer),
                raft::device_span<vertex_t const>(d_vertex_partition_range_offsets.data(),
                                                  d_vertex_partition_range_offsets.size()),
                std::min(static_cast<vertex_t>(allreduce_size_per_rank / sizeof(uint8_t)),
                         min_vertex_partition_size),
                subgroup_size);
          } else {  // priority = uint32_t
            std::tie(key_buffer, payload_buffer) =
              filter_buffer_elements<uint32_t, key_t, payload_t>(
                handle,
                std::move(key_buffer),
                std::move(payload_buffer),
                raft::device_span<vertex_t const>(d_vertex_partition_range_offsets.data(),
                                                  d_vertex_partition_range_offsets.size()),
                std::min(static_cast<vertex_t>(allreduce_size_per_rank / sizeof(uint32_t)),
                         min_vertex_partition_size),
                subgroup_size);
          }
        }
      }

      rmm::device_uvector<edge_t> d_tx_buffer_last_boundaries(major_comm_size, handle.get_stream());
      auto key_v_first =
        thrust_tuple_get_or_identity<decltype(get_dataframe_buffer_begin(key_buffer)), 0>(
          get_dataframe_buffer_begin(key_buffer));
      thrust::lower_bound(handle.get_thrust_policy(),
                          key_v_first,
                          key_v_first + size_dataframe_buffer(key_buffer),
                          d_vertex_partition_range_offsets.begin() + 1,
                          d_vertex_partition_range_offsets.end(),
                          d_tx_buffer_last_boundaries.begin());
      std::conditional_t<try_compression,
                         std::optional<rmm::device_uvector<uint32_t>>,
                         std::byte /* dummy */>
        compressed_v_buffer{};
      if constexpr (try_compression) {
        if (max_vertex_partition_size <= std::numeric_limits<uint32_t>::max()) {
          compressed_v_buffer =
            rmm::device_uvector<uint32_t>(size_dataframe_buffer(key_buffer), handle.get_stream());
          thrust::transform(
            handle.get_thrust_policy(),
            get_dataframe_buffer_begin(key_buffer),
            get_dataframe_buffer_end(key_buffer),
            (*compressed_v_buffer).begin(),
            cuda::proclaim_return_type<uint32_t>(
              [firsts = raft::device_span<vertex_t const>(d_vertex_partition_range_offsets.data(),
                                                          static_cast<size_t>(major_comm_size)),
               lasts  = raft::device_span<vertex_t const>(
                 d_vertex_partition_range_offsets.data() + 1,
                 static_cast<size_t>(major_comm_size))] __device__(auto v) {
                auto major_comm_rank = cuda::std::distance(
                  lasts.begin(), thrust::upper_bound(thrust::seq, lasts.begin(), lasts.end(), v));
                return static_cast<uint32_t>(v - firsts[major_comm_rank]);
              }));
          resize_dataframe_buffer(key_buffer, 0, handle.get_stream());
          shrink_to_fit_dataframe_buffer(key_buffer, handle.get_stream());
        }
      }
      std::vector<edge_t> h_tx_buffer_last_boundaries(d_tx_buffer_last_boundaries.size());
      raft::update_host(h_tx_buffer_last_boundaries.data(),
                        d_tx_buffer_last_boundaries.data(),
                        d_tx_buffer_last_boundaries.size(),
                        handle.get_stream());
      handle.sync_stream();
      std::vector<size_t> tx_counts(h_tx_buffer_last_boundaries.size());
      std::adjacent_difference(
        h_tx_buffer_last_boundaries.begin(), h_tx_buffer_last_boundaries.end(), tx_counts.begin());

      size_t min_element_size{cache_line_size};
      if constexpr (std::is_same_v<key_t, vertex_t>) {
        if constexpr (try_compression) {
          if (compressed_v_buffer) {
            min_element_size = std::min(sizeof(uint32_t), min_element_size);
          } else {
            min_element_size = std::min(sizeof(key_t), min_element_size);
          }
        } else {
          min_element_size = std::min(sizeof(key_t), min_element_size);
        }
      } else {
        static_assert(is_thrust_tuple_of_arithmetic<key_t>::value);
        min_element_size =
          std::min(cugraph::min_thrust_tuple_element_sizes<key_t>(), min_element_size);
      }
      if constexpr (!std::is_same_v<payload_t, void>) {
        if constexpr (std::is_arithmetic_v<payload_t>) {
          min_element_size = std::min(sizeof(payload_t), min_element_size);
        } else {
          static_assert(is_thrust_tuple_of_arithmetic<payload_t>::value);
          min_element_size =
            std::min(cugraph::min_thrust_tuple_element_sizes<payload_t>(), min_element_size);
        }
      }
      assert((cache_line_size % min_element_size) == 0);
      auto alignment = cache_line_size / min_element_size;
      std::optional<std::conditional_t<try_compression, std::variant<key_t, uint32_t>, key_t>>
        invalid_key{std::nullopt};

      if (avg_key_buffer_size >= alignment * size_t{128} /* 128 tuning parameter */) {
        if constexpr (std::is_same_v<key_t, vertex_t>) {
          if constexpr (try_compression) {
            if (compressed_v_buffer) {
              invalid_key = std::numeric_limits<uint32_t>::max();
            } else {
              invalid_key = invalid_vertex_id_v<vertex_t>;
            }
          } else {
            invalid_key = invalid_vertex_id_v<vertex_t>;
          }
        } else {
          invalid_key                  = key_t{};
          thrust::get<0>(*invalid_key) = invalid_vertex_id_v<vertex_t>;
        }

        if constexpr (try_compression) {
          if (compressed_v_buffer) {
            auto rx_compressed_v_buffer =
              allocate_dataframe_buffer<uint32_t>(size_t{0}, handle.get_stream());
            std::tie(rx_compressed_v_buffer,
                     std::ignore,
                     std::ignore,
                     std::ignore,
                     std::ignore,
                     std::ignore,
                     std::ignore) =
              shuffle_values(major_comm,
                             get_dataframe_buffer_begin(*compressed_v_buffer),
                             raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                             alignment,
                             std::make_optional(std::get<1>(*invalid_key)),
                             handle.get_stream());
            compressed_v_buffer = std::move(rx_compressed_v_buffer);
          } else {
            auto rx_key_buffer = allocate_dataframe_buffer<key_t>(size_t{0}, handle.get_stream());
            std::tie(rx_key_buffer,
                     std::ignore,
                     std::ignore,
                     std::ignore,
                     std::ignore,
                     std::ignore,
                     std::ignore) =
              shuffle_values(major_comm,
                             get_dataframe_buffer_begin(key_buffer),
                             raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                             alignment,
                             std::make_optional(std::get<0>(*invalid_key)),
                             handle.get_stream());
            key_buffer = std::move(rx_key_buffer);
          }
        } else {
          auto rx_key_buffer = allocate_dataframe_buffer<key_t>(size_t{0}, handle.get_stream());
          std::tie(rx_key_buffer,
                   std::ignore,
                   std::ignore,
                   std::ignore,
                   std::ignore,
                   std::ignore,
                   std::ignore) =
            shuffle_values(major_comm,
                           get_dataframe_buffer_begin(key_buffer),
                           raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                           alignment,
                           invalid_key,
                           handle.get_stream());
          key_buffer = std::move(rx_key_buffer);
        }
        if constexpr (!std::is_same_v<payload_t, void>) {
          auto rx_payload_buffer =
            allocate_dataframe_buffer<payload_t>(size_t{0}, handle.get_stream());
          std::tie(rx_payload_buffer,
                   std::ignore,
                   std::ignore,
                   std::ignore,
                   std::ignore,
                   std::ignore,
                   std::ignore) =
            shuffle_values(major_comm,
                           get_dataframe_buffer_begin(payload_buffer),
                           raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                           alignment,
                           std::nullopt,
                           handle.get_stream());
          payload_buffer = std::move(rx_payload_buffer);
        }
      } else {
        if constexpr (try_compression) {
          if (compressed_v_buffer) {
            auto rx_compressed_v_buffer =
              allocate_dataframe_buffer<uint32_t>(size_t{0}, handle.get_stream());
            std::tie(rx_compressed_v_buffer, std::ignore) =
              shuffle_values(major_comm,
                             get_dataframe_buffer_begin(*compressed_v_buffer),
                             raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                             handle.get_stream());
            compressed_v_buffer = std::move(rx_compressed_v_buffer);
          } else {
            auto rx_key_buffer = allocate_dataframe_buffer<key_t>(size_t{0}, handle.get_stream());
            std::tie(rx_key_buffer, std::ignore) =
              shuffle_values(major_comm,
                             get_dataframe_buffer_begin(key_buffer),
                             raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                             handle.get_stream());
            key_buffer = std::move(rx_key_buffer);
          }
        } else {
          auto rx_key_buffer = allocate_dataframe_buffer<key_t>(size_t{0}, handle.get_stream());
          std::tie(rx_key_buffer, std::ignore) =
            shuffle_values(major_comm,
                           get_dataframe_buffer_begin(key_buffer),
                           raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                           handle.get_stream());
          key_buffer = std::move(rx_key_buffer);
        }

        if constexpr (!std::is_same_v<payload_t, void>) {
          auto rx_payload_buffer =
            allocate_dataframe_buffer<payload_t>(size_t{0}, handle.get_stream());
          std::tie(rx_payload_buffer, std::ignore) =
            shuffle_values(major_comm,
                           get_dataframe_buffer_begin(payload_buffer),
                           raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                           handle.get_stream());
          payload_buffer = std::move(rx_payload_buffer);
        }
      }

      if constexpr (std::is_integral_v<key_t>) {
        vertex_range = std::make_tuple(graph_view.local_vertex_partition_range_first(),
                                       graph_view.local_vertex_partition_range_last());
      }
      if constexpr (try_compression) {
        if (compressed_v_buffer) {
          std::tie(key_buffer, payload_buffer) =
            detail::sort_and_reduce_buffer_elements<uint32_t, key_t, payload_t, ReduceOp>(
              handle,
              std::move(*compressed_v_buffer),
              std::move(payload_buffer),
              reduce_op,
              vertex_range,
              invalid_key ? std::make_optional(std::get<1>(*invalid_key)) : std::nullopt);
        } else {
          std::tie(key_buffer, payload_buffer) =
            detail::sort_and_reduce_buffer_elements<key_t, key_t, payload_t, ReduceOp>(
              handle,
              std::move(key_buffer),
              std::move(payload_buffer),
              reduce_op,
              vertex_range,
              invalid_key ? std::make_optional(std::get<0>(*invalid_key)) : std::nullopt);
        }
      } else {
        std::tie(key_buffer, payload_buffer) =
          detail::sort_and_reduce_buffer_elements<key_t, key_t, payload_t, ReduceOp>(
            handle,
            std::move(key_buffer),
            std::move(payload_buffer),
            reduce_op,
            vertex_range,
            invalid_key);
      }
    }
  }

  if constexpr (!std::is_same_v<payload_t, void>) {
    return std::make_tuple(std::move(key_buffer), std::move(payload_buffer));
  } else {
    return std::move(key_buffer);
  }
}

}  // namespace detail

template <typename GraphViewType, typename KeyBucketType>
size_t compute_num_out_nbrs_from_frontier(raft::handle_t const& handle,
                                          GraphViewType const& graph_view,
                                          KeyBucketType const& frontier)
{
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename KeyBucketType::key_type;

  size_t ret{0};

  auto local_frontier_vertex_first =
    thrust_tuple_get_or_identity<decltype(frontier.begin()), 0>(frontier.begin());

  std::vector<size_t> local_frontier_sizes{};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& minor_comm     = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    local_frontier_sizes = host_scalar_allgather(minor_comm, frontier.size(), handle.get_stream());
  } else {
    local_frontier_sizes = std::vector<size_t>{static_cast<size_t>(frontier.size())};
  }

  auto edge_mask_view = graph_view.edge_mask_view();

  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    auto edge_partition_e_mask =
      edge_mask_view
        ? cuda::std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : cuda::std::nullopt;

    if constexpr (GraphViewType::is_multi_gpu) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_rank = minor_comm.get_rank();

      rmm::device_uvector<vertex_t> edge_partition_frontier_vertices(local_frontier_sizes[i],
                                                                     handle.get_stream());
      device_bcast(minor_comm,
                   local_frontier_vertex_first,
                   edge_partition_frontier_vertices.data(),
                   local_frontier_sizes[i],
                   static_cast<int>(i),
                   handle.get_stream());

      if (edge_partition_e_mask) {
        ret +=
          edge_partition.compute_number_of_edges_with_mask((*edge_partition_e_mask).value_first(),
                                                           edge_partition_frontier_vertices.begin(),
                                                           edge_partition_frontier_vertices.end(),
                                                           handle.get_stream());
      } else {
        ret += edge_partition.compute_number_of_edges(edge_partition_frontier_vertices.begin(),
                                                      edge_partition_frontier_vertices.end(),
                                                      handle.get_stream());
      }
    } else {
      assert(i == 0);
      if (edge_partition_e_mask) {
        ret += edge_partition.compute_number_of_edges_with_mask(
          (*edge_partition_e_mask).value_first(),
          local_frontier_vertex_first,
          local_frontier_vertex_first + frontier.size(),
          handle.get_stream());
      } else {
        ret += edge_partition.compute_number_of_edges(local_frontier_vertex_first,
                                                      local_frontier_vertex_first + frontier.size(),
                                                      handle.get_stream());
      }
    }
  }

  return ret;
}

/**
 * @brief Iterate over outgoing edges from the current vertex frontier and reduce valid edge functor
 * outputs by (tagged-)destination ID.
 *
 * Edge functor outputs are evaluated if the predicate operator returns true. Vertices are assumed
 * to be tagged if KeyBucketType::key_type is a tuple of a vertex type and a tag type
 * (KeyBucketType::key_type is identical to a vertex type otherwise).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam KeyBucketType Type of the vertex frontier bucket class which abstracts the
 * current (tagged-)vertex frontier.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam PredOp Type of the quinary predicate operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param frontier KeyBucketType class object for the current vertex frontier.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either cugraph::edge_src_property_t::view()
 * (if @p e_op needs to access source property values) or cugraph::edge_src_dummy_property_t::view()
 * (if @p e_op does not access source property values). Use update_edge_src_property to fill the
 * wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_dst_property_t::view() (if @p e_op needs to access destination property values) or
 * cugraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either cugraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param e_op Quinary operator takes edge (tagged-)source, edge destination, property values for
 * the source, destination, and edge and 1) just returns (return value = void, if vertices are not
 * tagged and ReduceOp::value_type is void, in this case, @p e_op is dummy and won't be called); 2)
 * returns a value to be reduced (if vertices are not tagged and ReduceOp::value_type is not void);
 * 3) returns a tag (if vertices are tagged and ReduceOp::value_type is void); or 4) returns a tuple
 * of a tag and a value to be reduced (if vertices are tagged and ReduceOp::value_type is not void).
 * @param reduce_op Binary operator that takes two input arguments and reduce the two values to one.
 * There are pre-defined reduction operators in prims/reduce_op.cuh. It is
 * recommended to use the pre-defined reduction operators whenever possible as the current (and
 * future) implementations of graph primitives may check whether @p ReduceOp is a known type (or has
 * known member variables) to take a more optimized code path. See the documentation in the
 * reduce_op.cuh file for instructions on writing custom reduction operators.
 * @param pred_op Quinary predicate operator takes edge (tagged-)source, edge destination, property
 * values for the source, destination, and edge and returns whether this edge should be included (if
 * true is returned) or excluded.
 * @return Tuple of key values and payload values (if ReduceOp::value_type is not void) or just key
 * values (if ReduceOp::value_type is void). Keys in the return values are sorted in ascending order
 * using a vertex ID as the primary key and a tag (if relevant) as the secondary key.
 */
template <typename GraphViewType,
          typename KeyBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename ReduceOp,
          typename PredOp>
std::conditional_t<
  !std::is_same_v<typename ReduceOp::value_type, void>,
  std::tuple<dataframe_buffer_type_t<typename KeyBucketType::key_type>,
             detail::optional_dataframe_buffer_type_t<typename ReduceOp::value_type>>,
  dataframe_buffer_type_t<typename KeyBucketType::key_type>>
transform_reduce_if_v_frontier_outgoing_e_by_dst(raft::handle_t const& handle,
                                                 GraphViewType const& graph_view,
                                                 KeyBucketType const& frontier,
                                                 EdgeSrcValueInputWrapper edge_src_value_input,
                                                 EdgeDstValueInputWrapper edge_dst_value_input,
                                                 EdgeValueInputWrapper edge_value_input,
                                                 EdgeOp e_op,
                                                 ReduceOp reduce_op,
                                                 PredOp pred_op,
                                                 bool do_expensive_check = false)
{
  return detail::transform_reduce_if_v_frontier_outgoing_e_by_dst(handle,
                                                                  graph_view,
                                                                  frontier,
                                                                  edge_src_value_input,
                                                                  edge_dst_value_input,
                                                                  edge_value_input,
                                                                  e_op,
                                                                  reduce_op,
                                                                  pred_op,
                                                                  do_expensive_check);
}

}  // namespace cugraph

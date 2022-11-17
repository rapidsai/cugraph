/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <detail/graph_utils.cuh>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>

#include <tuple>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>& d_edgelist_majors,
  rmm::device_uvector<vertex_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<weight_t>>& d_edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>&
    d_edgelist_id_type_pairs,
  bool groupby_and_count_local_partition_by_minor)
{
  auto& comm               = handle.get_comms();
  auto const comm_size     = comm.get_size();
  auto const comm_rank     = comm.get_rank();
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_size = row_comm.get_size();
  auto const row_comm_rank = row_comm.get_rank();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();
  auto const col_comm_rank = col_comm.get_rank();

  auto total_global_mem = handle.get_device_properties().totalGlobalMem;
  auto element_size =
    sizeof(vertex_t) * 2 + (d_edgelist_weights ? sizeof(weight_t) : size_t{0}) +
    (d_edgelist_id_type_pairs ? (sizeof(edge_t) + sizeof(edge_type_t)) : size_t{0});
  auto constexpr mem_frugal_ratio =
    0.1;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
          // total_global_mem, switch to the memory frugal approach (thrust::sort is used to
          // group-by by default, and thrust::sort requires temporary buffer comparable to the input
          // data size)
  auto mem_frugal_threshold =
    static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

  auto pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(d_edgelist_majors.begin(), d_edgelist_minors.begin()));

  if (groupby_and_count_local_partition_by_minor) {
    auto local_partition_id_gpu_id_pair_op =
      [comm_size,
       row_comm_size,
       partition_id_key_func =
         cugraph::detail::compute_partition_id_from_ext_edge_endpoints_t<vertex_t>{
           comm_size, row_comm_size, col_comm_size},
       gpu_id_key_func = cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
         comm_size}] __device__(auto pair) {
        auto local_partition_id =
          partition_id_key_func(thrust::get<0>(pair), thrust::get<1>(pair)) /
          comm_size;  // global partition id to local partition id
        return local_partition_id * row_comm_size +
               (gpu_id_key_func(thrust::get<1>(pair)) % row_comm_size);
      };

    return d_edgelist_weights
             ? (d_edgelist_id_type_pairs
                  ? cugraph::groupby_and_count(pair_first,
                                               pair_first + d_edgelist_majors.size(),
                                               thrust::make_zip_iterator(thrust::make_tuple(
                                                 d_edgelist_weights->begin(),
                                                 std::get<0>(*d_edgelist_id_type_pairs).begin(),
                                                 std::get<1>(*d_edgelist_id_type_pairs).begin())),
                                               local_partition_id_gpu_id_pair_op,
                                               comm_size,
                                               mem_frugal_threshold,
                                               handle.get_stream())
                  : cugraph::groupby_and_count(pair_first,
                                               pair_first + d_edgelist_majors.size(),
                                               d_edgelist_weights->begin(),
                                               local_partition_id_gpu_id_pair_op,
                                               comm_size,
                                               mem_frugal_threshold,
                                               handle.get_stream()))
             : (d_edgelist_id_type_pairs
                  ? cugraph::groupby_and_count(
                      pair_first,
                      pair_first + d_edgelist_majors.size(),
                      get_dataframe_buffer_begin(*d_edgelist_id_type_pairs),
                      local_partition_id_gpu_id_pair_op,
                      comm_size,
                      mem_frugal_threshold,
                      handle.get_stream())
                  : cugraph::groupby_and_count(pair_first,
                                               pair_first + d_edgelist_majors.size(),
                                               local_partition_id_gpu_id_pair_op,
                                               comm_size,
                                               mem_frugal_threshold,
                                               handle.get_stream()));
  } else {
    auto local_partition_id_op =
      [comm_size,
       key_func = cugraph::detail::compute_partition_id_from_ext_edge_endpoints_t<vertex_t>{
         comm_size, row_comm_size, col_comm_size}] __device__(auto pair) {
        return key_func(thrust::get<0>(pair), thrust::get<1>(pair)) /
               comm_size;  // global partition id to local partition id
      };

    return d_edgelist_weights
             ? (d_edgelist_id_type_pairs
                  ? cugraph::groupby_and_count(pair_first,
                                               pair_first + d_edgelist_majors.size(),
                                               thrust::make_zip_iterator(thrust::make_tuple(
                                                 d_edgelist_weights->begin(),
                                                 std::get<0>(*d_edgelist_id_type_pairs).begin(),
                                                 std::get<1>(*d_edgelist_id_type_pairs).begin())),
                                               local_partition_id_op,
                                               col_comm_size,
                                               mem_frugal_threshold,
                                               handle.get_stream())
                  : cugraph::groupby_and_count(pair_first,
                                               pair_first + d_edgelist_majors.size(),
                                               d_edgelist_weights->begin(),
                                               local_partition_id_op,
                                               col_comm_size,
                                               mem_frugal_threshold,
                                               handle.get_stream()))
             : (d_edgelist_id_type_pairs
                  ? cugraph::groupby_and_count(
                      pair_first,
                      pair_first + d_edgelist_majors.size(),
                      get_dataframe_buffer_begin(*d_edgelist_id_type_pairs),
                      local_partition_id_op,
                      col_comm_size,
                      mem_frugal_threshold,
                      handle.get_stream())
                  : cugraph::groupby_and_count(pair_first,
                                               pair_first + d_edgelist_majors.size(),
                                               local_partition_id_op,
                                               col_comm_size,
                                               mem_frugal_threshold,
                                               handle.get_stream()));
  }
}

template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& d_edgelist_majors,
  rmm::device_uvector<int32_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<float>>& d_edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>&
    d_edgelist_id_type_pairs,
  bool groupby_and_counts_local_partition);

template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& d_edgelist_majors,
  rmm::device_uvector<int32_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<double>>& d_edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>&
    d_edgelist_id_type_pairs,
  bool groupby_and_counts_local_partition);

template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& d_edgelist_majors,
  rmm::device_uvector<int32_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<float>>& d_edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&
    d_edgelist_id_type_pairs,
  bool groupby_and_counts_local_partition);

template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& d_edgelist_majors,
  rmm::device_uvector<int32_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<double>>& d_edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&
    d_edgelist_id_type_pairs,
  bool groupby_and_counts_local_partition);

template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>& d_edgelist_majors,
  rmm::device_uvector<int64_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<float>>& d_edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&
    d_edgelist_id_type_pairs,
  bool groupby_and_counts_local_partition);

template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>& d_edgelist_majors,
  rmm::device_uvector<int64_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<double>>& d_edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&
    d_edgelist_id_type_pairs,
  bool groupby_and_counts_local_partition);

}  // namespace detail
}  // namespace cugraph

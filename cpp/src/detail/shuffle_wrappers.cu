/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cugraph/detail/graph_utils.cuh>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <rmm/exec_policy.hpp>

#include <tuple>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
shuffle_edgelist_by_gpu_id(raft::handle_t const& handle,
                           rmm::device_uvector<vertex_t>&& d_edgelist_majors,
                           rmm::device_uvector<vertex_t>&& d_edgelist_minors,
                           std::optional<rmm::device_uvector<weight_t>>&& d_edgelist_weights)
{
  auto& comm               = handle.get_comms();
  auto const comm_size     = comm.get_size();
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_size = row_comm.get_size();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();

  rmm::device_uvector<vertex_t> d_rx_edgelist_majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> d_rx_edgelist_minors(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> d_rx_edgelist_weights{std::nullopt};
  if (d_edgelist_weights) {
    auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(
      d_edgelist_majors.begin(), d_edgelist_minors.begin(), (*d_edgelist_weights).begin()));

    std::forward_as_tuple(
      std::tie(d_rx_edgelist_majors, d_rx_edgelist_minors, d_rx_edgelist_weights),
      std::ignore) =
      cugraph::groupby_gpuid_and_shuffle_values(
        comm,  // handle.get_comms(),
        edge_first,
        edge_first + d_edgelist_majors.size(),
        [key_func =
           cugraph::detail::compute_gpu_id_from_edge_t<vertex_t>{
             comm_size, row_comm_size, col_comm_size}] __device__(auto val) {
          return key_func(thrust::get<0>(val), thrust::get<1>(val));
        },
        handle.get_stream());
  } else {
    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(d_edgelist_majors.begin(), d_edgelist_minors.begin()));

    std::forward_as_tuple(std::tie(d_rx_edgelist_majors, d_rx_edgelist_minors),
                          std::ignore) =
      cugraph::groupby_gpuid_and_shuffle_values(
        comm,  // handle.get_comms(),
        edge_first,
        edge_first + d_edgelist_majors.size(),
        [key_func =
           cugraph::detail::compute_gpu_id_from_edge_t<vertex_t>{
             comm_size, row_comm_size, col_comm_size}] __device__(auto val) {
          return key_func(thrust::get<0>(val), thrust::get<1>(val));
        },
        handle.get_stream());
  }

  return std::make_tuple(std::move(d_rx_edgelist_majors),
                         std::move(d_rx_edgelist_minors),
                         std::move(d_rx_edgelist_weights));
}

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
shuffle_edgelist_by_gpu_id(raft::handle_t const& handle,
                           rmm::device_uvector<int32_t>&& d_edgelist_majors,
                           rmm::device_uvector<int32_t>&& d_edgelist_minors,
                           std::optional<rmm::device_uvector<float>>&& d_edgelist_weights);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
shuffle_edgelist_by_gpu_id(raft::handle_t const& handle,
                           rmm::device_uvector<int32_t>&& d_edgelist_majors,
                           rmm::device_uvector<int32_t>&& d_edgelist_minors,
                           std::optional<rmm::device_uvector<double>>&& d_edgelist_weights);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>>
shuffle_edgelist_by_gpu_id(raft::handle_t const& handle,
                           rmm::device_uvector<int64_t>&& d_edgelist_majors,
                           rmm::device_uvector<int64_t>&& d_edgelist_minors,
                           std::optional<rmm::device_uvector<float>>&& d_edgelist_weights);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>>
shuffle_edgelist_by_gpu_id(raft::handle_t const& handle,
                           rmm::device_uvector<int64_t>&& d_edgelist_majors,
                           rmm::device_uvector<int64_t>&& d_edgelist_minors,
                           std::optional<rmm::device_uvector<double>>&& d_edgelist_weights);

template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_vertices_by_gpu_id(raft::handle_t const& handle,
                                                         rmm::device_uvector<vertex_t>&& d_vertices)
{
  auto& comm           = handle.get_comms();
  auto const comm_size = comm.get_size();

  rmm::device_uvector<vertex_t> d_rx_vertices(0, handle.get_stream());
  std::tie(d_rx_vertices, std::ignore) = cugraph::groupby_gpuid_and_shuffle_values(
    comm,  // handle.get_comms(),
    d_vertices.begin(),
    d_vertices.end(),
    [key_func = cugraph::detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size}] __device__(
      auto val) { return key_func(val); },
    handle.get_stream());

  return d_rx_vertices;
}

template rmm::device_uvector<int32_t> shuffle_vertices_by_gpu_id(
  raft::handle_t const& handle, rmm::device_uvector<int32_t>&& d_vertices);

template rmm::device_uvector<int64_t> shuffle_vertices_by_gpu_id(
  raft::handle_t const& handle, rmm::device_uvector<int64_t>&& d_vertices);

template <typename vertex_t, typename weight_t>
rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>& d_edgelist_majors,
  rmm::device_uvector<vertex_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<weight_t>>& d_edgelist_weights,
  bool groupby_and_count_local_partition)
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

  auto pair_first = thrust::make_zip_iterator(
    thrust::make_tuple(d_edgelist_majors.begin(), d_edgelist_minors.begin()));

  if (groupby_and_count_local_partition) {
    auto local_partition_id_gpu_id_pair_op =
      [comm_size,
       row_comm_size,
       partition_id_key_func =
         cugraph::detail::compute_partition_id_from_edge_t<vertex_t>{
           comm_size, row_comm_size, col_comm_size},
       gpu_id_key_func =
         cugraph::detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size}] __device__(auto pair) {
        auto local_partition_id =
          partition_id_key_func(thrust::get<0>(pair), thrust::get<1>(pair)) /
          comm_size;  // global partition id to local partition id
        return local_partition_id * row_comm_size +
               (gpu_id_key_func(thrust::get<1>(pair)) % row_comm_size);
      };

    return d_edgelist_weights ? cugraph::groupby_and_count(pair_first,
                                                           pair_first + d_edgelist_majors.size(),
                                                           d_edgelist_weights->begin(),
                                                           local_partition_id_gpu_id_pair_op,
                                                           comm_size,
                                                           handle.get_stream())
                              : cugraph::groupby_and_count(pair_first,
                                                           pair_first + d_edgelist_majors.size(),
                                                           local_partition_id_gpu_id_pair_op,
                                                           comm_size,
                                                           handle.get_stream());
  } else {
    auto local_partition_id_op =
      [comm_size,
       key_func = cugraph::detail::compute_partition_id_from_edge_t<vertex_t>{
         comm_size, row_comm_size, col_comm_size}] __device__(auto pair) {
        return key_func(thrust::get<0>(pair), thrust::get<1>(pair)) /
               comm_size;  // global partition id to local partition id
      };

    return d_edgelist_weights ? cugraph::groupby_and_count(pair_first,
                                                           pair_first + d_edgelist_majors.size(),
                                                           d_edgelist_weights->begin(),
                                                           local_partition_id_op,
                                                           col_comm_size,
                                                           handle.get_stream())
                              : cugraph::groupby_and_count(pair_first,
                                                           pair_first + d_edgelist_majors.size(),
                                                           local_partition_id_op,
                                                           col_comm_size,
                                                           handle.get_stream());
  }
}

template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& d_edgelist_majors,
  rmm::device_uvector<int32_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<float>>& d_edgelist_weights,
  bool groupby_and_counts_local_partition);

template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& d_edgelist_majors,
  rmm::device_uvector<int32_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<double>>& d_edgelist_weights,
  bool groupby_and_counts_local_partition);

template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>& d_edgelist_majors,
  rmm::device_uvector<int64_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<float>>& d_edgelist_weights,
  bool groupby_and_counts_local_partition);

template rmm::device_uvector<size_t> groupby_and_count_edgelist_by_local_partition_id(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>& d_edgelist_majors,
  rmm::device_uvector<int64_t>& d_edgelist_minors,
  std::optional<rmm::device_uvector<double>>& d_edgelist_weights,
  bool groupby_and_counts_local_partition);

}  // namespace detail
}  // namespace cugraph

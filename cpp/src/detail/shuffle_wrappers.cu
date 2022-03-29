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
#include <cugraph/detail/graph_utils.cuh>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>
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

  auto total_global_mem = handle.get_device_properties().totalGlobalMem;
  auto element_size = sizeof(vertex_t) * 2 + (d_edgelist_weights ? sizeof(weight_t) : size_t{0});
  auto constexpr mem_frugal_ratio =
    0.1;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
          // total_global_mem, switch to the memory frugal approach (thrust::sort is used to
          // group-by by default, and thrust::sort requires temporary buffer comparable to the input
          // data size)
  auto mem_frugal_threshold =
    static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

  auto mem_frugal_flag =
    host_scalar_allreduce(comm,
                          d_edgelist_majors.size() > mem_frugal_threshold ? int{1} : int{0},
                          raft::comms::op_t::MAX,
                          handle.get_stream());

  // invoke groupby_and_count and shuffle values to pass mem_frugal_threshold instead of directly
  // calling groupby_gpu_id_and_shuffle_values there is no benefit in reducing peak memory as we
  // need to allocate a receive buffer anyways) but this reduces the maximum memory allocation size
  // by half or more (thrust::sort used inside the groupby_and_count allocates the entire temporary
  // buffer in a single chunk, and the pool allocator  often cannot handle a large single allocation
  // (due to fragmentation) even when the remaining free memory in aggregate is significantly larger
  // than the requested size).
  rmm::device_uvector<vertex_t> d_rx_edgelist_majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> d_rx_edgelist_minors(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> d_rx_edgelist_weights{std::nullopt};
  if (d_edgelist_weights) {
    auto edge_first = thrust::make_zip_iterator(thrust::make_tuple(
      d_edgelist_majors.begin(), d_edgelist_minors.begin(), (*d_edgelist_weights).begin()));

    auto d_tx_value_counts = cugraph::groupby_and_count(
      edge_first,
      edge_first + d_edgelist_majors.size(),
      [key_func =
         cugraph::detail::compute_gpu_id_from_edge_t<vertex_t>{
           comm_size, row_comm_size, col_comm_size}] __device__(auto val) {
        return key_func(thrust::get<0>(val), thrust::get<1>(val));
      },
      comm_size,
      mem_frugal_threshold,
      handle.get_stream());

    std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
    raft::update_host(h_tx_value_counts.data(),
                      d_tx_value_counts.data(),
                      d_tx_value_counts.size(),
                      handle.get_stream());
    handle.sync_stream();

    if (mem_frugal_flag) {  // trade-off potential parallelism to lower peak memory
      std::tie(d_rx_edgelist_majors, std::ignore) =
        shuffle_values(comm, d_edgelist_majors.begin(), h_tx_value_counts, handle.get_stream());
      d_edgelist_majors.resize(0, handle.get_stream());
      d_edgelist_majors.shrink_to_fit(handle.get_stream());

      std::tie(d_rx_edgelist_minors, std::ignore) =
        shuffle_values(comm, d_edgelist_minors.begin(), h_tx_value_counts, handle.get_stream());
      d_edgelist_minors.resize(0, handle.get_stream());
      d_edgelist_minors.shrink_to_fit(handle.get_stream());

      std::tie(d_rx_edgelist_weights, std::ignore) =
        shuffle_values(comm, (*d_edgelist_weights).begin(), h_tx_value_counts, handle.get_stream());
      (*d_edgelist_weights).resize(0, handle.get_stream());
      (*d_edgelist_weights).shrink_to_fit(handle.get_stream());
    } else {
      std::forward_as_tuple(
        std::tie(d_rx_edgelist_majors, d_rx_edgelist_minors, d_rx_edgelist_weights), std::ignore) =
        shuffle_values(comm, edge_first, h_tx_value_counts, handle.get_stream());
      d_edgelist_majors.resize(0, handle.get_stream());
      d_edgelist_majors.shrink_to_fit(handle.get_stream());
      d_edgelist_minors.resize(0, handle.get_stream());
      d_edgelist_minors.shrink_to_fit(handle.get_stream());
      (*d_edgelist_weights).resize(0, handle.get_stream());
      (*d_edgelist_weights).shrink_to_fit(handle.get_stream());
    }
  } else {
    auto edge_first = thrust::make_zip_iterator(
      thrust::make_tuple(d_edgelist_majors.begin(), d_edgelist_minors.begin()));

    auto d_tx_value_counts = cugraph::groupby_and_count(
      edge_first,
      edge_first + d_edgelist_majors.size(),
      [key_func =
         cugraph::detail::compute_gpu_id_from_edge_t<vertex_t>{
           comm_size, row_comm_size, col_comm_size}] __device__(auto val) {
        return key_func(thrust::get<0>(val), thrust::get<1>(val));
      },
      comm_size,
      mem_frugal_threshold,
      handle.get_stream());

    std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
    raft::update_host(h_tx_value_counts.data(),
                      d_tx_value_counts.data(),
                      d_tx_value_counts.size(),
                      handle.get_stream());
    handle.sync_stream();

    if (mem_frugal_flag) {  // trade-off potential parallelism to lower peak memory
      std::tie(d_rx_edgelist_majors, std::ignore) =
        shuffle_values(comm, d_edgelist_majors.begin(), h_tx_value_counts, handle.get_stream());
      d_edgelist_majors.resize(0, handle.get_stream());
      d_edgelist_majors.shrink_to_fit(handle.get_stream());

      std::tie(d_rx_edgelist_minors, std::ignore) =
        shuffle_values(comm, d_edgelist_minors.begin(), h_tx_value_counts, handle.get_stream());
      d_edgelist_minors.resize(0, handle.get_stream());
      d_edgelist_minors.shrink_to_fit(handle.get_stream());
    } else {
      std::forward_as_tuple(std::tie(d_rx_edgelist_majors, d_rx_edgelist_minors), std::ignore) =
        shuffle_values(comm, edge_first, h_tx_value_counts, handle.get_stream());
      d_edgelist_majors.resize(0, handle.get_stream());
      d_edgelist_majors.shrink_to_fit(handle.get_stream());
      d_edgelist_minors.resize(0, handle.get_stream());
      d_edgelist_minors.shrink_to_fit(handle.get_stream());
    }
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
  std::tie(d_rx_vertices, std::ignore) = cugraph::groupby_gpu_id_and_shuffle_values(
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
  auto element_size = sizeof(vertex_t) * 2 + (d_edgelist_weights ? sizeof(weight_t) : size_t{0});
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
                                                           mem_frugal_threshold,
                                                           handle.get_stream())
                              : cugraph::groupby_and_count(pair_first,
                                                           pair_first + d_edgelist_majors.size(),
                                                           local_partition_id_gpu_id_pair_op,
                                                           comm_size,
                                                           mem_frugal_threshold,
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
                                                           mem_frugal_threshold,
                                                           handle.get_stream())
                              : cugraph::groupby_and_count(pair_first,
                                                           pair_first + d_edgelist_majors.size(),
                                                           local_partition_id_op,
                                                           col_comm_size,
                                                           mem_frugal_threshold,
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

template <typename vertex_t, typename value_t, bool multi_gpu>
rmm::device_uvector<value_t> collect_local_vertex_values_from_ext_vertex_value_pairs(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& d_vertices,
  rmm::device_uvector<value_t>&& d_values,
  rmm::device_uvector<vertex_t> const& number_map,
  vertex_t local_vertex_first,
  vertex_t local_vertex_last,
  value_t default_value,
  bool do_expensive_check)
{
  rmm::device_uvector<value_t> d_local_values(0, handle.get_stream());

  if constexpr (multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();

    std::tie(d_vertices, d_values, std::ignore) = cugraph::groupby_gpu_id_and_shuffle_kv_pairs(
      comm,
      d_vertices.begin(),
      d_vertices.end(),
      d_values.begin(),
      cugraph::detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size},
      handle.get_stream());
  }

  // Now I can renumber locally
  renumber_local_ext_vertices<vertex_t, multi_gpu>(handle,
                                                   d_vertices.data(),
                                                   d_vertices.size(),
                                                   number_map.data(),
                                                   local_vertex_first,
                                                   local_vertex_last,
                                                   do_expensive_check);

  auto vertex_iterator = thrust::make_transform_iterator(
    d_vertices.begin(),
    [local_vertex_first] __device__(vertex_t v) { return v - local_vertex_first; });

  d_local_values.resize(local_vertex_last - local_vertex_first, handle.get_stream());
  thrust::fill(
    handle.get_thrust_policy(), d_local_values.begin(), d_local_values.end(), default_value);

  thrust::scatter(handle.get_thrust_policy(),
                  d_values.begin(),
                  d_values.end(),
                  vertex_iterator,
                  d_local_values.begin());

  return d_local_values;
}

template rmm::device_uvector<float>
collect_local_vertex_values_from_ext_vertex_value_pairs<int32_t, float, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<float>&& d_values,
  rmm::device_uvector<int32_t> const& number_map,
  int32_t local_vertex_first,
  int32_t local_vertex_last,
  float default_value,
  bool do_expensive_check);

template rmm::device_uvector<float>
collect_local_vertex_values_from_ext_vertex_value_pairs<int64_t, float, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<float>&& d_values,
  rmm::device_uvector<int64_t> const& number_map,
  int64_t local_vertex_first,
  int64_t local_vertex_last,
  float default_value,
  bool do_expensive_check);

template rmm::device_uvector<double>
collect_local_vertex_values_from_ext_vertex_value_pairs<int32_t, double, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<double>&& d_values,
  rmm::device_uvector<int32_t> const& number_map,
  int32_t local_vertex_first,
  int32_t local_vertex_last,
  double default_value,
  bool do_expensive_check);

template rmm::device_uvector<double>
collect_local_vertex_values_from_ext_vertex_value_pairs<int64_t, double, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<double>&& d_values,
  rmm::device_uvector<int64_t> const& number_map,
  int64_t local_vertex_first,
  int64_t local_vertex_last,
  double default_value,
  bool do_expensive_check);

template rmm::device_uvector<float>
collect_local_vertex_values_from_ext_vertex_value_pairs<int32_t, float, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<float>&& d_values,
  rmm::device_uvector<int32_t> const& number_map,
  int32_t local_vertex_first,
  int32_t local_vertex_last,
  float default_value,
  bool do_expensive_check);

template rmm::device_uvector<float>
collect_local_vertex_values_from_ext_vertex_value_pairs<int64_t, float, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<float>&& d_values,
  rmm::device_uvector<int64_t> const& number_map,
  int64_t local_vertex_first,
  int64_t local_vertex_last,
  float default_value,
  bool do_expensive_check);

template rmm::device_uvector<double>
collect_local_vertex_values_from_ext_vertex_value_pairs<int32_t, double, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<double>&& d_values,
  rmm::device_uvector<int32_t> const& number_map,
  int32_t local_vertex_first,
  int32_t local_vertex_last,
  double default_value,
  bool do_expensive_check);

template rmm::device_uvector<double>
collect_local_vertex_values_from_ext_vertex_value_pairs<int64_t, double, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<double>&& d_values,
  rmm::device_uvector<int64_t> const& number_map,
  int64_t local_vertex_first,
  int64_t local_vertex_last,
  double default_value,
  bool do_expensive_check);

}  // namespace detail
}  // namespace cugraph

/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <tuple>

namespace cugraph {

namespace {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename func_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>>
shuffle_vertex_pairs_by_gpu_id_impl(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::optional<rmm::device_uvector<weight_t>>&& weights,
  std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>&&
    edge_id_type_tuple,
  func_t func)
{
  auto& comm           = handle.get_comms();
  auto const comm_size = comm.get_size();

  auto total_global_mem = handle.get_device_properties().totalGlobalMem;
  auto element_size     = sizeof(vertex_t) * 2 + (weights ? sizeof(weight_t) : size_t{0});
  auto constexpr mem_frugal_ratio =
    0.1;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
          // total_global_mem, switch to the memory frugal approach (thrust::sort is used to
          // group-by by default, and thrust::sort requires temporary buffer comparable to the input
          // data size)
  auto mem_frugal_threshold =
    static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

  auto mem_frugal_flag =
    host_scalar_allreduce(comm,
                          majors.size() > mem_frugal_threshold ? int{1} : int{0},
                          raft::comms::op_t::MAX,
                          handle.get_stream());

  // invoke groupby_and_count and shuffle values to pass mem_frugal_threshold instead of directly
  // calling groupby_gpu_id_and_shuffle_values there is no benefit in reducing peak memory as we
  // need to allocate a receive buffer anyways) but this reduces the maximum memory allocation size
  // by half or more (thrust::sort used inside the groupby_and_count allocates the entire temporary
  // buffer in a single chunk, and the pool allocator  often cannot handle a large single allocation
  // (due to fragmentation) even when the remaining free memory in aggregate is significantly larger
  // than the requested size).
  rmm::device_uvector<vertex_t> rx_majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> rx_minors(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> rx_weights{std::nullopt};
  std::optional<rmm::device_uvector<edge_t>> rx_edge_ids{std::nullopt};
  std::optional<rmm::device_uvector<edge_type_t>> rx_edge_types{std::nullopt};

  rmm::device_uvector<size_t> d_tx_value_counts(0, handle.get_stream());

  if (weights) {
    if (edge_id_type_tuple) {
      auto edge_first = thrust::make_zip_iterator(majors.begin(),
                                                  minors.begin(),
                                                  weights->begin(),
                                                  std::get<0>(*edge_id_type_tuple).begin(),
                                                  std::get<1>(*edge_id_type_tuple).begin());

      d_tx_value_counts = cugraph::groupby_and_count(
        edge_first,
        edge_first + majors.size(),
        [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
        comm_size,
        mem_frugal_threshold,
        handle.get_stream());
    } else {
      auto edge_first = thrust::make_zip_iterator(majors.begin(), minors.begin(), weights->begin());

      d_tx_value_counts = cugraph::groupby_and_count(
        edge_first,
        edge_first + majors.size(),
        [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
        comm_size,
        mem_frugal_threshold,
        handle.get_stream());
    }
  } else {
    if (edge_id_type_tuple) {
      auto edge_first = thrust::make_zip_iterator(majors.begin(),
                                                  minors.begin(),
                                                  std::get<0>(*edge_id_type_tuple).begin(),
                                                  std::get<1>(*edge_id_type_tuple).begin());

      d_tx_value_counts = cugraph::groupby_and_count(
        edge_first,
        edge_first + majors.size(),
        [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
        comm_size,
        mem_frugal_threshold,
        handle.get_stream());
    } else {
      auto edge_first = thrust::make_zip_iterator(majors.begin(), minors.begin());

      d_tx_value_counts = cugraph::groupby_and_count(
        edge_first,
        edge_first + majors.size(),
        [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
        comm_size,
        mem_frugal_threshold,
        handle.get_stream());
    }
  }

  std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
  raft::update_host(h_tx_value_counts.data(),
                    d_tx_value_counts.data(),
                    d_tx_value_counts.size(),
                    handle.get_stream());
  handle.sync_stream();

  if (mem_frugal_flag) {  // trade-off potential parallelism to lower peak memory
    std::tie(rx_majors, std::ignore) =
      shuffle_values(comm, majors.begin(), h_tx_value_counts, handle.get_stream());
    majors.resize(0, handle.get_stream());
    majors.shrink_to_fit(handle.get_stream());

    std::tie(rx_minors, std::ignore) =
      shuffle_values(comm, minors.begin(), h_tx_value_counts, handle.get_stream());
    minors.resize(0, handle.get_stream());
    minors.shrink_to_fit(handle.get_stream());

    if (weights) {
      std::tie(rx_weights, std::ignore) =
        shuffle_values(comm, (*weights).begin(), h_tx_value_counts, handle.get_stream());
      (*weights).resize(0, handle.get_stream());
      (*weights).shrink_to_fit(handle.get_stream());
    }

    if (edge_id_type_tuple) {
      std::tie(rx_edge_ids, std::ignore) = shuffle_values(
        comm, std::get<0>(*edge_id_type_tuple).begin(), h_tx_value_counts, handle.get_stream());
      std::get<0>(*edge_id_type_tuple).resize(0, handle.get_stream());
      std::get<0>(*edge_id_type_tuple).shrink_to_fit(handle.get_stream());

      std::tie(rx_edge_types, std::ignore) = shuffle_values(
        comm, std::get<1>(*edge_id_type_tuple).begin(), h_tx_value_counts, handle.get_stream());
      std::get<1>(*edge_id_type_tuple).resize(0, handle.get_stream());
      std::get<1>(*edge_id_type_tuple).shrink_to_fit(handle.get_stream());
    }
  } else {
    if (weights) {
      if (edge_id_type_tuple) {
        std::forward_as_tuple(
          std::tie(rx_majors, rx_minors, rx_weights, rx_edge_ids, rx_edge_types), std::ignore) =
          shuffle_values(comm,
                         thrust::make_zip_iterator(majors.begin(),
                                                   minors.begin(),
                                                   weights->begin(),
                                                   std::get<0>(*edge_id_type_tuple).begin(),
                                                   std::get<1>(*edge_id_type_tuple).begin()),
                         h_tx_value_counts,
                         handle.get_stream());
        majors.resize(0, handle.get_stream());
        majors.shrink_to_fit(handle.get_stream());
        minors.resize(0, handle.get_stream());
        minors.shrink_to_fit(handle.get_stream());
        (*weights).resize(0, handle.get_stream());
        (*weights).shrink_to_fit(handle.get_stream());
        std::get<0>(*edge_id_type_tuple).resize(0, handle.get_stream());
        std::get<0>(*edge_id_type_tuple).shrink_to_fit(handle.get_stream());
        std::get<1>(*edge_id_type_tuple).resize(0, handle.get_stream());
        std::get<1>(*edge_id_type_tuple).resize(0, handle.get_stream());
      } else {
        std::forward_as_tuple(std::tie(rx_majors, rx_minors, rx_weights), std::ignore) =
          shuffle_values(
            comm,
            thrust::make_zip_iterator(majors.begin(), minors.begin(), weights->begin()),
            h_tx_value_counts,
            handle.get_stream());
        majors.resize(0, handle.get_stream());
        majors.shrink_to_fit(handle.get_stream());
        minors.resize(0, handle.get_stream());
        minors.shrink_to_fit(handle.get_stream());
        (*weights).resize(0, handle.get_stream());
        (*weights).shrink_to_fit(handle.get_stream());
      }
    } else {
      if (edge_id_type_tuple) {
        std::forward_as_tuple(std::tie(rx_majors, rx_minors, rx_edge_ids, rx_edge_types),
                              std::ignore) =
          shuffle_values(comm,
                         thrust::make_zip_iterator(majors.begin(),
                                                   minors.begin(),
                                                   std::get<0>(*edge_id_type_tuple).begin(),
                                                   std::get<1>(*edge_id_type_tuple).begin()),
                         h_tx_value_counts,
                         handle.get_stream());
        majors.resize(0, handle.get_stream());
        majors.shrink_to_fit(handle.get_stream());
        minors.resize(0, handle.get_stream());
        minors.shrink_to_fit(handle.get_stream());
        std::get<0>(*edge_id_type_tuple).resize(0, handle.get_stream());
        std::get<0>(*edge_id_type_tuple).shrink_to_fit(handle.get_stream());
        std::get<1>(*edge_id_type_tuple).resize(0, handle.get_stream());
        std::get<1>(*edge_id_type_tuple).resize(0, handle.get_stream());
      } else {
        std::forward_as_tuple(std::tie(rx_majors, rx_minors), std::ignore) =
          shuffle_values(comm,
                         thrust::make_zip_iterator(majors.begin(), minors.begin()),
                         h_tx_value_counts,
                         handle.get_stream());
        majors.resize(0, handle.get_stream());
        majors.shrink_to_fit(handle.get_stream());
        minors.resize(0, handle.get_stream());
        minors.shrink_to_fit(handle.get_stream());
      }
    }
  }

  if (edge_id_type_tuple)
    return std::make_tuple(
      std::move(rx_majors),
      std::move(rx_minors),
      std::move(rx_weights),
      std::make_optional(std::make_tuple(std::move(*rx_edge_ids), std::move(*rx_edge_types))));
  else
    return std::make_tuple(
      std::move(rx_majors), std::move(rx_minors), std::move(rx_weights), std::nullopt);
}

}  // namespace

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>>
shuffle_ext_vertex_pairs_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::optional<rmm::device_uvector<weight_t>>&& weights,
  std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>&&
    edge_id_type_tuple)
{
  auto& comm               = handle.get_comms();
  auto const comm_size     = comm.get_size();
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_size = row_comm.get_size();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();

  return shuffle_vertex_pairs_by_gpu_id_impl(
    handle,
    std::move(majors),
    std::move(minors),
    std::move(weights),
    std::move(edge_id_type_tuple),
    cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
      comm_size, row_comm_size, col_comm_size});
}

template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>>
shuffle_int_vertex_pairs_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::optional<rmm::device_uvector<weight_t>>&& weights,
  std::optional<std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_type_t>>>&&
    edge_id_type_tuple,
  std::vector<vertex_t> const& vertex_partition_range_lasts)
{
  auto& comm               = handle.get_comms();
  auto const comm_size     = comm.get_size();
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_size = row_comm.get_size();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();

  rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(vertex_partition_range_lasts.size(),
                                                               handle.get_stream());
  raft::update_device(d_vertex_partition_range_lasts.data(),
                      vertex_partition_range_lasts.data(),
                      vertex_partition_range_lasts.size(),
                      handle.get_stream());

  return shuffle_vertex_pairs_by_gpu_id_impl(
    handle,
    std::move(majors),
    std::move(minors),
    std::move(weights),
    std::move(edge_id_type_tuple),
    cugraph::detail::compute_gpu_id_from_int_edge_endpoints_t<vertex_t>{
      raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                        d_vertex_partition_range_lasts.size()),
      comm_size,
      row_comm_size,
      col_comm_size});
}

template std::tuple<
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>,
  std::optional<rmm::device_uvector<float>>,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>>
shuffle_ext_vertex_pairs_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<float>>&& weights,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>&&
    edge_id_type_tuple);

template std::tuple<
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>,
  std::optional<rmm::device_uvector<double>>,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>>
shuffle_ext_vertex_pairs_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<double>>&& weights,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>&&
    edge_id_type_tuple);

template std::tuple<
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>,
  std::optional<rmm::device_uvector<float>>,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>>
shuffle_ext_vertex_pairs_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<float>>&& weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&&
    edge_id_type_tuple);

template std::tuple<
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>,
  std::optional<rmm::device_uvector<double>>,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>>
shuffle_ext_vertex_pairs_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<double>>&& weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&&
    edge_id_type_tuple);

template std::tuple<
  rmm::device_uvector<int64_t>,
  rmm::device_uvector<int64_t>,
  std::optional<rmm::device_uvector<float>>,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>>
shuffle_ext_vertex_pairs_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& majors,
  rmm::device_uvector<int64_t>&& minors,
  std::optional<rmm::device_uvector<float>>&& weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&&
    edge_id_type_tuple);

template std::tuple<
  rmm::device_uvector<int64_t>,
  rmm::device_uvector<int64_t>,
  std::optional<rmm::device_uvector<double>>,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>>
shuffle_ext_vertex_pairs_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& majors,
  rmm::device_uvector<int64_t>&& minors,
  std::optional<rmm::device_uvector<double>>&& weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&&
    edge_id_type_tuple);

template std::tuple<
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>,
  std::optional<rmm::device_uvector<float>>,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>>
shuffle_int_vertex_pairs_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<float>>&& weights,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>&&
    edge_id_type_tuple,
  std::vector<int32_t> const& vertex_partition_range_lasts);

template std::tuple<
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>,
  std::optional<rmm::device_uvector<double>>,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>>
shuffle_int_vertex_pairs_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<double>>&& weights,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>&&
    edge_id_type_tuple,
  std::vector<int32_t> const& vertex_partition_range_lasts);

template std::tuple<
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>,
  std::optional<rmm::device_uvector<float>>,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>>
shuffle_int_vertex_pairs_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<float>>&& weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&&
    edge_id_type_tuple,
  std::vector<int32_t> const& vertex_partition_range_lasts);

template std::tuple<
  rmm::device_uvector<int32_t>,
  rmm::device_uvector<int32_t>,
  std::optional<rmm::device_uvector<double>>,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>>
shuffle_int_vertex_pairs_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<double>>&& weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&&
    edge_id_type_tuple,
  std::vector<int32_t> const& vertex_partition_range_lasts);

template std::tuple<
  rmm::device_uvector<int64_t>,
  rmm::device_uvector<int64_t>,
  std::optional<rmm::device_uvector<float>>,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>>
shuffle_int_vertex_pairs_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& majors,
  rmm::device_uvector<int64_t>&& minors,
  std::optional<rmm::device_uvector<float>>&& weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&&
    edge_id_type_tuple,
  std::vector<int64_t> const& vertex_partition_range_lasts);

template std::tuple<
  rmm::device_uvector<int64_t>,
  rmm::device_uvector<int64_t>,
  std::optional<rmm::device_uvector<double>>,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>>
shuffle_int_vertex_pairs_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& majors,
  rmm::device_uvector<int64_t>&& minors,
  std::optional<rmm::device_uvector<double>>&& weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&&
    edge_id_type_tuple,
  std::vector<int64_t> const& vertex_partition_range_lasts);

}  // namespace detail
}  // namespace cugraph

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

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

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
         cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
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
         cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
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

}  // namespace detail
}  // namespace cugraph

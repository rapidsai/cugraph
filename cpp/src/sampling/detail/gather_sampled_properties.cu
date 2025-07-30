/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "detail/graph_partition_utils.cuh"
#include "prims/edge_bucket.cuh"
#include "prims/transform_gather_e.cuh"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/shuffle_functions.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>

#include <rmm/device_uvector.hpp>

#include <variant>

namespace cugraph {
namespace detail {

namespace {

struct return_edge_property_t {
  template <typename key_t, typename vertex_t, typename T>
  T __device__ operator()(
    key_t src, vertex_t dst, cuda::std::nullopt_t, cuda::std::nullopt_t, T edge_property) const
  {
    return edge_property;
  }
};

}  // namespace

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<arithmetic_device_uvector_t>>
gather_sampled_properties(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  arithmetic_device_uvector_t&& multi_index,
  raft::host_span<edge_arithmetic_property_view_t<edge_t>> edge_property_views)
{
  const bool store_transposed = false;

  std::optional<cugraph::edge_multi_index_property_t<edge_t, vertex_t>> multi_edge_indices{
    std::nullopt};

  cugraph::edge_bucket_t<vertex_t, edge_t, !store_transposed, multi_gpu, false> edge_list(
    handle, graph_view.is_multigraph());

  // FIXME:  There is a mismatch here.  per_v_random_select_transform_outgoing_e shuffles the
  // sampled results so that the selected vertices are on the GPU of the major vertex.  But
  // transform_gather_e expects them to be shuffled by the location of the edge.  It seems like we
  // could make per_v_random_select_transform_outgoing_e output by the edge location and potentially
  // save this shuffle (and the one that follows transform_gather_e).  It's not clear that sampling
  // benefits from gathering the sampled edges in this way.

  //
  // Shuffle majors/minors/multi-index
  //
  if constexpr (multi_gpu) {
    std::vector<cugraph::arithmetic_device_uvector_t> edge_properties{};
    if (std::holds_alternative<rmm::device_uvector<edge_t>>(multi_index))
      edge_properties.push_back(std::move(multi_index));

    std::tie(majors, minors, edge_properties, std::ignore) =
      shuffle_int_edges(handle,
                        std::move(majors),
                        std::move(minors),
                        std::move(edge_properties),
                        store_transposed,
                        graph_view.vertex_partition_range_lasts(),
                        std::nullopt);
    if (edge_properties.size() > 0) multi_index = std::move(edge_properties[0]);
  }

  edge_list.insert(
    majors.begin(),
    majors.end(),
    minors.begin(),
    (std::holds_alternative<rmm::device_uvector<edge_t>>(multi_index))
      ? std::make_optional(std::get<rmm::device_uvector<edge_t>>(multi_index).begin())
      : std::nullopt);

  std::vector<arithmetic_device_uvector_t> result_properties{};

  std::for_each(edge_property_views.begin(),
                edge_property_views.end(),
                [&handle, &graph_view, &edge_list, &result_properties](auto edge_property_view) {
                  cugraph::variant_type_dispatch(
                    edge_property_view,
                    [&handle, &graph_view, &edge_list, &result_properties](auto property_view) {
                      using T = typename decltype(property_view)::value_type;

                      if constexpr (std::is_same_v<T, cuda::std::nullopt_t>) {
                        CUGRAPH_FAIL("Should not have a property of type cuda::std::nullopt");
                      } else {
                        rmm::device_uvector<T> tmp(edge_list.size(), handle.get_stream());

                        cugraph::transform_gather_e(handle,
                                                    graph_view,
                                                    edge_list,
                                                    edge_src_dummy_property_t{}.view(),
                                                    edge_dst_dummy_property_t{}.view(),
                                                    property_view,
                                                    return_edge_property_t{},
                                                    tmp.begin());

                        result_properties.push_back(arithmetic_device_uvector_t{std::move(tmp)});
                      }
                    });
                });

  // Now shuffle back
  if constexpr (multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    rmm::device_uvector<size_t> property_position(majors.size(), handle.get_stream());
    detail::sequence_fill(
      handle.get_stream(), property_position.data(), property_position.size(), size_t{0});

    rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
      graph_view.vertex_partition_range_lasts().size(), handle.get_stream());
    raft::update_device(d_vertex_partition_range_lasts.data(),
                        graph_view.vertex_partition_range_lasts().data(),
                        graph_view.vertex_partition_range_lasts().size(),
                        handle.get_stream());

    size_t element_size{sizeof(vertex_t) + sizeof(size_t)};
    auto total_global_mem = handle.get_device_properties().totalGlobalMem;
    auto constexpr mem_frugal_ratio =
      0.1;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
            // total_global_mem, switch to the memory frugal approach (thrust::sort is used to
            // group-by by default, and thrust::sort requires temporary buffer comparable to the
            // input data size)
    auto mem_frugal_threshold =
      static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

    auto d_tx_value_counts = cugraph::groupby_and_count(
      majors.begin(),
      majors.end(),
      property_position.begin(),
      cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
        raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                          d_vertex_partition_range_lasts.size()),
        major_comm_size,
        minor_comm_size},
      comm_size,
      mem_frugal_threshold,
      handle.get_stream());

    raft::device_span<size_t const> d_tx_value_counts_span{d_tx_value_counts.data(),
                                                           d_tx_value_counts.size()};

    std::tie(majors, std::ignore) =
      shuffle_values(comm, majors.begin(), d_tx_value_counts_span, handle.get_stream());

    {
      rmm::device_uvector<vertex_t> tmp(minors.size(), handle.get_stream());

      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     minors.begin(),
                     tmp.begin());

      std::tie(minors, std::ignore) =
        shuffle_values(comm, tmp.begin(), d_tx_value_counts_span, handle.get_stream());
    }

    std::for_each(
      result_properties.begin(),
      result_properties.end(),
      [&handle, &comm, &property_position, d_tx_value_counts_span](auto& property) {
        cugraph::variant_type_dispatch(
          property, [&handle, &comm, &property_position, d_tx_value_counts_span](auto& prop) {
            using T = typename std::remove_reference<decltype(prop)>::type::value_type;
            rmm::device_uvector<T> tmp(prop.size(), handle.get_stream());

            thrust::gather(handle.get_thrust_policy(),
                           property_position.begin(),
                           property_position.end(),
                           prop.begin(),
                           tmp.begin());

            std::tie(prop, std::ignore) =
              shuffle_values(comm, tmp.begin(), d_tx_value_counts_span, handle.get_stream());
          });
      });
  }

  return std::make_tuple(std::move(majors), std::move(minors), std::move(result_properties));
}

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::vector<arithmetic_device_uvector_t>>
gather_sampled_properties(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  arithmetic_device_uvector_t&& multi_index,
  raft::host_span<edge_arithmetic_property_view_t<int32_t>> edge_property_views);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::vector<arithmetic_device_uvector_t>>
gather_sampled_properties(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  rmm::device_uvector<int64_t>&& majors,
  rmm::device_uvector<int64_t>&& minors,
  arithmetic_device_uvector_t&& multi_index,
  raft::host_span<edge_arithmetic_property_view_t<int64_t>> edge_property_views);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::vector<arithmetic_device_uvector_t>>
gather_sampled_properties(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  arithmetic_device_uvector_t&& multi_index,
  raft::host_span<edge_arithmetic_property_view_t<int32_t>> edge_property_views);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::vector<arithmetic_device_uvector_t>>
gather_sampled_properties(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  rmm::device_uvector<int64_t>&& majors,
  rmm::device_uvector<int64_t>&& minors,
  arithmetic_device_uvector_t&& multi_index,
  raft::host_span<edge_arithmetic_property_view_t<int64_t>> edge_property_views);

}  // namespace detail
}  // namespace cugraph

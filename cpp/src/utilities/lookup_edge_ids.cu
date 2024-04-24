/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <thrust/for_each.h>
#include <thrust/tuple.h>

#include <iostream>
#include <string>
#include <tuple>

namespace cugraph {
namespace detail {

/**
 * @brief This function prints vertex and edge partitions.
 */
template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
std::
  tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
  lookup_edge_ids_impl(
    raft::handle_t const& handle,
    cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
    std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
    raft::device_span<edge_t const> edge_ids_to_lookup)
{
  auto const comm_rank = multi_gpu ? handle.get_comms().get_rank() : 0;
  bool has_edge_id     = false;
  if (edge_id_view.has_value()) { has_edge_id = true; }

  rmm::device_uvector<edge_t> sorted_edge_ids_to_lookup(edge_ids_to_lookup.size(),
                                                        handle.get_stream());

  raft::copy(sorted_edge_ids_to_lookup.begin(),
             edge_ids_to_lookup.begin(),
             edge_ids_to_lookup.size(),
             handle.get_stream());

  thrust::sort(
    handle.get_thrust_policy(), sorted_edge_ids_to_lookup.begin(), sorted_edge_ids_to_lookup.end());

  rmm::device_uvector<vertex_t> output_srcs(sorted_edge_ids_to_lookup.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> output_dsts(sorted_edge_ids_to_lookup.size(), handle.get_stream());

  auto constexpr invalid_partner = cugraph::invalid_vertex_id<vertex_t>::value;
  thrust::fill(handle.get_thrust_policy(), output_srcs.begin(), output_srcs.end(), invalid_partner);
  thrust::fill(handle.get_thrust_policy(), output_dsts.begin(), output_dsts.end(), invalid_partner);

  rmm::device_uvector<edge_t> matched_edge_indices(sorted_edge_ids_to_lookup.size(),
                                                   handle.get_stream());

  thrust::fill(handle.get_thrust_policy(),
               matched_edge_indices.begin(),
               matched_edge_indices.end(),
               cugraph::invalid_vertex_id<edge_t>::value);

  //
  // Read sources and destinations associated with ege ids
  //

  for (size_t ep_idx = 0; ep_idx < graph_view.number_of_local_edge_partitions(); ++ep_idx) {
    auto edge_partition_view = graph_view.local_edge_partition_view(ep_idx);

    auto number_of_edges_in_edge_partition = edge_partition_view.number_of_edges();
    auto offsets                           = edge_partition_view.offsets();
    auto indices                           = edge_partition_view.indices();

    assert(number_of_edges_in_edge_partition == indices.size());

    auto major_range_first = edge_partition_view.major_range_first();
    auto major_range_last  = edge_partition_view.major_range_last();

    auto major_hypersparse_first = edge_partition_view.major_hypersparse_first();
    auto dcs_nzd_vertices        = edge_partition_view.dcs_nzd_vertices();

    raft::device_span<edge_t const> ids_of_edges_stored_in_this_edge_partition{};

    if (has_edge_id) {
      auto value_firsts = edge_id_view->value_firsts();
      auto edge_counts  = edge_id_view->edge_counts();

      ids_of_edges_stored_in_this_edge_partition =
        raft::device_span<edge_t const>(value_firsts[ep_idx], edge_counts[ep_idx]);
    }

    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(size_t{ids_of_edges_stored_in_this_edge_partition.size()}),
      [has_edge_id,
       indices,
       stored_edge_ids = ids_of_edges_stored_in_this_edge_partition.begin(),
       sorted_edge_ids_to_lookup =
         raft::device_span<edge_t const>{sorted_edge_ids_to_lookup.begin(),
                                         sorted_edge_ids_to_lookup.size()},

       matched_edge_indices = matched_edge_indices.begin()] __device__(auto edge_idx) {
        if (has_edge_id) {
          auto ptr = thrust::lower_bound(thrust::seq,
                                         sorted_edge_ids_to_lookup.begin(),
                                         sorted_edge_ids_to_lookup.end(),
                                         stored_edge_ids[edge_idx]);
          if (*ptr == stored_edge_ids[edge_idx]) {
            matched_edge_indices[ptr - sorted_edge_ids_to_lookup.begin()] = edge_idx;

            auto w_at = static_cast<std::ptrdiff_t>(ptr - sorted_edge_ids_to_lookup.begin());
            printf("\neidx = %d dst = %d at = %d\n",
                   static_cast<int>(edge_idx),
                   static_cast<int>(indices[edge_idx]),
                   static_cast<int>(w_at));
          }
        }
      });

    auto last_in_non_hypersparse =
      (major_hypersparse_first ? (*major_hypersparse_first) : major_range_last) - major_range_first;
    rmm::device_uvector<vertex_t> uppers(matched_edge_indices.size(), handle.get_stream());

    thrust::fill(handle.get_thrust_policy(),
                 uppers.begin(),
                 uppers.end(),
                 cugraph::invalid_vertex_id<vertex_t>::value);

    thrust::upper_bound(handle.get_thrust_policy(),
                        offsets.begin(),
                        offsets.end(),
                        matched_edge_indices.begin(),
                        matched_edge_indices.end(),
                        uppers.begin());

#if 1
    RAFT_CUDA_TRY(cudaDeviceSynchronize());

    auto title = std::string("offsets_")
                   .append(std::to_string(comm_rank))
                   .append("_")
                   .append(std::to_string(ep_idx));

    raft::print_device_vector(title.c_str(), offsets.begin(), offsets.size(), std::cout);

    title = std::string("matched_")
              .append(std::to_string(comm_rank))
              .append("_")
              .append(std::to_string(ep_idx));

    raft::print_device_vector(
      title.c_str(), matched_edge_indices.begin(), matched_edge_indices.size(), std::cout);

    title = std::string("uppers_")
              .append(std::to_string(comm_rank))
              .append("_")
              .append(std::to_string(ep_idx));

    raft::print_device_vector(title.c_str(), uppers.begin(), uppers.size(), std::cout);

#endif

    bool is_hypersparse = major_hypersparse_first.has_value();

    thrust::transform(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(thrust::make_tuple(uppers.begin(), matched_edge_indices.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(uppers.end(), matched_edge_indices.end())),
      thrust::make_zip_iterator(thrust::make_tuple(output_srcs.begin(), output_dsts.begin())),
      [major_range_first,
       indices,
       stored_edge_ids = ids_of_edges_stored_in_this_edge_partition.begin(),
       invalid_id      = cugraph::invalid_vertex_id<vertex_t>::value,
       is_hypersparse,
       major_hypersparse_first = (*major_hypersparse_first),
       dcs_nzd_vertices        = (*dcs_nzd_vertices)] __device__(auto upper_and_eidx) {
        vertex_t idx_in_offsets = thrust::get<0>(upper_and_eidx) - 1;
        auto eidx               = thrust::get<1>(upper_and_eidx);
        vertex_t major;

        if (is_hypersparse && (idx_in_offsets >= (major_hypersparse_first - major_range_first))) {
          major = dcs_nzd_vertices[idx_in_offsets];
        } else {
          major = idx_in_offsets;
        }
        if (eidx >= 0) {
          printf("> %d (%d) : %d %d\n",
                 static_cast<int>(stored_edge_ids[eidx]),
                 static_cast<int>(eidx),
                 static_cast<int>(major),
                 static_cast<int>(indices[eidx]));

          return thrust::make_tuple(major, indices[eidx]);

        } else {
          return thrust::make_tuple(invalid_id, invalid_id);
        }
      });

#if 1
    // Edge property values
    if (edge_id_view) {
      auto value_firsts = edge_id_view->value_firsts();
      auto edge_counts  = edge_id_view->edge_counts();

      assert(number_of_edges_in_edge_partition == edge_counts[ep_idx]);

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      auto edge_ids_title = std::string("edge_ids_")
                              .append(std::to_string(comm_rank))
                              .append("_")
                              .append(std::to_string(ep_idx));
      raft::print_device_vector(
        edge_ids_title.c_str(), value_firsts[ep_idx], number_of_edges_in_edge_partition, std::cout);
    }
#endif
  }

  return std::make_tuple(
    std::move(sorted_edge_ids_to_lookup), std::move(output_srcs), std::move(output_dsts));
}

}  // namespace detail

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
std::
  tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
  lookup_edge_ids(
    raft::handle_t const& handle,
    cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
    std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
    raft::device_span<edge_t const> edge_ids_to_lookup)
{
  return detail::lookup_edge_ids_impl(handle, graph_view, edge_id_view, edge_ids_to_lookup);
}

template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
  lookup_edge_ids(raft::handle_t const& handle,
                  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
                  raft::device_span<int32_t const> edge_ids_to_lookup);

template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
  lookup_edge_ids(raft::handle_t const& handle,
                  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
                  std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
                  raft::device_span<int64_t const> edge_ids_to_lookup);

template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
  lookup_edge_ids(raft::handle_t const& handle,
                  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
                  std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
                  raft::device_span<int64_t const> edge_ids_to_lookup);

template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
  lookup_edge_ids(raft::handle_t const& handle,
                  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
                  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
                  raft::device_span<int32_t const> edge_ids_to_lookup);

template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
  lookup_edge_ids(raft::handle_t const& handle,
                  graph_view_t<int32_t, int64_t, false, false> const& graph_view,
                  std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
                  raft::device_span<int64_t const> edge_ids_to_lookup);

template std::
  tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
  lookup_edge_ids(raft::handle_t const& handle,
                  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
                  std::optional<edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
                  raft::device_span<int64_t const> edge_ids_to_lookup);

}  // namespace cugraph


/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <prims/fill_edge_src_dst_property.cuh>
#include <prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <community/detail/mis.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>

#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/remove.h>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <cmath>
#include <numeric>
#include <type_traits>
#include <utility>

namespace cugraph {

namespace detail {

const double EPSILON = 1e-4;

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<vertex_t> compute_mis(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view)
{
  using GraphViewType = cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>;
  using FlagType      = vertex_t;

  vertex_t local_vtx_partitoin_size = graph_view.local_vertex_partition_range_size();

  bool debug = graph_view.local_vertex_partition_range_size() < 40;
  if (debug) {
    auto offsets = graph_view.local_edge_partition_view(0).offsets();
    auto indices = graph_view.local_edge_partition_view(0).indices();
    cudaDeviceSynchronize();
    std::cout << "---- MIS Graph -----" << std::endl;
    raft::print_device_vector("offsets: ", offsets.data(), offsets.size(), std::cout);
    raft::print_device_vector("indices: ", indices.data(), indices.size(), std::cout);
  }

  rmm::device_uvector<vertex_t> remaining_vertices(local_vtx_partitoin_size, handle.get_stream());

  auto vertex_begin =
    thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first());
  auto vertex_end = thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last());

  auto out_degrees = graph_view.compute_out_degrees(handle);
  auto ranks       = compute_out_weight_sums<vertex_t, edge_t, weight_t, false, multi_gpu>(
    handle, graph_view, *edge_weight_view);

  // Each node as at most one outgoing edge in the decision graph
  // Vertices with non-zeor outdegree are possible candidates for MIS
  remaining_vertices.resize(
    thrust::distance(remaining_vertices.begin(),
                     thrust::copy_if(handle.get_thrust_policy(),
                                     vertex_begin,
                                     vertex_end,
                                     out_degrees.begin(),
                                     remaining_vertices.begin(),
                                     [] __device__(auto deg) { return deg > 0; })),
    handle.get_stream());

  rmm::device_uvector<vertex_t> mis(0, handle.get_stream());
  rmm::device_uvector<vertex_t> candidates(0, handle.get_stream());

  if (debug) {
    cudaDeviceSynchronize();
    raft::print_device_vector("degrees: ", out_degrees.data(), out_degrees.size(), std::cout);
    raft::print_device_vector(
      "remaining_vertices: ", remaining_vertices.data(), remaining_vertices.size(), std::cout);
  }

  // Rank caches

  edge_src_property_t<GraphViewType, weight_t> src_rank_cache(handle);
  edge_dst_property_t<GraphViewType, weight_t> dst_rank_cache(handle);
  rmm::device_uvector<weight_t> tmp_ranks(local_vtx_partitoin_size, handle.get_stream());

  size_t loop_counter = 0;
  while (true) {
    cudaDeviceSynchronize();
    std::cout << "Mis loop, counter: " << loop_counter << std::endl;
    loop_counter++;

    //
    // Copy ranks into temporary vector to begin with
    //
    thrust::copy(handle.get_thrust_policy(), ranks.begin(), ranks.end(), tmp_ranks.begin());

    if (debug) {
      cudaDeviceSynchronize();
      raft::print_device_vector("tmp_ranks: ", tmp_ranks.data(), tmp_ranks.size(), std::cout);
      raft::print_device_vector(
        "remaining_vertices: ", remaining_vertices.data(), remaining_vertices.size(), std::cout);
    }

    //
    // Select a random set of candidate vertices
    //
    thrust::default_random_engine g;
    g.seed(0);
    thrust::shuffle(
      handle.get_thrust_policy(), remaining_vertices.begin(), remaining_vertices.end(), g);

    vertex_t nr_candidates =
      std::max(vertex_t{1}, static_cast<vertex_t>(0.50 * remaining_vertices.size()));

    std::cout << "nr_candidates: " << nr_candidates
              << ", nr_remaining_vertices: " << remaining_vertices.size() << std::endl;

    candidates.resize(nr_candidates, handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 remaining_vertices.begin(),
                 remaining_vertices.begin() + nr_candidates,
                 candidates.begin());

    thrust::for_each(
      handle.get_thrust_policy(),
      remaining_vertices.begin() + nr_candidates,
      remaining_vertices.end(),
      [tmp_ranks = raft::device_span<weight_t>(tmp_ranks.data(), tmp_ranks.size()),
       v_first   = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
        //
        // if rank of a non-candidate vertex is not +Inf
        //(i.e. the vertex is not already in MIS), set it to -Inf
        //
        auto v_offset = v - v_first;
        if (tmp_ranks[v_offset] <
            std::numeric_limits<weight_t>::max()) {  // TODO: Find better way to compare
          tmp_ranks[v_offset] = std::numeric_limits<weight_t>::lowest();
        } else {
          // do nothing
        }
      });
    
    if (debug) {
      cudaDeviceSynchronize();
      raft::print_device_vector("candidates:", candidates.data(), candidates.size(), std::cout);
      raft::print_device_vector("tmp_ranks:", tmp_ranks.data(), tmp_ranks.size(), std::cout);
    }

    if constexpr (multi_gpu) {
      src_rank_cache = edge_src_property_t<GraphViewType, weight_t>(handle, graph_view);
      dst_rank_cache = edge_dst_property_t<GraphViewType, weight_t>(handle, graph_view);
      update_edge_src_property(handle, graph_view, tmp_ranks.begin(), src_rank_cache);
      update_edge_dst_property(handle, graph_view, tmp_ranks.begin(), dst_rank_cache);
    }

    //
    // Find outgoing max neighbor for each vertex
    // (In case of Leiden decision graph, each vertex has at most one outgoing edge)
    //
    auto outgoing_rank_id_pairs = allocate_dataframe_buffer<thrust::tuple<weight_t, vertex_t>>(
      graph_view.local_vertex_partition_range_size(), handle.get_stream());

    per_v_transform_reduce_outgoing_e(
      handle,
      graph_view,
      multi_gpu ? src_rank_cache.view()
                : detail::edge_major_property_view_t<vertex_t, weight_t const*>(tmp_ranks.data()),
      multi_gpu ? dst_rank_cache.view()
                : detail::edge_minor_property_view_t<vertex_t, weight_t const*>(tmp_ranks.data(),
                                                                                vertex_t{0}),
      edge_dummy_property_t{}.view(),
      [] __device__(auto src, auto dst, auto src_rank, auto dst_rank, auto wt) {
        return thrust::make_tuple(dst_rank, dst);
      },
      thrust::make_tuple(std::numeric_limits<weight_t>::lowest(),
                         invalid_vertex_id<vertex_t>::value),
      cugraph::reduce_op::maximum<thrust::tuple<weight_t, vertex_t>>{},
      cugraph::get_dataframe_buffer_begin(outgoing_rank_id_pairs));

    if (debug) {
      CUDA_TRY(cudaDeviceSynchronize());
      std::cout << "Outgoing max neighbor rank-ids: ";
      thrust::for_each(handle.get_thrust_policy(),
                       cugraph::get_dataframe_buffer_cbegin(outgoing_rank_id_pairs),
                       cugraph::get_dataframe_buffer_cend(outgoing_rank_id_pairs),
                       [] __device__(auto rank_id_tuple) {
                         auto rank = thrust::get<0>(rank_id_tuple);
                         auto id   = thrust::get<1>(rank_id_tuple);
                         printf("\n%f %d\n", rank, id);
                       });
    }

    //
    // Find incoming max neighbor for each vertex
    //

    auto incoming_rank_id_pairs = allocate_dataframe_buffer<thrust::tuple<weight_t, vertex_t>>(
      graph_view.local_vertex_partition_range_size(), handle.get_stream());

    per_v_transform_reduce_incoming_e(
      handle,
      graph_view,
      multi_gpu ? src_rank_cache.view()
                : detail::edge_major_property_view_t<vertex_t, weight_t const*>(tmp_ranks.data()),
      multi_gpu ? dst_rank_cache.view()
                : detail::edge_minor_property_view_t<vertex_t, weight_t const*>(tmp_ranks.data(),
                                                                                vertex_t{0}),
      edge_dummy_property_t{}.view(),
      [] __device__(auto src, auto dst, auto src_rank, auto dst_rank, auto wt) {
        return thrust::make_tuple(src_rank, src);
      },
      thrust::make_tuple(std::numeric_limits<weight_t>::lowest(),
                         invalid_vertex_id<vertex_t>::value),
      cugraph::reduce_op::maximum<thrust::tuple<weight_t, vertex_t>>{},
      cugraph::get_dataframe_buffer_begin(incoming_rank_id_pairs));

    if (debug) {
      CUDA_TRY(cudaDeviceSynchronize());
      std::cout << "Incoming max neighbor rank-ids: ";
      thrust::for_each(handle.get_thrust_policy(),
                       cugraph::get_dataframe_buffer_cbegin(incoming_rank_id_pairs),
                       cugraph::get_dataframe_buffer_cend(incoming_rank_id_pairs),
                       [] __device__(auto rank_id_tuple) {
                         auto rank = thrust::get<0>(rank_id_tuple);
                         auto id   = thrust::get<1>(rank_id_tuple);
                         printf("\n%f %d\n", rank, id);
                       });
    }

    //
    // Compute max of (outgoing max, Incoming max)
    //

    thrust::transform(handle.get_thrust_policy(),
                      cugraph::get_dataframe_buffer_cbegin(incoming_rank_id_pairs),
                      cugraph::get_dataframe_buffer_cend(incoming_rank_id_pairs),
                      cugraph::get_dataframe_buffer_cbegin(outgoing_rank_id_pairs),
                      cugraph::get_dataframe_buffer_begin(outgoing_rank_id_pairs),
                      thrust::maximum<thrust::tuple<weight_t, vertex_t>>());

    if (debug) {
      CUDA_TRY(cudaDeviceSynchronize());
      std::cout << "Max neighbor rank-ids: ";
      thrust::for_each(handle.get_thrust_policy(),
                       cugraph::get_dataframe_buffer_cbegin(outgoing_rank_id_pairs),
                       cugraph::get_dataframe_buffer_cend(outgoing_rank_id_pairs),
                       [] __device__(auto rank_id_tuple) {
                         auto rank = thrust::get<0>(rank_id_tuple);
                         auto id   = thrust::get<1>(rank_id_tuple);
                         printf("\n%f %d\n", rank, id);
                       });
    }

    //
    //
    //
    auto last = thrust::remove_if(
      handle.get_thrust_policy(),
      remaining_vertices.begin(),
      remaining_vertices.end(),
      [max_rank_id_pair_first = cugraph::get_dataframe_buffer_begin(outgoing_rank_id_pairs),
       tmp_ranks = raft::device_span<weight_t const>(tmp_ranks.data(), tmp_ranks.size()),
       ranks     = raft::device_span<weight_t>(ranks.data(), ranks.size()),
       v_first   = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
        auto v_offset         = v - v_first;
        auto max_rank_id_pair = *(max_rank_id_pair_first + v_offset);
        auto max_rank         = thrust::get<0>(max_rank_id_pair);
        auto max_id           = thrust::get<1>(max_rank_id_pair);
        auto v_rank           = tmp_ranks[v_offset];

        if (fabs(max_rank - std::numeric_limits<weight_t>::max()) < EPSILON) {
          // max neighbor is alreay in MIS, delete the current vertex v from remaining vertices
          ranks[v_offset] = std::numeric_limits<weight_t>::lowest();
          return true;
        } else if ((v_rank + EPSILON) > max_rank) {
          // TODO: Find a better way to compare float/double
          // include the current vertex v into MIS and set its (global) rank to +Inf
          // which indicates inclusion in MIS
          ranks[v_offset] = std::numeric_limits<weight_t>::max();
          return true;
        }
        return false;
      });

    remaining_vertices.resize(thrust::distance(remaining_vertices.begin(), last),
                              handle.get_stream());

    if (debug) {
      cudaDeviceSynchronize();
      raft::print_device_vector(
        "remaining_vertices*: ", remaining_vertices.data(), remaining_vertices.size(), std::cout);
      raft::print_device_vector("ranks*:", ranks.data(), ranks.size(), std::cout);
    }

    vertex_t nr_remaining_vertices_to_check = remaining_vertices.size();
    if (multi_gpu) {
      nr_remaining_vertices_to_check = host_scalar_allreduce(handle.get_comms(),
                                                             nr_remaining_vertices_to_check,
                                                             raft::comms::op_t::SUM,
                                                             handle.get_stream());
    }

    std::cout << " local_vtx_partitoin_size:       " << local_vtx_partitoin_size << std::endl;
    std::cout << "remaining_vts_to_check:   " << nr_remaining_vertices_to_check << std::endl;

    if (nr_remaining_vertices_to_check == 0) { break; }
  }

  if (debug) { raft::print_device_vector("ranks**:", ranks.data(), ranks.size(), std::cout); }

  //
  // Count number of vertices included in MIS, using (global) ranks.
  //  A rank of +Inf means that the corresponding vertex has been included in MIS
  //

  vertex_t nr_vertices_included_in_mis = thrust::count_if(
    handle.get_thrust_policy(), ranks.begin(), ranks.end(), [] __device__(auto v_rank) {
      return fabs(v_rank - std::numeric_limits<weight_t>::max()) < EPSILON;
      ;
    });

  mis.resize(nr_vertices_included_in_mis, handle.get_stream());
  thrust::copy_if(handle.get_thrust_policy(),
                  vertex_begin,
                  vertex_end,
                  ranks.begin(),
                  mis.begin(),
                  [] __device__(auto v_rank) {
                    return fabs(v_rank - std::numeric_limits<weight_t>::max()) < EPSILON;
                  });

  cudaDeviceSynchronize();
  std::cout << "Found mis of size " << mis.size() << std::endl;

  if (debug) {
    cudaDeviceSynchronize();
    raft::print_device_vector("mis", mis.data(), mis.size(), std::cout);
  }

  return mis;
}
}  // namespace detail
}  // namespace cugraph
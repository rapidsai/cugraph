
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

const double EPSILON = 1e-6;

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

  // Compute out-degree and ranks
  auto out_degrees = graph_view.compute_out_degrees(handle);
  auto ranks       = compute_out_weight_sums<vertex_t, edge_t, weight_t, false, multi_gpu>(
    handle, graph_view, *edge_weight_view);

  // Vertices with non-zero outdegree are possible candidates for MIS.
  remaining_vertices.resize(
    thrust::distance(remaining_vertices.begin(),
                     thrust::copy_if(handle.get_thrust_policy(),
                                     vertex_begin,
                                     vertex_end,
                                     out_degrees.begin(),
                                     remaining_vertices.begin(),
                                     [] __device__(auto deg) { return deg > 0; })),
    handle.get_stream());

  rmm::device_uvector<vertex_t> candidates(0, handle.get_stream());

  if (debug) {
    cudaDeviceSynchronize();
    raft::print_device_vector("degrees: ", out_degrees.data(), out_degrees.size(), std::cout);
    raft::print_device_vector(
      "remaining_vertices: ", remaining_vertices.data(), remaining_vertices.size(), std::cout);
  }

  // Caches for ranks
  edge_src_property_t<GraphViewType, weight_t> src_rank_cache(handle);
  edge_dst_property_t<GraphViewType, weight_t> dst_rank_cache(handle);
  rmm::device_uvector<weight_t> temporary_ranks(local_vtx_partitoin_size, handle.get_stream());

  size_t loop_counter = 0;
  while (true) {
    if (debug) {
      cudaDeviceSynchronize();
      std::cout << "Mis loop, counter: " << loop_counter << std::endl;
      loop_counter++;
    }
    //
    // Copy ranks into temporary vector to begin with
    //
    thrust::copy(handle.get_thrust_policy(), ranks.begin(), ranks.end(), temporary_ranks.begin());

    //
    // Select a random set of candidate vertices
    //
    thrust::default_random_engine g;
    g.seed(0);
    thrust::shuffle(
      handle.get_thrust_policy(), remaining_vertices.begin(), remaining_vertices.end(), g);

    vertex_t nr_candidates =
      std::max(vertex_t{1}, static_cast<vertex_t>(0.60 * remaining_vertices.size()));
    candidates.resize(nr_candidates, handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 remaining_vertices.begin(),
                 remaining_vertices.begin() + nr_candidates,
                 candidates.begin());

    // Set temporary ranks of non-candidate vertices to -Inf
    thrust::for_each(
      handle.get_thrust_policy(),
      remaining_vertices.begin() + nr_candidates,
      remaining_vertices.end(),
      [temporary_ranks =
         raft::device_span<weight_t>(temporary_ranks.data(), temporary_ranks.size()),
       v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
        //
        // if rank of a non-candidate vertex is not +Inf (i.e. the vertex
        // is not already in MIS), set it to -Inf
        //
        auto v_offset = v - v_first;
        if (temporary_ranks[v_offset] < (std::numeric_limits<weight_t>::max() - EPSILON)) {
          temporary_ranks[v_offset] = std::numeric_limits<weight_t>::lowest();
        }
      });

    if (debug) {
      cudaDeviceSynchronize();
      std::cout << "nr_remaining_vertices: " << remaining_vertices.size()
                << ", nr_candidates: " << nr_candidates << std::endl;
      raft::print_device_vector(
        "remaining_vertices:", remaining_vertices.data(), remaining_vertices.size(), std::cout);
      raft::print_device_vector("candidates:", candidates.data(), candidates.size(), std::cout);

      std::cout << "ID :    tempoary rank        " << std::endl;
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(thrust::make_tuple(vertex_begin, temporary_ranks.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(vertex_end, temporary_ranks.end())),
        [] __device__(auto id_rank_tuple) {
          auto id   = thrust::get<0>(id_rank_tuple);
          auto rank = thrust::get<1>(id_rank_tuple);
          printf("%d : %f\n", id, rank);
        });
    }

    // Update rank caches with temporary ranks
    if constexpr (multi_gpu) {
      src_rank_cache = edge_src_property_t<GraphViewType, weight_t>(handle, graph_view);
      dst_rank_cache = edge_dst_property_t<GraphViewType, weight_t>(handle, graph_view);
      update_edge_src_property(handle, graph_view, temporary_ranks.begin(), src_rank_cache);
      update_edge_dst_property(handle, graph_view, temporary_ranks.begin(), dst_rank_cache);
    }

    //
    // Find maximum rank outgoing neighbor for each vertex
    // (In case of Leiden decision graph, each vertex has at most one outgoing edge)
    //
    auto max_outgoing_rank_id_pairs = allocate_dataframe_buffer<thrust::tuple<weight_t, vertex_t>>(
      graph_view.local_vertex_partition_range_size(), handle.get_stream());

    per_v_transform_reduce_outgoing_e(
      handle,
      graph_view,
      multi_gpu
        ? src_rank_cache.view()
        : detail::edge_major_property_view_t<vertex_t, weight_t const*>(temporary_ranks.data()),
      multi_gpu ? dst_rank_cache.view()
                : detail::edge_minor_property_view_t<vertex_t, weight_t const*>(
                    temporary_ranks.data(), vertex_t{0}),
      edge_dummy_property_t{}.view(),
      [] __device__(auto src, auto dst, auto src_rank, auto dst_rank, auto wt) {
        return thrust::make_tuple(dst_rank, dst);
      },
      thrust::make_tuple(std::numeric_limits<weight_t>::lowest(),
                         invalid_vertex_id<vertex_t>::value),
      cugraph::reduce_op::maximum<thrust::tuple<weight_t, vertex_t>>{},
      cugraph::get_dataframe_buffer_begin(max_outgoing_rank_id_pairs));

    // if (debug) {
    //   CUDA_TRY(cudaDeviceSynchronize());
    //   std::cout << "Outgoing max neighbor rank-ids: ";
    //   thrust::for_each(handle.get_thrust_policy(),
    //                    cugraph::get_dataframe_buffer_cbegin(max_outgoing_rank_id_pairs),
    //                    cugraph::get_dataframe_buffer_cend(max_outgoing_rank_id_pairs),
    //                    [] __device__(auto rank_id_tuple) {
    //                      auto rank = thrust::get<0>(rank_id_tuple);
    //                      auto id   = thrust::get<1>(rank_id_tuple);
    //                      printf("\n%d %f\n", id, rank);
    //                    });
    // }

    //
    // Find maximum rank incoming neighbor for each vertex
    //

    auto max_incoming_rank_id_pairs = allocate_dataframe_buffer<thrust::tuple<weight_t, vertex_t>>(
      graph_view.local_vertex_partition_range_size(), handle.get_stream());

    per_v_transform_reduce_incoming_e(
      handle,
      graph_view,
      multi_gpu
        ? src_rank_cache.view()
        : detail::edge_major_property_view_t<vertex_t, weight_t const*>(temporary_ranks.data()),
      multi_gpu ? dst_rank_cache.view()
                : detail::edge_minor_property_view_t<vertex_t, weight_t const*>(
                    temporary_ranks.data(), vertex_t{0}),
      edge_dummy_property_t{}.view(),
      [] __device__(auto src, auto dst, auto src_rank, auto dst_rank, auto wt) {
        return thrust::make_tuple(src_rank, src);
      },
      thrust::make_tuple(std::numeric_limits<weight_t>::lowest(),
                         invalid_vertex_id<vertex_t>::value),
      cugraph::reduce_op::maximum<thrust::tuple<weight_t, vertex_t>>{},
      cugraph::get_dataframe_buffer_begin(max_incoming_rank_id_pairs));

    // if (debug) {
    //   CUDA_TRY(cudaDeviceSynchronize());
    //   std::cout << "Incoming max neighbor rank-ids: ";
    //   thrust::for_each(handle.get_thrust_policy(),
    //                    cugraph::get_dataframe_buffer_cbegin(max_incoming_rank_id_pairs),
    //                    cugraph::get_dataframe_buffer_cend(max_incoming_rank_id_pairs),
    //                    [] __device__(auto rank_id_tuple) {
    //                      auto rank = thrust::get<0>(rank_id_tuple);
    //                      auto id   = thrust::get<1>(rank_id_tuple);
    //                      printf("\n%d %f\n", id, rank);
    //                    });
    // }

    //
    // Compute max of outgoing and incoming
    //

    thrust::transform(handle.get_thrust_policy(),
                      cugraph::get_dataframe_buffer_cbegin(max_incoming_rank_id_pairs),
                      cugraph::get_dataframe_buffer_cend(max_incoming_rank_id_pairs),
                      cugraph::get_dataframe_buffer_cbegin(max_outgoing_rank_id_pairs),
                      cugraph::get_dataframe_buffer_begin(max_outgoing_rank_id_pairs),
                      thrust::maximum<thrust::tuple<weight_t, vertex_t>>());

    if (debug) {
      CUDA_TRY(cudaDeviceSynchronize());
      std::cout << "Id and rank of maximum rank neighbor" << std::endl;
      std::cout << "verterx: (id, rank) [of maximum rank neighbor]" << std::endl;

      auto pair_begin = cugraph::get_dataframe_buffer_cbegin(max_outgoing_rank_id_pairs);
      auto pair_end   = cugraph::get_dataframe_buffer_cend(max_outgoing_rank_id_pairs);

      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_zip_iterator(
                         thrust::make_tuple(vertex_begin,
                                            thrust::get<0>(pair_begin.get_iterator_tuple()),
                                            thrust::get<1>(pair_begin.get_iterator_tuple()))),
                       thrust::make_zip_iterator(
                         thrust::make_tuple(vertex_begin,
                                            thrust::get<0>(pair_end.get_iterator_tuple()),
                                            thrust::get<1>(pair_end.get_iterator_tuple()))),
                       [] __device__(auto triple) {
                         auto v                 = thrust::get<0>(triple);
                         auto max_neighbor_rank = thrust::get<1>(triple);
                         auto max_neighbor_id   = thrust::get<2>(triple);
                         printf("\n%d: %d %f\n", v, max_neighbor_id, max_neighbor_rank);
                       });
    }

    //
    // If the max neighbor of a vertex is already in MIS (i.e. has +Inf rank), discard it
    // Otherwise, include the vertex if it has larger rank than its maximum rank neighbor
    // TODO: Find a better way to compare float/double
    //
    auto last = thrust::remove_if(
      handle.get_thrust_policy(),
      remaining_vertices.begin(),
      remaining_vertices.end(),
      [max_rank_id_pair_first = cugraph::get_dataframe_buffer_begin(max_outgoing_rank_id_pairs),
       temporary_ranks =
         raft::device_span<weight_t const>(temporary_ranks.data(), temporary_ranks.size()),
       ranks   = raft::device_span<weight_t>(ranks.data(), ranks.size()),
       v_first = graph_view.local_vertex_partition_range_first(),
       debug   = debug] __device__(auto v) {
        auto v_offset                      = v - v_first;
        auto max_neighbor_rank_and_id_pair = *(max_rank_id_pair_first + v_offset);
        auto max_neighbor_rank             = thrust::get<0>(max_neighbor_rank_and_id_pair);
        auto max_neighbor_id               = thrust::get<1>(max_neighbor_rank_and_id_pair);
        auto tmp_rank_of_v                 = temporary_ranks[v_offset];

        if (debug) {
          printf("(%d, %f) ==> (%d, %f)", v, tmp_rank_of_v, max_neighbor_id, max_neighbor_rank);
        }

        if (fabs(max_neighbor_rank - std::numeric_limits<weight_t>::max()) < EPSILON) {
          if (debug) { printf("---> discarding %d\n", v); }

          // Maximum rank neighbor is alreay in MIS
          // Discard current vertex by setting (global) rank to -Inf
          ranks[v_offset] = std::numeric_limits<weight_t>::lowest();
          return true;

        } else if ((tmp_rank_of_v > (EPSILON + std::numeric_limits<weight_t>::lowest())) &&
                   (thrust::make_tuple(tmp_rank_of_v, v) >
                    thrust::make_tuple(max_neighbor_rank, max_neighbor_id))) {
          if (debug) { printf("---> including %d\n", v); }

          // Mark it included by setting (global) rank to +Inf
          ranks[v_offset] = std::numeric_limits<weight_t>::max();
          return true;
        }
        if (debug) { printf("\n"); }
        return false;
      });

    remaining_vertices.resize(thrust::distance(remaining_vertices.begin(), last),
                              handle.get_stream());

    if (debug) {
      cudaDeviceSynchronize();
      raft::print_device_vector(
        "remaining_vertices*: ", remaining_vertices.data(), remaining_vertices.size(), std::cout);
      std::cout << "ID :    Rank (persistent, after marking included (+Inf) and discarded (-Inf)"
                << std::endl;
      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_zip_iterator(thrust::make_tuple(vertex_begin, ranks.begin())),
                       thrust::make_zip_iterator(thrust::make_tuple(vertex_end, ranks.end())),
                       [] __device__(auto id_rank_tuple) {
                         auto id   = thrust::get<0>(id_rank_tuple);
                         auto rank = thrust::get<1>(id_rank_tuple);
                         printf("%d : %f\n", id, rank);
                       });
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

  if (debug) {
    std::cout << "ID :    Rank (final)" << std::endl;
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_zip_iterator(thrust::make_tuple(vertex_begin, ranks.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(vertex_end, ranks.end())),
                     [] __device__(auto id_rank_tuple) {
                       auto id   = thrust::get<0>(id_rank_tuple);
                       auto rank = thrust::get<1>(id_rank_tuple);
                       printf("%d : %f\n", id, rank);
                     });
  }

  //
  // Count number of vertices included in MIS
  // A rank of +Inf means that the corresponding vertex has been included in MIS
  //
  rmm::device_uvector<vertex_t> mis(0, handle.get_stream());
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
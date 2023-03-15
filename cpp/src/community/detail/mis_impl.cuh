
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

  if (debug) {
    cudaDeviceSynchronize();
    raft::print_device_vector("degrees: ", out_degrees.data(), out_degrees.size(), std::cout);
    raft::print_device_vector(
      "remaining_vertices: ", remaining_vertices.data(), remaining_vertices.size(), std::cout);
  }

  // Each vertex's ID is its rank
  rmm::device_uvector<vertex_t> ranks(local_vtx_partitoin_size, handle.get_stream());
  thrust::copy(handle.get_thrust_policy(), vertex_begin, vertex_end, ranks.begin());

  thrust::for_each(
    handle.get_thrust_policy(),
    vertex_begin,
    vertex_end,
    [out_degrees = raft::device_span<edge_t const>(out_degrees.data(), out_degrees.size()),
     ranks       = raft::device_span<vertex_t>(ranks.data(), ranks.size()),
     v_first     = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
      auto v_offset = v - v_first;

      if (out_degrees[v_offset] == 0) { ranks[v_offset] = std::numeric_limits<vertex_t>::lowest(); }
    });

  if (debug) { raft::print_device_vector("ranks: ", ranks.data(), ranks.size(), std::cout); }

  size_t loop_counter                              = 0;
  vertex_t nr_remaining_vertices_in_last_iteration = 0;
  while (true) {
    loop_counter++;
    // if (debug) {
    cudaDeviceSynchronize();
    std::cout << "------------Mis loop, counter: ---------- " << loop_counter << std::endl;
    // }
    //
    // Copy ranks into temporary vector to begin with
    //
    // Caches for ranks

    rmm::device_uvector<vertex_t> temporary_ranks(local_vtx_partitoin_size, handle.get_stream());
    thrust::copy(handle.get_thrust_policy(), ranks.begin(), ranks.end(), temporary_ranks.begin());

    //
    // Select a random set of candidate vertices
    //
    thrust::default_random_engine g;
    g.seed(0);
    thrust::shuffle(
      handle.get_thrust_policy(), remaining_vertices.begin(), remaining_vertices.end(), g);

    vertex_t nr_candidates = std::max(
      vertex_t{1},
      std::min(static_cast<vertex_t>((0.50 + 0.25 * loop_counter) * remaining_vertices.size()),
               vertex_t{remaining_vertices.size()}));

    // Set temporary ranks of non-candidate vertices to -Inf
    thrust::for_each(handle.get_thrust_policy(),
                     remaining_vertices.begin(),
                     remaining_vertices.end() - nr_candidates,
                     [temporary_ranks =
                        raft::device_span<vertex_t>(temporary_ranks.data(), temporary_ranks.size()),
                      v_first = graph_view.local_vertex_partition_range_first(),
                      debug   = debug] __device__(auto v) {
                       //
                       // if rank of a non-candidate vertex is not +Inf (i.e. the vertex
                       // is not already in MIS), set it to -Inf
                       //
                       auto v_offset = v - v_first;
                       if (temporary_ranks[v_offset] < std::numeric_limits<vertex_t>::max()) {
                         temporary_ranks[v_offset] = std::numeric_limits<vertex_t>::lowest();
                         if (debug)
                           printf("Setting rank of %d to %d\n", v, temporary_ranks[v_offset]);
                       }
                     });

    std::cout << "nr_remaining_vertices: " << remaining_vertices.size()
              << ", nr_candidates: " << nr_candidates << std::endl;

    if (debug) {
      cudaDeviceSynchronize();
      std::cout << "nr_remaining_vertices: " << remaining_vertices.size()
                << ", nr_candidates: " << nr_candidates << std::endl;
      raft::print_device_vector(
        "remaining_vertices:", remaining_vertices.data(), remaining_vertices.size(), std::cout);
      raft::print_device_vector(
        "candidates:",
        remaining_vertices.data() + remaining_vertices.size() - nr_candidates,
        nr_candidates,
        std::cout);

      std::cout << " vertex id : temporary rank " << std::endl;
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(thrust::make_tuple(vertex_begin, temporary_ranks.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(vertex_end, temporary_ranks.end())),
        [debug = debug] __device__(auto vertex_id_rank_pair) {
          auto id   = thrust::get<0>(vertex_id_rank_pair);
          auto rank = thrust::get<1>(vertex_id_rank_pair);
          if (debug) printf("%d : %d\n", id, rank);
        });
    }

    // Update rank caches with temporary ranks
    edge_src_property_t<GraphViewType, vertex_t> src_rank_cache(handle);
    edge_dst_property_t<GraphViewType, vertex_t> dst_rank_cache(handle);
    if constexpr (multi_gpu) {
      src_rank_cache = edge_src_property_t<GraphViewType, vertex_t>(handle, graph_view);
      dst_rank_cache = edge_dst_property_t<GraphViewType, vertex_t>(handle, graph_view);
      update_edge_src_property(handle, graph_view, temporary_ranks.begin(), src_rank_cache);
      update_edge_dst_property(handle, graph_view, temporary_ranks.begin(), dst_rank_cache);
    }

    //
    // Find maximum rank outgoing neighbor for each vertex
    // (In case of Leiden decision graph, each vertex has at most one outgoing edge)
    //

    rmm::device_uvector<vertex_t> max_outgoing_ranks(local_vtx_partitoin_size, handle.get_stream());

    per_v_transform_reduce_outgoing_e(
      handle,
      graph_view,
      multi_gpu
        ? src_rank_cache.view()
        : detail::edge_major_property_view_t<vertex_t, vertex_t const*>(temporary_ranks.data()),
      multi_gpu ? dst_rank_cache.view()
                : detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                    temporary_ranks.data(), vertex_t{0}),
      edge_dummy_property_t{}.view(),
      [] __device__(auto src, auto dst, auto src_rank, auto dst_rank, auto wt) { return dst_rank; },
      std::numeric_limits<vertex_t>::lowest(),
      cugraph::reduce_op::maximum<vertex_t>{},
      max_outgoing_ranks.begin());

    //
    // Find maximum rank incoming neighbor for each vertex
    //

    rmm::device_uvector<vertex_t> max_incoming_ranks(local_vtx_partitoin_size, handle.get_stream());

    per_v_transform_reduce_incoming_e(
      handle,
      graph_view,
      multi_gpu
        ? src_rank_cache.view()
        : detail::edge_major_property_view_t<vertex_t, vertex_t const*>(temporary_ranks.data()),
      multi_gpu ? dst_rank_cache.view()
                : detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                    temporary_ranks.data(), vertex_t{0}),
      edge_dummy_property_t{}.view(),
      [] __device__(auto src, auto dst, auto src_rank, auto dst_rank, auto wt) { return src_rank; },
      std::numeric_limits<vertex_t>::lowest(),
      cugraph::reduce_op::maximum<vertex_t>{},
      max_incoming_ranks.begin());

    if (debug) std::cout << "Check outgoing max" << std::endl;
    thrust::for_each(handle.get_thrust_policy(),
                     max_outgoing_ranks.begin(),
                     max_outgoing_ranks.end(),
                     [ranks           = ranks.data(),
                      temporary_ranks = temporary_ranks.data(),
                      v_first         = graph_view.local_vertex_partition_range_first(),
                      debug           = debug] __device__(auto max_neighbor_rank) {
                       if (debug) printf("%d \n", max_neighbor_rank);
                       if ((max_neighbor_rank < std::numeric_limits<vertex_t>::max()) &&
                           (max_neighbor_rank > std::numeric_limits<vertex_t>::lowest())) {
                         if (max_neighbor_rank != temporary_ranks[max_neighbor_rank - v_first]) {
                           printf("?  %d : %d != %d (r = %d ) \n",
                                  max_neighbor_rank,
                                  max_neighbor_rank,
                                  temporary_ranks[max_neighbor_rank - v_first],
                                  ranks[max_neighbor_rank - v_first]);
                         }
                       }
                     });

    if (debug) std::cout << "Check incoming max" << std::endl;
    thrust::for_each(handle.get_thrust_policy(),
                     max_incoming_ranks.begin(),
                     max_incoming_ranks.end(),
                     [ranks           = ranks.data(),
                      temporary_ranks = temporary_ranks.data(),
                      v_first         = graph_view.local_vertex_partition_range_first(),
                      debug           = debug] __device__(auto max_neighbor_rank) {
                       if (debug) printf("%d \n", max_neighbor_rank);
                       if ((max_neighbor_rank < std::numeric_limits<vertex_t>::max()) &&
                           (max_neighbor_rank > std::numeric_limits<vertex_t>::lowest())) {
                         if (max_neighbor_rank != temporary_ranks[max_neighbor_rank - v_first]) {
                           printf("?  %d : %d != %d (r = %d ) \n",
                                  max_neighbor_rank,
                                  max_neighbor_rank,
                                  temporary_ranks[max_neighbor_rank - v_first],
                                  ranks[max_neighbor_rank - v_first]);
                         }
                       }
                     });

    //
    // Compute max of outgoing and incoming
    //

    // temporary_ranks.resize(0, handle.get_stream());
    // temporary_ranks.shrink_to_fit(handle.get_stream());

    thrust::transform(handle.get_thrust_policy(),
                      max_incoming_ranks.begin(),
                      max_incoming_ranks.end(),
                      max_outgoing_ranks.begin(),
                      max_outgoing_ranks.begin(),
                      thrust::maximum<vertex_t>());

    // cugraph::resize_dataframe_buffer(max_incoming_ranks, size_t{0}, handle.get_stream());
    // cugraph::shrink_to_fit_dataframe_buffer(max_incoming_ranks, handle.get_stream());

    if (debug) std::cout << "Check max (incoming, outgoing):" << std::endl;
    thrust::for_each(handle.get_thrust_policy(),
                     max_outgoing_ranks.begin(),
                     max_outgoing_ranks.end(),
                     [ranks           = ranks.data(),
                      temporary_ranks = temporary_ranks.data(),
                      v_first         = graph_view.local_vertex_partition_range_first(),
                      debug           = debug] __device__(auto max_neighbor_rank) {
                       if (debug) printf("%d \n", max_neighbor_rank);
                       if ((max_neighbor_rank < std::numeric_limits<vertex_t>::max()) &&
                           (max_neighbor_rank > std::numeric_limits<vertex_t>::lowest())) {
                         if (max_neighbor_rank != temporary_ranks[max_neighbor_rank - v_first]) {
                           printf("?  %d : %d != %d (r = %d ) \n",
                                  max_neighbor_rank,
                                  max_neighbor_rank,
                                  temporary_ranks[max_neighbor_rank - v_first],
                                  ranks[max_neighbor_rank - v_first]);
                         }
                       }
                     });

    if (debug) {
      CUDA_TRY(cudaDeviceSynchronize());
      std::cout << "Rank of maximum rank neighbors" << std::endl;
      thrust::for_each(
        handle.get_thrust_policy(),
        max_outgoing_ranks.begin(),
        max_outgoing_ranks.end(),
        [] __device__(auto max_rank_neighbor) { printf("%d \n", max_rank_neighbor); });
    }

    if (debug) std::cout << "----------------count-------------------" << std::endl;
    vertex_t nr_candidates_to_remove = thrust::count_if(
      handle.get_thrust_policy(),
      remaining_vertices.end() - nr_candidates,
      remaining_vertices.end(),
      [max_rank_neighbor_first = max_outgoing_ranks.begin(),
       temporary_ranks =
         raft::device_span<vertex_t const>(temporary_ranks.data(), temporary_ranks.size()),
       ranks   = raft::device_span<vertex_t>(ranks.data(), ranks.size()),
       v_first = graph_view.local_vertex_partition_range_first(),
       debug   = debug] __device__(auto v) {
        auto v_offset          = v - v_first;
        auto max_neighbor_rank = *(max_rank_neighbor_first + v_offset);

        auto rank_of_v     = ranks[v_offset];
        auto tmp_rank_of_v = temporary_ranks[v_offset];  // debug

        if (debug) {
          bool valid = (max_neighbor_rank < std::numeric_limits<vertex_t>::max()) &&
                       (max_neighbor_rank > std::numeric_limits<vertex_t>::lowest());
          printf("%d, (r= %d, t= %d)  ==> %d [t= %d, r= %d]\n",
                 v,
                 rank_of_v,
                 tmp_rank_of_v,
                 max_neighbor_rank,
                 valid ? temporary_ranks[max_neighbor_rank - v_first] : max_neighbor_rank,
                 valid ? ranks[max_neighbor_rank - v_first] : max_neighbor_rank);
        }

        if (max_neighbor_rank >= std::numeric_limits<vertex_t>::max()) {
          if (debug) { printf("---> to discard %d\n", v); }

          // Maximum rank neighbor is alreay in MIS
          // Discard current vertex by setting (global) rank to -Inf
          return true;
        }

        if (rank_of_v > max_neighbor_rank) {
          if (debug) { printf("---> to include %d\n", v); }
          // Mark it included by setting (global) rank to +Inf
          return true;
        }
        if (debug) { printf("\n"); }
        return false;
      });

    std::cout << "----------> nr_candidates_to_remove: " << nr_candidates_to_remove << std::endl;

    std::cout << "---------check ranks, temporary_ranks --------" << nr_candidates_to_remove
              << std::endl;

    thrust::for_each(
      handle.get_thrust_policy(),
      vertex_begin,
      vertex_end,
      [out_degrees = raft::device_span<edge_t const>(out_degrees.data(), out_degrees.size()),
       ranks       = raft::device_span<vertex_t>(ranks.data(), ranks.size()),
       temporary_ranks =
         raft::device_span<vertex_t const>(temporary_ranks.data(), temporary_ranks.size()),
       v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
        auto v_offset = v - v_first;

        if (v == 0) printf("\nChecking ranks and temporary ranks\n");

        auto tmp_rank = temporary_ranks[v_offset];

        if ((tmp_rank != std::numeric_limits<vertex_t>::lowest()) &&
            (tmp_rank != std::numeric_limits<vertex_t>::max()) && (tmp_rank != v)) {
          printf("?? %d %d\n", v, temporary_ranks[v_offset]);
        }

        auto rank = ranks[v_offset];

        if ((rank != std::numeric_limits<vertex_t>::lowest()) &&
            (rank != std::numeric_limits<vertex_t>::max()) && (rank != v)) {
          printf("??? %d %d\n", v, ranks[v_offset]);
        }
      });

    if (nr_candidates_to_remove == 0) {
      std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
      // sort
      thrust::sort(handle.get_thrust_policy(),
                   remaining_vertices.end() - nr_candidates,
                   remaining_vertices.end());

      std::cout << " Incoming  ............." << std::endl;

      thrust::count_if(
        handle.get_thrust_policy(),
        remaining_vertices.end() - 5,
        remaining_vertices.end(),
        [max_rank_neighbor_first = max_incoming_ranks.begin(),
         temporary_ranks =
           raft::device_span<vertex_t const>(temporary_ranks.data(), temporary_ranks.size()),
         ranks   = raft::device_span<vertex_t>(ranks.data(), ranks.size()),
         v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
          auto v_offset          = v - v_first;
          auto max_neighbor_rank = *(max_rank_neighbor_first + v_offset);

          auto rank_of_v     = ranks[v_offset];
          auto tmp_rank_of_v = temporary_ranks[v_offset];  // debug

          uint8_t debug = 1;

          if (debug) {
            bool valid = (max_neighbor_rank < std::numeric_limits<vertex_t>::max()) &&
                         (max_neighbor_rank > std::numeric_limits<vertex_t>::lowest());
            printf("%d, (r= %d, t= %d)  ==> %d [t= %d, r= %d]\n",
                   v,
                   rank_of_v,
                   tmp_rank_of_v,
                   max_neighbor_rank,
                   valid ? temporary_ranks[max_neighbor_rank - v_first] : max_neighbor_rank,
                   valid ? ranks[max_neighbor_rank - v_first] : max_neighbor_rank);
          }

          return false;
        });

      std::cout << " Max (incoming, outgoing) ............." << std::endl;
      thrust::count_if(
        handle.get_thrust_policy(),
        remaining_vertices.end() - 5,
        remaining_vertices.end(),
        [max_rank_neighbor_first = max_outgoing_ranks.begin(),
         temporary_ranks =
           raft::device_span<vertex_t const>(temporary_ranks.data(), temporary_ranks.size()),
         ranks   = raft::device_span<vertex_t>(ranks.data(), ranks.size()),
         v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
          auto v_offset          = v - v_first;
          auto max_neighbor_rank = *(max_rank_neighbor_first + v_offset);

          auto rank_of_v     = ranks[v_offset];
          auto tmp_rank_of_v = temporary_ranks[v_offset];  // debug
          auto max_of_max    = *(max_rank_neighbor_first + max_neighbor_rank - v_first);

          uint8_t debug = 1;

          if (debug) {
            bool valid = (max_neighbor_rank < std::numeric_limits<vertex_t>::max()) &&
                         (max_neighbor_rank > std::numeric_limits<vertex_t>::lowest());
            printf("%d, (r= %d, t= %d)  ==> %d [t= %d, r= %d], max_of_max= %d \n",
                   v,
                   rank_of_v,
                   tmp_rank_of_v,
                   max_neighbor_rank,
                   valid ? temporary_ranks[max_neighbor_rank - v_first] : max_neighbor_rank,
                   valid ? ranks[max_neighbor_rank - v_first] : max_neighbor_rank,
                   max_of_max);
          }
          return false;
        });

      std::cout << "-------------- Before removing -------------------" << std::endl;
      // just print
      thrust::count_if(
        handle.get_thrust_policy(),
        remaining_vertices.end() - 5,
        remaining_vertices.end(),
        [max_rank_neighbor_first = max_outgoing_ranks.begin(),
         temporary_ranks =
           raft::device_span<vertex_t const>(temporary_ranks.data(), temporary_ranks.size()),
         ranks   = raft::device_span<vertex_t>(ranks.data(), ranks.size()),
         v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
          auto v_offset          = v - v_first;
          auto max_neighbor_rank = *(max_rank_neighbor_first + v_offset);

          auto rank_of_v     = ranks[v_offset];
          auto tmp_rank_of_v = temporary_ranks[v_offset];  // debug

          bool debug = true;

          if (debug) {
            bool valid = (max_neighbor_rank < std::numeric_limits<vertex_t>::max()) &&
                         (max_neighbor_rank > std::numeric_limits<vertex_t>::lowest());
            printf("%d, (r= %d, t= %d)  ==> %d [t= %d, r= %d]\n",
                   v,
                   rank_of_v,
                   tmp_rank_of_v,
                   max_neighbor_rank,
                   valid ? temporary_ranks[max_neighbor_rank - v_first] : max_neighbor_rank,
                   valid ? ranks[max_neighbor_rank - v_first] : max_neighbor_rank);
          }

          if (max_neighbor_rank >= std::numeric_limits<vertex_t>::max()) {
            if (debug) { printf("---> to discard %d\n", v); }

            // Maximum rank neighbor is alreay in MIS
            // Discard current vertex by setting (global) rank to -Inf
            return true;
          }

          if (rank_of_v > max_neighbor_rank) {
            if (debug) { printf("---> to include %d\n", v); }
            // Mark it included by setting (global) rank to +Inf
            return true;
          }
          if (debug) { printf("\n"); }
          return false;
        });

      std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    }

    // temporary_ranks.resize(0, handle.get_stream());
    // temporary_ranks.shrink_to_fit(handle.get_stream());

    max_incoming_ranks.resize(0, handle.get_stream());
    max_incoming_ranks.shrink_to_fit(handle.get_stream());

    //
    // If the max neighbor of a vertex is already in MIS (i.e. has +Inf rank), discard it
    // Otherwise, include the vertex if it has larger rank than its maximum rank neighbor
    // TODO: Find a better way to compare float/double
    //
    auto last = thrust::remove_if(
      handle.get_thrust_policy(),
      remaining_vertices.end() - nr_candidates,
      remaining_vertices.end(),
      [max_rank_neighbor_first = max_outgoing_ranks.begin(),
       temporary_ranks =
         raft::device_span<vertex_t const>(temporary_ranks.data(), temporary_ranks.size()),
       ranks   = raft::device_span<vertex_t>(ranks.data(), ranks.size()),
       v_first = graph_view.local_vertex_partition_range_first(),
       debug   = debug] __device__(auto v) {
        auto v_offset          = v - v_first;
        auto max_neighbor_rank = *(max_rank_neighbor_first + v_offset);

        auto rank_of_v     = ranks[v_offset];
        auto tmp_rank_of_v = temporary_ranks[v_offset];  // debug

        if (debug) {
          bool valid = (max_neighbor_rank < std::numeric_limits<vertex_t>::max()) &&
                       (max_neighbor_rank > std::numeric_limits<vertex_t>::lowest());
          printf("%d, (r= %d, t= %d)  ==> %d [t= %d, r= %d]\n",
                 v,
                 rank_of_v,
                 tmp_rank_of_v,
                 max_neighbor_rank,
                 valid ? temporary_ranks[max_neighbor_rank - v_first] : max_neighbor_rank,
                 valid ? ranks[max_neighbor_rank - v_first] : max_neighbor_rank);
        }

        if (max_neighbor_rank >= std::numeric_limits<vertex_t>::max()) {
          if (debug) { printf("---> discarding %d\n", v); }

          // Maximum rank neighbor is alreay in MIS
          // Discard current vertex by setting (global) rank to -Inf
          ranks[v_offset] = std::numeric_limits<vertex_t>::lowest();
          return true;
        }

        if (rank_of_v > max_neighbor_rank) {
          if (debug) { printf("---> including %d\n", v); }
          // Mark it included by setting (global) rank to +Inf
          ranks[v_offset] = std::numeric_limits<vertex_t>::max();
          return true;
        }
        if (debug) { printf("\n"); }
        return false;
      });

    temporary_ranks.resize(0, handle.get_stream());
    temporary_ranks.shrink_to_fit(handle.get_stream());

    max_outgoing_ranks.resize(0, handle.get_stream());
    max_outgoing_ranks.shrink_to_fit(handle.get_stream());

    std::cout << "---------->nr of removed candidates: "
              << thrust::distance(last, remaining_vertices.end()) << std::endl;

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
                       [] __device__(auto vertex_id_rank_pair) {
                         auto vertex_id = thrust::get<0>(vertex_id_rank_pair);
                         auto rank      = thrust::get<1>(vertex_id_rank_pair);
                         printf("%d : %d\n", vertex_id, rank);
                       });
    }

    vertex_t nr_remaining_vertices_to_check = remaining_vertices.size();
    if (multi_gpu) {
      nr_remaining_vertices_to_check = host_scalar_allreduce(handle.get_comms(),
                                                             nr_remaining_vertices_to_check,
                                                             raft::comms::op_t::SUM,
                                                             handle.get_stream());
    }

    std::cout << "local_vtx_partitoin_size:       " << local_vtx_partitoin_size << std::endl;
    std::cout << "remaining_vts_to_check:   " << nr_remaining_vertices_to_check << std::endl;

    if (nr_remaining_vertices_to_check == 0) { break; }
    if (nr_remaining_vertices_in_last_iteration == nr_remaining_vertices_to_check) {
      // break;
    } else {
      nr_remaining_vertices_in_last_iteration = nr_remaining_vertices_to_check;
    }
  }

  if (debug) {
    std::cout << "ID :    Rank (final)" << std::endl;
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_zip_iterator(thrust::make_tuple(vertex_begin, ranks.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(vertex_end, ranks.end())),
                     [] __device__(auto vertex_id_rank_pair) {
                       auto vertex_id = thrust::get<0>(vertex_id_rank_pair);
                       auto rank      = thrust::get<1>(vertex_id_rank_pair);
                       printf("%d : %d\n", vertex_id, rank);
                     });
  }

  //
  // Count number of vertices included in MIS
  // A rank of +Inf means that the corresponding vertex has been included in MIS
  //
  vertex_t nr_vertices_included_in_mis = thrust::count_if(
    handle.get_thrust_policy(), ranks.begin(), ranks.end(), [] __device__(auto v_rank) {
      return v_rank >= std::numeric_limits<vertex_t>::max();
    });

  rmm::device_uvector<vertex_t> mis(nr_vertices_included_in_mis, handle.get_stream());
  thrust::copy_if(
    handle.get_thrust_policy(),
    vertex_begin,
    vertex_end,
    ranks.begin(),
    mis.begin(),
    [] __device__(auto v_rank) { return v_rank >= std::numeric_limits<vertex_t>::max(); });

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
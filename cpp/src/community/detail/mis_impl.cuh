
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

  // Out degree must be 1 to be included in MIS
  auto out_degrees = graph_view.compute_out_degrees(handle);

  // Each vertex has at most one outgoing edge,
  auto ranks = compute_out_weight_sums<vertex_t, edge_t, weight_t, false, multi_gpu>(
    handle, graph_view, *edge_weight_view);

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

  size_t loop_counter = 0;
  rmm::device_uvector<weight_t> tmp_ranks(local_vtx_partitoin_size, handle.get_stream());
  while (true) {
    // if (debug) {
    cudaDeviceSynchronize();
    std::cout << "Mis loop, counter: " << loop_counter << std::endl;
    loop_counter++;
    // }

    // thrust::sort(handle.get_thrust_policy(), mis.begin(), mis.end());

    // if (debug) {
    //   cudaDeviceSynchronize();
    //   raft::print_device_vector("mis(sorted): ", mis.data(), mis.size(), std::cout);
    // }

    // thrust::transform(handle.get_thrust_policy(),
    //                   vertex_begin,
    //                   vertex_end,
    //                   ranks.begin(),
    //                   tmp_ranks.begin(),
    //                   [mis = mis.data(), mis_size = mis.size()] __device__(auto v, auto v_rank) {
    //                     bool is_included =
    //                       thrust::binary_search(thrust::seq, mis, mis + mis_size, v);
    //                     if (is_included) {
    //                       return std::numeric_limits<weight_t>::max();
    //                     } else {
    //                       return v_rank;
    //                     }
    //                   });

    thrust::copy(handle.get_thrust_policy(), ranks.begin(), ranks.end(), tmp_ranks.begin());

    if (debug) {
      cudaDeviceSynchronize();
      raft::print_device_vector("tmp_ranks: ", tmp_ranks.data(), tmp_ranks.size(), std::cout);
    }

    if (debug) {
      cudaDeviceSynchronize();
      raft::print_device_vector(
        "remaining_vertices: ", remaining_vertices.data(), remaining_vertices.size(), std::cout);
    }

    // Select a random set of eligible vertices
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

    // Set tmp_ranks of non-candidate to std::numeric_limits<weight_t>::lowest()

    thrust::sort(handle.get_thrust_policy(), candidates.begin(), candidates.end());

    if (debug) {
      cudaDeviceSynchronize();
      raft::print_device_vector(
        "candidates(sorted): ", candidates.data(), candidates.size(), std::cout);
    }

    // Set temporary rank of each non-candidate vertex to -Inf
    thrust::transform(
      handle.get_thrust_policy(),
      vertex_begin,
      vertex_end,
      tmp_ranks.begin(),
      tmp_ranks.begin(),
      [candidates    = candidates.data(),
       nr_candidates = nr_candidates,
       v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v, auto v_rank) {
        bool is_candidate =
          thrust::binary_search(thrust::seq, candidates, candidates + nr_candidates, v);

        // printf("\nv=%d, rank= %f is_candidate = %d max_r = %f\n ",
        //        v,
        //        v_rank,
        //        vertex_t{is_candidate},
        //        std::numeric_limits<weight_t>::max());

        if (v_rank < std::numeric_limits<weight_t>::max()) {
          if (!is_candidate) { return std::numeric_limits<weight_t>::lowest(); }
        }
        return v_rank;
      });

    // mm::device_uvector<weight_t> ranks_of_noncandidates(remaining_vertices.size() -
    // nr_candidates,
    //                                                     handle.get_stream());
    // thrust::transform(
    //   handle.get_thrust_policy(),
    //   remaining_vertices.begin() + nr_candidates,
    //   remaining_vertices.end(),
    //   ranks_of_noncandidates
    //     .begin()[tmp_ranks = tmp_ranks.data(),
    //              v_first =
    //                graph_view.local_edge_partition_src_range_first()] __device__(auto v) {
    //       printf("\nv=%d, rank= %f", v, tmp_ranks[v - v_first]);

    //       if (tmp_ranks[v - v_first] < std::numeric_limits<weight_t>::max())
    //         return std::numeric_limits<weight_t>::lowest();
    //     }
    // );

    if (debug) {
      cudaDeviceSynchronize();
      raft::print_device_vector("candidates:", candidates.data(), candidates.size(), std::cout);
      raft::print_device_vector("tmp_ranks:", tmp_ranks.data(), tmp_ranks.size(), std::cout);
    }

    edge_src_property_t<GraphViewType, weight_t> src_rank_cache(handle);
    edge_dst_property_t<GraphViewType, weight_t> dst_rank_cache(handle);

    if constexpr (multi_gpu) {
      src_rank_cache = edge_src_property_t<GraphViewType, weight_t>(handle, graph_view);
      dst_rank_cache = edge_dst_property_t<GraphViewType, weight_t>(handle, graph_view);
      update_edge_src_property(handle, graph_view, tmp_ranks.begin(), src_rank_cache);
      update_edge_dst_property(handle, graph_view, tmp_ranks.begin(), dst_rank_cache);
    }

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

      auto out_max_cbegin = cugraph::get_dataframe_buffer_cbegin(outgoing_rank_id_pairs);
      raft::print_device_vector("max outgoing tmp_ranks:",
                                thrust::get<0>(out_max_cbegin.get_iterator_tuple()),
                                local_vtx_partitoin_size,
                                std::cout);
      raft::print_device_vector("max outgoing id   :",
                                thrust::get<1>(out_max_cbegin.get_iterator_tuple()),
                                local_vtx_partitoin_size,
                                std::cout);

      std::cout << "Outgoin rank-ids: ";
      thrust::for_each(handle.get_thrust_policy(),
                       cugraph::get_dataframe_buffer_cbegin(outgoing_rank_id_pairs),
                       cugraph::get_dataframe_buffer_cend(outgoing_rank_id_pairs),
                       [] __device__(auto rank_id_tuple) {
                         auto rank = thrust::get<0>(rank_id_tuple);
                         auto id   = thrust::get<1>(rank_id_tuple);
                         printf("\n%f %d\n", rank, id);
                       });
    }

    // if constexpr (multi_gpu) {
    //   update_edge_src_property(handle, graph_view, tmp_ranks.begin(), src_rank_cache);
    //   update_edge_dst_property(handle, graph_view, tmp_ranks.begin(), dst_rank_cache);
    // }

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

      auto in_max_cbegin = cugraph::get_dataframe_buffer_cbegin(incoming_rank_id_pairs);
      raft::print_device_vector("max incoming tmp_ranks:",
                                thrust::get<0>(in_max_cbegin.get_iterator_tuple()),
                                local_vtx_partitoin_size,
                                std::cout);
      raft::print_device_vector("max incoming id   :",
                                thrust::get<1>(in_max_cbegin.get_iterator_tuple()),
                                local_vtx_partitoin_size,
                                std::cout);

      std::cout << "Incoming rank-ids: ";
      thrust::for_each(handle.get_thrust_policy(),
                       cugraph::get_dataframe_buffer_cbegin(incoming_rank_id_pairs),
                       cugraph::get_dataframe_buffer_cend(incoming_rank_id_pairs),
                       [] __device__(auto rank_id_tuple) {
                         auto rank = thrust::get<0>(rank_id_tuple);
                         auto id   = thrust::get<1>(rank_id_tuple);
                         printf("\n%f %d\n", rank, id);
                       });
    }

    thrust::transform(handle.get_thrust_policy(),
                      cugraph::get_dataframe_buffer_cbegin(incoming_rank_id_pairs),
                      cugraph::get_dataframe_buffer_cend(incoming_rank_id_pairs),
                      cugraph::get_dataframe_buffer_cbegin(outgoing_rank_id_pairs),
                      cugraph::get_dataframe_buffer_begin(outgoing_rank_id_pairs),
                      thrust::maximum<thrust::tuple<weight_t, vertex_t>>());

    if (debug) {
      CUDA_TRY(cudaDeviceSynchronize());

      auto out_max_cbegin = cugraph::get_dataframe_buffer_cbegin(outgoing_rank_id_pairs);
      raft::print_device_vector("max neighbhor tmp_ranks:",
                                thrust::get<0>(out_max_cbegin.get_iterator_tuple()),
                                local_vtx_partitoin_size,
                                std::cout);
      raft::print_device_vector("max neighbhor id   :",
                                thrust::get<1>(out_max_cbegin.get_iterator_tuple()),
                                local_vtx_partitoin_size,
                                std::cout);

      std::cout << "   Max rank-ids: ";
      thrust::for_each(handle.get_thrust_policy(),
                       cugraph::get_dataframe_buffer_cbegin(outgoing_rank_id_pairs),
                       cugraph::get_dataframe_buffer_cend(outgoing_rank_id_pairs),
                       [] __device__(auto rank_id_tuple) {
                         auto rank = thrust::get<0>(rank_id_tuple);
                         auto id   = thrust::get<1>(rank_id_tuple);
                         printf("\n%f %d\n", rank, id);
                       });
    }

    if (debug) {
      thrust::for_each(
        handle.get_thrust_policy(),
        remaining_vertices.begin(),
        remaining_vertices.end(),
        [max_rank_id_pair_first = cugraph::get_dataframe_buffer_begin(outgoing_rank_id_pairs),
         tmp_ranks = raft::device_span<weight_t const>(tmp_ranks.data(), tmp_ranks.size()),
         ranks     = raft::device_span<weight_t>(ranks.data(), ranks.size()),
         v_first   = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
          auto max_rank_id_pair    = *(max_rank_id_pair_first + v - v_first);
          auto rank_of_max_neigbor = thrust::get<0>(max_rank_id_pair);
          auto id_of_max_neigbor   = thrust::get<1>(max_rank_id_pair);
          auto candidate_rank      = tmp_ranks[v - v_first];

          printf("\n(v:%d, rank: %f) => (%d, %f)",
                 v,
                 candidate_rank,
                 id_of_max_neigbor,
                 rank_of_max_neigbor);
        });
    }
    // vertex_t mis_size = remaining_vertices.size();
    auto last = thrust::remove_if(
      handle.get_thrust_policy(),
      remaining_vertices.begin(),
      remaining_vertices.end(),
      [max_rank_id_pair_first = cugraph::get_dataframe_buffer_begin(outgoing_rank_id_pairs),
       tmp_ranks = raft::device_span<weight_t const>(tmp_ranks.data(), tmp_ranks.size()),
       ranks     = raft::device_span<weight_t>(ranks.data(), ranks.size()),
       v_first   = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
        auto max_rank_id_pair    = *(max_rank_id_pair_first + v - v_first);
        auto rank_of_max_neigbor = thrust::get<0>(max_rank_id_pair);
        auto id_of_max_neigbor   = thrust::get<1>(max_rank_id_pair);
        auto candidate_rank      = tmp_ranks[v - v_first];

        // printf("\n(v:%d, rank: %f) => (%d, %f)",
        //        v,
        //        candidate_rank,
        //        id_of_max_neigbor,
        //        rank_of_max_neigbor);

        if (rank_of_max_neigbor == std::numeric_limits<weight_t>::max()) {
          // printf("  -->discarding %d\n", v);
          return true;
        } else if (rank_of_max_neigbor < candidate_rank) {
          // printf("  -->inclding %d\n", v);
          ranks[v - v_first] = std::numeric_limits<weight_t>::max();
          return true;
        }
        return false;
      });
    remaining_vertices.resize(thrust::distance(remaining_vertices.begin(), last),
                              handle.get_stream());
    // mis_size -= remaining_vertices.size();

    // mis.resize(mis_size, handle.get_stream());

    if (debug) {
      cudaDeviceSynchronize();
      raft::print_device_vector(
        "remaining_vertices*: ", remaining_vertices.data(), remaining_vertices.size(), std::cout);
      raft::print_device_vector("tmp_ranks*:", tmp_ranks.data(), tmp_ranks.size(), std::cout);
      raft::print_device_vector("mis*:", mis.data(), mis.size(), std::cout);
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

  vertex_t nr_vertices_included_in_mis = thrust::count_if(
    handle.get_thrust_policy(), ranks.begin(), ranks.end(), [] __device__(auto v_rank) {
      auto flag = fabs(v_rank - std::numeric_limits<weight_t>::max()) < EPSILON;
      // printf("\n%f %d\n", v_rank, vertex_t{flag});
      return flag;
    });

  if (debug) { raft::print_device_vector("ranks*:", ranks.data(), ranks.size(), std::cout); }

  mis.resize(nr_vertices_included_in_mis, handle.get_stream());

  if (debug) { std::cout << "copy to mis:" << std::endl; }

  thrust::copy_if(handle.get_thrust_policy(),
                  vertex_begin,
                  vertex_end,
                  ranks.begin(),
                  mis.begin(),
                  [] __device__(auto v_rank) {
                    auto flag = fabs(v_rank - std::numeric_limits<weight_t>::max()) < EPSILON;
                    // printf("\n%f %d\n", v_rank, vertex_t{flag});
                    return flag;
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
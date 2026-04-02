/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/fill_edge_property.cuh>
#include <cugraph/prims/fill_edge_src_dst_property.cuh>
#include <cugraph/prims/make_initialized_edge_property.cuh>
#include <cugraph/prims/transform_e.cuh>
#include <cugraph/prims/transform_reduce_v_frontier_outgoing_e_by_dst.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/prims/update_v_frontier.cuh>
#include <cugraph/prims/vertex_frontier.cuh>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/collect_comm.cuh>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

namespace cugraph {

namespace {

// Iteratively peel vertices with zero in-degree or zero out-degree.
// If the peel process does not terminate within max_iterations, attempt to find chains from or to
// peeled vertices.
template <typename GraphViewType, typename KVStoreViewType>
rmm::device_uvector<typename GraphViewType::vertex_type> find_trivial_singleton_scc_vertices(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  GraphViewType const& inverse_graph_view,
  KVStoreViewType const&
    to_inverse_renumber_map /* graph_view vertex ID => local inverse_graph_view vertex ID */,
  raft::device_span<typename GraphViewType::vertex_type const>
    from_inverse_renumber_map /* local inverse_graph_view vertex ID => graph_view vertex ID */,
  raft::device_span<typename GraphViewType::vertex_type const> candidate_vertices /* not sorted */,
  rmm::device_uvector<typename GraphViewType::edge_type>&&
    in_degrees /* for the entire local vertex partition */,
  rmm::device_uvector<typename GraphViewType::edge_type>&&
    out_degrees /* for the entire local vertex partition */,
  size_t max_iterations = 50 /* maximum number of iterations */)
{
  using vertex_t           = typename GraphViewType::vertex_type;
  using edge_t             = typename GraphViewType::edge_type;
  constexpr bool multi_gpu = GraphViewType::is_multi_gpu;

  auto num_aggregate_candidate_vertices = candidate_vertices.size();
  if constexpr (multi_gpu) {
    num_aggregate_candidate_vertices = host_scalar_allreduce(handle.get_comms(),
                                                             num_aggregate_candidate_vertices,
                                                             raft::comms::op_t::SUM,
                                                             handle.get_stream());
  }

  if (num_aggregate_candidate_vertices == 0) {
    return rmm::device_uvector<vertex_t>(0, handle.get_stream());
  }

  rmm::device_uvector<uint32_t> bitmap(
    packed_bool_size(graph_view.local_vertex_partition_range_size()),
    handle.get_stream());  // to enusre candidate vertices enter frontier only once.
  thrust::fill(handle.get_thrust_policy(), bitmap.begin(), bitmap.end(), packed_bool_empty_mask());

  auto cur_in_degrees  = std::move(in_degrees);
  auto cur_out_degrees = std::move(out_degrees);

  rmm::device_uvector<vertex_t> frontier_vertices(candidate_vertices.size(), handle.get_stream());
  frontier_vertices.resize(
    cuda::std::distance(
      frontier_vertices.begin(),
      thrust::copy_if(
        handle.get_thrust_policy(),
        candidate_vertices.begin(),
        candidate_vertices.end(),
        frontier_vertices.begin(),
        cuda::proclaim_return_type<bool>(
          [cur_in_degrees =
             raft::device_span<edge_t const>(cur_in_degrees.data(), cur_in_degrees.size()),
           cur_out_degrees =
             raft::device_span<edge_t const>(cur_out_degrees.data(), cur_out_degrees.size()),
           v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
            auto v_offset = v - v_first;
            return (cur_in_degrees[v_offset] == 0) || (cur_out_degrees[v_offset] == 0);
          }))),
    handle.get_stream());
  thrust::sort(handle.get_thrust_policy(), frontier_vertices.begin(), frontier_vertices.end());

  rmm::device_uvector<vertex_t> peeled_vertices(0, handle.get_stream());
  peeled_vertices.reserve(candidate_vertices.size(), handle.get_stream());

  size_t aggregate_frontier_size{0};
  size_t iter{0};
  while (true) {
    aggregate_frontier_size = frontier_vertices.size();
    if constexpr (multi_gpu) {
      aggregate_frontier_size = host_scalar_allreduce(
        handle.get_comms(), aggregate_frontier_size, raft::comms::op_t::SUM, handle.get_stream());
    }
    if (aggregate_frontier_size == 0) { break; }

    thrust::for_each(
      handle.get_thrust_policy(),
      frontier_vertices.begin(),
      frontier_vertices.end(),
      [bitmap  = raft::device_span<uint32_t>(bitmap.data(), bitmap.size()),
       v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
        auto v_offset = v - v_first;
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(
          bitmap[packed_bool_offset(v_offset)]);
        word.fetch_or(packed_bool_mask(v_offset), cuda::std::memory_order_relaxed);
      });

    if (iter >= max_iterations) { break; }

    // 1. Append frontier vertices to peeled vertices (if iter >= max_iterations, add
    // frontier_vertices to peeled_vertices after computing chain candidate vertices (one_vertices)'
    // acestors & descendants)

    auto old_size = peeled_vertices.size();
    peeled_vertices.resize(old_size + frontier_vertices.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 frontier_vertices.begin(),
                 frontier_vertices.end(),
                 peeled_vertices.begin() + old_size);

    // 2. Forward pass: traverse outgoing edges from frontier, decrement in_degrees of destinations

    rmm::device_uvector<vertex_t> fwd_dst_vertices(0, handle.get_stream());
    {
      key_bucket_view_t<vertex_t, void, multi_gpu, true> frontier(
        handle,
        raft::device_span<vertex_t const>(frontier_vertices.data(), frontier_vertices.size()));

      rmm::device_uvector<edge_t> decrement_counts(0, handle.get_stream());
      std::tie(fwd_dst_vertices, decrement_counts) =
        cugraph::transform_reduce_v_frontier_outgoing_e_by_dst(
          handle,
          graph_view,
          frontier,
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          edge_dummy_property_t{}.view(),
          cuda::proclaim_return_type<edge_t>(
            [iter] __device__(auto src, auto dst, auto, auto, auto) { return edge_t{1}; }),
          reduce_op::plus<edge_t>());

      auto pair_first =
        thrust::make_zip_iterator(fwd_dst_vertices.begin(), decrement_counts.begin());
      thrust::for_each(
        handle.get_thrust_policy(),
        pair_first,
        pair_first + fwd_dst_vertices.size(),
        [cur_in_degrees = raft::device_span<edge_t>(cur_in_degrees.data(), cur_in_degrees.size()),
         v_first        = graph_view.local_vertex_partition_range_first()] __device__(auto pair) {
          auto v_offset        = cuda::std::get<0>(pair) - v_first;
          auto decrement_count = cuda::std::get<1>(pair);
          cur_in_degrees[v_offset] -= decrement_count;
        });
    }

    // 3. Inverse pass: traverse incoming edges (via inverse graph), decrement out_degrees of
    // sources

    rmm::device_uvector<vertex_t> inv_dst_vertices(0, handle.get_stream());
    {
      auto inv_frontier_vertices = std::move(frontier_vertices);
      if constexpr (multi_gpu) {
        std::tie(inv_frontier_vertices, std::ignore) =
          shuffle_ext_vertices(handle,
                               std::move(inv_frontier_vertices),
                               std::vector<cugraph::arithmetic_device_uvector_t>{});
      }

      to_inverse_renumber_map.find(
        inv_frontier_vertices.begin(),
        inv_frontier_vertices.end(),
        inv_frontier_vertices.begin(),
        handle.get_stream()); /* functionally identical to renumber_local_ext_vertices but to avoid
                                 repetitively rebuilding a kv_store_t object for the entire local
                                 vertex partition range */

      thrust::sort(
        handle.get_thrust_policy(), inv_frontier_vertices.begin(), inv_frontier_vertices.end());

      key_bucket_t<vertex_t, void, multi_gpu, true> inv_frontier(handle,
                                                                 std::move(inv_frontier_vertices));

      rmm::device_uvector<edge_t> inv_decrement_counts(0, handle.get_stream());
      std::tie(inv_dst_vertices, inv_decrement_counts) =
        cugraph::transform_reduce_v_frontier_outgoing_e_by_dst(
          handle,
          inverse_graph_view,
          inv_frontier,
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          edge_dummy_property_t{}.view(),
          cuda::proclaim_return_type<edge_t>(
            [renumber_map = raft::device_span<vertex_t const>(from_inverse_renumber_map.data(),
                                                              from_inverse_renumber_map.size()),
             iter] __device__(auto src, auto dst, auto, auto, auto) { return edge_t{1}; }),
          reduce_op::plus<edge_t>());

      unrenumber_local_int_vertices(handle,
                                    inv_dst_vertices.data(),
                                    inv_dst_vertices.size(),
                                    from_inverse_renumber_map.data(),
                                    inverse_graph_view.local_vertex_partition_range_first(),
                                    inverse_graph_view.local_vertex_partition_range_last());

      if constexpr (multi_gpu) {
        std::vector<arithmetic_device_uvector_t> properties{};
        properties.push_back(std::move(inv_decrement_counts));
        std::tie(inv_dst_vertices, properties) =
          shuffle_int_vertices(handle,
                               std::move(inv_dst_vertices),
                               std::move(properties),
                               graph_view.vertex_partition_range_lasts());
        inv_decrement_counts = std::move(std::get<rmm::device_uvector<edge_t>>(properties[0]));
      }

      auto pair_first =
        thrust::make_zip_iterator(inv_dst_vertices.begin(), inv_decrement_counts.begin());
      thrust::for_each(
        handle.get_thrust_policy(),
        pair_first,
        pair_first + inv_dst_vertices.size(),
        [cur_out_degrees =
           raft::device_span<vertex_t>(cur_out_degrees.data(), cur_out_degrees.size()),
         v_first = graph_view.local_vertex_partition_range_first()] __device__(auto pair) {
          auto v_offset        = cuda::std::get<0>(pair) - v_first;
          auto decrement_count = cuda::std::get<1>(pair);
          cur_out_degrees[v_offset] -= decrement_count;
        });
    }

    // 4. Build frontier vertices with newly peeled vertices with zero in- or out-degrees

    auto is_newly_peeled = cuda::proclaim_return_type<bool>(
      [bitmap = raft::device_span<uint32_t const>(bitmap.data(), bitmap.size()),
       cur_in_degrees =
         raft::device_span<edge_t const>(cur_in_degrees.data(), cur_in_degrees.size()),
       cur_out_degrees =
         raft::device_span<edge_t const>(cur_out_degrees.data(), cur_out_degrees.size()),
       v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
        auto v_offset = v - v_first;
        auto word     = bitmap[packed_bool_offset(v_offset)];
        if ((word & packed_bool_mask(v_offset)) != packed_bool_empty_mask()) {  // previously peeled
          return false;
        } else {
          return ((cur_in_degrees[v_offset] == 0) || (cur_out_degrees[v_offset] == 0));
        }
      });
    frontier_vertices = rmm::device_uvector<vertex_t>(
      fwd_dst_vertices.size() + inv_dst_vertices.size(), handle.get_stream());
    auto last = thrust::copy_if(handle.get_thrust_policy(),
                                fwd_dst_vertices.begin(),
                                fwd_dst_vertices.end(),
                                frontier_vertices.begin(),
                                is_newly_peeled);
    last      = thrust::copy_if(handle.get_thrust_policy(),
                           inv_dst_vertices.begin(),
                           inv_dst_vertices.end(),
                           last,
                           is_newly_peeled);
    frontier_vertices.resize(cuda::std::distance(frontier_vertices.begin(), last),
                             handle.get_stream());
    thrust::sort(handle.get_thrust_policy(), frontier_vertices.begin(), frontier_vertices.end());
    frontier_vertices.resize(cuda::std::distance(frontier_vertices.begin(),
                                                 thrust::unique(handle.get_thrust_policy(),
                                                                frontier_vertices.begin(),
                                                                frontier_vertices.end())),
                             handle.get_stream());
    ++iter;
  }
  thrust::sort(handle.get_thrust_policy(), peeled_vertices.begin(), peeled_vertices.end());

  if (aggregate_frontier_size > 0) {  // check for chains from or to peeled vertices
    // find in-degree 1 and out-degree 1 vertices and their predecessors and successors

    rmm::device_uvector<vertex_t> one_vertices(
      candidate_vertices.size(),
      handle.get_stream());  // vertices with one in-degree and one out-degree
    one_vertices.resize(
      cuda::std::distance(
        one_vertices.begin(),
        thrust::copy_if(
          handle.get_thrust_policy(),
          candidate_vertices.begin(),
          candidate_vertices.end(),
          one_vertices.begin(),
          cuda::proclaim_return_type<bool>(
            [cur_in_degrees =
               raft::device_span<edge_t const>(cur_in_degrees.data(), cur_in_degrees.size()),
             cur_out_degrees =
               raft::device_span<edge_t const>(cur_out_degrees.data(), cur_out_degrees.size()),
             v_first = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
              auto v_offset = v - v_first;
              return (cur_in_degrees[v_offset] == edge_t{1}) &&
                     (cur_out_degrees[v_offset] == edge_t{1});
            }))),
      handle.get_stream());

    rmm::device_uvector<vertex_t> descendants(one_vertices.size(), handle.get_stream());
    {
      thrust::sort(handle.get_thrust_policy(), peeled_vertices.begin(), peeled_vertices.end());
      thrust::sort(handle.get_thrust_policy(), one_vertices.begin(), one_vertices.end());
      auto edge_dst_peeled_flags = make_initialized_edge_dst_property(handle, graph_view, false);
      fill_edge_dst_property(
        handle,
        graph_view,
        peeled_vertices.begin(),
        peeled_vertices.end(),
        edge_dst_peeled_flags.mutable_view(),
        true); /* exclude the most recently peeled vertices to include the edges to them. */
      per_v_transform_reduce_if_outgoing_e(
        handle,
        graph_view,
        key_bucket_view_t<vertex_t, void, multi_gpu, true>(
          handle, raft::device_span<vertex_t const>(one_vertices.data(), one_vertices.size())),
        edge_src_dummy_property_t{}.view(),
        edge_dst_peeled_flags.view(),
        edge_dummy_property_t{}.view(),
        cuda::proclaim_return_type<vertex_t>(
          [] __device__(auto, auto dst, auto, auto, auto) { return dst; }),
        invalid_vertex_id_v<vertex_t>,
        reduce_op::any<vertex_t>(),
        cuda::proclaim_return_type<bool>(
          [] __device__(auto, auto, auto, auto flag, auto) { return !flag; }),
        descendants.begin());
    }

    rmm::device_uvector<vertex_t> ancestors(0, handle.get_stream());
    {
      rmm::device_uvector<vertex_t> inv_peeled_vertices(peeled_vertices.size(),
                                                        handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   peeled_vertices.begin(),
                   peeled_vertices.end(),
                   inv_peeled_vertices.begin());
      rmm::device_uvector<vertex_t> inv_one_vertices(one_vertices.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   one_vertices.begin(),
                   one_vertices.end(),
                   inv_one_vertices.begin());
      if constexpr (multi_gpu) {
        std::tie(inv_peeled_vertices, std::ignore) =
          shuffle_ext_vertices(handle,
                               std::move(inv_peeled_vertices),
                               std::vector<cugraph::arithmetic_device_uvector_t>{});
        std::tie(inv_one_vertices, std::ignore) = shuffle_ext_vertices(
          handle, std::move(inv_one_vertices), std::vector<cugraph::arithmetic_device_uvector_t>{});
      }

      to_inverse_renumber_map.find(
        inv_peeled_vertices.begin(),
        inv_peeled_vertices.end(),
        inv_peeled_vertices.begin(),
        handle.get_stream()); /* functionally identical to renumber_local_ext_vertices but to avoid
                                 repetitively rebuilding a kv_store_t object for the entire local
                                 vertex partition range */
      to_inverse_renumber_map.find(
        inv_one_vertices.begin(),
        inv_one_vertices.end(),
        inv_one_vertices.begin(),
        handle.get_stream()); /* functionally identical to renumber_local_ext_vertices but to avoid
                                 repetitively rebuilding a kv_store_t object for the entire local
                                 vertex partition range */
      thrust::sort(
        handle.get_thrust_policy(), inv_peeled_vertices.begin(), inv_peeled_vertices.end());
      thrust::sort(handle.get_thrust_policy(), inv_one_vertices.begin(), inv_one_vertices.end());
      auto edge_dst_peeled_flags =
        make_initialized_edge_dst_property(handle, inverse_graph_view, false);
      fill_edge_dst_property(
        handle,
        inverse_graph_view,
        inv_peeled_vertices.begin(),
        inv_peeled_vertices.end(),
        edge_dst_peeled_flags.mutable_view(),
        true); /* exclude the most recently peeled vertices to include the edges to them. */
      ancestors.resize(inv_one_vertices.size(), handle.get_stream());
      per_v_transform_reduce_if_outgoing_e(
        handle,
        inverse_graph_view,
        key_bucket_view_t<vertex_t, void, multi_gpu, true>(
          handle,
          raft::device_span<vertex_t const>(inv_one_vertices.data(), inv_one_vertices.size())),
        edge_src_dummy_property_t{}.view(),
        edge_dst_peeled_flags.view(),
        edge_dummy_property_t{}.view(),
        cuda::proclaim_return_type<vertex_t>(
          [] __device__(auto, auto dst, auto, auto, auto) { return dst; }),
        invalid_vertex_id_v<vertex_t>,
        reduce_op::any<vertex_t>(),
        cuda::proclaim_return_type<bool>(
          [] __device__(auto, auto, auto, auto flag, auto) { return !flag; }),
        ancestors.begin());

      unrenumber_local_int_vertices(handle,
                                    inv_one_vertices.data(),
                                    inv_one_vertices.size(),
                                    from_inverse_renumber_map.data(),
                                    inverse_graph_view.local_vertex_partition_range_first(),
                                    inverse_graph_view.local_vertex_partition_range_last());
      unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle,
        ancestors.data(),
        ancestors.size(),
        from_inverse_renumber_map.data(),
        inverse_graph_view.vertex_partition_range_lasts());

      if constexpr (multi_gpu) {
        std::vector<arithmetic_device_uvector_t> properties{};
        properties.push_back(std::move(ancestors));
        std::tie(inv_one_vertices, properties) =
          shuffle_int_vertices(handle,
                               std::move(inv_one_vertices),
                               std::move(properties),
                               graph_view.vertex_partition_range_lasts());
        ancestors = std::move(std::get<rmm::device_uvector<vertex_t>>(properties[0]));
      }
      thrust::sort_by_key(handle.get_thrust_policy(),
                          inv_one_vertices.begin(),
                          inv_one_vertices.end(),
                          ancestors.begin());
    }

    // now add the most recently peeled vertices to peeled_vertices

    {
      auto old_size = peeled_vertices.size();
      peeled_vertices.resize(old_size + frontier_vertices.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   frontier_vertices.begin(),
                   frontier_vertices.end(),
                   peeled_vertices.begin() + old_size);
      frontier_vertices.resize(0, handle.get_stream());
      frontier_vertices.shrink_to_fit(handle.get_stream());
    }

    // pointer jumping to find a chain of vertices from/to an already peeled vertex

    rmm::device_uvector<vertex_t> indices(one_vertices.size(), handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(), indices.begin(), indices.end());

    std::optional<kv_store_t<vertex_t, bool, true>> peeled_vertex_store{std::nullopt};
    std::optional<rmm::device_uvector<vertex_t>> d_vertex_partition_range_lasts{std::nullopt};
    std::optional<kv_store_t<vertex_t, vertex_t, false>> ancestor_store{std::nullopt};
    std::optional<kv_store_t<vertex_t, vertex_t, false>> descendant_store{std::nullopt};
    if constexpr (multi_gpu) {
      auto h_vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();
      d_vertex_partition_range_lasts =
        rmm::device_uvector<vertex_t>(h_vertex_partition_range_lasts.size(), handle.get_stream());
      raft::update_device(d_vertex_partition_range_lasts->data(),
                          h_vertex_partition_range_lasts.data(),
                          h_vertex_partition_range_lasts.size(),
                          handle.get_stream());

      peeled_vertex_store = kv_store_t<vertex_t, bool, true>(peeled_vertices.begin(),
                                                             peeled_vertices.end(),
                                                             cuda::make_constant_iterator(true),
                                                             false,
                                                             true,
                                                             handle.get_stream());

      ancestor_store   = kv_store_t<vertex_t, vertex_t, false>(one_vertices.begin(),
                                                             one_vertices.end(),
                                                             ancestors.begin(),
                                                             invalid_vertex_id_v<vertex_t>,
                                                             invalid_vertex_id_v<vertex_t>,
                                                             handle.get_stream());
      descendant_store = kv_store_t<vertex_t, vertex_t, false>(one_vertices.begin(),
                                                               one_vertices.end(),
                                                               descendants.begin(),
                                                               invalid_vertex_id_v<vertex_t>,
                                                               invalid_vertex_id_v<vertex_t>,
                                                               handle.get_stream());
    }

    while (true) {
      // trivial singleton SCC if ancestor or descendant is in peeled_vertices

      auto new_trivial_first = indices.begin();
      if constexpr (multi_gpu) {
        auto major_comm_size =
          handle.get_subcomm(cugraph::partition_manager::major_comm_name()).get_size();
        auto minor_comm_size =
          handle.get_subcomm(cugraph::partition_manager::minor_comm_name()).get_size();
        auto key_to_gpu_id = detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
          raft::device_span<vertex_t const>(d_vertex_partition_range_lasts->data(),
                                            d_vertex_partition_range_lasts->size()),
          major_comm_size,
          minor_comm_size};
        auto collect_key_first = cuda::make_transform_iterator(
          indices.begin(), detail::indirection_t<vertex_t, vertex_t const*>{ancestors.data()});
        auto ancestor_peeled_flags =
          detail::collect_values_for_keys(handle.get_comms(),
                                          peeled_vertex_store->view(),
                                          collect_key_first,
                                          collect_key_first + indices.size(),
                                          key_to_gpu_id,
                                          handle.get_stream());
        collect_key_first = cuda::make_transform_iterator(
          indices.begin(), detail::indirection_t<vertex_t, vertex_t const*>{descendants.data()});
        auto descendant_peeled_flags =
          detail::collect_values_for_keys(handle.get_comms(),
                                          peeled_vertex_store->view(),
                                          collect_key_first,
                                          collect_key_first + indices.size(),
                                          key_to_gpu_id,
                                          handle.get_stream());
        auto triplet_first = thrust::make_zip_iterator(
          indices.begin(), ancestor_peeled_flags.begin(), descendant_peeled_flags.begin());
        auto triplet_last = thrust::partition(
          handle.get_thrust_policy(),
          triplet_first,
          triplet_first + indices.size(),
          cuda::proclaim_return_type<bool>([] __device__(auto triplet) {
            return (cuda::std::get<1>(triplet) != true) && (cuda::std::get<2>(triplet) != true);
          }));
        new_trivial_first = indices.begin() + cuda::std::distance(triplet_first, triplet_last);
      } else {
        new_trivial_first = thrust::partition(
          handle.get_thrust_policy(),
          indices.begin(),
          indices.end(),
          cuda::proclaim_return_type<bool>(
            [one_vertices =
               raft::device_span<vertex_t const>(one_vertices.data(), one_vertices.size()),
             bitmap      = raft::device_span<uint32_t const>(bitmap.data(), bitmap.size()),
             ancestors   = raft::device_span<vertex_t const>(ancestors.data(), ancestors.size()),
             descendants = raft::device_span<vertex_t const>(
               descendants.data(), descendants.size())] __device__(auto i) {
              auto a = ancestors[i];
              auto d = descendants[i];
              return ((bitmap[packed_bool_offset(a)] & packed_bool_mask(a)) ==
                      packed_bool_empty_mask()) &&
                     ((bitmap[packed_bool_offset(d)] & packed_bool_mask(d)) ==
                      packed_bool_empty_mask());
            }));
      }
      auto new_trivial_size =
        static_cast<size_t>(cuda::std::distance(new_trivial_first, indices.end()));
      if (new_trivial_size > 0) {
        auto old_size = peeled_vertices.size();
        peeled_vertices.resize(peeled_vertices.size() + new_trivial_size, handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       new_trivial_first,
                       new_trivial_first + new_trivial_size,
                       one_vertices.begin(),
                       peeled_vertices.begin() + old_size);
        indices.resize(cuda::std::distance(indices.begin(), new_trivial_first),
                       handle.get_stream());
      }

      if constexpr (multi_gpu) {
        new_trivial_size = host_scalar_allreduce(
          handle.get_comms(), new_trivial_size, raft::comms::op_t::SUM, handle.get_stream());
      }
      if (new_trivial_size == 0) { break; }

      // update ancestor/descendant
      // not a trivial singleton SCC if no ancestor/descendant is in one_vertices (not a part of a
      // chain from/to one of the already peeled vertices)

      rmm::device_uvector<vertex_t> new_ancestors(0, handle.get_stream());
      rmm::device_uvector<vertex_t> new_descendants(0, handle.get_stream());
      if constexpr (multi_gpu) {
        auto major_comm_size =
          handle.get_subcomm(cugraph::partition_manager::major_comm_name()).get_size();
        auto minor_comm_size =
          handle.get_subcomm(cugraph::partition_manager::minor_comm_name()).get_size();
        auto key_to_gpu_id = detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
          raft::device_span<vertex_t const>(d_vertex_partition_range_lasts->data(),
                                            d_vertex_partition_range_lasts->size()),
          major_comm_size,
          minor_comm_size};

        auto ancestor_first = cuda::make_transform_iterator(
          indices.begin(), detail::indirection_t<vertex_t, vertex_t const*>{ancestors.data()});
        new_ancestors         = detail::collect_values_for_keys(handle.get_comms(),
                                                        ancestor_store->view(),
                                                        ancestor_first,
                                                        ancestor_first + indices.size(),
                                                        key_to_gpu_id,
                                                        handle.get_stream());
        auto descendant_first = cuda::make_transform_iterator(
          indices.begin(), detail::indirection_t<vertex_t, vertex_t const*>{descendants.data()});
        new_descendants = detail::collect_values_for_keys(handle.get_comms(),
                                                          descendant_store->view(),
                                                          descendant_first,
                                                          descendant_first + indices.size(),
                                                          key_to_gpu_id,
                                                          handle.get_stream());
      } else {
        new_ancestors.resize(indices.size(), handle.get_stream());
        new_descendants.resize(indices.size(), handle.get_stream());
        thrust::transform(
          handle.get_thrust_policy(),
          indices.begin(),
          indices.end(),
          thrust::make_zip_iterator(new_ancestors.begin(), new_descendants.begin()),
          cuda::proclaim_return_type<cuda::std::tuple<vertex_t, vertex_t>>(
            [one_vertices =
               raft::device_span<vertex_t const>(one_vertices.data(), one_vertices.size()),
             ancestors = raft::device_span<vertex_t const>(ancestors.data(), ancestors.size()),
             descendants =
               raft::device_span<vertex_t const>(descendants.data(), descendants.size()),
             invalid_v = invalid_vertex_id_v<vertex_t>] __device__(auto i) {
              auto a = ancestors[i];
              auto it =
                thrust::lower_bound(thrust::seq, one_vertices.begin(), one_vertices.end(), a);
              if (it != one_vertices.end() && *it == a) {
                a = ancestors[cuda::std::distance(one_vertices.begin(), it)];
              } else {
                a = invalid_v;
              }
              auto d = descendants[i];
              it = thrust::lower_bound(thrust::seq, one_vertices.begin(), one_vertices.end(), d);
              if (it != one_vertices.end() && *it == d) {
                d = descendants[cuda::std::distance(one_vertices.begin(), it)];
              } else {
                d = invalid_v;
              }
              return cuda::std::make_tuple(a, d);
            }));
      }
      auto triplet_first =
        thrust::make_zip_iterator(indices.begin(), new_ancestors.begin(), new_descendants.begin());
      indices.resize(
        cuda::std::distance(
          triplet_first,
          thrust::remove_if(handle.get_thrust_policy(),
                            triplet_first,
                            triplet_first + indices.size(),
                            cuda::proclaim_return_type<bool>(
                              [invalid_v = invalid_vertex_id_v<vertex_t>] __device__(auto triplet) {
                                return (cuda::std::get<1>(triplet) == invalid_v) &&
                                       (cuda::std::get<2>(triplet) == invalid_v);
                              }))),
        handle.get_stream());
      new_ancestors.resize(indices.size(), handle.get_stream());
      new_descendants.resize(indices.size(), handle.get_stream());
      triplet_first =
        thrust::make_zip_iterator(indices.begin(), new_ancestors.begin(), new_descendants.begin());
      thrust::for_each(
        handle.get_thrust_policy(),
        triplet_first,
        triplet_first + indices.size(),
        [ancestors   = raft::device_span<vertex_t>(ancestors.data(), ancestors.size()),
         descendants = raft::device_span<vertex_t>(descendants.data(), descendants.size()),
         invalid_v   = invalid_vertex_id_v<vertex_t>] __device__(auto triplet) {
          auto idx = cuda::std::get<0>(triplet);
          auto a   = cuda::std::get<1>(triplet);
          auto d   = cuda::std::get<2>(triplet);
          if (a != invalid_v) { ancestors[idx] = a; }
          if (d != invalid_v) { descendants[idx] = d; }
        });

      if constexpr (multi_gpu) {
        auto map_first = cuda::make_transform_iterator(
          indices.begin(), detail::indirection_t<vertex_t, vertex_t const*>{one_vertices.data()});
        ancestor_store->insert_and_assign_if(
          map_first,
          map_first + indices.size(),
          new_ancestors.begin(),
          new_ancestors.begin(),
          cuda::proclaim_return_type<bool>([invalid_v = invalid_vertex_id_v<vertex_t>] __device__(
                                             auto v) { return v != invalid_v; }),
          handle.get_stream());

        descendant_store->insert_and_assign_if(
          map_first,
          map_first + indices.size(),
          new_descendants.begin(),
          new_descendants.begin(),
          cuda::proclaim_return_type<bool>([invalid_v = invalid_vertex_id_v<vertex_t>] __device__(
                                             auto v) { return v != invalid_v; }),
          handle.get_stream());
      }
    }

    thrust::sort(handle.get_thrust_policy(), peeled_vertices.begin(), peeled_vertices.end());
  }

  peeled_vertices.shrink_to_fit(handle.get_stream());

  return peeled_vertices;
}

// find pivots;returns (pivot vertices, pivot unresolved component indexes) pairs; the returned
// pairs should be sorted by pivot vertex. Currently, we return one pivot per unresolved component
// (or zero if all the vertices in the unresolved component are excluded). We may update this code
// to return more than one pivot per unresolved component to extract additional parallelism for
// large diameter graphs (but this will complicate the reachable_sets function implementation).
template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>>
find_pivots(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  raft::device_span<typename GraphViewType::vertex_type const> unresolved_component_offsets,
  raft::device_span<typename GraphViewType::vertex_type const> unresolved_component_vertices,
  raft::device_span<typename GraphViewType::edge_type const> unresolved_component_vertex_in_degrees,
  raft::device_span<typename GraphViewType::edge_type const>
    unresolved_component_vertex_out_degrees,
  raft::device_span<typename GraphViewType::vertex_type const>
    sorted_excluded_vertices /* should not be selected as pivots */)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  // reduce key (component index) value (pivot, priority) pairs (find highest priority (pivot,
  // priority) pairs for each component ID)

  rmm::device_uvector<vertex_t> component_idxs(unresolved_component_offsets.size() - 1,
                                               handle.get_stream());
  rmm::device_uvector<vertex_t> pivots(component_idxs.size(), handle.get_stream());
  rmm::device_uvector<edge_t> priorities(component_idxs.size(), handle.get_stream());
  {
    auto component_idx_first = cuda::make_transform_iterator(
      thrust::make_counting_iterator(vertex_t{0}),
      detail::segment_id_t<vertex_t>{raft::device_span<vertex_t const>(
        unresolved_component_offsets.data() + 1, unresolved_component_offsets.size() - 1)});
    auto priority_first = cuda::make_transform_iterator(
      thrust::make_zip_iterator(unresolved_component_vertices.begin(),
                                unresolved_component_vertex_in_degrees.begin(),
                                unresolved_component_vertex_out_degrees.begin()),
      cuda::proclaim_return_type<edge_t>(
        [excluded_vertices = raft::device_span<vertex_t const>(
           sorted_excluded_vertices.data(),
           sorted_excluded_vertices.size())] __device__(auto triplet) {
          auto excluded = thrust::binary_search(thrust::seq,
                                                excluded_vertices.begin(),
                                                excluded_vertices.end(),
                                                cuda::std::get<0>(triplet));
          if (excluded)
            return edge_t{0};
          else
            return cuda::std::min(cuda::std::get<1>(triplet), cuda::std::get<2>(triplet));
        }));
    auto ret = thrust::reduce_by_key(
      handle.get_thrust_policy(),
      component_idx_first,
      component_idx_first + unresolved_component_vertices.size(),
      thrust::make_zip_iterator(unresolved_component_vertices.begin(), priority_first),
      component_idxs.begin(),
      thrust::make_zip_iterator(pivots.begin(), priorities.begin()),
      cuda::std::equal_to<vertex_t>{},
      cuda::proclaim_return_type<cuda::std::tuple<vertex_t, edge_t>>(
        [] __device__(auto lhs, auto rhs) {
          return cuda::std::get<1>(lhs) >= cuda::std::get<1>(rhs) ? lhs : rhs;
        }));
    component_idxs.resize(cuda::std::distance(component_idxs.begin(), ret.first),
                          handle.get_stream());
    pivots.resize(component_idxs.size(), handle.get_stream());
    priorities.resize(component_idxs.size(), handle.get_stream());
    component_idxs.shrink_to_fit(handle.get_stream());
    pivots.shrink_to_fit(handle.get_stream());
    priorities.shrink_to_fit(handle.get_stream());
  }

  // remove 0 priority key value pairs (0 priority means excluded)

  {
    auto triplet_first =
      thrust::make_zip_iterator(component_idxs.begin(), pivots.begin(), priorities.begin());
    component_idxs.resize(
      cuda::std::distance(
        triplet_first,
        thrust::remove_if(handle.get_thrust_policy(),
                          triplet_first,
                          triplet_first + component_idxs.size(),
                          cuda::proclaim_return_type<bool>([] __device__(auto triplet) {
                            return cuda::std::get<2>(triplet) == 0;
                          }))),
      handle.get_stream());
    pivots.resize(component_idxs.size(), handle.get_stream());
    priorities.resize(component_idxs.size(), handle.get_stream());
    component_idxs.shrink_to_fit(handle.get_stream());
    pivots.shrink_to_fit(handle.get_stream());
    priorities.shrink_to_fit(handle.get_stream());
  }

  if constexpr (GraphViewType::is_multi_gpu) {
    // multi-GPU reduction

    std::vector<arithmetic_device_uvector_t> vertex_properties{};
    vertex_properties.push_back(std::move(pivots));
    vertex_properties.push_back(std::move(priorities));
    std::tie(component_idxs, vertex_properties) =
      shuffle_ext_vertices(handle, std::move(component_idxs), std::move(vertex_properties));
    pivots     = std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
    priorities = std::move(std::get<rmm::device_uvector<edge_t>>(vertex_properties[1]));
    thrust::sort_by_key(handle.get_thrust_policy(),
                        component_idxs.begin(),
                        component_idxs.end(),
                        thrust::make_zip_iterator(pivots.begin(), priorities.begin()));
    rmm::device_uvector<vertex_t> tmp_component_idxs(unresolved_component_offsets.size() - 1,
                                                     handle.get_stream());
    rmm::device_uvector<vertex_t> tmp_pivots(tmp_component_idxs.size(), handle.get_stream());
    rmm::device_uvector<edge_t> tmp_priorities(tmp_component_idxs.size(), handle.get_stream());
    auto ret =
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            component_idxs.begin(),
                            component_idxs.end(),
                            thrust::make_zip_iterator(pivots.begin(), priorities.begin()),
                            tmp_component_idxs.begin(),
                            thrust::make_zip_iterator(tmp_pivots.begin(), tmp_priorities.begin()),
                            cuda::std::equal_to<vertex_t>{},
                            cuda::proclaim_return_type<cuda::std::tuple<vertex_t, edge_t>>(
                              [] __device__(auto lhs, auto rhs) {
                                return cuda::std::get<1>(lhs) >= cuda::std::get<1>(rhs) ? lhs : rhs;
                              }));
    tmp_priorities.resize(0, handle.get_stream());
    tmp_priorities.shrink_to_fit(handle.get_stream());
    tmp_component_idxs.resize(cuda::std::distance(tmp_component_idxs.begin(), ret.first),
                              handle.get_stream());
    tmp_pivots.resize(tmp_component_idxs.size(), handle.get_stream());
    tmp_component_idxs.shrink_to_fit(handle.get_stream());
    tmp_pivots.shrink_to_fit(handle.get_stream());

    vertex_properties.clear();
    vertex_properties.push_back(std::move(tmp_component_idxs));
    std::tie(pivots, vertex_properties) =
      shuffle_int_vertices(handle,
                           std::move(tmp_pivots),
                           std::move(vertex_properties),
                           graph_view.vertex_partition_range_lasts());
    component_idxs = std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
  }

  thrust::sort_by_key(
    handle.get_thrust_policy(), pivots.begin(), pivots.end(), component_idxs.begin());

  return std::make_tuple(std::move(pivots), std::move(component_idxs));
}

// return (offsets, reachable vertices) paris (offsets.size() = num_unresolved_components + 1)
template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>>
reachable_sets(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  raft::device_span<typename GraphViewType::vertex_type const> sorted_starting_vertices,
  raft::device_span<typename GraphViewType::vertex_type const>
    sorted_starting_vertex_unresolved_component_idxs,
  typename GraphViewType::vertex_type num_unresolved_components)
{
  using vertex_t = typename GraphViewType::vertex_type;

  rmm::device_uvector<vertex_t> vertices(0, handle.get_stream());
  rmm::device_uvector<vertex_t> idxs(0, handle.get_stream());
  {
    // run multi-source BFS

    rmm::device_uvector<vertex_t> predecessors(graph_view.local_vertex_partition_range_size(),
                                               handle.get_stream());
    {
      rmm::device_uvector<vertex_t> distances(graph_view.local_vertex_partition_range_size(),
                                              handle.get_stream());
      bfs(handle,
          graph_view,
          distances.data(),
          predecessors.data(),
          sorted_starting_vertices.data(),
          sorted_starting_vertices.size());
      thrust::scatter(
        handle.get_thrust_policy(),
        sorted_starting_vertices.begin(),
        sorted_starting_vertices.end(),
        cuda::make_transform_iterator(
          sorted_starting_vertices.begin(),
          cugraph::detail::shift_left_t<vertex_t>{graph_view.local_vertex_partition_range_first()}),
        predecessors
          .begin());  // bfs sets predecessors of starting vertices to invalid_vertex_id_v<vertex_t>
    }

    // back-track to the starting vertices

    rmm::device_uvector<vertex_t> remaining_vertices(graph_view.local_vertex_partition_range_size(),
                                                     handle.get_stream());
    remaining_vertices.resize(
      cuda::std::distance(
        remaining_vertices.begin(),
        thrust::copy_if(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
          thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
          predecessors.begin(),
          remaining_vertices.begin(),
          cuda::proclaim_return_type<bool>(
            [invalid_vertex = invalid_vertex_id_v<vertex_t>] __device__(auto pred) {
              return pred != invalid_vertex;
            }))),
      handle.get_stream());
    vertices.resize(remaining_vertices.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 remaining_vertices.begin(),
                 remaining_vertices.end(),
                 vertices.begin());

    auto ancestors = std::move(predecessors);
    while (true) {
      auto aggregated_size = remaining_vertices.size();
      if constexpr (GraphViewType::is_multi_gpu) {
        aggregated_size = host_scalar_allreduce(
          handle.get_comms(), aggregated_size, raft::comms::op_t::SUM, handle.get_stream());
      }
      if (aggregated_size == size_t{0}) { break; }

      rmm::device_uvector<vertex_t> new_remaining_vertex_ancestors(0, handle.get_stream());
      auto remaining_vertex_ancestor_first = cuda::make_transform_iterator(
        remaining_vertices.begin(),
        cuda::proclaim_return_type<vertex_t>(
          [ancestors = raft::device_span<vertex_t const>(ancestors.data(), ancestors.size()),
           v_first   = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
            return ancestors[v - v_first];
          }));
      if constexpr (GraphViewType::is_multi_gpu) {
        new_remaining_vertex_ancestors = collect_values_for_int_vertices(
          handle,
          remaining_vertex_ancestor_first,
          remaining_vertex_ancestor_first + remaining_vertices.size(),
          ancestors.begin(),
          graph_view.vertex_partition_range_lasts(),
          graph_view.local_vertex_partition_range_first());
      } else {
        new_remaining_vertex_ancestors.resize(remaining_vertices.size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       remaining_vertex_ancestor_first,
                       remaining_vertex_ancestor_first + remaining_vertices.size(),
                       ancestors.begin(),
                       new_remaining_vertex_ancestors.begin());
      }
      rmm::device_uvector<bool> updated_flags(remaining_vertices.size(), handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        updated_flags.begin(),
        updated_flags.end(),
        cuda::proclaim_return_type<bool>(
          [remaining_vertices = raft::device_span<vertex_t const>(remaining_vertices.data(),
                                                                  remaining_vertices.size()),
           new_remaining_vertex_ancestors = raft::device_span<vertex_t const>(
             new_remaining_vertex_ancestors.data(), new_remaining_vertex_ancestors.size()),
           ancestors = raft::device_span<vertex_t const>(ancestors.data(), ancestors.size()),
           v_first   = graph_view.local_vertex_partition_range_first()] __device__(auto i) {
            auto v_offset     = remaining_vertices[i] - v_first;
            auto old_ancestor = ancestors[v_offset];
            auto new_ancestor = new_remaining_vertex_ancestors[i];
            return old_ancestor != new_ancestor;
          }));
      thrust::scatter_if(
        handle.get_thrust_policy(),
        new_remaining_vertex_ancestors.begin(),
        new_remaining_vertex_ancestors.end(),
        cuda::make_transform_iterator(
          remaining_vertices.begin(),
          detail::shift_left_t<vertex_t>{graph_view.local_vertex_partition_range_first()}),
        updated_flags.begin(),
        ancestors.begin(),
        cuda::std::identity());
      remaining_vertices.resize(
        cuda::std::distance(remaining_vertices.begin(),
                            thrust::remove_if(handle.get_thrust_policy(),
                                              remaining_vertices.begin(),
                                              remaining_vertices.end(),
                                              updated_flags.begin(),
                                              detail::is_equal_t<bool>{false})),
        handle.get_stream());
    }

    cugraph::kv_store_t<vertex_t, vertex_t, true> starting_vertex_unresolved_component_idx_store(
      sorted_starting_vertices.begin(),
      sorted_starting_vertices.end(),
      sorted_starting_vertex_unresolved_component_idxs.begin(),
      invalid_component_id_v<vertex_t>,
      true /* key_sorted */,
      handle.get_stream());
    auto starting_vertex_unresolved_component_idx_store_view =
      starting_vertex_unresolved_component_idx_store.view();

    idxs.resize(vertices.size(), handle.get_stream());
    auto ancestor_first = cuda::make_transform_iterator(
      vertices.begin(),
      cuda::proclaim_return_type<vertex_t>(
        [ancestors = raft::device_span<vertex_t const>(ancestors.data(), ancestors.size()),
         v_first   = graph_view.local_vertex_partition_range_first()] __device__(auto v) {
          return ancestors[v - v_first];
        }));
    starting_vertex_unresolved_component_idx_store_view.find(
      ancestor_first, ancestor_first + vertices.size(), idxs.begin(), handle.get_stream());
  }

  {
    auto pair_first = thrust::make_zip_iterator(idxs.begin(), vertices.begin());
    thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + idxs.size());
  }

  // starting_vertices => component indices

  rmm::device_uvector<vertex_t> offsets(num_unresolved_components + 1, handle.get_stream());
  offsets.set_element_to_zero_async(0, handle.get_stream());
  thrust::upper_bound(handle.get_thrust_policy(),
                      idxs.begin(),
                      idxs.end(),
                      thrust::make_counting_iterator(vertex_t{0}),
                      thrust::make_counting_iterator(num_unresolved_components),
                      offsets.begin() + 1);

  return std::make_tuple(std::move(offsets), std::move(vertices));
}

// Input vertices in each segment should be pre-sorted.
// return new (unresovled_component_ids, unresolved_component_offsets,
// unresolved_component_vertices) and new (scc_component_ids, scc_component_offsets, and
// scc_component_vertices).
template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>>
intersect_reachable_sets(
  raft::handle_t const& handle,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& unresolved_component_offsets,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& unresolved_component_vertices,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& forward_set_offsets,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& forward_set_vertices,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& backward_set_offsets,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& backward_set_vertices,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& trivial_singleton_scc_vertices)
{
  using vertex_t = typename GraphViewType::vertex_type;

  auto num_old_unresolved_components =
    static_cast<vertex_t>(unresolved_component_offsets.size() - 1);

  rmm::device_uvector<vertex_t> fwd_only_component_idxs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> fwd_only_vertices(0, handle.get_stream());
  rmm::device_uvector<vertex_t> bwd_only_component_idxs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> bwd_only_vertices(0, handle.get_stream());
  rmm::device_uvector<vertex_t> scc_component_idxs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> scc_vertices(0, handle.get_stream());
  rmm::device_uvector<vertex_t> remaining_component_idxs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> remaining_vertices(0, handle.get_stream());
  {
    // Find the component_idx for each vertex based on CSR offsets
    auto unresolved_component_idxs_first = cuda::make_transform_iterator(
      thrust::make_counting_iterator(vertex_t{0}),
      detail::segment_id_t<vertex_t>{raft::device_span<vertex_t const>(
        unresolved_component_offsets.data() + 1, unresolved_component_offsets.size() - 1)});
    auto forward_set_component_idxs_first = cuda::make_transform_iterator(
      thrust::make_counting_iterator(vertex_t{0}),
      detail::segment_id_t<vertex_t>{raft::device_span<vertex_t const>(
        forward_set_offsets.data() + 1, forward_set_offsets.size() - 1)});
    auto backward_set_component_idxs_first = cuda::make_transform_iterator(
      thrust::make_counting_iterator(vertex_t{0}),
      detail::segment_id_t<vertex_t>{raft::device_span<vertex_t const>(
        backward_set_offsets.data() + 1, backward_set_offsets.size() - 1)});

    auto unresolved_component_pairs_first = thrust::make_zip_iterator(
      unresolved_component_idxs_first, unresolved_component_vertices.begin());
    auto forward_set_pairs_first =
      thrust::make_zip_iterator(forward_set_component_idxs_first, forward_set_vertices.begin());
    auto backward_set_pairs_first =
      thrust::make_zip_iterator(backward_set_component_idxs_first, backward_set_vertices.begin());

    auto remove_trivial_singleton_scc_vertices =
      [&handle,
       trivial_singleton_scc_vertices = raft::device_span<vertex_t const>(
         trivial_singleton_scc_vertices.data(), trivial_singleton_scc_vertices.size())](
        rmm::device_uvector<vertex_t>& component_idxs,
        rmm::device_uvector<vertex_t>& component_vertices) {
        auto zip_first =
          thrust::make_zip_iterator(component_idxs.begin(), component_vertices.begin());
        auto zip_end =
          thrust::remove_if(handle.get_thrust_policy(),
                            zip_first,
                            zip_first + component_vertices.size(),
                            [trivial_singleton_scc_vertices] __device__(auto pair) {
                              auto v = cuda::std::get<1>(pair);
                              return thrust::binary_search(thrust::seq,
                                                           trivial_singleton_scc_vertices.begin(),
                                                           trivial_singleton_scc_vertices.end(),
                                                           v);
                            });
        auto count = static_cast<vertex_t>(cuda::std::distance(zip_first, zip_end));
        component_idxs.resize(count, handle.get_stream());
        component_vertices.resize(count, handle.get_stream());
      };

    // 1. FWD_ONLY = FWD \ BWD

    fwd_only_component_idxs.resize(forward_set_vertices.size(), handle.get_stream());
    fwd_only_vertices.resize(forward_set_vertices.size(), handle.get_stream());
    {
      auto out_first =
        thrust::make_zip_iterator(fwd_only_component_idxs.begin(), fwd_only_vertices.begin());
      auto out_end = thrust::set_difference(handle.get_thrust_policy(),
                                            forward_set_pairs_first,
                                            forward_set_pairs_first + forward_set_vertices.size(),
                                            backward_set_pairs_first,
                                            backward_set_pairs_first + backward_set_vertices.size(),
                                            out_first);
      auto count   = static_cast<vertex_t>(cuda::std::distance(out_first, out_end));
      fwd_only_component_idxs.resize(count, handle.get_stream());
      fwd_only_vertices.resize(count, handle.get_stream());

      remove_trivial_singleton_scc_vertices(fwd_only_component_idxs, fwd_only_vertices);

      fwd_only_component_idxs.shrink_to_fit(handle.get_stream());
      fwd_only_vertices.shrink_to_fit(handle.get_stream());
    }

    // 2. BWD_ONLY = BWD \ FWD

    bwd_only_component_idxs.resize(backward_set_vertices.size(), handle.get_stream());
    bwd_only_vertices.resize(backward_set_vertices.size(), handle.get_stream());
    {
      auto out_first =
        thrust::make_zip_iterator(bwd_only_component_idxs.begin(), bwd_only_vertices.begin());
      auto out_end = thrust::set_difference(handle.get_thrust_policy(),
                                            backward_set_pairs_first,
                                            backward_set_pairs_first + backward_set_vertices.size(),
                                            forward_set_pairs_first,
                                            forward_set_pairs_first + forward_set_vertices.size(),
                                            out_first);
      auto count   = static_cast<vertex_t>(cuda::std::distance(out_first, out_end));
      bwd_only_component_idxs.resize(count, handle.get_stream());
      bwd_only_vertices.resize(count, handle.get_stream());

      remove_trivial_singleton_scc_vertices(bwd_only_component_idxs, bwd_only_vertices);

      bwd_only_component_idxs.shrink_to_fit(handle.get_stream());
      bwd_only_vertices.shrink_to_fit(handle.get_stream());
    }

    // 3. REMAINDER = UC \ (FWD ∪ BWD)

    remaining_component_idxs.resize(unresolved_component_vertices.size(), handle.get_stream());
    remaining_vertices.resize(remaining_component_idxs.size(), handle.get_stream());
    {
      rmm::device_uvector<vertex_t> union_component_idxs(
        forward_set_vertices.size() + backward_set_vertices.size(), handle.get_stream());
      rmm::device_uvector<vertex_t> union_vertices(union_component_idxs.size(),
                                                   handle.get_stream());
      {
        auto out_first =
          thrust::make_zip_iterator(union_component_idxs.begin(), union_vertices.begin());
        auto out_end = thrust::set_union(handle.get_thrust_policy(),
                                         forward_set_pairs_first,
                                         forward_set_pairs_first + forward_set_vertices.size(),
                                         backward_set_pairs_first,
                                         backward_set_pairs_first + backward_set_vertices.size(),
                                         out_first);
        auto count   = static_cast<vertex_t>(cuda::std::distance(out_first, out_end));
        union_component_idxs.resize(count, handle.get_stream());
        union_vertices.resize(count, handle.get_stream());
        union_component_idxs.shrink_to_fit(handle.get_stream());
        union_vertices.shrink_to_fit(handle.get_stream());
      }

      auto union_pairs_first =
        thrust::make_zip_iterator(union_component_idxs.begin(), union_vertices.begin());
      auto out_first =
        thrust::make_zip_iterator(remaining_component_idxs.begin(), remaining_vertices.begin());
      auto out_end = thrust::set_difference(
        handle.get_thrust_policy(),
        unresolved_component_pairs_first,
        unresolved_component_pairs_first + unresolved_component_vertices.size(),
        union_pairs_first,
        union_pairs_first + union_vertices.size(),
        out_first);
      auto count = static_cast<vertex_t>(cuda::std::distance(out_first, out_end));
      remaining_component_idxs.resize(count, handle.get_stream());
      remaining_vertices.resize(count, handle.get_stream());

      remove_trivial_singleton_scc_vertices(remaining_component_idxs, remaining_vertices);

      remaining_component_idxs.shrink_to_fit(handle.get_stream());
      remaining_vertices.shrink_to_fit(handle.get_stream());
    }

    // 4. SCC = FWD ∩ BWD

    scc_component_idxs.resize(std::min(forward_set_vertices.size(), backward_set_vertices.size()),
                              handle.get_stream());
    scc_vertices.resize(scc_component_idxs.size(), handle.get_stream());
    {
      auto out_first = thrust::make_zip_iterator(scc_component_idxs.begin(), scc_vertices.begin());
      auto out_end =
        thrust::set_intersection(handle.get_thrust_policy(),
                                 forward_set_pairs_first,
                                 forward_set_pairs_first + forward_set_vertices.size(),
                                 backward_set_pairs_first,
                                 backward_set_pairs_first + backward_set_vertices.size(),
                                 out_first);
      auto count = static_cast<vertex_t>(cuda::std::distance(out_first, out_end));
      scc_component_idxs.resize(count, handle.get_stream());
      scc_component_idxs.shrink_to_fit(handle.get_stream());
      scc_vertices.resize(count, handle.get_stream());
      scc_vertices.shrink_to_fit(handle.get_stream());
    }
  }
  unresolved_component_offsets.resize(0, handle.get_stream());
  unresolved_component_vertices.resize(0, handle.get_stream());
  forward_set_offsets.resize(0, handle.get_stream());
  forward_set_vertices.resize(0, handle.get_stream());
  backward_set_offsets.resize(0, handle.get_stream());
  backward_set_vertices.resize(0, handle.get_stream());
  unresolved_component_offsets.shrink_to_fit(handle.get_stream());
  unresolved_component_vertices.shrink_to_fit(handle.get_stream());
  forward_set_offsets.shrink_to_fit(handle.get_stream());
  forward_set_vertices.shrink_to_fit(handle.get_stream());
  backward_set_offsets.shrink_to_fit(handle.get_stream());
  backward_set_vertices.shrink_to_fit(handle.get_stream());

  rmm::device_uvector<vertex_t> component_global_min_vertex_ids(
    0,
    handle
      .get_stream());  // size = 4 * num_unresolved_components (fwd_only, bwd_only, remaining, scc)
  rmm::device_uvector<vertex_t> component_local_sizes(
    0,
    handle.get_stream());  // size = 3 * num_unresolved_components (fwd_only, bwd_only, remaining)
  rmm::device_uvector<bool> component_global_size1_flags(
    0,
    handle.get_stream());  // size = 3 * num_unresolved_components (fwd_only, bwd_only, remaining)
  rmm::device_uvector<vertex_t> unique_fwd_only_size1_component_idxs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> unique_fwd_only_size1_component_sizes(0, handle.get_stream());
  rmm::device_uvector<vertex_t> unique_bwd_only_size1_component_idxs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> unique_bwd_only_size1_component_sizes(0, handle.get_stream());
  rmm::device_uvector<vertex_t> unique_remaining_size1_component_idxs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> unique_remaining_size1_component_sizes(0, handle.get_stream());
  rmm::device_uvector<vertex_t> unique_scc_component_idxs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> unique_scc_component_sizes(0, handle.get_stream());
  vertex_t num_fwd_only_size1_vertices{0};
  vertex_t num_bwd_only_size1_vertices{0};
  vertex_t num_remaining_size1_vertices{0};
  {
    // For each subset (fwd, bwd, remainder, scc), find the local min vertex id per component idx
    // on the current GPU and perform global reduction.

    auto find_min_vertex_id_per_component = [&handle](
                                              raft::device_span<vertex_t const> component_idxs,
                                              raft::device_span<vertex_t const> vertices) {
      rmm::device_uvector<vertex_t> unique_idxs(component_idxs.size(), handle.get_stream());
      rmm::device_uvector<vertex_t> min_vertex_ids(component_idxs.size(), handle.get_stream());

      auto [unique_end, min_end] = thrust::reduce_by_key(handle.get_thrust_policy(),
                                                         component_idxs.begin(),
                                                         component_idxs.end(),
                                                         vertices.begin(),
                                                         unique_idxs.begin(),
                                                         min_vertex_ids.begin(),
                                                         thrust::equal_to<vertex_t>{},
                                                         thrust::minimum<vertex_t>());

      unique_idxs.resize(cuda::std::distance(unique_idxs.begin(), unique_end), handle.get_stream());
      min_vertex_ids.resize(cuda::std::distance(min_vertex_ids.begin(), min_end),
                            handle.get_stream());
      unique_idxs.shrink_to_fit(handle.get_stream());
      min_vertex_ids.shrink_to_fit(handle.get_stream());

      return std::make_pair(std::move(unique_idxs), std::move(min_vertex_ids));
    };

    auto [unique_fwd_only_component_idxs, unique_fwd_only_component_min_vertex_ids] =
      find_min_vertex_id_per_component(
        raft::device_span<vertex_t const>(fwd_only_component_idxs.data(),
                                          fwd_only_component_idxs.size()),
        raft::device_span<vertex_t const>(fwd_only_vertices.data(), fwd_only_vertices.size()));
    auto [unique_bwd_only_component_idxs, unique_bwd_only_component_min_vertex_ids] =
      find_min_vertex_id_per_component(
        raft::device_span<vertex_t const>(bwd_only_component_idxs.data(),
                                          bwd_only_component_idxs.size()),
        raft::device_span<vertex_t const>(bwd_only_vertices.data(), bwd_only_vertices.size()));
    auto [unique_remaining_component_idxs, unique_remaining_component_min_vertex_ids] =
      find_min_vertex_id_per_component(
        raft::device_span<vertex_t const>(remaining_component_idxs.data(),
                                          remaining_component_idxs.size()),
        raft::device_span<vertex_t const>(remaining_vertices.data(), remaining_vertices.size()));
    rmm::device_uvector<vertex_t> unique_scc_component_min_vertex_ids(0, handle.get_stream());
    std::tie(unique_scc_component_idxs, unique_scc_component_min_vertex_ids) =
      find_min_vertex_id_per_component(
        raft::device_span<vertex_t const>(scc_component_idxs.data(), scc_component_idxs.size()),
        raft::device_span<vertex_t const>(scc_vertices.data(), scc_vertices.size()));

    // Build a flat array of 4 * num_unresolved_components to hold the min vertex id for each
    // (component, subset) pair [fwd_only_0, fwd_only_1, ..., bwd_only_0, bwd_only_1, ...,
    // remaining_0, remaining_1, ..., scc_0, scc_1, ...]

    component_global_min_vertex_ids.resize(4 * num_old_unresolved_components, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 component_global_min_vertex_ids.begin(),
                 component_global_min_vertex_ids.end(),
                 std::numeric_limits<vertex_t>::max());

    // Scatter each subset's local min vertex ids into the interleaved array.
    auto scatter_mins = [&handle,
                         component_min_vertex_ids =
                           raft::device_span<vertex_t>(component_global_min_vertex_ids.data(),
                                                       component_global_min_vertex_ids.size()),
                         num_old_unresolved_components](
                          raft::device_span<vertex_t const> unique_component_idxs,
                          raft::device_span<vertex_t const> min_vertex_ids,
                          vertex_t subset_offset) {
      thrust::scatter(handle.get_thrust_policy(),
                      min_vertex_ids.begin(),
                      min_vertex_ids.end(),
                      cuda::make_transform_iterator(
                        unique_component_idxs.begin(),
                        cuda::proclaim_return_type<vertex_t>(
                          [num_old_unresolved_components, subset_offset] __device__(vertex_t idx) {
                            return subset_offset * num_old_unresolved_components + idx;
                          })),
                      component_min_vertex_ids.begin());
    };

    scatter_mins(raft::device_span<vertex_t const>(unique_fwd_only_component_idxs.data(),
                                                   unique_fwd_only_component_idxs.size()),
                 raft::device_span<vertex_t const>(unique_fwd_only_component_min_vertex_ids.data(),
                                                   unique_fwd_only_component_min_vertex_ids.size()),
                 0);
    scatter_mins(raft::device_span<vertex_t const>(unique_bwd_only_component_idxs.data(),
                                                   unique_bwd_only_component_idxs.size()),
                 raft::device_span<vertex_t const>(unique_bwd_only_component_min_vertex_ids.data(),
                                                   unique_bwd_only_component_min_vertex_ids.size()),
                 1);
    scatter_mins(
      raft::device_span<vertex_t const>(unique_remaining_component_idxs.data(),
                                        unique_remaining_component_idxs.size()),
      raft::device_span<vertex_t const>(unique_remaining_component_min_vertex_ids.data(),
                                        unique_remaining_component_min_vertex_ids.size()),
      2);
    scatter_mins(raft::device_span<vertex_t const>(unique_scc_component_idxs.data(),
                                                   unique_scc_component_idxs.size()),
                 raft::device_span<vertex_t const>(unique_scc_component_min_vertex_ids.data(),
                                                   unique_scc_component_min_vertex_ids.size()),
                 3);

    // Multi-GPU: reduce across GPUs to get global min vertex id per sub-component.
    if constexpr (GraphViewType::is_multi_gpu) {
      device_allreduce(handle.get_comms(),
                       component_global_min_vertex_ids.begin(),
                       component_global_min_vertex_ids.begin(),
                       component_global_min_vertex_ids.size(),
                       raft::comms::op_t::MIN,
                       handle.get_stream());
    }

    // For fwd only, bwd only and remaining components, if global component size is 1
    // we can treat them as trivial SCCs.

    auto find_size_per_component = [&handle](
                                     raft::device_span<vertex_t const> component_idxs,
                                     raft::device_span<vertex_t const> unique_component_idxs) {
      rmm::device_uvector<vertex_t> component_lasts(unique_component_idxs.size(),
                                                    handle.get_stream());
      thrust::upper_bound(handle.get_thrust_policy(),
                          component_idxs.begin(),
                          component_idxs.end(),
                          unique_component_idxs.begin(),
                          unique_component_idxs.end(),
                          component_lasts.begin());
      rmm::device_uvector<vertex_t> component_sizes(unique_component_idxs.size(),
                                                    handle.get_stream());
      thrust::adjacent_difference(handle.get_thrust_policy(),
                                  component_lasts.begin(),
                                  component_lasts.end(),
                                  component_sizes.begin());
      return component_sizes;
    };

    auto unique_fwd_only_component_sizes = find_size_per_component(
      raft::device_span<vertex_t const>(fwd_only_component_idxs.data(),
                                        fwd_only_component_idxs.size()),
      raft::device_span<vertex_t const>(unique_fwd_only_component_idxs.data(),
                                        unique_fwd_only_component_idxs.size()));
    auto unique_bwd_only_component_sizes = find_size_per_component(
      raft::device_span<vertex_t const>(bwd_only_component_idxs.data(),
                                        bwd_only_component_idxs.size()),
      raft::device_span<vertex_t const>(unique_bwd_only_component_idxs.data(),
                                        unique_bwd_only_component_idxs.size()));
    auto unique_remaining_component_sizes = find_size_per_component(
      raft::device_span<vertex_t const>(remaining_component_idxs.data(),
                                        remaining_component_idxs.size()),
      raft::device_span<vertex_t const>(unique_remaining_component_idxs.data(),
                                        unique_remaining_component_idxs.size()));
    unique_scc_component_sizes = find_size_per_component(
      raft::device_span<vertex_t const>(scc_component_idxs.data(), scc_component_idxs.size()),
      raft::device_span<vertex_t const>(unique_scc_component_idxs.data(),
                                        unique_scc_component_idxs.size()));

    // [fwd_only_0, fwd_only_1, ..., bwd_only_0, bwd_only_1, ..., remaining_0, remaining_1, ...]
    component_local_sizes.resize(3 * num_old_unresolved_components, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 component_local_sizes.begin(),
                 component_local_sizes.end(),
                 vertex_t{0});

    auto scatter_sizes = [&handle,
                          component_sizes = raft::device_span<vertex_t>(
                            component_local_sizes.data(), component_local_sizes.size()),
                          num_old_unresolved_components](
                           raft::device_span<vertex_t const> unique_component_idxs,
                           raft::device_span<vertex_t const> unique_component_sizes,
                           vertex_t subset_offset) {
      thrust::scatter(handle.get_thrust_policy(),
                      unique_component_sizes.begin(),
                      unique_component_sizes.end(),
                      cuda::make_transform_iterator(
                        unique_component_idxs.begin(),
                        cuda::proclaim_return_type<vertex_t>(
                          [num_old_unresolved_components, subset_offset] __device__(vertex_t idx) {
                            return subset_offset * num_old_unresolved_components + idx;
                          })),
                      component_sizes.begin());
    };

    scatter_sizes(raft::device_span<vertex_t const>(unique_fwd_only_component_idxs.data(),
                                                    unique_fwd_only_component_idxs.size()),
                  raft::device_span<vertex_t const>(unique_fwd_only_component_sizes.data(),
                                                    unique_fwd_only_component_sizes.size()),
                  0);
    scatter_sizes(raft::device_span<vertex_t const>(unique_bwd_only_component_idxs.data(),
                                                    unique_bwd_only_component_idxs.size()),
                  raft::device_span<vertex_t const>(unique_bwd_only_component_sizes.data(),
                                                    unique_bwd_only_component_sizes.size()),
                  1);
    scatter_sizes(raft::device_span<vertex_t const>(unique_remaining_component_idxs.data(),
                                                    unique_remaining_component_idxs.size()),
                  raft::device_span<vertex_t const>(unique_remaining_component_sizes.data(),
                                                    unique_remaining_component_sizes.size()),
                  2);

    component_global_size1_flags.resize(component_local_sizes.size(), handle.get_stream());
    if constexpr (GraphViewType::is_multi_gpu) {
      rmm::device_uvector<vertex_t> component_global_sizes(component_local_sizes.size(),
                                                           handle.get_stream());
      device_allreduce(handle.get_comms(),
                       component_local_sizes.begin(),
                       component_global_sizes.begin(),
                       component_global_sizes.size(),
                       raft::comms::op_t::SUM,
                       handle.get_stream());
      thrust::transform(handle.get_thrust_policy(),
                        component_global_sizes.begin(),
                        component_global_sizes.end(),
                        component_global_size1_flags.begin(),
                        cuda::proclaim_return_type<bool>(
                          [] __device__(vertex_t size) { return size <= vertex_t{1}; }));
    } else {
      thrust::transform(handle.get_thrust_policy(),
                        component_local_sizes.begin(),
                        component_local_sizes.end(),
                        component_global_size1_flags.begin(),
                        cuda::proclaim_return_type<bool>(
                          [] __device__(vertex_t size) { return size <= vertex_t{1}; }));
    }

    // Partition fwd/bwd/remaining vertices: size == 1 left, unresolved (size > 1)
    // right. Size-0 sub-components have no vertices, so they are naturally absent.

    auto partition_by_size =
      [&handle,
       component_size1_flags = raft::device_span<bool const>(component_global_size1_flags.data(),
                                                             component_global_size1_flags.size()),
       num_old_unresolved_components](rmm::device_uvector<vertex_t>&& unique_component_idxs,
                                      rmm::device_uvector<vertex_t>&& unique_component_sizes,
                                      rmm::device_uvector<vertex_t>&& component_idxs,
                                      rmm::device_uvector<vertex_t>&& component_vertices,
                                      vertex_t subset_offset) {
        auto pair_first =
          thrust::make_zip_iterator(unique_component_idxs.begin(), unique_component_sizes.begin());
        auto partition_point = thrust::stable_partition(
          handle.get_thrust_policy(),
          pair_first,
          pair_first + unique_component_idxs.size(),
          [component_size1_flags, num_old_unresolved_components, subset_offset] __device__(
            auto pair) {
            auto idx = cuda::std::get<0>(pair);
            return component_size1_flags[subset_offset * num_old_unresolved_components + idx];
          });
        auto num_size1_components =
          static_cast<vertex_t>(cuda::std::distance(pair_first, partition_point));
        unique_component_idxs.resize(num_size1_components, handle.get_stream());
        unique_component_sizes.resize(num_size1_components, handle.get_stream());
        unique_component_idxs.shrink_to_fit(handle.get_stream());
        unique_component_sizes.shrink_to_fit(handle.get_stream());

        pair_first = thrust::make_zip_iterator(component_idxs.begin(), component_vertices.begin());
        partition_point = thrust::stable_partition(
          handle.get_thrust_policy(),
          pair_first,
          pair_first + component_vertices.size(),
          [component_size1_flags, num_old_unresolved_components, subset_offset] __device__(
            auto pair) {
            auto idx = cuda::std::get<0>(pair);
            return component_size1_flags[subset_offset * num_old_unresolved_components + idx];
          });
        auto num_size1_vertices =
          static_cast<vertex_t>(cuda::std::distance(pair_first, partition_point));
        return std::make_tuple(std::move(unique_component_idxs),
                               std::move(unique_component_sizes),
                               std::move(component_vertices),
                               num_size1_vertices);
      };

    std::tie(unique_fwd_only_size1_component_idxs,
             unique_fwd_only_size1_component_sizes,
             fwd_only_vertices,
             num_fwd_only_size1_vertices) =
      partition_by_size(std::move(unique_fwd_only_component_idxs),
                        std::move(unique_fwd_only_component_sizes),
                        std::move(fwd_only_component_idxs),
                        std::move(fwd_only_vertices),
                        0);
    std::tie(unique_bwd_only_size1_component_idxs,
             unique_bwd_only_size1_component_sizes,
             bwd_only_vertices,
             num_bwd_only_size1_vertices) =
      partition_by_size(std::move(unique_bwd_only_component_idxs),
                        std::move(unique_bwd_only_component_sizes),
                        std::move(bwd_only_component_idxs),
                        std::move(bwd_only_vertices),
                        1);
    std::tie(unique_remaining_size1_component_idxs,
             unique_remaining_size1_component_sizes,
             remaining_vertices,
             num_remaining_size1_vertices) =
      partition_by_size(std::move(unique_remaining_component_idxs),
                        std::move(unique_remaining_component_sizes),
                        std::move(remaining_component_idxs),
                        std::move(remaining_vertices),
                        2);
  }

  rmm::device_uvector<vertex_t> new_unresolved_component_ids(0, handle.get_stream());
  rmm::device_uvector<vertex_t> new_unresolved_component_offsets(0, handle.get_stream());
  rmm::device_uvector<vertex_t> new_unresolved_component_vertices(0, handle.get_stream());
  rmm::device_uvector<vertex_t> new_scc_component_ids(0, handle.get_stream());
  rmm::device_uvector<vertex_t> new_scc_component_offsets(0, handle.get_stream());
  rmm::device_uvector<vertex_t> new_scc_component_vertices(0, handle.get_stream());
  {
    auto concatenate3 = [&handle](raft::device_span<vertex_t const> src0,
                                  raft::device_span<vertex_t const> src1,
                                  raft::device_span<vertex_t const> src2) {
      rmm::device_uvector<vertex_t> dst(src0.size() + src1.size() + src2.size(),
                                        handle.get_stream());
      thrust::copy(handle.get_thrust_policy(), src0.begin(), src0.end(), dst.begin());
      thrust::copy(handle.get_thrust_policy(), src1.begin(), src1.end(), dst.begin() + src0.size());
      thrust::copy(handle.get_thrust_policy(),
                   src2.begin(),
                   src2.end(),
                   dst.begin() + src0.size() + src1.size());
      return dst;
    };

    new_unresolved_component_ids.resize(num_old_unresolved_components * 3, handle.get_stream());
    rmm::device_uvector<vertex_t> new_unresolved_component_counts(
      new_unresolved_component_ids.size(), handle.get_stream());
    auto input_pair_first  = thrust::make_zip_iterator(component_global_min_vertex_ids.begin(),
                                                      component_local_sizes.begin());
    auto output_pair_first = thrust::make_zip_iterator(new_unresolved_component_ids.begin(),
                                                       new_unresolved_component_counts.begin());
    new_unresolved_component_ids.resize(
      cuda::std::distance(output_pair_first,
                          thrust::copy_if(handle.get_thrust_policy(),
                                          input_pair_first,
                                          input_pair_first + num_old_unresolved_components * 3,
                                          component_global_size1_flags.begin(),
                                          output_pair_first,
                                          cuda::proclaim_return_type<bool>(
                                            [] __device__(auto flag) { return !flag; }))),
      handle.get_stream());
    new_unresolved_component_counts.resize(new_unresolved_component_ids.size(),
                                           handle.get_stream());
    new_unresolved_component_offsets.resize(new_unresolved_component_ids.size() + 1,
                                            handle.get_stream());
    new_unresolved_component_offsets.set_element_to_zero_async(0, handle.get_stream());
    thrust::inclusive_scan(handle.get_thrust_policy(),
                           new_unresolved_component_counts.begin(),
                           new_unresolved_component_counts.end(),
                           new_unresolved_component_offsets.begin() + 1);

    new_unresolved_component_vertices = concatenate3(
      raft::device_span<vertex_t const>(fwd_only_vertices.data() + num_fwd_only_size1_vertices,
                                        fwd_only_vertices.size() - num_fwd_only_size1_vertices),
      raft::device_span<vertex_t const>(bwd_only_vertices.data() + num_bwd_only_size1_vertices,
                                        bwd_only_vertices.size() - num_bwd_only_size1_vertices),
      raft::device_span<vertex_t const>(remaining_vertices.data() + num_remaining_size1_vertices,
                                        remaining_vertices.size() - num_remaining_size1_vertices));

    auto concatenate5 = [&handle](raft::device_span<vertex_t const> src0,
                                  raft::device_span<vertex_t const> src1,
                                  raft::device_span<vertex_t const> src2,
                                  raft::device_span<vertex_t const> src3,
                                  raft::device_span<vertex_t const> src4) {
      rmm::device_uvector<vertex_t> dst(
        src0.size() + src1.size() + src2.size() + src3.size() + src4.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(), src0.begin(), src0.end(), dst.begin());
      thrust::copy(handle.get_thrust_policy(), src1.begin(), src1.end(), dst.begin() + src0.size());
      thrust::copy(handle.get_thrust_policy(),
                   src2.begin(),
                   src2.end(),
                   dst.begin() + src0.size() + src1.size());
      thrust::copy(handle.get_thrust_policy(),
                   src3.begin(),
                   src3.end(),
                   dst.begin() + src0.size() + src1.size() + src2.size());
      thrust::copy(handle.get_thrust_policy(),
                   src4.begin(),
                   src4.end(),
                   dst.begin() + src0.size() + src1.size() + src2.size() + src3.size());
      return dst;
    };

    new_scc_component_ids.resize(
      unique_fwd_only_size1_component_idxs.size() + unique_bwd_only_size1_component_idxs.size() +
        unique_remaining_size1_component_idxs.size() + unique_scc_component_idxs.size() +
        trivial_singleton_scc_vertices.size(),
      handle.get_stream());
    thrust::gather(handle.get_thrust_policy(),
                   unique_fwd_only_size1_component_idxs.begin(),
                   unique_fwd_only_size1_component_idxs.end(),
                   component_global_min_vertex_ids.begin(),
                   new_scc_component_ids.begin());
    thrust::gather(handle.get_thrust_policy(),
                   unique_bwd_only_size1_component_idxs.begin(),
                   unique_bwd_only_size1_component_idxs.end(),
                   component_global_min_vertex_ids.begin() + num_old_unresolved_components,
                   new_scc_component_ids.begin() + unique_fwd_only_size1_component_idxs.size());
    thrust::gather(handle.get_thrust_policy(),
                   unique_remaining_size1_component_idxs.begin(),
                   unique_remaining_size1_component_idxs.end(),
                   component_global_min_vertex_ids.begin() + num_old_unresolved_components * 2,
                   new_scc_component_ids.begin() + unique_fwd_only_size1_component_idxs.size() +
                     unique_bwd_only_size1_component_idxs.size());
    thrust::gather(handle.get_thrust_policy(),
                   unique_scc_component_idxs.begin(),
                   unique_scc_component_idxs.end(),
                   component_global_min_vertex_ids.begin() + num_old_unresolved_components * 3,
                   new_scc_component_ids.begin() + unique_fwd_only_size1_component_idxs.size() +
                     unique_bwd_only_size1_component_idxs.size() +
                     unique_remaining_size1_component_idxs.size());
    thrust::copy(handle.get_thrust_policy(),
                 trivial_singleton_scc_vertices.begin(),
                 trivial_singleton_scc_vertices.end(),
                 new_scc_component_ids.begin() + unique_fwd_only_size1_component_idxs.size() +
                   unique_bwd_only_size1_component_idxs.size() +
                   unique_remaining_size1_component_idxs.size() + unique_scc_component_idxs.size());
    rmm::device_uvector<vertex_t> ones(trivial_singleton_scc_vertices.size(), handle.get_stream());
    thrust::fill(handle.get_thrust_policy(), ones.begin(), ones.end(), vertex_t{1});
    auto new_scc_component_counts =
      concatenate5(raft::device_span<vertex_t const>(unique_fwd_only_size1_component_sizes.data(),
                                                     unique_fwd_only_size1_component_sizes.size()),
                   raft::device_span<vertex_t const>(unique_bwd_only_size1_component_sizes.data(),
                                                     unique_bwd_only_size1_component_sizes.size()),
                   raft::device_span<vertex_t const>(unique_remaining_size1_component_sizes.data(),
                                                     unique_remaining_size1_component_sizes.size()),
                   raft::device_span<vertex_t const>(unique_scc_component_sizes.data(),
                                                     unique_scc_component_sizes.size()),
                   raft::device_span<vertex_t const>(ones.data(), ones.size()));
    new_scc_component_vertices = concatenate5(
      raft::device_span<vertex_t const>(fwd_only_vertices.data(), num_fwd_only_size1_vertices),
      raft::device_span<vertex_t const>(bwd_only_vertices.data(), num_bwd_only_size1_vertices),
      raft::device_span<vertex_t const>(remaining_vertices.data(), num_remaining_size1_vertices),
      raft::device_span<vertex_t const>(scc_vertices.data(), scc_vertices.size()),
      raft::device_span<vertex_t const>(trivial_singleton_scc_vertices.data(),
                                        trivial_singleton_scc_vertices.size()));
    new_scc_component_offsets.resize(new_scc_component_counts.size() + 1, handle.get_stream());
    new_scc_component_offsets.set_element_to_zero_async(0, handle.get_stream());
    thrust::inclusive_scan(handle.get_thrust_policy(),
                           new_scc_component_counts.begin(),
                           new_scc_component_counts.end(),
                           new_scc_component_offsets.begin() + 1);
  }

  return std::make_tuple(std::move(new_unresolved_component_ids),
                         std::move(new_unresolved_component_offsets),
                         std::move(new_unresolved_component_vertices),
                         std::move(new_scc_component_ids),
                         std::move(new_scc_component_offsets),
                         std::move(new_scc_component_vertices));
}

// return component_ids, component_offsets, component_vertices, num_unresolved_components
template <typename GraphViewType, typename KVStoreViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>>
forward_backward_intersect(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  GraphViewType const& inverse_graph_view,
  KVStoreViewType const&
    to_inverse_renumber_map /* graph_view vertex ID => local inverse_graph_view vertex ID */,
  raft::device_span<typename GraphViewType::vertex_type const>
    from_inverse_renumber_map /* local inverse_graph_view vertex ID => graph_view vertex ID */,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& unresolved_component_offsets,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& unresolved_component_vertices)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  constexpr bool multi_gpu = GraphViewType::is_multi_gpu;

  auto in_degrees = inverse_graph_view.compute_out_degrees(handle);
  {
    auto tmp_degrees = rmm::device_uvector<edge_t>(graph_view.local_vertex_partition_range_size(),
                                                   handle.get_stream());
    if constexpr (multi_gpu) {
      auto tmp_vertices =
        rmm::device_uvector<vertex_t>(from_inverse_renumber_map.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   from_inverse_renumber_map.begin(),
                   from_inverse_renumber_map.end(),
                   tmp_vertices.begin());
      std::vector<arithmetic_device_uvector_t> vertex_properties{};
      vertex_properties.push_back(std::move(in_degrees));
      std::tie(tmp_vertices, vertex_properties) =
        shuffle_int_vertices(handle,
                             std::move(tmp_vertices),
                             std::move(vertex_properties),
                             graph_view.vertex_partition_range_lasts());
      in_degrees = std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
      thrust::scatter(
        handle.get_thrust_policy(),
        in_degrees.begin(),
        in_degrees.end(),
        cuda::make_transform_iterator(
          tmp_vertices.begin(),
          detail::shift_left_t<vertex_t>{graph_view.local_vertex_partition_range_first()}),
        tmp_degrees.begin());
    } else {
      thrust::scatter(handle.get_thrust_policy(),
                      in_degrees.begin(),
                      in_degrees.end(),
                      from_inverse_renumber_map.begin(),
                      tmp_degrees.begin());
    }
    in_degrees = std::move(tmp_degrees);
  }
  auto out_degrees = graph_view.compute_out_degrees(handle);

  rmm::device_uvector<edge_t> unresolved_component_vertex_in_degrees(
    unresolved_component_vertices.size(), handle.get_stream());
  rmm::device_uvector<edge_t> unresolved_component_vertex_out_degrees(
    unresolved_component_vertices.size(), handle.get_stream());
  {
    auto map_first = cuda::make_transform_iterator(
      unresolved_component_vertices.begin(),
      detail::shift_left_t<vertex_t>{graph_view.local_vertex_partition_range_first()});
    thrust::gather(handle.get_thrust_policy(),
                   map_first,
                   map_first + unresolved_component_vertices.size(),
                   in_degrees.begin(),
                   unresolved_component_vertex_in_degrees.begin());
    thrust::gather(handle.get_thrust_policy(),
                   map_first,
                   map_first + unresolved_component_vertices.size(),
                   out_degrees.begin(),
                   unresolved_component_vertex_out_degrees.begin());
  }

  auto trivial_singleton_scc_vertices = find_trivial_singleton_scc_vertices(
    handle,
    graph_view,
    inverse_graph_view,
    to_inverse_renumber_map,
    from_inverse_renumber_map,
    raft::device_span<vertex_t const>(unresolved_component_vertices.data(),
                                      unresolved_component_vertices.size()),
    std::move(in_degrees),
    std::move(out_degrees));

  auto [pivots, pivot_unresolved_component_idxs] =
    find_pivots(handle,
                graph_view,
                raft::device_span<vertex_t const>(unresolved_component_offsets.data(),
                                                  unresolved_component_offsets.size()),
                raft::device_span<vertex_t const>(unresolved_component_vertices.data(),
                                                  unresolved_component_vertices.size()),
                raft::device_span<edge_t const>(unresolved_component_vertex_in_degrees.data(),
                                                unresolved_component_vertex_in_degrees.size()),
                raft::device_span<edge_t const>(unresolved_component_vertex_out_degrees.data(),
                                                unresolved_component_vertex_out_degrees.size()),
                raft::device_span<vertex_t const>(trivial_singleton_scc_vertices.data(),
                                                  trivial_singleton_scc_vertices.size()));
  unresolved_component_vertex_in_degrees.resize(0, handle.get_stream());
  unresolved_component_vertex_in_degrees.shrink_to_fit(handle.get_stream());
  unresolved_component_vertex_out_degrees.resize(0, handle.get_stream());
  unresolved_component_vertex_out_degrees.shrink_to_fit(handle.get_stream());

  auto num_aggregate_pivots = pivots.size();
  if constexpr (GraphViewType::is_multi_gpu) {
    num_aggregate_pivots = host_scalar_allreduce(
      handle.get_comms(), num_aggregate_pivots, raft::comms::op_t::SUM, handle.get_stream());
  }

  rmm::device_uvector<vertex_t> forward_set_offsets(0, handle.get_stream());
  rmm::device_uvector<vertex_t> forward_set_vertices(0, handle.get_stream());
  if (num_aggregate_pivots > 0) {
    std::tie(forward_set_offsets, forward_set_vertices) =
      reachable_sets(handle,
                     graph_view,
                     raft::device_span<vertex_t const>(pivots.data(), pivots.size()),
                     raft::device_span<vertex_t const>(pivot_unresolved_component_idxs.data(),
                                                       pivot_unresolved_component_idxs.size()),
                     static_cast<vertex_t>(unresolved_component_offsets.size() - 1));
  } else {
    forward_set_offsets.resize(unresolved_component_offsets.size() - 1, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 forward_set_offsets.begin(),
                 forward_set_offsets.end(),
                 vertex_t{0});
  }

  rmm::device_uvector<vertex_t> backward_set_offsets(0, handle.get_stream());
  rmm::device_uvector<vertex_t> backward_set_vertices(0, handle.get_stream());
  if (num_aggregate_pivots > 0) {
    if constexpr (multi_gpu) {
      std::vector<arithmetic_device_uvector_t> vertex_properties{};
      vertex_properties.push_back(std::move(pivot_unresolved_component_idxs));
      std::tie(pivots, vertex_properties) =
        shuffle_ext_vertices(handle, std::move(pivots), std::move(vertex_properties));
      pivot_unresolved_component_idxs =
        std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
    }
    to_inverse_renumber_map.find(
      pivots.begin(),
      pivots.end(),
      pivots.begin(),
      handle.get_stream()); /* functionally identical to renumber_local_ext_vertices but to avoid
                               repetitively rebuilding a kv_store_t object for the entire local
                               vertex partition range */
    thrust::sort_by_key(handle.get_thrust_policy(),
                        pivots.begin(),
                        pivots.end(),
                        pivot_unresolved_component_idxs.begin());

    std::tie(backward_set_offsets, backward_set_vertices) =
      reachable_sets(handle,
                     inverse_graph_view,
                     raft::device_span<vertex_t const>(pivots.data(), pivots.size()),
                     raft::device_span<vertex_t const>(pivot_unresolved_component_idxs.data(),
                                                       pivot_unresolved_component_idxs.size()),
                     static_cast<vertex_t>(unresolved_component_offsets.size() - 1));

    unrenumber_local_int_vertices(handle,
                                  backward_set_vertices.data(),
                                  backward_set_vertices.size(),
                                  from_inverse_renumber_map.data(),
                                  inverse_graph_view.local_vertex_partition_range_first(),
                                  inverse_graph_view.local_vertex_partition_range_last());
    rmm::device_uvector<vertex_t> tmp_unresolved_component_idxs(backward_set_vertices.size(),
                                                                handle.get_stream());
    auto component_idx_first = cuda::make_transform_iterator(
      thrust::make_counting_iterator(vertex_t{0}),
      detail::segment_id_t<vertex_t>{raft::device_span<vertex_t const>(
        backward_set_offsets.data() + 1, backward_set_offsets.size() - 1)});
    thrust::copy(handle.get_thrust_policy(),
                 component_idx_first,
                 component_idx_first + backward_set_vertices.size(),
                 tmp_unresolved_component_idxs.begin());
    if constexpr (multi_gpu) {
      std::vector<arithmetic_device_uvector_t> vertex_properties{};
      vertex_properties.push_back(std::move(tmp_unresolved_component_idxs));
      std::tie(backward_set_vertices, vertex_properties) =
        shuffle_int_vertices(handle,
                             std::move(backward_set_vertices),
                             std::move(vertex_properties),
                             graph_view.vertex_partition_range_lasts());
      tmp_unresolved_component_idxs =
        std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
      auto pair_first = thrust::make_zip_iterator(tmp_unresolved_component_idxs.begin(),
                                                  backward_set_vertices.begin());
      thrust::sort(
        handle.get_thrust_policy(), pair_first, pair_first + tmp_unresolved_component_idxs.size());
      backward_set_offsets.set_element_to_zero_async(0, handle.get_stream());
      thrust::upper_bound(
        handle.get_thrust_policy(),
        tmp_unresolved_component_idxs.begin(),
        tmp_unresolved_component_idxs.end(),
        thrust::make_counting_iterator(vertex_t{0}),
        thrust::make_counting_iterator(static_cast<vertex_t>(backward_set_offsets.size() - 1)),
        backward_set_offsets.begin() + 1);
    } else {
      auto pair_first = thrust::make_zip_iterator(tmp_unresolved_component_idxs.begin(),
                                                  backward_set_vertices.begin());
      thrust::sort(
        handle.get_thrust_policy(), pair_first, pair_first + tmp_unresolved_component_idxs.size());
    }
  } else {
    backward_set_offsets.resize(unresolved_component_offsets.size() - 1, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 backward_set_offsets.begin(),
                 backward_set_offsets.end(),
                 vertex_t{0});
  }

  return intersect_reachable_sets<GraphViewType>(handle,
                                                 std::move(unresolved_component_offsets),
                                                 std::move(unresolved_component_vertices),
                                                 std::move(forward_set_offsets),
                                                 std::move(forward_set_vertices),
                                                 std::move(backward_set_offsets),
                                                 std::move(backward_set_vertices),
                                                 std::move(trivial_singleton_scc_vertices));
}

template <typename GraphViewType>
void strongly_connected_components_impl(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  raft::device_span<typename GraphViewType::vertex_type> components,
  bool do_expensive_check)
{
  using vertex_t                  = typename GraphViewType::vertex_type;
  using edge_t                    = typename GraphViewType::edge_type;
  using weight_t                  = float;    // dummy
  using edge_type_t               = int32_t;  // dummy
  constexpr bool store_transposed = GraphViewType::is_storage_transposed;
  constexpr bool multi_gpu        = GraphViewType::is_multi_gpu;

  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

  auto const num_vertices = graph_view.number_of_vertices();
  if (num_vertices == 0) { return; }

  // 1. check input arguments

  CUGRAPH_EXPECTS(
    !graph_view.is_symmetric(),
    "Invalid input argument>: call weakly_connnected_components instead for symmetric graphs.");

  if (do_expensive_check) {
    // nothing to do
  }

  // 2. initialize component IDs (initially, every vertex belongs to one unresolved component)

  thrust::fill(handle.get_thrust_policy(), components.begin(), components.end(), vertex_t{0});

  // 3. create an edge mask and mask out self-loops & multi-edges (except for the first one); this
  // edge mask will be used to mask out edges between different components

  auto forward_graph_view = graph_view;
  edge_property_t<edge_t, bool> edge_mask(handle, forward_graph_view);
  {
    if (forward_graph_view.has_edge_mask()) { forward_graph_view.clear_edge_mask(); }
    cugraph::fill_edge_property(handle, forward_graph_view, edge_mask.mutable_view(), false);
    if (graph_view.is_multigraph()) {
      edge_multi_index_property_t<edge_t, vertex_t> edge_multi_indices(handle, graph_view);
      transform_e(handle,
                  graph_view,
                  edge_src_dummy_property_t{}.view(),
                  edge_dst_dummy_property_t{}.view(),
                  edge_multi_indices.view(),
                  cuda::proclaim_return_type<bool>(
                    [] __device__(auto src, auto dst, auto, auto, auto multi_edge_index) {
                      return (src != dst) && (multi_edge_index == 0);
                    }),
                  edge_mask.mutable_view());
    } else {
      transform_e(handle,
                  graph_view,
                  edge_src_dummy_property_t{}.view(),
                  edge_dst_dummy_property_t{}.view(),
                  edge_dummy_property_t{}.view(),
                  cuda::proclaim_return_type<bool>(
                    [] __device__(auto src, auto dst, auto, auto, auto) { return (src != dst); }),
                  edge_mask.mutable_view());
    }
    forward_graph_view.attach_edge_mask(edge_mask.view());
  }

  // 4. create an inverse graph

  graph_t<vertex_t, edge_t, store_transposed, multi_gpu> inverse_graph(handle);
  rmm::device_uvector<vertex_t> from_inverse_renumber_map(0, handle.get_stream());
  {
    rmm::device_uvector<vertex_t> vertices(graph_view.local_vertex_partition_range_size(),
                                           handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(),
                     vertices.begin(),
                     vertices.end(),
                     graph_view.local_vertex_partition_range_first());
    if constexpr (multi_gpu) {
      std::tie(vertices, std::ignore) = shuffle_ext_vertices(
        handle, std::move(vertices), std::vector<cugraph::arithmetic_device_uvector_t>{});
    }

    rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
    std::tie(edgelist_srcs, edgelist_dsts, std::ignore, std::ignore, std::ignore) =
      decompress_to_edgelist<vertex_t, edge_t, weight_t, edge_type_t, store_transposed, multi_gpu>(
        handle, forward_graph_view, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    std::swap(edgelist_srcs, edgelist_dsts);
    if constexpr (multi_gpu) {
      std::tie(edgelist_srcs, edgelist_dsts, std::ignore) =
        shuffle_ext_edges(handle,
                          std::move(edgelist_srcs),
                          std::move(edgelist_dsts),
                          std::vector<cugraph::arithmetic_device_uvector_t>{},
                          store_transposed);
    }
    std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};
    std::tie(inverse_graph, std::ignore, tmp_renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, store_transposed, multi_gpu>(
        handle,
        std::move(vertices),
        std::move(edgelist_srcs),
        std::move(edgelist_dsts),
        std::vector<cugraph::arithmetic_device_uvector_t>{},
        graph_properties_t{false, false},
        true);
    from_inverse_renumber_map = std::move(*tmp_renumber_map);
  }
  auto inverse_graph_view = inverse_graph.view();
  kv_store_t<vertex_t, vertex_t, false> to_inverse_renumber_map(
    from_inverse_renumber_map.begin(),
    from_inverse_renumber_map.end(),
    thrust::make_counting_iterator(inverse_graph_view.local_vertex_partition_range_first()),
    invalid_vertex_id_v<vertex_t>,
    invalid_vertex_id_v<vertex_t>,
    handle.get_stream());
  auto to_inverse_renumber_map_view = to_inverse_renumber_map.view();

  edge_property_t<edge_t, bool> inverse_edge_mask(handle, inverse_graph_view);
  cugraph::fill_edge_property(handle, inverse_graph_view, inverse_edge_mask.mutable_view(), true);
  inverse_graph_view.attach_edge_mask(inverse_edge_mask.view());

  // 5. prepare for recursvie forward-backward SCC: set unresolved_component_offsets and
  // unresolved_component_vertices and initialzie edge_src|dst_components

  rmm::device_uvector<vertex_t> unresolved_component_offsets(2, handle.get_stream());
  rmm::device_uvector<vertex_t> unresolved_component_vertices(
    forward_graph_view.local_vertex_partition_range_size(), handle.get_stream());
  unresolved_component_offsets.set_element_to_zero_async(0, handle.get_stream());
  unresolved_component_offsets.set_element(
    1, forward_graph_view.local_vertex_partition_range_size(), handle.get_stream());
  thrust::sequence(handle.get_thrust_policy(),
                   unresolved_component_vertices.begin(),
                   unresolved_component_vertices.end(),
                   forward_graph_view.local_vertex_partition_range_first());

  auto edge_src_components = multi_gpu
                               ? edge_src_property_t<vertex_t, vertex_t>(handle, forward_graph_view)
                               : edge_src_property_t<vertex_t, vertex_t>(handle);
  auto edge_dst_components = multi_gpu
                               ? edge_dst_property_t<vertex_t, vertex_t>(handle, forward_graph_view)
                               : edge_dst_property_t<vertex_t, vertex_t>(handle);
  auto inverse_edge_src_components =
    multi_gpu ? edge_src_property_t<vertex_t, vertex_t>(handle, inverse_graph_view)
              : edge_src_property_t<vertex_t, vertex_t>(handle);
  auto inverse_edge_dst_components =
    multi_gpu ? edge_dst_property_t<vertex_t, vertex_t>(handle, inverse_graph_view)
              : edge_dst_property_t<vertex_t, vertex_t>(handle);
  if constexpr (multi_gpu) {
    fill_edge_src_property(
      handle, forward_graph_view, edge_src_components.mutable_view(), vertex_t{0});
    fill_edge_dst_property(
      handle, forward_graph_view, edge_dst_components.mutable_view(), vertex_t{0});
    fill_edge_src_property(
      handle, inverse_graph_view, inverse_edge_src_components.mutable_view(), vertex_t{0});
    fill_edge_dst_property(
      handle, inverse_graph_view, inverse_edge_dst_components.mutable_view(), vertex_t{0});
  }

  // 6. recursive forward-backward SCC

  while ((unresolved_component_offsets.size() - 1) > 0) {
    // 6-1. perform forward-backward SCC

    rmm::device_uvector<vertex_t> unresolved_component_ids(0, handle.get_stream());
    rmm::device_uvector<vertex_t> scc_component_ids(0, handle.get_stream());
    rmm::device_uvector<vertex_t> scc_component_offsets(0, handle.get_stream());
    rmm::device_uvector<vertex_t> scc_component_vertices(0, handle.get_stream());
    std::tie(unresolved_component_ids,
             unresolved_component_offsets,
             unresolved_component_vertices,
             scc_component_ids,
             scc_component_offsets,
             scc_component_vertices) =
      forward_backward_intersect(
        handle,
        forward_graph_view,
        inverse_graph_view,
        to_inverse_renumber_map_view,
        raft::device_span<vertex_t const>(from_inverse_renumber_map.data(),
                                          from_inverse_renumber_map.size()),
        std::move(unresolved_component_offsets),
        std::move(unresolved_component_vertices));

    // 6-2. update components

    auto scatter_component_ids =
      [&handle,
       components = raft::device_span<vertex_t>(components.data(), components.size()),
       v_first    = forward_graph_view.local_vertex_partition_range_first()](
        raft::device_span<vertex_t const> component_offsets,
        raft::device_span<vertex_t const> component_ids,
        raft::device_span<vertex_t const> component_vertices) {
        auto component_id_first = cuda::make_transform_iterator(
          thrust::make_counting_iterator(vertex_t{0}),
          cuda::proclaim_return_type<vertex_t>(
            [component_offsets, component_ids] __device__(vertex_t i) {
              auto idx = cuda::std::distance(
                component_offsets.begin() + 1,
                thrust::upper_bound(
                  thrust::seq, component_offsets.begin() + 1, component_offsets.end(), i));
              return component_ids[idx];
            }));
        auto map_first = cuda::make_transform_iterator(component_vertices.begin(),
                                                       detail::shift_left_t<vertex_t>{v_first});
        thrust::scatter(handle.get_thrust_policy(),
                        component_id_first,
                        component_id_first + component_vertices.size(),
                        map_first,
                        components.begin());
      };

    scatter_component_ids(raft::device_span<vertex_t const>(unresolved_component_offsets.data(),
                                                            unresolved_component_offsets.size()),
                          raft::device_span<vertex_t const>(unresolved_component_ids.data(),
                                                            unresolved_component_ids.size()),
                          raft::device_span<vertex_t const>(unresolved_component_vertices.data(),
                                                            unresolved_component_vertices.size()));
    unresolved_component_ids.resize(0, handle.get_stream());
    unresolved_component_ids.shrink_to_fit(handle.get_stream());

    scatter_component_ids(
      raft::device_span<vertex_t const>(scc_component_offsets.data(), scc_component_offsets.size()),
      raft::device_span<vertex_t const>(scc_component_ids.data(), scc_component_ids.size()),
      raft::device_span<vertex_t const>(scc_component_vertices.data(),
                                        scc_component_vertices.size()));
    scc_component_ids.resize(0, handle.get_stream());
    scc_component_offsets.resize(0, handle.get_stream());
    scc_component_vertices.resize(0, handle.get_stream());
    scc_component_ids.shrink_to_fit(handle.get_stream());
    scc_component_offsets.shrink_to_fit(handle.get_stream());
    scc_component_vertices.shrink_to_fit(handle.get_stream());

    // 6-3. mask out edges between different components

    if constexpr (multi_gpu) {
      rmm::device_uvector<vertex_t> tmp_vertices(
        unresolved_component_vertices.size() + scc_component_vertices.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   unresolved_component_vertices.begin(),
                   unresolved_component_vertices.end(),
                   tmp_vertices.begin());
      thrust::copy(handle.get_thrust_policy(),
                   scc_component_vertices.begin(),
                   scc_component_vertices.end(),
                   tmp_vertices.begin() + unresolved_component_vertices.size());
      thrust::sort(handle.get_thrust_policy(), tmp_vertices.begin(), tmp_vertices.end());
      rmm::device_uvector<vertex_t> tmp_components(tmp_vertices.size(), handle.get_stream());
      auto map_first = cuda::make_transform_iterator(
        tmp_vertices.begin(),
        detail::shift_left_t<vertex_t>{forward_graph_view.local_vertex_partition_range_first()});
      thrust::gather(handle.get_thrust_policy(),
                     map_first,
                     map_first + tmp_vertices.size(),
                     components.begin(),
                     tmp_components.begin());

      update_edge_src_property(handle,
                               forward_graph_view,
                               tmp_vertices.begin(),
                               tmp_vertices.end(),
                               tmp_components.begin(),
                               edge_src_components.mutable_view());
      update_edge_dst_property(handle,
                               forward_graph_view,
                               tmp_vertices.begin(),
                               tmp_vertices.end(),
                               tmp_components.begin(),
                               edge_dst_components.mutable_view());

      std::vector<arithmetic_device_uvector_t> vertex_properties{};
      vertex_properties.push_back(arithmetic_device_uvector_t{std::move(tmp_components)});
      std::tie(tmp_vertices, vertex_properties) =
        shuffle_ext_vertices(handle, std::move(tmp_vertices), std::move(vertex_properties));
      tmp_components = std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
      to_inverse_renumber_map_view.find(
        tmp_vertices.begin(),
        tmp_vertices.end(),
        tmp_vertices.begin(),
        handle.get_stream()); /* functionally identical to renumber_local_ext_vertices but to avoid
                                 repetitively rebuilding a kv_store_t object for the entire local
                                 vertex partition range */
      thrust::sort_by_key(handle.get_thrust_policy(),
                          tmp_vertices.begin(),
                          tmp_vertices.end(),
                          tmp_components.begin());

      update_edge_src_property(handle,
                               inverse_graph_view,
                               tmp_vertices.begin(),
                               tmp_vertices.end(),
                               tmp_components.begin(),
                               inverse_edge_src_components.mutable_view());
      update_edge_dst_property(handle,
                               inverse_graph_view,
                               tmp_vertices.begin(),
                               tmp_vertices.end(),
                               tmp_components.begin(),
                               inverse_edge_dst_components.mutable_view());
    }

    auto edge_src_component_view = multi_gpu
                                     ? edge_src_components.view()
                                     : make_edge_src_property_view<vertex_t, vertex_t>(
                                         forward_graph_view,
                                         components.begin(),
                                         forward_graph_view.local_vertex_partition_range_size());
    auto edge_dst_component_view = multi_gpu
                                     ? edge_dst_components.view()
                                     : make_edge_dst_property_view<vertex_t, vertex_t>(
                                         forward_graph_view,
                                         components.begin(),
                                         forward_graph_view.local_vertex_partition_range_size());

    auto new_edge_mask = make_initialized_edge_property(handle, forward_graph_view, false);
    transform_e(handle,
                forward_graph_view,
                edge_src_component_view,
                edge_dst_component_view,
                edge_dummy_property_t{}.view(),
                cuda::proclaim_return_type<bool>(
                  [] __device__(auto, auto, auto src_component, auto dst_component, auto) {
                    return src_component == dst_component;
                  }),
                new_edge_mask.mutable_view());
    if (forward_graph_view.has_edge_mask()) { forward_graph_view.clear_edge_mask(); }
    edge_mask = std::move(new_edge_mask);
    forward_graph_view.attach_edge_mask(edge_mask.view());

    {
      std::optional<rmm::device_uvector<vertex_t>> inverse_components{std::nullopt};
      if constexpr (!multi_gpu) {
        inverse_components = rmm::device_uvector<vertex_t>(
          inverse_graph_view.local_vertex_partition_range_size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       from_inverse_renumber_map.begin(),
                       from_inverse_renumber_map.end(),
                       components.begin(),
                       inverse_components->begin());
      }

      auto inverse_edge_src_component_view =
        multi_gpu ? inverse_edge_src_components.view()
                  : make_edge_src_property_view<vertex_t, vertex_t>(
                      inverse_graph_view,
                      inverse_components->begin(),
                      inverse_graph_view.local_vertex_partition_range_size());
      auto inverse_edge_dst_component_view =
        multi_gpu ? inverse_edge_dst_components.view()
                  : make_edge_dst_property_view<vertex_t, vertex_t>(
                      inverse_graph_view,
                      inverse_components->begin(),
                      inverse_graph_view.local_vertex_partition_range_size());

      auto new_inverse_edge_mask =
        make_initialized_edge_property(handle, inverse_graph_view, false);
      transform_e(handle,
                  inverse_graph_view,
                  inverse_edge_src_component_view,
                  inverse_edge_dst_component_view,
                  edge_dummy_property_t{}.view(),
                  cuda::proclaim_return_type<bool>(
                    [] __device__(auto, auto, auto src_component, auto dst_component, auto) {
                      return src_component == dst_component;
                    }),
                  new_inverse_edge_mask.mutable_view());
      if (inverse_graph_view.has_edge_mask()) { inverse_graph_view.clear_edge_mask(); }
      inverse_edge_mask = std::move(new_inverse_edge_mask);
      inverse_graph_view.attach_edge_mask(inverse_edge_mask.view());
    }
  }

  return;
}

}  // namespace

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> strongly_connected_components(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  bool do_expensive_check)
{
  rmm::device_uvector<vertex_t> components(graph_view.local_vertex_partition_range_size(),
                                           handle.get_stream());
  strongly_connected_components_impl(
    handle,
    graph_view,
    raft::device_span<vertex_t>(components.data(), components.size()),
    do_expensive_check);

  return std::move(components);
}

}  // namespace cugraph

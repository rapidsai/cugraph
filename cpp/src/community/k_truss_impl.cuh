/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/device_comm_wrapper.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/prims/edge_bucket.cuh>
#include <cugraph/prims/extract_transform_if_e.cuh>
#include <cugraph/prims/fill_edge_property.cuh>
#include <cugraph/prims/make_initialized_edge_property.cuh>
#include <cugraph/prims/per_v_pair_dst_nbr_intersection.cuh>
#include <cugraph/prims/per_v_pair_dst_nbr_intersection_for_each.cuh>
#include <cugraph/prims/transform_e.cuh>
#include <cugraph/prims/transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/util/integer_utils.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

namespace cugraph {

template <typename edge_t>
struct is_k_or_greater_t {
  edge_t k{};
  __device__ bool operator()(edge_t core_number) const { return core_number >= edge_t{k}; }
};

template <typename vertex_t, typename edge_t>
struct extract_triangles_endpoints {
  size_t chunk_start{};
  raft::device_span<size_t const> intersection_offsets{};
  raft::device_span<vertex_t const> intersection_indices{};
  raft::device_span<vertex_t const> weak_srcs{};
  raft::device_span<vertex_t const> weak_dsts{};

  __device__ cuda::std::tuple<vertex_t, vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto itr = thrust::upper_bound(
      thrust::seq, intersection_offsets.begin() + 1, intersection_offsets.end(), i);
    auto idx = cuda::std::distance(intersection_offsets.begin() + 1, itr);

    auto endpoints = cuda::std::make_tuple(weak_srcs[chunk_start + idx],  // p
                                           weak_dsts[chunk_start + idx],  // q
                                           intersection_indices[i]        // r
    );

    auto p = weak_srcs[chunk_start + idx];
    auto q = weak_dsts[chunk_start + idx];
    auto r = intersection_indices[i];
    // Re-order the endpoints such that p < q < r in order to identify duplicate triangles
    // which will cause overcompensation. comparing the vertex IDs is cheaper than comparing the
    // degrees (d(p) < d(q) < d(r)) which will be done once in the latter stage to retrieve the
    // direction of the edges once the triplet dependency is broken.
    if (p > q) cuda::std::swap(p, q);
    if (p > r) cuda::std::swap(p, r);
    if (q > r) cuda::std::swap(q, r);

    return cuda::std::make_tuple(p, q, r);
  }
};

namespace {

template <typename vertex_t>
struct exclude_self_loop_t {
  __device__ cuda::std::optional<cuda::std::tuple<vertex_t, vertex_t>> operator()(
    vertex_t src,
    vertex_t dst,
    cuda::std::nullopt_t,
    cuda::std::nullopt_t,
    cuda::std::nullopt_t) const
  {
    return src != dst
             ? cuda::std::optional<cuda::std::tuple<vertex_t, vertex_t>>{cuda::std::make_tuple(src,
                                                                                               dst)}
             : cuda::std::nullopt;
  }
};

template <typename vertex_t, typename edge_t>
struct extract_low_to_high_degree_edges_from_endpoints_e_op_t {
  raft::device_span<vertex_t const> srcs{};
  raft::device_span<vertex_t const> dsts{};
  raft::device_span<edge_t const> count{};
  __device__ cuda::std::tuple<vertex_t, vertex_t, edge_t> operator()(vertex_t src,
                                                                     vertex_t dst,
                                                                     edge_t src_out_degree,
                                                                     edge_t dst_out_degree,
                                                                     cuda::std::nullopt_t) const
  {
    auto itr = thrust::lower_bound(thrust::seq,
                                   thrust::make_zip_iterator(srcs.begin(), dsts.begin()),
                                   thrust::make_zip_iterator(srcs.end(), dsts.end()),
                                   cuda::std::make_tuple(src, dst));

    auto idx = cuda::std::distance(thrust::make_zip_iterator(srcs.begin(), dsts.begin()), itr);

    if (src_out_degree < dst_out_degree) {
      return cuda::std::make_tuple(src, dst, count[idx]);
    } else if (dst_out_degree < src_out_degree) {
      return cuda::std::make_tuple(dst, src, count[idx]);
    } else {  // src_out_degree == dst_out_degree
      if (src < dst /* tie-breaking using vertex ID */) {
        return cuda::std::make_tuple(src, dst, count[idx]);
      } else {
        return cuda::std::make_tuple(dst, src, count[idx]);
      }
    }
  }
};

template <typename vertex_t, typename edge_t>
struct extract_low_to_high_degree_edges_from_endpoints_pred_op_t {
  raft::device_span<vertex_t const> srcs{};
  raft::device_span<vertex_t const> dsts{};
  __device__ bool operator()(vertex_t src, vertex_t dst, edge_t, edge_t, cuda::std::nullopt_t) const
  {
    return thrust::binary_search(thrust::seq,
                                 thrust::make_zip_iterator(srcs.begin(), dsts.begin()),
                                 thrust::make_zip_iterator(srcs.end(), dsts.end()),
                                 cuda::std::make_tuple(src, dst));
  }
};

template <typename vertex_t, typename edge_t>
struct decrement_edge_triangle_count_t {
  raft::device_span<vertex_t const> edge_srcs{};
  raft::device_span<vertex_t const> edge_dsts{};
  raft::device_span<edge_t const> decrease_counts{};

  __device__ edge_t operator()(
    vertex_t src, vertex_t dst, cuda::std::nullopt_t, cuda::std::nullopt_t, edge_t count) const
  {
    auto edge_first = thrust::make_zip_iterator(edge_srcs.begin(), edge_dsts.begin());
    auto edge_last  = thrust::make_zip_iterator(edge_srcs.end(), edge_dsts.end());
    auto itr_pair =
      thrust::lower_bound(thrust::seq, edge_first, edge_last, cuda::std::make_tuple(src, dst));
    auto idx_pair = cuda::std::distance(edge_first, itr_pair);
    return count - decrease_counts[idx_pair];
  }
};

// Sets the bit corresponding to each weak edge (p, q) at its DODG offset in
// @p bitmask.
template <typename vertex_t, typename edge_t>
struct set_weak_bitmask_t {
  vertex_t const* srcs;
  vertex_t const* dsts;
  edge_t const* offsets_ptr;
  vertex_t const* indices_ptr;
  uint32_t* bitmask;

  __device__ void operator()(size_t i) const
  {
    auto p     = srcs[i];
    auto q     = dsts[i];
    auto start = offsets_ptr[p];
    auto end   = offsets_ptr[p + 1];
    auto pos   = thrust::lower_bound(thrust::seq, indices_ptr + start, indices_ptr + end, q);
    if (pos != indices_ptr + end && *pos == q) {
      auto eid = static_cast<edge_t>(pos - indices_ptr);
      atomicOr(&bitmask[packed_bool_offset(eid)], packed_bool_mask(eid));
    }
  }
};

// For each triangle (p, q, r) discovered from a weak edge (p, q):
//   1. Orient (p, r) and (q, r) to their DODG direction, using O(log D)
//      binary search when the orientation is reversed.
//   2. Test weakness of (p, r) and (q, r) via the packed weak bitmask.
//   3. Apply a lexicographic ownership rule on (min, max) endpoints so each
//      triangle is decremented by exactly one weak edge.
//   4. If (p, q) owns the triangle, atomically subtract one from counts at
//      the DODG offsets of all three sides.
template <typename vertex_t, typename edge_t>
struct decrement_weak_selective_t {
  edge_t*         counts;
  uint32_t const* weak_bitmask;
  edge_t   const* offsets_ptr;
  vertex_t const* indices_ptr;
  edge_t   const* out_degrees;

  __device__ edge_t find_dodg_offset(vertex_t u, vertex_t v) const
  {
    if (out_degrees[u] > out_degrees[v] ||
        (out_degrees[u] == out_degrees[v] && u > v)) {
      auto tmp = u; u = v; v = tmp;
    }
    auto start = offsets_ptr[u];
    auto end   = offsets_ptr[u + 1];
    auto pos   = thrust::lower_bound(thrust::seq, indices_ptr + start, indices_ptr + end, v);
    return static_cast<edge_t>(pos - indices_ptr);
  }

  __device__ bool is_weak(edge_t dodg_off) const
  {
    return (weak_bitmask[packed_bool_offset(dodg_off)] & packed_bool_mask(dodg_off)) != 0u;
  }

  __device__ void operator()(vertex_t p, vertex_t q, vertex_t r,
                             edge_t pq_off, edge_t pr_off, edge_t qr_off) const
  {
    edge_t dodg_pr = (out_degrees[p] < out_degrees[r] ||
                      (out_degrees[p] == out_degrees[r] && p < r))
                       ? pr_off
                       : find_dodg_offset(r, p);
    edge_t dodg_qr = (out_degrees[q] < out_degrees[r] ||
                      (out_degrees[q] == out_degrees[r] && q < r))
                       ? qr_off
                       : find_dodg_offset(r, q);

    bool pr_weak = is_weak(dodg_pr);
    bool qr_weak = is_weak(dodg_qr);

    auto pq_lo = min(p, q);
    auto pq_hi = max(p, q);
    if (pr_weak) {
      auto pr_lo = min(p, r);
      auto pr_hi = max(p, r);
      if (pr_lo < pq_lo || (pr_lo == pq_lo && pr_hi < pq_hi)) return;
    }
    if (qr_weak) {
      auto qr_lo = min(q, r);
      auto qr_hi = max(q, r);
      if (qr_lo < pq_lo || (qr_lo == pq_lo && qr_hi < pq_hi)) return;
    }

    atomicAdd(&counts[pq_off],  edge_t{-1});
    atomicAdd(&counts[dodg_pr], edge_t{-1});
    atomicAdd(&counts[dodg_qr], edge_t{-1});
  }
};

// For each edge offset @p e in the DODG, append (src, dst) to the weak edge
// list when count(e) < k - 2 and count(e) != 0.  When @p update_masks is
// true, also clear dodg_mask + weak_edges_mask (both directions) in the same
// pass.
template <typename vertex_t, typename edge_t, bool update_masks>
struct extract_weak_and_update_masks_t {
  edge_t   const* offsets_ptr;
  vertex_t const* indices_ptr;
  edge_t   const* counts_ptr;
  uint32_t*       dodg_mask_ptr;
  uint32_t*       weak_mask_ptr;
  edge_t          k;
  vertex_t*       out_srcs;
  vertex_t*       out_dsts;
  size_t*         out_counter;
  vertex_t        num_vertices;

  __device__ void operator()(edge_t e) const
  {
    if (!(dodg_mask_ptr[packed_bool_offset(e)] & packed_bool_mask(e))) return;

    auto c = counts_ptr[e];
    if (c >= k - 2 || c == 0) return;

    edge_t lo = 0, hi = static_cast<edge_t>(num_vertices);
    while (lo < hi) {
      auto mid = lo + (hi - lo) / 2;
      if (offsets_ptr[mid + 1] <= e) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    vertex_t src = static_cast<vertex_t>(lo);
    vertex_t dst = indices_ptr[e];

    if constexpr (update_masks) {
      atomicAnd(&dodg_mask_ptr[packed_bool_offset(e)], ~packed_bool_mask(e));
      atomicAnd(&weak_mask_ptr[packed_bool_offset(e)], ~packed_bool_mask(e));

      auto rev_start = offsets_ptr[dst];
      auto rev_end   = offsets_ptr[dst + 1];
      auto pos = thrust::lower_bound(
        thrust::seq, indices_ptr + rev_start, indices_ptr + rev_end, src);
      if (pos != indices_ptr + rev_end && *pos == src) {
        auto rev_e = static_cast<edge_t>(pos - indices_ptr);
        atomicAnd(&weak_mask_ptr[packed_bool_offset(rev_e)], ~packed_bool_mask(rev_e));
      }
    }

    auto idx = atomicAdd(out_counter, size_t{1});
    out_srcs[idx] = src;
    out_dsts[idx] = dst;
  }
};

// For each triangle (p, q, r), atomically increment the triangle count of
// all three edges at their CSR offsets.
template <typename vertex_t, typename edge_t>
struct increment_triangle_counts_t {
  edge_t* counts;
  __device__ void operator()(vertex_t, vertex_t, vertex_t,
                             edge_t pq_offset, edge_t pr_offset, edge_t qr_offset) const
  {
    atomicAdd(&counts[pq_offset], edge_t{1});
    atomicAdd(&counts[pr_offset], edge_t{1});
    atomicAdd(&counts[qr_offset], edge_t{1});
  }
};

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
k_truss(raft::handle_t const& handle,
        graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
        std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
        edge_t k,
        bool do_expensive_check)
{
  // 1. Check input arguments.

  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "Invalid input arguments: K-truss currently supports undirected graphs only.");
  CUGRAPH_EXPECTS(!graph_view.is_multigraph(),
                  "Invalid input arguments: K-truss currently does not support multi-graphs.");

  if (do_expensive_check) {
    // nothing to do
  }

  // 2. Exclude self-loops and edges that do not belong to (k-1)-core

  auto cur_graph_view = graph_view;

  // mask for self-loops and edges not part of k-1 core
  cugraph::edge_property_t<edge_t, bool> undirected_mask(handle);
  {
    // 2.1 Exclude self-loops

    if (cur_graph_view.count_self_loops(handle) > edge_t{0}) {
      // 2.1. Exclude self-loops

      auto self_loop_edge_mask = make_initialized_edge_property(handle, cur_graph_view, false);

      transform_e(handle,
                  cur_graph_view,
                  edge_src_dummy_property_t{}.view(),
                  edge_dst_dummy_property_t{}.view(),
                  edge_dummy_property_t{}.view(),
                  cuda::proclaim_return_type<bool>(
                    [] __device__(auto src, auto dst, auto, auto, auto) { return src != dst; }),
                  self_loop_edge_mask.mutable_view());

      undirected_mask = std::move(self_loop_edge_mask);
      if (cur_graph_view.has_edge_mask()) { cur_graph_view.clear_edge_mask(); }
      cur_graph_view.attach_edge_mask(undirected_mask.view());
    }

    // 2.2 Find (k-1)-core and exclude edges that do not belong to (k-1)-core
    {
      rmm::device_uvector<edge_t> core_numbers(cur_graph_view.number_of_vertices(),
                                               handle.get_stream());
      core_number(handle,
                  cur_graph_view,
                  core_numbers.data(),
                  k_core_degree_type_t::OUT,
                  size_t{2},
                  size_t{2});

      edge_src_property_t<vertex_t, bool> edge_src_in_k_minus_1_cores(handle, cur_graph_view);
      edge_dst_property_t<vertex_t, bool> edge_dst_in_k_minus_1_cores(handle, cur_graph_view);
      auto in_k_minus_1_core_first =
        cuda::make_transform_iterator(core_numbers.begin(), is_k_or_greater_t<edge_t>{k - 1});
      rmm::device_uvector<bool> in_k_minus_1_core_flags(core_numbers.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   in_k_minus_1_core_first,
                   in_k_minus_1_core_first + core_numbers.size(),
                   in_k_minus_1_core_flags.begin());
      update_edge_src_property(handle,
                               cur_graph_view,
                               in_k_minus_1_core_flags.begin(),
                               edge_src_in_k_minus_1_cores.mutable_view());
      update_edge_dst_property(handle,
                               cur_graph_view,
                               in_k_minus_1_core_flags.begin(),
                               edge_dst_in_k_minus_1_cores.mutable_view());

      auto in_k_minus_1_core_edge_mask =
        make_initialized_edge_property(handle, cur_graph_view, false);

      transform_e(
        handle,
        cur_graph_view,
        edge_src_in_k_minus_1_cores.view(),
        edge_dst_in_k_minus_1_cores.view(),
        edge_dummy_property_t{}.view(),
        cuda::proclaim_return_type<bool>(
          [] __device__(auto, auto, auto src_in_k_minus_1_core, auto dst_in_k_minus_1_core, auto) {
            return src_in_k_minus_1_core && dst_in_k_minus_1_core;
          }),
        in_k_minus_1_core_edge_mask.mutable_view());

      undirected_mask = std::move(in_k_minus_1_core_edge_mask);
      if (cur_graph_view.has_edge_mask()) { cur_graph_view.clear_edge_mask(); }
      cur_graph_view.attach_edge_mask(undirected_mask.view());
    }
  }

  // 3. Keep only the edges from a low-degree vertex to a high-degree vertex.

  edge_src_property_t<vertex_t, edge_t> edge_src_out_degrees(handle, cur_graph_view);
  edge_dst_property_t<vertex_t, edge_t> edge_dst_out_degrees(handle, cur_graph_view);

  // Per-vertex out-degree, computed before the DODG mask is attached and
  // reused in section 4 to determine DODG orientation.
  auto out_degrees = cur_graph_view.compute_out_degrees(handle);

  auto dodg_mask = make_initialized_edge_property(handle, cur_graph_view, false);
  {
    update_edge_src_property(
      handle, cur_graph_view, out_degrees.begin(), edge_src_out_degrees.mutable_view());
    update_edge_dst_property(
      handle, cur_graph_view, out_degrees.begin(), edge_dst_out_degrees.mutable_view());

    cugraph::transform_e(
      handle,
      cur_graph_view,
      edge_src_out_degrees.view(),
      edge_dst_out_degrees.view(),
      edge_dummy_property_t{}.view(),
      cuda::proclaim_return_type<bool>(
        [] __device__(auto src, auto dst, auto src_out_degree, auto dst_out_degree, auto) {
          return (src_out_degree < dst_out_degree) ? true
                 : ((src_out_degree == dst_out_degree) &&
                    (src < dst) /* tie-breaking using vertex ID */)
                   ? true
                   : false;
        }),
      dodg_mask.mutable_view(),
      do_expensive_check);

    if (cur_graph_view.has_edge_mask()) { cur_graph_view.clear_edge_mask(); }
    cur_graph_view.attach_edge_mask(dodg_mask.view());
  }

  // 4. Compute triangle count using nbr_intersection and unroll weak edges

  {
    // Mask self loops and edges not being part of k-1 core
    auto weak_edges_mask = std::move(undirected_mask);

    auto edge_triangle_counts =
      edge_triangle_count<vertex_t, edge_t, multi_gpu>(handle, cur_graph_view, false);

    cugraph::edge_bucket_t<vertex_t, edge_t, true, multi_gpu, true> edgelist_weak(
      handle, false /* multigraph */);
    cugraph::edge_bucket_t<vertex_t, edge_t, true, multi_gpu, true> edges_to_decrement_count(
      handle, false /* multigraph */);
    size_t prev_chunk_size = 0;  // FIXME: Add support for chunking
    int peel_iter = 0;

    while (true) {
      rmm::device_uvector<vertex_t> weak_edgelist_srcs(0, handle.get_stream());
      rmm::device_uvector<vertex_t> weak_edgelist_dsts(0, handle.get_stream());

      // Compute edge count BEFORE extraction (SG iter 0 fused extract modifies
      // the mask in place, so capturing this afterwards would invalidate the
      // convergence check).
      cur_graph_view.clear_edge_mask();
      cur_graph_view.attach_edge_mask(weak_edges_mask.view());
      auto prev_number_of_edges = cur_graph_view.compute_number_of_edges(handle);

      // Extract weak edges
      if constexpr (multi_gpu) {
        std::tie(weak_edgelist_srcs, weak_edgelist_dsts) = extract_transform_if_e(
          handle,
          cur_graph_view,
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          edge_triangle_counts.view(),
          cuda::proclaim_return_type<cuda::std::tuple<vertex_t, vertex_t>>(
            [] __device__(vertex_t src, vertex_t dst, auto, auto, auto) {
              return cuda::std::make_tuple(src, dst);
            }),
          cuda::proclaim_return_type<bool>([k] __device__(auto, auto, auto, auto, edge_t count) {
            return ((count < k - 2) && (count != 0));
          }));
      } else {
        auto edge_partition  = cur_graph_view.local_edge_partition_view(size_t{0});
        auto offsets_ptr     = edge_partition.offsets().data();
        auto indices_ptr     = edge_partition.indices().data();
        auto num_edges_total = edge_partition.number_of_edges();
        auto num_vertices    = cur_graph_view.local_vertex_partition_range_size();

        weak_edgelist_srcs.resize(num_edges_total, handle.get_stream());
        weak_edgelist_dsts.resize(num_edges_total, handle.get_stream());
        rmm::device_scalar<size_t> weak_count(size_t{0}, handle.get_stream());

        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(edge_t{0}),
          thrust::make_counting_iterator(static_cast<edge_t>(num_edges_total)),
          extract_weak_and_update_masks_t<vertex_t, edge_t, false>{
            offsets_ptr,
            indices_ptr,
            edge_triangle_counts.view().value_firsts()[0],
            dodg_mask.mutable_view().value_firsts()[0],
            weak_edges_mask.mutable_view().value_firsts()[0],
            k,
            weak_edgelist_srcs.data(),
            weak_edgelist_dsts.data(),
            weak_count.data(),
            static_cast<vertex_t>(num_vertices)});

        size_t num_weak = weak_count.value(handle.get_stream());
        weak_edgelist_srcs.resize(num_weak, handle.get_stream());
        weak_edgelist_dsts.resize(num_weak, handle.get_stream());
      }

      auto weak_edgelist_first =
        thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin());
      auto weak_edgelist_last =
        thrust::make_zip_iterator(weak_edgelist_srcs.end(), weak_edgelist_dsts.end());

      thrust::sort(handle.get_thrust_policy(), weak_edgelist_first, weak_edgelist_last);

      // Perform nbr_intersection of the weak edges from the undirected
      // graph view
      cur_graph_view.clear_edge_mask();

      // Attach the weak edge mask
      cur_graph_view.attach_edge_mask(weak_edges_mask.view());

      if constexpr (multi_gpu) {
      auto [intersection_offsets, intersection_indices] = per_v_pair_dst_nbr_intersection(
        handle, cur_graph_view, weak_edgelist_first, weak_edgelist_last, do_expensive_check);

      // This array stores (p, q, r) which are endpoints for the triangles with weak edges

      auto triangles_endpoints =
        allocate_dataframe_buffer<cuda::std::tuple<vertex_t, vertex_t, vertex_t>>(
          intersection_indices.size(), handle.get_stream());

      // Extract endpoints for triangles with weak edges
      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(triangles_endpoints),
        get_dataframe_buffer_end(triangles_endpoints),
        extract_triangles_endpoints<vertex_t, edge_t>{
          prev_chunk_size,
          raft::device_span<size_t const>(intersection_offsets.data(), intersection_offsets.size()),
          raft::device_span<vertex_t const>(intersection_indices.data(),
                                            intersection_indices.size()),
          raft::device_span<vertex_t const>(weak_edgelist_srcs.data(), weak_edgelist_srcs.size()),
          raft::device_span<vertex_t const>(weak_edgelist_dsts.data(), weak_edgelist_dsts.size())});

      thrust::sort(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(triangles_endpoints),
                   get_dataframe_buffer_end(triangles_endpoints));

      auto unique_triangle_end = thrust::unique(handle.get_thrust_policy(),
                                                get_dataframe_buffer_begin(triangles_endpoints),
                                                get_dataframe_buffer_end(triangles_endpoints));

      auto num_unique_triangles =
        cuda::std::distance(  // Triangles are represented by their endpoints
          get_dataframe_buffer_begin(triangles_endpoints),
          unique_triangle_end);

      resize_dataframe_buffer(triangles_endpoints, num_unique_triangles, handle.get_stream());

      if constexpr (multi_gpu) {
        auto& comm           = handle.get_comms();
        auto const comm_size = comm.get_size();
        auto& major_comm     = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
        auto const major_comm_size = major_comm.get_size();
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_size = minor_comm.get_size();

        auto vertex_partition_range_lasts = cur_graph_view.vertex_partition_range_lasts();

        {
          std::vector<cugraph::arithmetic_device_uvector_t> edge_properties{};
          edge_properties.push_back(std::move(std::get<2>(triangles_endpoints)));
          std::tie(
            std::get<0>(triangles_endpoints), std::get<1>(triangles_endpoints), edge_properties) =
            shuffle_int_edges(handle,
                              std::move(std::get<0>(triangles_endpoints)),
                              std::move(std::get<1>(triangles_endpoints)),
                              std::move(edge_properties),
                              false /* store_transposed */,
                              vertex_partition_range_lasts);
          std::get<2>(triangles_endpoints) =
            std::move(std::get<rmm::device_uvector<vertex_t>>(edge_properties[0]));
        }

        thrust::sort(handle.get_thrust_policy(),
                     get_dataframe_buffer_begin(triangles_endpoints),
                     get_dataframe_buffer_end(triangles_endpoints));

        unique_triangle_end = thrust::unique(handle.get_thrust_policy(),
                                             get_dataframe_buffer_begin(triangles_endpoints),
                                             get_dataframe_buffer_end(triangles_endpoints));

        num_unique_triangles =
          cuda::std::distance(get_dataframe_buffer_begin(triangles_endpoints), unique_triangle_end);
        resize_dataframe_buffer(triangles_endpoints, num_unique_triangles, handle.get_stream());
      }

      auto edgelist_to_update_count =
        allocate_dataframe_buffer<cuda::std::tuple<vertex_t, vertex_t>>(3 * num_unique_triangles,
                                                                        handle.get_stream());

      // The order no longer matters since duplicated triangles have been removed
      // Flatten the endpoints to a list of egdes.
      thrust::transform(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(size_dataframe_buffer(edgelist_to_update_count)),
        get_dataframe_buffer_begin(edgelist_to_update_count),
        cuda::proclaim_return_type<cuda::std::tuple<vertex_t, vertex_t>>(
          [num_unique_triangles,
           triangles_endpoints =
             get_dataframe_buffer_begin(triangles_endpoints)] __device__(auto idx) {
            auto idx_triangle           = idx % num_unique_triangles;
            auto idx_vertex_in_triangle = idx / num_unique_triangles;
            auto triangle               = (triangles_endpoints + idx_triangle).get_iterator_tuple();
            vertex_t src;
            vertex_t dst;

            if (idx_vertex_in_triangle == 0) {
              src = *(cuda::std::get<0>(triangle));
              dst = *(cuda::std::get<1>(triangle));
            }

            if (idx_vertex_in_triangle == 1) {
              src = *(cuda::std::get<0>(triangle));
              dst = *(cuda::std::get<2>(triangle));
            }

            if (idx_vertex_in_triangle == 2) {
              src = *(cuda::std::get<1>(triangle));
              dst = *(cuda::std::get<2>(triangle));
            }

            return cuda::std::make_tuple(src, dst);
          }));

      if constexpr (multi_gpu) {
        std::vector<cugraph::arithmetic_device_uvector_t> edge_properties{};

        std::tie(std::get<0>(edgelist_to_update_count),
                 std::get<1>(edgelist_to_update_count),
                 std::ignore) = shuffle_int_edges(handle,
                                                  std::move(std::get<0>(edgelist_to_update_count)),
                                                  std::move(std::get<1>(edgelist_to_update_count)),
                                                  std::move(edge_properties),
                                                  false,
                                                  cur_graph_view.vertex_partition_range_lasts());
      }

      thrust::sort(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(edgelist_to_update_count),
                   get_dataframe_buffer_end(edgelist_to_update_count));

      auto unique_pair_count =
        thrust::unique_count(handle.get_thrust_policy(),
                             get_dataframe_buffer_begin(edgelist_to_update_count),
                             get_dataframe_buffer_end(edgelist_to_update_count));

      auto vertex_pair_buffer_unique =
        allocate_dataframe_buffer<cuda::std::tuple<vertex_t, vertex_t>>(unique_pair_count,
                                                                        handle.get_stream());

      rmm::device_uvector<edge_t> decrease_count(unique_pair_count, handle.get_stream());

      // FIXME: thrust::reduce_by_key produces corrupted output when compiled with
      // nvcc 13.0 + CCCL 3.4.0 on Blackwell (sm_120). The bug does not occur in nvcc 13.1+.
      // See https://github.com/rapidsai/cugraph/issues/5494
#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && \
  (__CUDACC_VER_MAJOR__ == 13) && (__CUDACC_VER_MINOR__ == 0)
      thrust::unique_copy(handle.get_thrust_policy(),
                          get_dataframe_buffer_begin(edgelist_to_update_count),
                          get_dataframe_buffer_end(edgelist_to_update_count),
                          get_dataframe_buffer_begin(vertex_pair_buffer_unique));

      thrust::transform(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(vertex_pair_buffer_unique),
        get_dataframe_buffer_begin(vertex_pair_buffer_unique) + unique_pair_count,
        decrease_count.begin(),
        cuda::proclaim_return_type<edge_t>(
          [edgelist_first = get_dataframe_buffer_begin(edgelist_to_update_count),
           edgelist_last =
             get_dataframe_buffer_end(edgelist_to_update_count)] __device__(auto pair) {
            return static_cast<edge_t>(cuda::std::distance(
              thrust::lower_bound(thrust::seq, edgelist_first, edgelist_last, pair),
              thrust::upper_bound(thrust::seq, edgelist_first, edgelist_last, pair)));
          }));
#else
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            get_dataframe_buffer_begin(edgelist_to_update_count),
                            get_dataframe_buffer_end(edgelist_to_update_count),
                            cuda::make_constant_iterator(size_t{1}),
                            get_dataframe_buffer_begin(vertex_pair_buffer_unique),
                            decrease_count.begin(),
                            cuda::std::equal_to<cuda::std::tuple<vertex_t, vertex_t>>{});
#endif

      std::tie(std::get<0>(vertex_pair_buffer_unique),
               std::get<1>(vertex_pair_buffer_unique),
               decrease_count) =
        extract_transform_if_e(
          handle,
          cur_graph_view,
          edge_src_out_degrees.view(),
          edge_dst_out_degrees.view(),
          edge_dummy_property_t{}.view(),
          extract_low_to_high_degree_edges_from_endpoints_e_op_t<vertex_t, edge_t>{
            raft::device_span<vertex_t const>(std::get<0>(vertex_pair_buffer_unique).data(),
                                              std::get<0>(vertex_pair_buffer_unique).size()),
            raft::device_span<vertex_t const>(std::get<1>(vertex_pair_buffer_unique).data(),
                                              std::get<1>(vertex_pair_buffer_unique).size()),
            raft::device_span<edge_t const>(decrease_count.data(), decrease_count.size())},
          extract_low_to_high_degree_edges_from_endpoints_pred_op_t<vertex_t, edge_t>{
            raft::device_span<vertex_t const>(std::get<0>(vertex_pair_buffer_unique).data(),
                                              std::get<0>(vertex_pair_buffer_unique).size()),
            raft::device_span<vertex_t const>(std::get<1>(vertex_pair_buffer_unique).data(),
                                              std::get<1>(vertex_pair_buffer_unique).size())});

      if constexpr (multi_gpu) {
        auto& comm           = handle.get_comms();
        auto const comm_size = comm.get_size();
        auto& major_comm     = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
        auto const major_comm_size = major_comm.get_size();
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_size        = minor_comm.get_size();
        auto vertex_partition_range_lasts = cur_graph_view.vertex_partition_range_lasts();

        std::vector<cugraph::arithmetic_device_uvector_t> edge_properties{};
        edge_properties.push_back(std::move(decrease_count));
        std::tie(std::get<0>(vertex_pair_buffer_unique),
                 std::get<1>(vertex_pair_buffer_unique),
                 edge_properties) =
          shuffle_int_edges(handle,
                            std::move(std::get<0>(vertex_pair_buffer_unique)),
                            std::move(std::get<1>(vertex_pair_buffer_unique)),
                            std::move(edge_properties),
                            false /* store_transposed */,
                            vertex_partition_range_lasts);
        decrease_count = std::move(std::get<rmm::device_uvector<edge_t>>(edge_properties[0]));
      }

      thrust::sort_by_key(handle.get_thrust_policy(),
                          get_dataframe_buffer_begin(vertex_pair_buffer_unique),
                          get_dataframe_buffer_end(vertex_pair_buffer_unique),
                          decrease_count.begin());

      // Update count of weak edges
      edges_to_decrement_count.clear();

      edges_to_decrement_count.insert(std::get<0>(vertex_pair_buffer_unique).begin(),
                                      std::get<0>(vertex_pair_buffer_unique).end(),
                                      std::get<1>(vertex_pair_buffer_unique).begin(),
                                      std::optional<edge_t const*>{std::nullopt});

      cur_graph_view.clear_edge_mask();
      // Check for edge existance on the directed graph view
      cur_graph_view.attach_edge_mask(dodg_mask.view());

      // Update count of weak edges from the DODG view
      cugraph::transform_e(
        handle,
        cur_graph_view,
        edges_to_decrement_count,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        edge_triangle_counts.view(),
        decrement_edge_triangle_count_t<vertex_t, edge_t>{
          raft::device_span<vertex_t const>(std::get<0>(vertex_pair_buffer_unique).data(),
                                            std::get<0>(vertex_pair_buffer_unique).size()),
          raft::device_span<vertex_t const>(std::get<1>(vertex_pair_buffer_unique).data(),
                                            std::get<1>(vertex_pair_buffer_unique).size()),
          raft::device_span<edge_t const>(decrease_count.data(), decrease_count.size())},
        edge_triangle_counts.mutable_view(),
        do_expensive_check);
      } else {
        // Decrement triangle counts via neighbor intersection.
        auto edge_partition = cur_graph_view.local_edge_partition_view(size_t{0});
        auto offsets_ptr    = edge_partition.offsets().data();
        auto indices_ptr    = edge_partition.indices().data();

        // Build a weak-edge bitmask indexed by DODG offset.
        auto weak_bitmask = make_initialized_edge_property(handle, cur_graph_view, false);

        thrust::for_each(handle.get_thrust_policy(),
                         thrust::make_counting_iterator<size_t>(0),
                         thrust::make_counting_iterator<size_t>(weak_edgelist_srcs.size()),
                         set_weak_bitmask_t<vertex_t, edge_t>{
                           weak_edgelist_srcs.data(),
                           weak_edgelist_dsts.data(),
                           offsets_ptr,
                           indices_ptr,
                           weak_bitmask.mutable_view().value_firsts()[0]});

        per_v_pair_dst_nbr_intersection_for_each(
          handle,
          cur_graph_view,
          weak_edgelist_first,
          weak_edgelist_last,
          decrement_weak_selective_t<vertex_t, edge_t>{
            edge_triangle_counts.mutable_view().value_firsts()[0],
            weak_bitmask.view().value_firsts()[0],
            offsets_ptr,
            indices_ptr,
            out_degrees.data()},
          do_expensive_check);
      }

      edgelist_weak.clear();

      thrust::sort(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(weak_edgelist_srcs.begin(), weak_edgelist_dsts.begin()),
        thrust::make_zip_iterator(weak_edgelist_srcs.end(), weak_edgelist_dsts.end()));

      edgelist_weak.insert(weak_edgelist_srcs.begin(),
                           weak_edgelist_srcs.end(),
                           weak_edgelist_dsts.begin(),
                           std::optional<edge_t const*>{std::nullopt});

      // Get undirected graph view
      cur_graph_view.clear_edge_mask();
      cur_graph_view.attach_edge_mask(weak_edges_mask.view());

      cugraph::transform_e(
        handle,
        cur_graph_view,
        edgelist_weak,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        cuda::proclaim_return_type<bool>(
          [] __device__(
            auto src, auto dst, cuda::std::nullopt_t, cuda::std::nullopt_t, cuda::std::nullopt_t) {
            return false;
          }),
        weak_edges_mask.mutable_view(),
        do_expensive_check);

      edgelist_weak.clear();

      // shuffle the edges if multi_gpu
      if constexpr (multi_gpu) {
        std::vector<cugraph::arithmetic_device_uvector_t> edge_properties{};

        std::tie(weak_edgelist_dsts, weak_edgelist_srcs, std::ignore) =
          shuffle_int_edges(handle,
                            std::move(weak_edgelist_dsts),
                            std::move(weak_edgelist_srcs),
                            std::move(edge_properties),
                            false,
                            cur_graph_view.vertex_partition_range_lasts());
      }

      thrust::sort(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(weak_edgelist_dsts.begin(), weak_edgelist_srcs.begin()),
        thrust::make_zip_iterator(weak_edgelist_dsts.end(), weak_edgelist_srcs.end()));

      edgelist_weak.insert(weak_edgelist_dsts.begin(),
                           weak_edgelist_dsts.end(),
                           weak_edgelist_srcs.begin(),
                           std::optional<edge_t const*>{std::nullopt});

      cugraph::transform_e(
        handle,
        cur_graph_view,
        edgelist_weak,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        cuda::proclaim_return_type<bool>(
          [] __device__(
            auto src, auto dst, cuda::std::nullopt_t, cuda::std::nullopt_t, cuda::std::nullopt_t) {
            return false;
          }),
        weak_edges_mask.mutable_view(),
        do_expensive_check);

      cur_graph_view.attach_edge_mask(weak_edges_mask.view());

      if (prev_number_of_edges == cur_graph_view.compute_number_of_edges(handle)) { break; }

      cur_graph_view.clear_edge_mask();
      cur_graph_view.attach_edge_mask(dodg_mask.view());
    }

    cur_graph_view.clear_edge_mask();
    cur_graph_view.attach_edge_mask(dodg_mask.view());

    cugraph::transform_e(
      handle,
      cur_graph_view,
      cugraph::edge_src_dummy_property_t{}.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      edge_triangle_counts.view(),
      cuda::proclaim_return_type<bool>(
        [] __device__(auto src, auto dst, cuda::std::nullopt_t, cuda::std::nullopt_t, auto count) {
          return count == 0 ? false : true;
        }),
      dodg_mask.mutable_view(),
      do_expensive_check);

    rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> edgelist_wgts{std::nullopt};

    std::tie(edgelist_srcs, edgelist_dsts, edgelist_wgts, std::ignore, std::ignore) =
      decompress_to_edgelist(
        handle,
        cur_graph_view,
        edge_weight_view,
        std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
        std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
        std::optional<raft::device_span<vertex_t const>>{std::nullopt});

    std::tie(edgelist_srcs,
             edgelist_dsts,
             edgelist_wgts,
             std::ignore,
             std::ignore,
             std::ignore,
             std::ignore) =
      symmetrize_edgelist<vertex_t, edge_t, weight_t, int32_t, int32_t, false, multi_gpu>(
        handle,
        std::move(edgelist_srcs),
        std::move(edgelist_dsts),
        std::move(edgelist_wgts),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        false);

    return std::make_tuple(
      std::move(edgelist_srcs), std::move(edgelist_dsts), std::move(edgelist_wgts));
  }
}
}  // namespace cugraph

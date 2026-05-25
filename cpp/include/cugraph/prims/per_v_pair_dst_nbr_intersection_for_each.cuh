/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "prims/detail/nbr_intersection_for_each.cuh"

#include <cugraph/graph_view.hpp>

#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>


#include <cuda_runtime_api.h>

namespace cugraph {

template <typename edge_t>
struct edge_active_flag_t {
  uint32_t const* mask_ptr;

  __device__ edge_t operator()(edge_t e) const
  {
    return (mask_ptr[packed_bool_offset(e)] & packed_bool_mask(e)) ? edge_t{1} : edge_t{0};
  }
};

/**
 * @brief Iterate over each input vertex pair, compute the destination neighbor intersection,
 * and apply an operator to each common neighbor.
 *
 * For single-GPU, intersection work is dispatched by min(degree(src), degree(dst)) to thread,
 * warp, or block-level kernels.
 *
 * For multi-GPU, falls back to per_v_pair_dst_nbr_intersection and replays the result through
 * the operator.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexPairIterator Type of the iterator for input vertex pairs.
 * @tparam IntersectionOp Device-callable senary operator with signature
 *   void(vertex_t src, vertex_t dst, vertex_t common_nbr, edge_t pq_offset, edge_t pr_offset,
 *   edge_t qr_offset). Invoked once per triangle discovered through each input vertex pair.
 * @param handle RAFT handle object.
 * @param graph_view Non-owning graph object.
 * @param vertex_pair_first Iterator pointing to the first (inclusive) input vertex pair.
 * @param vertex_pair_last Iterator pointing to the last (exclusive) input vertex pair.
 * @param intersection_op Senary operator invoked for each (src, dst, common_nbr) triple with
 *   the corresponding edge offsets.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @param use_compaction If true and an edge mask is present, build a compact CSR for faster
 *   neighbor access.
 */
template <typename GraphViewType, typename VertexPairIterator, typename IntersectionOp>
void per_v_pair_dst_nbr_intersection_for_each(raft::handle_t const& handle,
                                              GraphViewType const& graph_view,
                                              VertexPairIterator vertex_pair_first,
                                              VertexPairIterator vertex_pair_last,
                                              IntersectionOp intersection_op,
                                              bool do_expensive_check = false,
                                              bool use_compaction = false)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  static_assert(!GraphViewType::is_storage_transposed);

  size_t input_size = static_cast<size_t>(
    cuda::std::distance(vertex_pair_first, vertex_pair_last));
  if (input_size == 0) return;

  if constexpr (!GraphViewType::is_multi_gpu) {
    auto edge_mask_view = graph_view.edge_mask_view();

    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, false>(
        graph_view.local_edge_partition_view(size_t{0}));
    auto edge_partition_e_mask =
      edge_mask_view
        ? cuda::std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, 0)
        : cuda::std::nullopt;

    auto stream = handle.get_stream();

      uint32_t const* mask_ptr = edge_partition_e_mask
        ? (*edge_partition_e_mask).value_first()
        : nullptr;

      bool const has_mask = (mask_ptr != nullptr);

      rmm::device_uvector<edge_t> vertex_degrees(0, stream);
      if (has_mask) {
        vertex_degrees = edge_partition.compute_local_degrees_with_mask(mask_ptr, stream);
      } else {
        vertex_degrees = edge_partition.compute_local_degrees(stream);
      }

      // Optional: build a temporary compact CSR containing only active (unmasked) edges.
      // The original CSR is a non-owning immutable view (graph_view_t); it cannot be modified
      // or shrunk in place. Reconstructing a new graph_t each iteration would require a full
      // sort + compress, which is more expensive than building this lightweight compact copy.
      auto num_vertices = edge_partition.major_range_size();
      auto num_edges    = edge_partition.number_of_edges();
      auto offsets_ptr  = edge_partition.offsets();
      auto indices_ptr  = edge_partition.indices();

      rmm::device_uvector<edge_t> compact_offsets_vec(0, stream);
      rmm::device_uvector<vertex_t> compact_indices_vec(0, stream);
      // It maps each compact edge position back to its original CSR offsets.
      // This is needed when writting the result of the intersection kernel
      // back to the original CSR.
      rmm::device_uvector<edge_t> compact_edge_map_vec(0, stream);
      edge_t const* c_off = nullptr;
      vertex_t const* c_ind = nullptr;
      edge_t const* c_map = nullptr;

      if (has_mask && use_compaction) {
        compact_offsets_vec.resize(num_vertices + 1, stream);
        thrust::fill_n(handle.get_thrust_policy(), compact_offsets_vec.data(), 1, edge_t{0});
        thrust::inclusive_scan(handle.get_thrust_policy(),
                               vertex_degrees.begin(),
                               vertex_degrees.end(),
                               compact_offsets_vec.data() + 1);
        
        // The last element of compact_offsets_vec is the number of active edges.
        // We use this to resize the compact_edge_map_vec and compact_indices_vec.
        auto num_active_edges = compact_offsets_vec.element(num_vertices, stream);
        compact_edge_map_vec.resize(num_active_edges, stream);
        compact_indices_vec.resize(num_active_edges, stream);

        auto edge_counting = thrust::make_counting_iterator(edge_t{0});
        // iterates over all edge indices and copies only the active (unmasked) edges
        // to the compact_edge_map_vec.
        thrust::copy_if(
          handle.get_thrust_policy(),
          edge_counting,
          edge_counting + num_edges,
          compact_edge_map_vec.data(),
          edge_active_flag_t<edge_t>{mask_ptr});

        thrust::gather(
          handle.get_thrust_policy(),
          compact_edge_map_vec.begin(),
          compact_edge_map_vec.begin() + num_active_edges,
          indices_ptr,
          compact_indices_vec.data());

        c_off = compact_offsets_vec.data();
        c_ind = compact_indices_vec.data();
        c_map = compact_edge_map_vec.data();
      }


      // Gather src/dst degrees for each edge pair, then compute min(src_deg, dst_deg).
      auto src_iter = cuda::std::get<0>(vertex_pair_first.get_iterator_tuple());
      auto dst_iter = cuda::std::get<1>(vertex_pair_first.get_iterator_tuple());

      rmm::device_uvector<edge_t> src_degrees(input_size, stream);
      rmm::device_uvector<edge_t> dst_degrees(input_size, stream);
      thrust::gather(handle.get_thrust_policy(),
                     src_iter, src_iter + input_size,
                     vertex_degrees.begin(), src_degrees.begin());
      thrust::gather(handle.get_thrust_policy(),
                     dst_iter, dst_iter + input_size,
                     vertex_degrees.begin(), dst_degrees.begin());

      rmm::device_uvector<edge_t> min_degrees(input_size, stream);
      // compute the minimum of the src and dst degrees for each edge pair
      thrust::transform(handle.get_thrust_policy(),
                        src_degrees.begin(), src_degrees.end(),
                        dst_degrees.begin(),
                        min_degrees.begin(),
                        thrust::minimum<edge_t>());

      auto min_deg_ptr = min_degrees.data();
      auto counting = thrust::make_counting_iterator(size_t{0});

      constexpr size_t ep_low_threshold = 128;
      constexpr size_t ep_mid_threshold = detail::mid_degree_threshold;

      auto num_low = thrust::count_if(
        handle.get_thrust_policy(), counting, counting + input_size,
        [min_deg_ptr] __device__(size_t i) {
          return min_deg_ptr[i] < static_cast<edge_t>(ep_low_threshold);
        });
      auto num_high = thrust::count_if(
        handle.get_thrust_policy(), counting, counting + input_size,
        [min_deg_ptr] __device__(size_t i) {
          return min_deg_ptr[i] >= static_cast<edge_t>(ep_mid_threshold);
        });
      auto num_mid = static_cast<decltype(num_low)>(input_size) - num_low - num_high;

      rmm::device_uvector<size_t> low_indices(num_low, stream);
      rmm::device_uvector<size_t> mid_indices(num_mid, stream);
      rmm::device_uvector<size_t> high_indices(num_high, stream);

      // Copy the indices(positions) of the edge pairs whose
      // min_degree < ep_*_threshold.
      if (num_low > 0) {
        // Copy the indices(positions) of the edge pairs whose
        // min_degree < ep_low_threshold(128)
        thrust::copy_if(
          handle.get_thrust_policy(), counting, counting + input_size, low_indices.data(),
          [min_deg_ptr] __device__(size_t i) {
            return min_deg_ptr[i] < static_cast<edge_t>(ep_low_threshold);
          });
      }
      if (num_mid > 0) {
        thrust::copy_if(
          handle.get_thrust_policy(), counting, counting + input_size, mid_indices.data(),
          [min_deg_ptr] __device__(size_t i) {
            return min_deg_ptr[i] >= static_cast<edge_t>(ep_low_threshold) &&
                   min_deg_ptr[i] < static_cast<edge_t>(ep_mid_threshold);
          });
      }
      if (num_high > 0) {
        thrust::copy_if(
          handle.get_thrust_policy(), counting, counting + input_size, high_indices.data(),
          [min_deg_ptr] __device__(size_t i) {
            return min_deg_ptr[i] >= static_cast<edge_t>(ep_mid_threshold);
          });
      }


      auto max_grid_size = handle.get_device_properties().maxGridSize[0];

      // Launch low-degree kernel (thread-per-pair)
      if (num_low > 0) {
        raft::grid_1d_thread_t grid(
          num_low, detail::intersection_kernel_block_size, max_grid_size);
        if (c_off != nullptr) {
          detail::intersection_low_degree<false, true, vertex_t, edge_t>
            <<<grid.num_blocks, grid.block_size, 0, stream>>>(
              edge_partition, vertex_pair_first, low_indices.data(),
              static_cast<size_t>(num_low), intersection_op, mask_ptr,
              c_off, c_ind, c_map);
        } else if (has_mask) {
          detail::intersection_low_degree<true, false, vertex_t, edge_t>
            <<<grid.num_blocks, grid.block_size, 0, stream>>>(
              edge_partition, vertex_pair_first, low_indices.data(),
              static_cast<size_t>(num_low), intersection_op, mask_ptr);
        } else {
          detail::intersection_low_degree<false, false, vertex_t, edge_t>
            <<<grid.num_blocks, grid.block_size, 0, stream>>>(
              edge_partition, vertex_pair_first, low_indices.data(),
              static_cast<size_t>(num_low), intersection_op, mask_ptr);
        }
      }

      // Launch mid-degree kernel (warp-per-pair, binary search)
      if (num_mid > 0) {
        raft::grid_1d_warp_t grid(
          num_mid, detail::intersection_kernel_block_size, max_grid_size);
        if (c_off != nullptr) {
          detail::intersection_mid_degree<false, true, vertex_t, edge_t>
            <<<grid.num_blocks, grid.block_size, 0, stream>>>(
              edge_partition, vertex_pair_first, mid_indices.data(),
              static_cast<size_t>(num_mid), intersection_op, mask_ptr,
              c_off, c_ind, c_map);
        } else if (has_mask) {
          detail::intersection_mid_degree<true, false, vertex_t, edge_t>
            <<<grid.num_blocks, grid.block_size, 0, stream>>>(
              edge_partition, vertex_pair_first, mid_indices.data(),
              static_cast<size_t>(num_mid), intersection_op, mask_ptr);
        } else {
          detail::intersection_mid_degree<false, false, vertex_t, edge_t>
            <<<grid.num_blocks, grid.block_size, 0, stream>>>(
              edge_partition, vertex_pair_first, mid_indices.data(),
              static_cast<size_t>(num_mid), intersection_op, mask_ptr);
        }
      }

      // Launch high-degree kernel (block-per-pair)
      if (num_high > 0) {
        raft::grid_1d_block_t grid(
          num_high, detail::intersection_kernel_block_size, max_grid_size);
        if (c_off != nullptr) {
          detail::intersection_high_degree<false, true, vertex_t, edge_t>
            <<<grid.num_blocks, grid.block_size, 0, stream>>>(
              edge_partition, vertex_pair_first, high_indices.data(),
              static_cast<size_t>(num_high), intersection_op, mask_ptr,
              c_off, c_ind, c_map);
        } else if (has_mask) {
          detail::intersection_high_degree<true, false, vertex_t, edge_t>
            <<<grid.num_blocks, grid.block_size, 0, stream>>>(
              edge_partition, vertex_pair_first, high_indices.data(),
              static_cast<size_t>(num_high), intersection_op, mask_ptr);
        } else {
          detail::intersection_high_degree<false, false, vertex_t, edge_t>
            <<<grid.num_blocks, grid.block_size, 0, stream>>>(
              edge_partition, vertex_pair_first, high_indices.data(),
              static_cast<size_t>(num_high), intersection_op, mask_ptr);
        }
      }
  } else {
    // MG fallback: materialize intersections, then replay via functor
    auto [nbr_intersection_offsets, nbr_intersection_indices] =
      detail::nbr_intersection(handle,
                               graph_view,
                               cugraph::edge_dummy_property_t{}.view(),
                               vertex_pair_first,
                               vertex_pair_last,
                               std::array<bool, 2>{true, true},
                               do_expensive_check);

    auto offsets_span =
      raft::device_span<size_t const>(nbr_intersection_offsets.data(),
                                      nbr_intersection_offsets.size());
    auto indices_span =
      raft::device_span<vertex_t const>(nbr_intersection_indices.data(),
                                        nbr_intersection_indices.size());

    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(input_size),
      [vertex_pair_first, offsets_span, indices_span, intersection_op] __device__(size_t i) {
        auto pair  = *(vertex_pair_first + i);
        vertex_t p = cuda::std::get<0>(pair);
        vertex_t q = cuda::std::get<1>(pair);
        for (auto j = offsets_span[i]; j < offsets_span[i + 1]; ++j) {
          intersection_op(p, q, indices_span[j]);
        }
      });
  }
}

template <typename vertex_t, typename edge_t>
struct csr_to_pair_t {
  edge_t const* offsets;
  vertex_t const* indices;
  vertex_t num_vertices;

  __device__ cuda::std::tuple<vertex_t, vertex_t> operator()(edge_t i) const
  {
    auto src = static_cast<vertex_t>(
      thrust::upper_bound(thrust::seq, offsets, offsets + num_vertices + 1, i) - offsets - 1);
    return cuda::std::make_tuple(src, indices[i]);
  }
};

template <typename vertex_t, typename edge_t>
struct compute_min_degree_t {
  edge_t const* offsets;
  vertex_t const* indices;
  edge_t const* vertex_degrees;
  vertex_t num_vertices;

  __device__ edge_t operator()(edge_t i) const
  {
    auto src = static_cast<vertex_t>(
      thrust::upper_bound(thrust::seq, offsets, offsets + num_vertices + 1, i) - offsets - 1);
    auto dst = indices[i];
    return min(vertex_degrees[src], vertex_degrees[dst]);
  }
};

template <typename vertex_t, typename edge_t>
struct scatter_active_edges_t {
  uint32_t const* mask_ptr;
  vertex_t const* indices_ptr;
  edge_t const* scan_ptr;
  vertex_t* compact_indices;
  edge_t* compact_edge_map;

  __device__ void operator()(edge_t e) const
  {
    if (mask_ptr[packed_bool_offset(e)] & packed_bool_mask(e)) {
      auto pos = scan_ptr[e];
      compact_indices[pos] = indices_ptr[e];
      compact_edge_map[pos] = e;
    }
  }
};


/**
 * @brief Iterate over all active edges in the graph, compute the destination neighbor intersection
 * for each edge, and apply an operator to each common neighbor.
 *
 * Intersection work is dispatched by min(degree(src), degree(dst)) to thread, warp, or
 * block-level kernels.  Single-GPU only.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam IntersectionOp Device-callable senary operator with signature
 *   void(vertex_t src, vertex_t dst, vertex_t common_nbr, edge_t pq_offset, edge_t pr_offset,
 *   edge_t qr_offset). Invoked once per triangle discovered through each edge.
 * @param handle RAFT handle object.
 * @param graph_view Non-owning graph object.
 * @param intersection_op Senary operator invoked for each (src, dst, common_nbr) triple with
 *   the corresponding edge offsets.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType, typename IntersectionOp>
void per_v_pair_dst_nbr_intersection_for_each(raft::handle_t const& handle,
                                              GraphViewType const& graph_view,
                                              IntersectionOp intersection_op,
                                              bool do_expensive_check = false,
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  static_assert(!GraphViewType::is_storage_transposed);
  static_assert(!GraphViewType::is_multi_gpu);

  auto edge_mask_view = graph_view.edge_mask_view();

  auto edge_partition =
    edge_partition_device_view_t<vertex_t, edge_t, false>(
      graph_view.local_edge_partition_view(size_t{0}));
  auto edge_partition_e_mask =
    edge_mask_view
      ? cuda::std::make_optional<
          detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
          *edge_mask_view, 0)
      : cuda::std::nullopt;

  auto stream = handle.get_stream();
  auto num_vertices = edge_partition.major_range_size();
  auto num_edges    = edge_partition.number_of_edges();

  if (num_edges == 0) return;

  uint32_t const* mask_ptr = edge_partition_e_mask
    ? (*edge_partition_e_mask).value_first()
    : nullptr;
  bool const has_mask = (mask_ptr != nullptr);

  auto offsets_ptr = edge_partition.offsets();
  auto indices_ptr = edge_partition.indices();


  // NOTE: vertex_pair_first is a transform iterator that resolves (src, dst) on-the-fly via
  // binary search on offsets (O(log V) per access). This avoids allocating a full src_idx array
  // (saves E * sizeof(vertex_t) memory) at the cost of repeated binary searches per thread.
  // For large-scale graphs, pre-computing src_idx externally and passing it in can eliminate
  // this overhead when the same graph is processed multiple times.
  auto vertex_pair_first = thrust::make_transform_iterator(
    thrust::make_counting_iterator(edge_t{0}),
    csr_to_pair_t<vertex_t, edge_t>{offsets_ptr, indices_ptr, static_cast<vertex_t>(num_vertices)});




  // Step 2: Compute per-vertex degrees, then per-edge min-degree via inline binary search.
  constexpr bool use_compaction = true;
  constexpr bool use_parallel_compaction = true;

  rmm::device_uvector<edge_t> vertex_degrees(0, stream);
  if (has_mask) {
    vertex_degrees = edge_partition.compute_local_degrees_with_mask(mask_ptr, stream);
  } else {
    vertex_degrees = edge_partition.compute_local_degrees(stream);
  }

  // Step 2b: Optionally build compact CSR (only active edges, no gaps).
  rmm::device_uvector<edge_t> compact_offsets_vec(0, stream);
  rmm::device_uvector<vertex_t> compact_indices_vec(0, stream);
  rmm::device_uvector<edge_t> compact_edge_map_vec(0, stream);
  edge_t const* c_off = nullptr;
  vertex_t const* c_ind = nullptr;
  edge_t const* c_map = nullptr;

  if (has_mask && use_compaction) {
    compact_offsets_vec.resize(num_vertices + 1, stream);
    thrust::fill_n(handle.get_thrust_policy(), compact_offsets_vec.data(), 1, edge_t{0});
    thrust::inclusive_scan(handle.get_thrust_policy(),
                           vertex_degrees.begin(),
                           vertex_degrees.end(),
                           compact_offsets_vec.data() + 1);

    // FIXME(NVCC BUG): DO NOT use `if constexpr` here.
    // NVCC 12.9 (V12.9.86, sm_90) has a bug where `if constexpr` around blocks
    // containing device lambdas corrupts the name mangling / kernel registration
    // of device lambdas that appear LATER in the same function (e.g. in
    // thrust::count_if / thrust::copy_if calls in the binning step below).
    // The downstream calls fail with:
    //   cudaErrorInvalidResourceHandle: invalid resource handle
    // at CUB's "determine reduce temporary storage size" step.
    // This happens even when both branches contain IDENTICAL code.
    // Workaround: use a runtime `if` so both branches are always compiled.
    if (use_parallel_compaction) {
      auto num_active_edges = compact_offsets_vec.element(num_vertices, stream);
      compact_edge_map_vec.resize(num_active_edges, stream);
      compact_indices_vec.resize(num_active_edges, stream);

      auto edge_counting = thrust::make_counting_iterator(edge_t{0});

      thrust::copy_if(
        handle.get_thrust_policy(),
        edge_counting,
        edge_counting + num_edges,
        compact_edge_map_vec.data(),
        edge_active_flag_t<edge_t>{mask_ptr});

      thrust::gather(
        handle.get_thrust_policy(),
        compact_edge_map_vec.begin(),
        compact_edge_map_vec.begin() + num_active_edges,
        indices_ptr,
        compact_indices_vec.data());
    } else {
      compact_indices_vec.resize(num_edges, stream);
      compact_edge_map_vec.resize(num_edges, stream);

      auto c_off_ptr = compact_offsets_vec.data();
      auto c_ind_ptr = compact_indices_vec.data();
      auto c_map_ptr = compact_edge_map_vec.data();

      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(vertex_t{0}),
        thrust::make_counting_iterator(static_cast<vertex_t>(num_vertices)),
        [offsets_ptr, indices_ptr, mask_ptr,
         c_off_ptr, c_ind_ptr, c_map_ptr] __device__(vertex_t v) {
          auto orig_offset = offsets_ptr[v];
          auto degree      = offsets_ptr[v + 1] - offsets_ptr[v];
          auto csr_start   = c_off_ptr[v];
          edge_t j         = 0;
          for (edge_t k = 0; k < degree; ++k) {
            auto eidx = orig_offset + k;
            if (mask_ptr[packed_bool_offset(eidx)] & packed_bool_mask(eidx)) {
              c_ind_ptr[csr_start + j] = indices_ptr[eidx];
              c_map_ptr[csr_start + j] = eidx;
              ++j;
            }
          }
        });
    }

    c_off = compact_offsets_vec.data();
    c_ind = compact_indices_vec.data();
    c_map = compact_edge_map_vec.data();
  }

  rmm::device_uvector<edge_t> min_degrees(num_edges, stream);
  thrust::transform(handle.get_thrust_policy(),
                    thrust::make_counting_iterator(edge_t{0}),
                    thrust::make_counting_iterator(static_cast<edge_t>(num_edges)),
                    min_degrees.begin(),
                    compute_min_degree_t<vertex_t, edge_t>{
                      offsets_ptr, indices_ptr, vertex_degrees.data(),
                      static_cast<vertex_t>(num_vertices)});

  vertex_degrees.resize(0, stream);
  vertex_degrees.shrink_to_fit(stream);

  // Step 3: Bin edges by min-degree (only unmasked edges when mask present).
  auto min_deg_ptr = min_degrees.data();
  auto counting    = thrust::make_counting_iterator(size_t{0});

  auto is_active = [mask_ptr] __device__(size_t e) -> bool {
    if (mask_ptr == nullptr) return true;
    return static_cast<bool>(mask_ptr[packed_bool_offset(e)] & packed_bool_mask(e));
  };

  auto num_low = thrust::count_if(
    handle.get_thrust_policy(), counting, counting + num_edges,
    [min_deg_ptr, is_active] __device__(size_t i) {
      return is_active(i) && min_deg_ptr[i] < static_cast<edge_t>(detail::low_degree_threshold);
    });
  auto num_high = thrust::count_if(
    handle.get_thrust_policy(), counting, counting + num_edges,
    [min_deg_ptr, is_active] __device__(size_t i) {
      return is_active(i) && min_deg_ptr[i] >= static_cast<edge_t>(detail::mid_degree_threshold);
    });
  auto num_active = thrust::count_if(
    handle.get_thrust_policy(), counting, counting + num_edges, is_active);

  auto num_mid = num_active - num_low - num_high;

  rmm::device_uvector<size_t> low_indices(num_low, stream);
  rmm::device_uvector<size_t> mid_indices(num_mid, stream);
  rmm::device_uvector<size_t> high_indices(num_high, stream);

  if (num_low > 0) {
    thrust::copy_if(
      handle.get_thrust_policy(), counting, counting + num_edges, low_indices.data(),
      [min_deg_ptr, is_active] __device__(size_t i) {
        return is_active(i) && min_deg_ptr[i] < static_cast<edge_t>(detail::low_degree_threshold);
      });
  }
  if (num_mid > 0) {
    thrust::copy_if(
      handle.get_thrust_policy(), counting, counting + num_edges, mid_indices.data(),
      [min_deg_ptr, is_active] __device__(size_t i) {
        return is_active(i) &&
               min_deg_ptr[i] >= static_cast<edge_t>(detail::low_degree_threshold) &&
               min_deg_ptr[i] < static_cast<edge_t>(detail::mid_degree_threshold);
      });
  }
  if (num_high > 0) {
    thrust::copy_if(
      handle.get_thrust_policy(), counting, counting + num_edges, high_indices.data(),
      [min_deg_ptr, is_active] __device__(size_t i) {
        return is_active(i) && min_deg_ptr[i] >= static_cast<edge_t>(detail::mid_degree_threshold);
      });
  }

  // Step 4: Launch intersection kernels.
  auto max_grid_size = handle.get_device_properties().maxGridSize[0];

  if (num_low > 0) {
    raft::grid_1d_thread_t grid(
      num_low, detail::intersection_kernel_block_size, max_grid_size);
    if (c_off != nullptr) {
      detail::intersection_low_degree<false, true, vertex_t, edge_t>
        <<<grid.num_blocks, grid.block_size, 0, stream>>>(
          edge_partition, vertex_pair_first, low_indices.data(),
          static_cast<size_t>(num_low), intersection_op, mask_ptr,
          c_off, c_ind, c_map);
    } else if (has_mask) {
      detail::intersection_low_degree<true, false, vertex_t, edge_t>
        <<<grid.num_blocks, grid.block_size, 0, stream>>>(
          edge_partition, vertex_pair_first, low_indices.data(),
          static_cast<size_t>(num_low), intersection_op, mask_ptr);
    } else {
      detail::intersection_low_degree<false, false, vertex_t, edge_t>
        <<<grid.num_blocks, grid.block_size, 0, stream>>>(
          edge_partition, vertex_pair_first, low_indices.data(),
          static_cast<size_t>(num_low), intersection_op, mask_ptr);
    }
  }
  // double ms_low = sync_and_ms(t);

  if (num_mid > 0) {
    raft::grid_1d_warp_t grid(
      num_mid, detail::intersection_kernel_block_size, max_grid_size);
    if (c_off != nullptr) {
      detail::intersection_mid_degree<false, true, vertex_t, edge_t>
        <<<grid.num_blocks, grid.block_size, 0, stream>>>(
          edge_partition, vertex_pair_first, mid_indices.data(),
          static_cast<size_t>(num_mid), intersection_op, mask_ptr,
          c_off, c_ind, c_map);
    } else if (has_mask) {
      detail::intersection_mid_degree<true, false, vertex_t, edge_t>
        <<<grid.num_blocks, grid.block_size, 0, stream>>>(
          edge_partition, vertex_pair_first, mid_indices.data(),
          static_cast<size_t>(num_mid), intersection_op, mask_ptr);
    } else {
      detail::intersection_mid_degree<false, false, vertex_t, edge_t>
        <<<grid.num_blocks, grid.block_size, 0, stream>>>(
          edge_partition, vertex_pair_first, mid_indices.data(),
          static_cast<size_t>(num_mid), intersection_op, mask_ptr);
    }
  }
  // double ms_mid = sync_and_ms(t);

  if (num_high > 0) {
    raft::grid_1d_block_t grid(
      num_high, detail::intersection_kernel_block_size, max_grid_size);
    if (c_off != nullptr) {
      detail::intersection_high_degree<false, true, vertex_t, edge_t>
        <<<grid.num_blocks, grid.block_size, 0, stream>>>(
          edge_partition, vertex_pair_first, high_indices.data(),
          static_cast<size_t>(num_high), intersection_op, mask_ptr,
          c_off, c_ind, c_map);
    } else if (has_mask) {
      detail::intersection_high_degree<true, false, vertex_t, edge_t>
        <<<grid.num_blocks, grid.block_size, 0, stream>>>(
          edge_partition, vertex_pair_first, high_indices.data(),
          static_cast<size_t>(num_high), intersection_op, mask_ptr);
    } else {
      detail::intersection_high_degree<false, false, vertex_t, edge_t>
        <<<grid.num_blocks, grid.block_size, 0, stream>>>(
          edge_partition, vertex_pair_first, high_indices.data(),
          static_cast<size_t>(num_high), intersection_op, mask_ptr);
    }
  }


}

}  // namespace cugraph

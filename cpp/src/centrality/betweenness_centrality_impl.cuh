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
#pragma once

#include "prims/count_if_v.cuh"
#include "prims/detail/prim_functors.cuh"
#include "prims/edge_bucket.cuh"
#include "prims/extract_transform_if_e.cuh"
#include "prims/extract_transform_if_v_frontier_outgoing_e.cuh"
#include "prims/fill_edge_property.cuh"
#include "prims/per_v_transform_reduce_incoming_outgoing_e.cuh"
#include "prims/transform_e.cuh"
#include "prims/transform_reduce_if_v_frontier_outgoing_e_by_dst.cuh"
#include "prims/transform_reduce_v.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/update_v_frontier.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/core/handle.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda/atomic>
#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

//
// The formula for BC(v) is the sum over all (s,t) where s != v != t of
// sigma_st(v) / sigma_st.  Sigma_st(v) is the number of shortest paths
// that pass through vertex v, whereas sigma_st is the total number of shortest
// paths.
namespace {

// Memory measurement utilities
struct MemoryInfo {
  size_t free_memory;
  size_t used_memory;
  size_t total_memory;

  static MemoryInfo get_device_memory()
  {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return {free, total - free, total};
  }

  void print(const char* label) const
  {
    double memory_usage = static_cast<double>(used_memory) / total_memory;
    printf("[Memory] %s: Free: %.1fGB, Used: %.1fGB, Total: %.1fGB (%.1f%%)\n",
           label,
           free_memory / (1024.0 * 1024.0 * 1024.0),
           used_memory / (1024.0 * 1024.0 * 1024.0),
           total_memory / (1024.0 * 1024.0 * 1024.0),
           memory_usage * 100.0);
  }
};

// 4-Stage Memory Allocation Strategy
enum class MemoryStage {
  RAMP_UP,      // Stage 1: Double until ceiling
  OSCILLATION,  // Stage 2: Throttle down/up until convergence
  FINE_TUNE,    // Stage 3: Slowly add +5 until 98%
  CONVERGED     // Stage 4: Locked in constant batch size
};

class AdaptiveMemoryManager {
 private:
  MemoryStage current_stage;
  size_t current_batch_size;
  size_t min_batch_size;
  size_t max_batch_size;
  size_t successful_batches;
  size_t oom_batches;
  size_t total_batches;
  double memory_threshold;
  size_t oscillation_count;
  size_t last_throttle_direction;        // 0 = none, 1 = down, 2 = up
  size_t converged_batch_size;           // Locked-in constant batch size once converged
  size_t consecutive_same_size_batches;  // Track stuck detection
  size_t last_batch_size;                // Track stuck detection
  static constexpr size_t CONVERGENCE_THRESHOLD = 3;
  static constexpr size_t STUCK_THRESHOLD       = 5;  // Force transition if stuck

 public:
  AdaptiveMemoryManager(size_t initial_batch_size = 50,
                        size_t min_batch          = 10,
                        size_t max_batch          = 1000,
                        double threshold          = 0.85)
    : current_stage(MemoryStage::RAMP_UP),
      current_batch_size(initial_batch_size),
      min_batch_size(min_batch),
      max_batch_size(max_batch),
      successful_batches(0),
      oom_batches(0),
      total_batches(0),
      memory_threshold(threshold),
      oscillation_count(0),
      last_throttle_direction(0),
      converged_batch_size(0),
      consecutive_same_size_batches(0),
      last_batch_size(initial_batch_size)
  {
  }

  size_t get_batch_size() const { return current_batch_size; }
  MemoryStage get_stage() const { return current_stage; }

  // Record batch result and handle OOMs immediately
  void record_batch_result(bool success)
  {
    total_batches++;

    if (success) {
      successful_batches++;
      oom_batches = 0;  // Reset OOM counter on success
    } else {
      oom_batches++;
      // If we hit OOM, reduce batch size immediately (regardless of stage)
      current_batch_size = std::max(current_batch_size / 2, min_batch_size);
      printf("[OOM] Batch size halved to %zu\n", current_batch_size);
    }
  }

  void update_batch_size(bool success, const MemoryInfo& mem_info)
  {
    // Note: OOM handling is now done in record_batch_result()
    // This method only handles stage transitions and success-based increases

    double memory_usage = static_cast<double>(mem_info.used_memory) / mem_info.total_memory;

    // 4-stage adaptive batch sizing logic
    switch (current_stage) {
      case MemoryStage::RAMP_UP: {
        // Stage 1: Double until ceiling, but use 1.5x if we've had OOMs
        if (memory_usage < memory_threshold) {
          // If we've had OOMs, be more conservative with 1.5x instead of doubling
          if (oom_batches > 0) {
            // OOMs detected - use 1.5x instead of doubling
            size_t new_batch_size = static_cast<size_t>(current_batch_size * 1.5);
            if (new_batch_size <= max_batch_size) {
              current_batch_size = new_batch_size;
              printf("RAMP_UP: OOMs detected, conservative increase to %zu (1.5x)\n",
                     current_batch_size);
            }
          } else {
            // No OOMs - safe to double
            size_t new_batch_size = current_batch_size * 2;
            if (new_batch_size <= max_batch_size) {
              current_batch_size = new_batch_size;
              printf("RAMP_UP: Safe doubling to %zu\n", current_batch_size);
            }
          }
        } else {
          // Hit ceiling - transition to oscillation stage
          current_stage = MemoryStage::OSCILLATION;
          printf("RAMP_UP → OSCILLATION: Hit memory ceiling at %zu sources\n", current_batch_size);
        }
        break;
      }

      case MemoryStage::OSCILLATION: {
        // Stage 2: Throttle down/up until convergence using single 85% threshold

        // Update stuck detection tracking
        if (current_batch_size == last_batch_size) {
          consecutive_same_size_batches++;
        } else {
          consecutive_same_size_batches = 0;
        }
        last_batch_size = current_batch_size;

        // Check if we're stuck (same batch size for too long)
        if (consecutive_same_size_batches >= STUCK_THRESHOLD) {
          current_stage = MemoryStage::FINE_TUNE;
          printf(
            "OSCILLATION → FINE_TUNE: Forced transition due to stuck at batch size %zu for %zu "
            "consecutive batches\n",
            current_batch_size,
            consecutive_same_size_batches);
          break;
        }

        if (memory_usage > memory_threshold) {
          // Memory > 85%: Throttle down
          size_t new_batch_size =
            std::max(static_cast<size_t>(current_batch_size * 0.8), min_batch_size);
          if (new_batch_size < current_batch_size) {
            current_batch_size      = new_batch_size;
            last_throttle_direction = 1;  // Down
            printf("OSCILLATION: Memory %.1f%% > %.1f%%, throttling down to %zu (0.8x)\n",
                   memory_usage * 100.0,
                   memory_threshold * 100.0,
                   current_batch_size);
          }
        } else {
          // Memory < 85%: Throttle up
          size_t new_batch_size = static_cast<size_t>(current_batch_size * 1.2);
          if (new_batch_size <= max_batch_size) {
            current_batch_size = new_batch_size;

            // Check if this completes a down→up cycle
            if (last_throttle_direction == 1) {  // Previous was down
              oscillation_count++;

              // Check if we've converged
              if (oscillation_count >= CONVERGENCE_THRESHOLD) {
                current_stage = MemoryStage::FINE_TUNE;
                printf(
                  "OSCILLATION → FINE_TUNE: %zu down→up cycles completed, starting fine-tuning\n",
                  oscillation_count);
              }
            }

            last_throttle_direction = 2;  // Up
            printf("OSCILLATION: Memory %.1f%% < %.1f%%, throttling up to %zu (1.2x)\n",
                   memory_usage * 100.0,
                   memory_threshold * 100.0,
                   current_batch_size);
          }
        }
        break;
      }

      case MemoryStage::FINE_TUNE: {
        // Stage 3: Slowly add +5 until close to 98%
        if (memory_usage < 0.98) {  // Below 98% memory usage
          size_t new_batch_size = current_batch_size + 5;
          if (new_batch_size <= max_batch_size) {
            current_batch_size = new_batch_size;
            printf("FINE_TUNE: Memory %.1f%% < 98%%, increasing to %zu (+5)\n",
                   memory_usage * 100.0,
                   current_batch_size);
          }
        } else {
          // Hit 98% memory usage - lock in this batch size as constant maximum
          converged_batch_size = current_batch_size;
          current_stage        = MemoryStage::CONVERGED;
          printf(
            "FINE_TUNE → CONVERGED: Hit 98%% memory usage, locked in constant batch size: %zu\n",
            converged_batch_size);
        }
        break;
      }

      case MemoryStage::CONVERGED: {
        // Stage 4: Use locked-in constant batch size for all future processing
        // This is the ultimate maximum - never change it
        if (converged_batch_size > 0 && converged_batch_size <= max_batch_size) {
          current_batch_size = converged_batch_size;
        } else {
          // Safety fallback - if converged_batch_size is invalid, reset to oscillation
          printf("CONVERGED: Invalid converged batch size %zu, falling back to oscillation\n",
                 converged_batch_size);
          current_stage        = MemoryStage::OSCILLATION;
          current_batch_size   = std::min(current_batch_size, max_batch_size / 2);
          converged_batch_size = 0;
        }
        break;
      }
    }
  }

  void print_stats() const
  {
    printf(
      "Performance Stats: Total=%zu, Successful=%zu, OOM=%zu, "
      "Success Rate=%.1f%%, Stage=%s, Oscillations=%zu\n",
      total_batches,
      successful_batches,
      oom_batches,
      (total_batches > 0) ? (100.0 * successful_batches / total_batches) : 0.0,
      stage_to_string(current_stage).c_str(),
      oscillation_count);

    if (current_stage == MemoryStage::CONVERGED) {
      printf("CONVERGED: Using locked-in constant batch size: %zu (ultimate maximum)\n",
             converged_batch_size);
    }
  }

 private:
  static std::string stage_to_string(MemoryStage stage)
  {
    switch (stage) {
      case MemoryStage::RAMP_UP: return "RAMP_UP";
      case MemoryStage::OSCILLATION: return "OSCILLATION";
      case MemoryStage::FINE_TUNE: return "FINE_TUNE";
      case MemoryStage::CONVERGED: return "CONVERGED";
      default: return "UNKNOWN";
    }
  }
};

// Memory cleanup utilities
struct MemoryCleanup {
  static void cleanup_batch_memory(raft::handle_t const& handle)
  {
    handle.sync_stream();

    // Try to free memory from RMM pools
    auto* current_resource = rmm::mr::get_current_device_resource();
    if (current_resource) { current_resource->deallocate(nullptr, 0, handle.get_stream()); }

    // Synchronize CUDA operations
    cudaDeviceSynchronize();

    // Try to free any cached memory
    cudaFree(0);
  }

  static void cleanup_test_memory(raft::handle_t const& handle)
  {
    handle.sync_stream();

    // More aggressive cleanup for test completion
    auto* current_resource = rmm::mr::get_current_device_resource();
    if (current_resource) { current_resource->deallocate(nullptr, 0, handle.get_stream()); }

    cudaDeviceSynchronize();
    cudaFree(0);
  }
};

template <typename vertex_t>
struct brandes_e_op_t {
  template <typename value_t, typename ignore_t>
  __device__ value_t operator()(vertex_t, vertex_t, value_t src_sigma, vertex_t, ignore_t) const
  {
    return src_sigma;
  }
};

template <typename vertex_t>
struct brandes_pred_op_t {
  const vertex_t invalid_distance_{std::numeric_limits<vertex_t>::max()};

  template <typename value_t, typename ignore_t>
  __device__ bool operator()(
    vertex_t, vertex_t, value_t src_sigma, vertex_t dst_distance, ignore_t) const
  {
    return (dst_distance == invalid_distance_);
  }
};

template <typename vertex_t>
struct extract_edge_e_op_t {
  template <typename edge_t, typename weight_t>
  __device__ cuda::std::tuple<vertex_t, vertex_t> operator()(
    vertex_t src,
    vertex_t dst,
    cuda::std::tuple<vertex_t, edge_t, weight_t> src_props,
    cuda::std::tuple<vertex_t, edge_t, weight_t> dst_props,
    cuda::std::nullopt_t) const
  {
    return cuda::std::make_tuple(src, dst);
  }

  template <typename edge_t, typename weight_t>
  __device__ cuda::std::tuple<vertex_t, vertex_t, edge_t> operator()(
    vertex_t src,
    vertex_t dst,
    cuda::std::tuple<vertex_t, edge_t, weight_t> src_props,
    cuda::std::tuple<vertex_t, edge_t, weight_t> dst_props,
    edge_t edge_multi_index) const
  {
    return cuda::std::make_tuple(src, dst, edge_multi_index);
  }
};

template <typename vertex_t>
struct extract_edge_pred_op_t {
  vertex_t d{};

  template <typename edge_t, typename weight_t>
  __device__ bool operator()(vertex_t src,
                             vertex_t dst,
                             cuda::std::tuple<vertex_t, edge_t, weight_t> src_props,
                             cuda::std::tuple<vertex_t, edge_t, weight_t> dst_props,
                             cuda::std::nullopt_t) const
  {
    return ((cuda::std::get<0>(src_props) == (d - 1)) && (cuda::std::get<0>(dst_props) == d));
  }

  template <typename edge_t, typename weight_t>
  __device__ bool operator()(vertex_t src,
                             vertex_t dst,
                             cuda::std::tuple<vertex_t, edge_t, weight_t> src_props,
                             cuda::std::tuple<vertex_t, edge_t, weight_t> dst_props,
                             edge_t edge_multi_index) const
  {
    return ((cuda::std::get<0>(src_props) == (d - 1)) && (cuda::std::get<0>(dst_props) == d));
  }
};

}  // namespace

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<edge_t>> brandes_bfs(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  vertex_frontier_t<vertex_t, void, multi_gpu, true>& vertex_frontier,
  bool do_expensive_check)
{
  //
  // Do BFS with a multi-output.  If we're on hop k and multiple vertices arrive at vertex v,
  // add all predecessors to the predecessor list, don't just arbitrarily pick one.
  //
  // Predecessors could be a CSR if that's helpful for doing the backwards tracing
  constexpr vertex_t invalid_distance = std::numeric_limits<vertex_t>::max();
  constexpr size_t bucket_idx_cur{0};
  constexpr size_t bucket_idx_next{1};

  rmm::device_uvector<edge_t> sigmas(graph_view.local_vertex_partition_range_size(),
                                     handle.get_stream());
  rmm::device_uvector<vertex_t> distances(graph_view.local_vertex_partition_range_size(),
                                          handle.get_stream());
  detail::scalar_fill(handle, distances.data(), distances.size(), invalid_distance);
  detail::scalar_fill(handle, sigmas.data(), sigmas.size(), edge_t{0});

  edge_src_property_t<vertex_t, edge_t> src_sigmas(handle, graph_view);
  edge_dst_property_t<vertex_t, vertex_t> dst_distances(handle, graph_view);

  auto vertex_partition =
    vertex_partition_device_view_t<vertex_t, multi_gpu>(graph_view.local_vertex_partition_view());

  if (vertex_frontier.bucket(bucket_idx_cur).size() > 0) {
    thrust::for_each(
      handle.get_thrust_policy(),
      vertex_frontier.bucket(bucket_idx_cur).begin(),
      vertex_frontier.bucket(bucket_idx_cur).end(),
      [d_sigma = sigmas.begin(), d_distance = distances.begin(), vertex_partition] __device__(
        auto v) {
        auto offset        = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
        d_distance[offset] = 0;
        d_sigma[offset]    = 1;
      });
  }

  edge_t hop{0};

  while (true) {
    update_edge_src_property(handle, graph_view, sigmas.begin(), src_sigmas.mutable_view());
    update_edge_dst_property(handle, graph_view, distances.begin(), dst_distances.mutable_view());

    auto [new_frontier, new_sigma] = cugraph::transform_reduce_if_v_frontier_outgoing_e_by_dst(
      handle,
      graph_view,
      vertex_frontier.bucket(bucket_idx_cur),
      src_sigmas.view(),
      dst_distances.view(),
      cugraph::edge_dummy_property_t{}.view(),
      brandes_e_op_t<vertex_t>{},
      reduce_op::plus<vertex_t>(),
      brandes_pred_op_t<vertex_t>{});

    auto next_frontier_bucket_indices = std::vector<size_t>{bucket_idx_next};
    update_v_frontier(handle,
                      graph_view,
                      std::move(new_frontier),
                      std::move(new_sigma),
                      vertex_frontier,
                      raft::host_span<size_t const>(next_frontier_bucket_indices.data(),
                                                    next_frontier_bucket_indices.size()),
                      thrust::make_zip_iterator(distances.begin(), sigmas.begin()),
                      thrust::make_zip_iterator(distances.begin(), sigmas.begin()),
                      [hop] __device__(auto v, auto old_values, auto v_sigma) {
                        return cuda::std::make_tuple(
                          cuda::std::make_optional(bucket_idx_next),
                          cuda::std::make_optional(cuda::std::make_tuple(hop + 1, v_sigma)));
                      });

    vertex_frontier.bucket(bucket_idx_cur).clear();
    vertex_frontier.bucket(bucket_idx_cur).shrink_to_fit();
    vertex_frontier.swap_buckets(bucket_idx_cur, bucket_idx_next);
    if (vertex_frontier.bucket(bucket_idx_cur).aggregate_size() == 0) { break; }

    ++hop;
  }

  return std::make_tuple(std::move(distances), std::move(sigmas));
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void accumulate_vertex_results(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  raft::device_span<weight_t> centralities,
  rmm::device_uvector<vertex_t>&& distances,
  rmm::device_uvector<edge_t>&& sigmas,
  bool with_endpoints,
  bool do_expensive_check)
{
  constexpr vertex_t invalid_distance = std::numeric_limits<vertex_t>::max();

  vertex_t diameter = transform_reduce_v(
    handle,
    graph_view,
    distances.begin(),
    [] __device__(auto, auto d) { return (d == invalid_distance) ? vertex_t{0} : d; },
    vertex_t{0},
    reduce_op::maximum<vertex_t>{},
    do_expensive_check);

  if (with_endpoints) {
    vertex_t count = count_if_v(
      handle,
      graph_view,
      distances.begin(),
      [] __device__(auto, auto d) { return (d != invalid_distance); },
      do_expensive_check);

    thrust::transform(handle.get_thrust_policy(),
                      distances.begin(),
                      distances.end(),
                      centralities.begin(),
                      centralities.begin(),
                      [count] __device__(auto d, auto centrality) {
                        if (d == vertex_t{0}) {
                          return centrality + static_cast<weight_t>(count - 1);
                        } else if (d == invalid_distance) {
                          return centrality;
                        } else {
                          return centrality + weight_t{1};
                        }
                      });
  }

  edge_src_property_t<vertex_t, edge_t> src_sigmas(handle, graph_view);
  edge_dst_property_t<vertex_t, vertex_t> dst_distances(handle, graph_view);
  edge_dst_property_t<vertex_t, edge_t> dst_sigmas(handle, graph_view);
  edge_dst_property_t<vertex_t, weight_t> dst_deltas(handle, graph_view);

  // Update all 3 properties initially (deltas start as 0)
  update_edge_src_property(handle, graph_view, sigmas.begin(), src_sigmas.mutable_view());
  update_edge_dst_property(handle,
                           graph_view,
                           thrust::make_zip_iterator(distances.begin(), sigmas.begin()),
                           view_concat(dst_distances.mutable_view(), dst_sigmas.mutable_view()));
  fill_edge_dst_property(handle, graph_view, dst_deltas.mutable_view(), weight_t{0.0});

  // Use binary search method to find frontier boundaries more efficiently
  std::vector<vertex_t> h_bounds{};
  rmm::device_uvector<vertex_t> vertices_sorted(distances.size(), handle.get_stream());
  {
    // Create distance_keys for sorting
    rmm::device_uvector<vertex_t> distance_keys(distances.size(), handle.get_stream());

    // Copy distances for sorting (preserve original)
    raft::copy(distance_keys.data(), distances.data(), distances.size(), handle.get_stream());

    // Use thrust::sequence instead of thrust::copy for vertices
    thrust::sequence(handle.get_thrust_policy(),
                     vertices_sorted.begin(),
                     vertices_sorted.end(),
                     graph_view.local_vertex_partition_range_first());

    // Sort vertices by distance using stable_sort_by_key
    thrust::stable_sort_by_key(handle.get_thrust_policy(),
                               distance_keys.begin(),
                               distance_keys.end(),     // keys (copied distances)
                               vertices_sorted.begin()  // values (vertices)
    );

    rmm::device_uvector<vertex_t> d_bounds(diameter + 1, handle.get_stream());

    // Single vectorized thrust call to compute all bounds for distances 0 to diameter
    thrust::lower_bound(
      handle.get_thrust_policy(),
      distance_keys.begin(),
      distance_keys.end(),                          // sorted distances
      thrust::make_counting_iterator<vertex_t>(0),  // search keys: [0, 1, 2, 3, ...]
      thrust::make_counting_iterator<vertex_t>(diameter + 1),
      d_bounds.data());

    // Copy bounds to host for use in delta loop
    h_bounds.resize(d_bounds.size());
    raft::update_host(h_bounds.data(), d_bounds.data(), d_bounds.size(), handle.get_stream());
    handle.sync_stream();
  }

  // Calculate max frontier size using the precomputed bounds
  vertex_t max_frontier_size = 0;
  for (size_t d = 0; d < h_bounds.size() - 1; ++d) {
    vertex_t frontier_count = h_bounds[d + 1] - h_bounds[d];
    max_frontier_size       = std::max(max_frontier_size, frontier_count);
  }

  // Pre-allocate reusable buffers to avoid repeated allocations (optimized for max frontier size)
  rmm::device_uvector<weight_t> reusable_delta_buffer(max_frontier_size, handle.get_stream());

  // Based on Brandes algorithm, we want to follow back pointers in non-increasing
  // distance from S to compute delta
  for (vertex_t d = diameter; d > 1; --d) {
    vertex_t frontier_count = h_bounds[d] - h_bounds[d - 1];
    if constexpr (multi_gpu) {
      frontier_count = host_scalar_allreduce(
        handle.get_comms(), frontier_count, raft::comms::op_t::SUM, handle.get_stream());
    }

    if (frontier_count > 0) {
      // Create key_bucket_t from the frontier vertices directly
      key_bucket_t<vertex_t, void, multi_gpu, true> vertex_list(
        handle,
        raft::device_span<vertex_t const>(vertices_sorted.data() + h_bounds[d - 1],
                                          h_bounds[d] - h_bounds[d - 1]));

      // Compute deltas for frontier vertices
      per_v_transform_reduce_outgoing_e(
        handle,
        graph_view,
        vertex_list,
        src_sigmas.view(),
        view_concat(dst_distances.view(), dst_sigmas.view(), dst_deltas.view()),
        cugraph::edge_dummy_property_t{}.view(),
        [d] __device__(auto, auto, auto src_sigma, auto dst_props, auto) {
          if (cuda::std::get<0>(dst_props) == d) {
            auto sigma_v = src_sigma;
            auto sigma_w = static_cast<weight_t>(cuda::std::get<1>(dst_props));
            auto delta_w = cuda::std::get<2>(dst_props);
            return (sigma_v / sigma_w) * (1 + delta_w);
          } else {
            return weight_t{0};
          }
        },
        weight_t{0},
        reduce_op::plus<weight_t>{},
        reusable_delta_buffer.begin(),
        do_expensive_check);

      // Only update deltas for vertices in vertex_list
      update_edge_dst_property(handle,
                               graph_view,
                               vertex_list.cbegin(),
                               vertex_list.cend(),
                               reusable_delta_buffer.begin(),
                               dst_deltas.mutable_view());

      // Update centralities - both vertices_sorted and centralities use local vertex IDs
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(vertices_sorted.begin() + h_bounds[d - 1],
                                  reusable_delta_buffer.begin()),
        thrust::make_zip_iterator(vertices_sorted.begin() + h_bounds[d],
                                  reusable_delta_buffer.begin()),
        [centralities = centralities.data(),
         v_first      = graph_view.local_vertex_partition_range_first()] __device__(auto pair) {
          auto v        = cuda::std::get<0>(pair);
          auto delta    = cuda::std::get<1>(pair);
          auto v_offset = v - v_first;
          centralities[v_offset] += delta;
        });
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void accumulate_edge_results(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  edge_property_view_t<edge_t, weight_t*> centralities_view,
  rmm::device_uvector<vertex_t>&& distances,
  rmm::device_uvector<edge_t>&& sigmas,
  bool do_expensive_check)
{
  constexpr vertex_t invalid_distance = std::numeric_limits<vertex_t>::max();

  vertex_t diameter = transform_reduce_v(
    handle,
    graph_view,
    distances.begin(),
    [] __device__(auto, auto d) { return (d == invalid_distance) ? vertex_t{0} : d; },
    vertex_t{0},
    reduce_op::maximum<vertex_t>{},
    do_expensive_check);

  rmm::device_uvector<weight_t> deltas(sigmas.size(), handle.get_stream());
  detail::scalar_fill(handle, deltas.data(), deltas.size(), weight_t{0});

  edge_src_property_t<vertex_t, cuda::std::tuple<vertex_t, edge_t, weight_t>> src_properties(
    handle, graph_view);
  edge_dst_property_t<vertex_t, cuda::std::tuple<vertex_t, edge_t, weight_t>> dst_properties(
    handle, graph_view);

  // Note: deltas are included here even though they start as 0, because the original approach
  // updates all properties at once. Deltas will be overwritten iteratively in the delta loop.
  update_edge_src_property(
    handle,
    graph_view,
    thrust::make_zip_iterator(distances.begin(), sigmas.begin(), deltas.begin()),
    src_properties.mutable_view());
  update_edge_dst_property(
    handle,
    graph_view,
    thrust::make_zip_iterator(distances.begin(), sigmas.begin(), deltas.begin()),
    dst_properties.mutable_view());

  //
  //   For now this will do a O(E) pass over all edges over the diameter
  //   of the graph.
  //
  // Based on Brandes algorithm, we want to follow back pointers in non-increasing
  // distance from S to compute delta
  //
  for (vertex_t d = diameter; d > 0; --d) {
    //
    //  Populate edge_list with edges where `cuda::std::get<0>(dst_props) == d`
    //  and `cuda::std::get<0>(dst_props) == (d-1)`
    //
    cugraph::edge_bucket_t<vertex_t, edge_t, true, multi_gpu, true> edge_list(
      handle, graph_view.is_multigraph());

    rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<edge_t>> indices{std::nullopt};
    if (graph_view.is_multigraph()) {
      edge_multi_index_property_t<edge_t, vertex_t> edge_multi_indices(handle, graph_view);
      std::tie(srcs, dsts, indices) = extract_transform_if_e(handle,
                                                             graph_view,
                                                             src_properties.view(),
                                                             dst_properties.view(),
                                                             edge_multi_indices.view(),
                                                             extract_edge_e_op_t<vertex_t>{},
                                                             extract_edge_pred_op_t<vertex_t>{d},
                                                             do_expensive_check);

      auto triplet_first = thrust::make_zip_iterator(srcs.begin(), dsts.begin(), indices->begin());
      thrust::sort(handle.get_thrust_policy(), triplet_first, triplet_first + srcs.size());
    } else {
      std::tie(srcs, dsts) = extract_transform_if_e(handle,
                                                    graph_view,
                                                    src_properties.view(),
                                                    dst_properties.view(),
                                                    edge_dummy_property_t{}.view(),
                                                    extract_edge_e_op_t<vertex_t>{},
                                                    extract_edge_pred_op_t<vertex_t>{d},
                                                    do_expensive_check);
      auto pair_first      = thrust::make_zip_iterator(srcs.begin(), dsts.begin());
      thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + srcs.size());
    }
    edge_list.insert(srcs.begin(),
                     srcs.end(),
                     dsts.begin(),
                     indices ? std::make_optional(indices->begin()) : std::nullopt);

    transform_e(
      handle,
      graph_view,
      edge_list,
      src_properties.view(),
      dst_properties.view(),
      centralities_view,
      [d] __device__(auto src, auto dst, auto src_props, auto dst_props, auto edge_centrality) {
        if ((cuda::std::get<0>(dst_props) == d) && (cuda::std::get<0>(src_props) == (d - 1))) {
          auto sigma_v = static_cast<weight_t>(cuda::std::get<1>(src_props));
          auto sigma_w = static_cast<weight_t>(cuda::std::get<1>(dst_props));
          auto delta_w = cuda::std::get<2>(dst_props);

          return edge_centrality + (sigma_v / sigma_w) * (1 + delta_w);
        } else {
          return edge_centrality;
        }
      },
      centralities_view,
      do_expensive_check);

    per_v_transform_reduce_outgoing_e(
      handle,
      graph_view,
      src_properties.view(),
      dst_properties.view(),
      cugraph::edge_dummy_property_t{}.view(),
      [d] __device__(auto, auto, auto src_props, auto dst_props, auto) {
        if ((cuda::std::get<0>(dst_props) == d) && (cuda::std::get<0>(src_props) == (d - 1))) {
          auto sigma_v = static_cast<weight_t>(cuda::std::get<1>(src_props));
          auto sigma_w = static_cast<weight_t>(cuda::std::get<1>(dst_props));
          auto delta_w = cuda::std::get<2>(dst_props);

          return (sigma_v / sigma_w) * (1 + delta_w);
        } else {
          return weight_t{0};
        }
      },
      weight_t{0},
      reduce_op::plus<weight_t>{},
      deltas.begin(),
      do_expensive_check);

    update_edge_src_property(
      handle,
      graph_view,
      thrust::make_zip_iterator(distances.begin(), sigmas.begin(), deltas.begin()),
      src_properties.mutable_view());
    update_edge_dst_property(
      handle,
      graph_view,
      thrust::make_zip_iterator(distances.begin(), sigmas.begin(), deltas.begin()),
      dst_properties.mutable_view());
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename VertexIterator>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<edge_t>> multisource_bfs(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  VertexIterator vertex_first,
  VertexIterator vertex_last,
  bool do_expensive_check)
{
  constexpr vertex_t invalid_distance = std::numeric_limits<vertex_t>::max();
  constexpr size_t bucket_idx_cur     = 0;
  constexpr size_t bucket_idx_next    = 1;
  constexpr size_t num_buckets        = 2;

  using origin_t = uint32_t;  // Source index type

  // Check that number of sources doesn't overflow origin_t
  CUGRAPH_EXPECTS(
    cuda::std::distance(vertex_first, vertex_last) <= std::numeric_limits<origin_t>::max(),
    "Number of sources exceeds maximum value for origin_t (uint32_t), would cause overflow");

  // Use 2D arrays to track per-source distances and sigmas
  // Layout: [source_idx * num_vertices + vertex_idx]
  auto num_vertices = graph_view.local_vertex_partition_range_size();
  auto num_sources  = cuda::std::distance(vertex_first, vertex_last);

  rmm::device_uvector<edge_t> sigmas_2d(num_sources * num_vertices, handle.get_stream());
  rmm::device_uvector<vertex_t> distances_2d(num_sources * num_vertices, handle.get_stream());
  detail::scalar_fill(handle, sigmas_2d.data(), sigmas_2d.size(), edge_t{0});
  detail::scalar_fill(handle, distances_2d.data(), distances_2d.size(), invalid_distance);

  // Create tagged frontier with origin indices
  vertex_frontier_t<vertex_t, origin_t, multi_gpu, true> vertex_frontier(handle, num_buckets);

  auto vertex_partition =
    vertex_partition_device_view_t<vertex_t, multi_gpu>(graph_view.local_vertex_partition_view());

  // Initialize sources with their origins using zip iterator approach
  if (num_sources > 0) {
    // Create zip iterator for (vertex, origin) pairs
    auto pair_first =
      thrust::make_zip_iterator(vertex_first, thrust::make_counting_iterator(origin_t{0}));
    auto pair_last = pair_first + num_sources;

    // Insert tagged sources into frontier
    vertex_frontier.bucket(bucket_idx_cur).insert(pair_first, pair_last);

    // Initialize distances and sigmas for sources
    thrust::for_each(handle.get_thrust_policy(),
                     pair_first,
                     pair_last,
                     [d_sigma_2d    = sigmas_2d.begin(),
                      d_distance_2d = distances_2d.begin(),
                      vertex_partition,
                      num_vertices] __device__(auto tagged_source) {
                       auto v      = thrust::get<0>(tagged_source);
                       auto origin = thrust::get<1>(tagged_source);
                       auto offset =
                         vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
                       auto idx           = origin * num_vertices + offset;
                       d_distance_2d[idx] = 0;
                       d_sigma_2d[idx]    = 1;
                     });
  }

  edge_t hop{0};

  while (vertex_frontier.bucket(bucket_idx_cur).aggregate_size() > 0) {
    // Step 1: Extract ALL edges from frontier (filtered by unvisited vertices)
    using bfs_edge_tuple_t = thrust::tuple<vertex_t, origin_t, edge_t>;

    auto new_frontier_tagged_vertex_buffer = extract_transform_if_v_frontier_outgoing_e(
      handle,
      graph_view,
      vertex_frontier.bucket(bucket_idx_cur),
      edge_src_dummy_property_t{}.view(),
      edge_dst_dummy_property_t{}.view(),
      edge_dummy_property_t{}.view(),
      cuda::proclaim_return_type<bfs_edge_tuple_t>(
        [d_sigma_2d = sigmas_2d.begin(), num_vertices, vertex_partition] __device__(
          auto tagged_src, auto dst, auto, auto, auto) {
          auto src        = thrust::get<0>(tagged_src);
          auto origin     = thrust::get<1>(tagged_src);
          auto src_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(src);
          auto src_idx    = origin * num_vertices + src_offset;
          auto src_sigma  = static_cast<edge_t>(d_sigma_2d[src_idx]);

          return thrust::make_tuple(dst, origin, src_sigma);
        }),
      // PREDICATE: only process edges to unvisited vertices
      cuda::proclaim_return_type<bool>(
        [d_distances_2d = distances_2d.begin(),
         num_vertices,
         vertex_partition,
         invalid_distance] __device__(auto tagged_src, auto dst, auto, auto, auto) {
          auto origin     = thrust::get<1>(tagged_src);
          auto dst_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(dst);
          auto dst_idx    = origin * num_vertices + dst_offset;
          return d_distances_2d[dst_idx] == invalid_distance;
        }));

    // Access individual vectors directly to avoid transform iterator issues
    auto& frontier_vertices = std::get<0>(new_frontier_tagged_vertex_buffer);
    auto& frontier_origins  = std::get<1>(new_frontier_tagged_vertex_buffer);
    auto& sigmas            = std::get<2>(new_frontier_tagged_vertex_buffer);

    // Step 2: Reduce by (destination, origin) - sums sigmas for multiple paths
    // Sort by (destination, origin) pairs
    thrust::sort_by_key(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(frontier_vertices.begin(), frontier_origins.begin()),
      thrust::make_zip_iterator(frontier_vertices.end(), frontier_origins.end()),
      sigmas.begin());

    // Reduce by key to sum sigmas for identical (destination, origin) pairs
    auto num_unique = thrust::unique_count(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(frontier_vertices.begin(), frontier_origins.begin()),
      thrust::make_zip_iterator(frontier_vertices.end(), frontier_origins.end()),
      [] __device__(auto const& a, auto const& b) { return a == b; });

    // Use in-place reduction to avoid temporaries
    auto reduced_result = thrust::reduce_by_key(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(frontier_vertices.begin(), frontier_origins.begin()),
      thrust::make_zip_iterator(frontier_vertices.end(), frontier_origins.end()),
      sigmas.begin(),
      thrust::make_zip_iterator(frontier_vertices.begin(),
                                frontier_origins.begin()),  // Output keys (overwrite input)
      sigmas.begin(),                                       // Output values (overwrite input)
      thrust::equal_to<thrust::tuple<vertex_t, origin_t>>{},
      thrust::plus<edge_t>{});

    // Step 3: Manual array updates using in-place reduced data
    // Get count from the values output since keys output is a zip iterator
    size_t num_reduced = thrust::distance(sigmas.begin(), reduced_result.second);
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_zip_iterator(
                       frontier_vertices.begin(), frontier_origins.begin(), sigmas.begin()),
                     thrust::make_zip_iterator(frontier_vertices.begin() + num_reduced,
                                               frontier_origins.begin() + num_reduced,
                                               sigmas.begin() + num_reduced),
                     [d_distances_2d = distances_2d.begin(),
                      d_sigmas_2d    = sigmas_2d.begin(),
                      num_vertices,
                      hop,
                      vertex_partition] __device__(auto tuple) {
                       auto v      = thrust::get<0>(tuple);
                       auto origin = thrust::get<1>(tuple);
                       auto sigma  = thrust::get<2>(tuple);
                       auto offset =
                         vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
                       auto idx = origin * num_vertices + offset;

                       // Direct assignment - no atomics needed because reduction already handled
                       // duplicates
                       d_distances_2d[idx] = hop + 1;
                       d_sigmas_2d[idx]    = sigma;
                     });

    // Step 4: Update frontier for next iteration using in-place reduced data
    vertex_frontier.bucket(bucket_idx_cur).clear();
    vertex_frontier.bucket(bucket_idx_next)
      .insert(thrust::make_zip_iterator(frontier_vertices.begin(), frontier_origins.begin()),
              thrust::make_zip_iterator(frontier_vertices.begin() + num_reduced,
                                        frontier_origins.begin() + num_reduced));

    vertex_frontier.swap_buckets(bucket_idx_cur, bucket_idx_next);
    ++hop;
  }

  return std::make_tuple(std::move(distances_2d), std::move(sigmas_2d));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename VertexIterator>
void multisource_backward_pass(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  raft::device_span<weight_t> centralities,
  rmm::device_uvector<vertex_t>&& distances_2d,
  rmm::device_uvector<edge_t>&& sigmas_2d,
  VertexIterator sources_first,
  VertexIterator sources_last,
  bool include_endpoints,
  bool do_expensive_check)
{
  auto num_vertices = static_cast<size_t>(graph_view.local_vertex_partition_range_size());
  auto num_sources  = cuda::std::distance(sources_first, sources_last);

  using origin_t = uint32_t;  // Source index type

  // Initialize centrality array to zero
  thrust::fill(handle.get_thrust_policy(), centralities.begin(), centralities.end(), weight_t{0});

  // Allocate delta accumulation buffer for all sources
  // This tracks accumulated deltas per vertex per source: deltas[source_idx][vertex]
  rmm::device_uvector<weight_t> delta_buffer(num_sources * num_vertices, handle.get_stream());
  thrust::fill(handle.get_thrust_policy(), delta_buffer.begin(), delta_buffer.end(), weight_t{0});

  // Find global maximum distance across all sources
  constexpr vertex_t invalid_distance = std::numeric_limits<vertex_t>::max();

  auto d_first = thrust::make_transform_iterator(
    distances_2d.begin(),
    cuda::proclaim_return_type<vertex_t>(
      [invalid_distance] __device__(auto d) { return d == invalid_distance ? vertex_t{0} : d; }));
  vertex_t global_max_distance = thrust::reduce(handle.get_thrust_policy(),
                                                d_first,
                                                d_first + distances_2d.size(),
                                                vertex_t{0},
                                                thrust::maximum<vertex_t>());

  // PRE-COMPUTE: Partition all (vertex, source) pairs by distance ONCE
  // This eliminates the need to scan the distance array global_max_distance times

  // Create buckets for each distance level
  std::vector<rmm::device_uvector<vertex_t>> distance_buckets_vertices;
  std::vector<rmm::device_uvector<origin_t>> distance_buckets_sources;

  // Reserve space and create empty buckets
  distance_buckets_vertices.reserve(global_max_distance + 1);
  distance_buckets_sources.reserve(global_max_distance + 1);

  for (vertex_t d = 0; d <= global_max_distance; ++d) {
    distance_buckets_vertices.emplace_back(0, handle.get_stream());
    distance_buckets_sources.emplace_back(0, handle.get_stream());
  }

  // Count vertices at each distance level first
  rmm::device_uvector<size_t> distance_counts(global_max_distance + 1, handle.get_stream());
  thrust::fill(
    handle.get_thrust_policy(), distance_counts.begin(), distance_counts.end(), size_t{0});

  // Step 1: Count vertices at each distance level
  thrust::for_each_n(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator<size_t>(0),
    num_sources * num_vertices,
    [distances_2d = distances_2d.data(),
     num_vertices,
     distance_counts = distance_counts.data(),
     global_max_distance] __device__(size_t global_idx) {
      size_t source_idx         = global_idx / num_vertices;
      vertex_t v_offset         = global_idx % num_vertices;
      const vertex_t* distances = distances_2d + source_idx * num_vertices;
      vertex_t dist             = distances[v_offset];

      if (dist >= 0 && dist <= global_max_distance) {
        cuda::atomic_ref<size_t, cuda::thread_scope_device> counter(distance_counts[dist]);
        counter.fetch_add(size_t{1}, cuda::std::memory_order_relaxed);
      }
    });

  // Copy counts to host and allocate buckets
  std::vector<size_t> host_distance_counts(global_max_distance + 1);
  raft::update_host(host_distance_counts.data(),
                    distance_counts.data(),
                    global_max_distance + 1,
                    handle.get_stream());
  handle.sync_stream();

  // Allocate exact-sized buckets
  for (vertex_t d = 0; d <= global_max_distance; ++d) {
    if (host_distance_counts[d] > 0) {
      distance_buckets_vertices[d].resize(host_distance_counts[d], handle.get_stream());
      distance_buckets_sources[d].resize(host_distance_counts[d], handle.get_stream());
    }
  }

  // Reset counts for use as offsets
  thrust::fill(
    handle.get_thrust_policy(), distance_counts.begin(), distance_counts.end(), size_t{0});

  auto v_first = graph_view.local_vertex_partition_range_first();

  // Create arrays of raw pointers for device access
  std::vector<vertex_t*> host_bucket_vertices_ptrs(global_max_distance + 1);
  std::vector<origin_t*> host_bucket_sources_ptrs(global_max_distance + 1);

  for (vertex_t d = 0; d <= global_max_distance; ++d) {
    host_bucket_vertices_ptrs[d] = distance_buckets_vertices[d].data();
    host_bucket_sources_ptrs[d]  = distance_buckets_sources[d].data();
  }

  // Copy pointer arrays to device
  rmm::device_uvector<vertex_t*> device_bucket_vertices_ptrs(global_max_distance + 1,
                                                             handle.get_stream());
  rmm::device_uvector<origin_t*> device_bucket_sources_ptrs(global_max_distance + 1,
                                                            handle.get_stream());

  raft::update_device(device_bucket_vertices_ptrs.data(),
                      host_bucket_vertices_ptrs.data(),
                      global_max_distance + 1,
                      handle.get_stream());
  raft::update_device(device_bucket_sources_ptrs.data(),
                      host_bucket_sources_ptrs.data(),
                      global_max_distance + 1,
                      handle.get_stream());

  // Ensure pointer arrays are copied before kernel launch
  handle.sync_stream();

  // Populate buckets - single scan of distance array
  thrust::for_each_n(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator<size_t>(0),
    num_sources * num_vertices,
    [distances_2d = distances_2d.data(),
     num_vertices,
     distance_counts      = distance_counts.data(),
     bucket_vertices_ptrs = device_bucket_vertices_ptrs.data(),
     bucket_sources_ptrs  = device_bucket_sources_ptrs.data(),
     v_first,
     global_max_distance] __device__(size_t global_idx) {
      size_t source_idx         = global_idx / num_vertices;
      vertex_t v_offset         = global_idx % num_vertices;
      const vertex_t* distances = distances_2d + source_idx * num_vertices;
      vertex_t dist             = distances[v_offset];

      if (dist >= 0 && dist <= global_max_distance) {
        cuda::atomic_ref<size_t, cuda::thread_scope_device> counter(distance_counts[dist]);
        size_t offset = counter.fetch_add(size_t{1}, cuda::std::memory_order_relaxed);
        bucket_vertices_ptrs[dist][offset] = v_first + v_offset;
        bucket_sources_ptrs[dist][offset]  = source_idx;
      }
    });

  // Process distance levels using pre-computed buckets
  for (vertex_t d = global_max_distance; d > 1; --d) {
    // Step 1: Create vertex frontier with all vertices at distance d-1 for all sources
    // Use tagged vertices with (vertex, source_idx) pairs
    using tagged_vertex_t = thrust::tuple<vertex_t, size_t>;

    // Get vertices at distance d-1 from pre-computed buckets (O(1) lookup)
    auto& frontier_vertices            = distance_buckets_vertices[d - 1];
    auto& frontier_sources             = distance_buckets_sources[d - 1];
    size_t total_vertices_at_d_minus_1 = frontier_vertices.size();

    if (total_vertices_at_d_minus_1 > 0) {
      // Step 3: Use extract_transform_if_v_frontier_e to enumerate all qualifying edges
      // This extracts (src, tag, dst) triplets as recommended

      // Create a proper frontier object for the tagged vertices
      vertex_frontier_t<vertex_t, origin_t, multi_gpu, true> frontier(handle, 1);

      // Insert tagged vertices directly using zip iterator (no temporary needed)
      auto pair_first =
        thrust::make_zip_iterator(frontier_vertices.begin(), frontier_sources.begin());
      frontier.bucket(0).insert(pair_first, pair_first + frontier_vertices.size());

      auto edge_tuples_buffer = extract_transform_if_v_frontier_outgoing_e(
        handle,
        graph_view,
        frontier.bucket(0),
        edge_src_dummy_property_t{}.view(),
        edge_dst_dummy_property_t{}.view(),
        edge_dummy_property_t{}.view(),
        cuda::proclaim_return_type<thrust::tuple<vertex_t, origin_t, vertex_t, weight_t>>(
          [d,
           distances_2d = distances_2d.data(),
           sigmas_2d    = sigmas_2d.data(),
           delta_buffer = delta_buffer.data(),
           num_vertices,
           invalid_distance,
           v_first] __device__(auto tagged_src, auto dst, auto, auto, auto) {
            auto src        = thrust::get<0>(tagged_src);
            auto source_idx = thrust::get<1>(tagged_src);

            // Calculate delta using Brandes formula with accumulated deltas
            const vertex_t* distances = distances_2d + source_idx * num_vertices;
            const edge_t* sigmas      = sigmas_2d + source_idx * num_vertices;
            const weight_t* deltas    = delta_buffer + source_idx * num_vertices;

            auto src_offset = src - v_first;
            auto dst_offset = dst - v_first;

            // Calculate delta using Brandes formula with accumulated deltas
            auto sigma_v = static_cast<weight_t>(sigmas[src_offset]);
            auto sigma_w = static_cast<weight_t>(sigmas[dst_offset]);

            // Get accumulated delta for destination vertex
            weight_t delta_w = deltas[dst_offset];
            weight_t delta   = (sigma_v / sigma_w) * (1 + delta_w);

            return thrust::make_tuple(src, source_idx, dst, delta);
          }),
        // PREDICATE: only process edges where dst is at distance d
        cuda::proclaim_return_type<bool>(
          [d, distances_2d = distances_2d.data(), num_vertices, v_first] __device__(
            auto tagged_src, auto dst, auto, auto, auto) {
            auto source_idx           = thrust::get<1>(tagged_src);
            const vertex_t* distances = distances_2d + source_idx * num_vertices;
            auto dst_offset           = dst - v_first;
            return distances[dst_offset] == d;
          }));

      // Work directly with the result buffer (no temporaries needed)
      size_t num_edges = size_dataframe_buffer(edge_tuples_buffer);

      if (num_edges > 0) {
        // Access individual vectors directly to avoid transform iterator issues
        auto& srcs           = std::get<0>(edge_tuples_buffer);
        auto& source_indices = std::get<1>(edge_tuples_buffer);
        auto& dsts           = std::get<2>(edge_tuples_buffer);
        auto& deltas         = std::get<3>(edge_tuples_buffer);

        // Step 4: Sort using (src, source_index) as composite key for efficient reduction
        thrust::stable_sort_by_key(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(srcs.begin(), source_indices.begin()),  // Composite key
          thrust::make_zip_iterator(srcs.end(), source_indices.end()),
          deltas.begin());  // Values to sort

        // Step 5: Use reduce_by_key with in-place reduction (no temporaries needed)

        // Reduce by key and get count in one operation - overwrite input buffers
        auto reduced_result = thrust::reduce_by_key(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(srcs.begin(), source_indices.begin()),
          thrust::make_zip_iterator(srcs.end(), source_indices.end()),
          deltas.begin(),
          thrust::make_zip_iterator(srcs.begin(),
                                    source_indices.begin()),      // Output keys (overwrite input)
          deltas.begin(),                                         // Output values (overwrite input)
          thrust::equal_to<thrust::tuple<vertex_t, origin_t>>{},  // BinaryPredicate: compare keys
          thrust::plus<weight_t>{});                              // BinaryFunction: sum values

        // Get num_unique from the result
        size_t num_unique = reduced_result.second - deltas.begin();

        // Step 6: Update centralities and deltas from the in-place reduced results
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator<size_t>(0),
          thrust::make_counting_iterator<size_t>(num_unique),
          [srcs           = srcs.data(),
           source_indices = source_indices.data(),
           deltas         = deltas.data(),
           centralities   = centralities.data(),
           delta_buffer   = delta_buffer.data(),
           num_vertices,
           v_first = graph_view.local_vertex_partition_range_first()] __device__(size_t i) {
            auto src        = srcs[i];
            auto source_idx = source_indices[i];
            auto delta      = deltas[i];

            // Update centrality using atomic for floating point
            auto src_offset = src - v_first;
            cuda::atomic_ref<weight_t, cuda::thread_scope_device> centrality_counter(
              centralities[src_offset]);
            centrality_counter.fetch_add(delta, cuda::std::memory_order_relaxed);

            // Accumulate delta for next iteration using atomic for floating point
            weight_t* source_deltas = delta_buffer + source_idx * num_vertices;
            cuda::atomic_ref<weight_t, cuda::thread_scope_device> delta_counter(
              source_deltas[src_offset]);
            delta_counter.fetch_add(delta, cuda::std::memory_order_relaxed);
          });
      }
    }
  }

  // Handle source and destination vertex contributions if include_endpoints is true
  if (include_endpoints) {
    auto v_first = graph_view.local_vertex_partition_range_first();

    // Create small temporary buffer for source vertex access (needed for 2D array indexing)
    rmm::device_uvector<vertex_t> sources_buffer(num_sources, handle.get_stream());
    thrust::copy(handle.get_thrust_policy(), sources_first, sources_last, sources_buffer.begin());

    // Handle source vertex contributions
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(num_sources),
      [distances_2d = distances_2d.data(),
       sigmas_2d    = sigmas_2d.data(),
       sources      = sources_buffer.data(),
       centralities = centralities.data(),
       num_vertices,
       v_first] __device__(size_t source_idx) {
        const vertex_t* distances = distances_2d + source_idx * num_vertices;
        const edge_t* sigmas      = sigmas_2d + source_idx * num_vertices;
        vertex_t source_vertex    = sources[source_idx];

        // Source vertex contribution: count of reachable vertices (excluding self)
        weight_t source_contribution = 0;
        for (vertex_t v = 0; v < num_vertices; ++v) {
          if (v != source_vertex && distances[v] != std::numeric_limits<vertex_t>::max()) {
            source_contribution += 1.0;
          }
        }
        // Convert global vertex ID to local offset
        auto source_offset = source_vertex - v_first;
        cuda::atomic_ref<weight_t, cuda::thread_scope_device> centrality_counter(
          centralities[source_offset]);
        centrality_counter.fetch_add(source_contribution, cuda::std::memory_order_relaxed);
      });

    // Handle destination vertex contributions
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(num_sources),
      [distances_2d = distances_2d.data(),
       sigmas_2d    = sigmas_2d.data(),
       sources      = sources_buffer.data(),
       centralities = centralities.data(),
       num_vertices,
       v_first] __device__(size_t source_idx) {
        const vertex_t* distances = distances_2d + source_idx * num_vertices;
        const edge_t* sigmas      = sigmas_2d + source_idx * num_vertices;
        vertex_t source_vertex    = sources[source_idx];

        // Destination vertex contributions: each reachable vertex contributes to its own centrality
        for (vertex_t v = 0; v < num_vertices; ++v) {
          if (v != source_vertex && distances[v] != std::numeric_limits<vertex_t>::max()) {
            // Each destination vertex contributes 1 to its own centrality
            auto dest_offset = v - v_first;
            cuda::atomic_ref<weight_t, cuda::thread_scope_device> centrality_counter(
              centralities[dest_offset]);
            centrality_counter.fetch_add(1.0, cuda::std::memory_order_relaxed);
          }
        }
      });
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename VertexIterator>
rmm::device_uvector<weight_t> betweenness_centrality(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  VertexIterator vertices_begin,
  VertexIterator vertices_end,
  bool const normalized,
  bool const include_endpoints,
  bool const do_expensive_check)
{
  //
  // Betweenness Centrality algorithm based on the Brandes Algorithm (2001)
  //
  if (do_expensive_check) {
    auto vertex_partition =
      vertex_partition_device_view_t<vertex_t, multi_gpu>(graph_view.local_vertex_partition_view());
    auto num_invalid_vertices =
      thrust::count_if(handle.get_thrust_policy(),
                       vertices_begin,
                       vertices_end,
                       [vertex_partition] __device__(auto val) {
                         return !(vertex_partition.is_valid_vertex(val) &&
                                  vertex_partition.in_local_vertex_partition_range_nocheck(val));
                       });
    if constexpr (multi_gpu) {
      num_invalid_vertices = host_scalar_allreduce(
        handle.get_comms(), num_invalid_vertices, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(num_invalid_vertices == 0,
                    "Invalid input argument: sources have invalid vertex IDs.");
  }

  rmm::device_uvector<weight_t> centralities(graph_view.local_vertex_partition_range_size(),
                                             handle.get_stream());
  detail::scalar_fill(handle, centralities.data(), centralities.size(), weight_t{0});

  size_t num_sources = cuda::std::distance(vertices_begin, vertices_end);
  std::vector<size_t> source_offsets{{0, num_sources}};
  int my_rank = 0;

  if constexpr (multi_gpu) {
    auto source_counts =
      host_scalar_allgather(handle.get_comms(), num_sources, handle.get_stream());

    num_sources = std::accumulate(source_counts.begin(), source_counts.end(), 0);
    source_offsets.resize(source_counts.size() + 1);
    source_offsets[0] = 0;
    std::inclusive_scan(source_counts.begin(), source_counts.end(), source_offsets.begin() + 1);
    my_rank = handle.get_comms().get_rank();
  }

  // Initialize memory management and measurement
  MemoryInfo initial_mem = MemoryInfo::get_device_memory();
  initial_mem.print("Before BFS processing");

  AdaptiveMemoryManager memory_manager(100, 10, 10000, 0.85);
  size_t processed_sources = 0;
  size_t batch_number      = 0;

  if constexpr (multi_gpu) {
    // Multi-GPU: Use sequential brandes_bfs (more reliable for cross-GPU)
    printf("[DEBUG] Running SEQUENTIAL version (multi-GPU mode)\n");

    // Process each source individually using brandes_bfs
    for (size_t source_idx = 0; source_idx < num_sources; ++source_idx) {
      //
      //  BFS
      //
      constexpr size_t bucket_idx_cur = 0;
      constexpr size_t num_buckets    = 2;

      vertex_frontier_t<vertex_t, void, multi_gpu, true> vertex_frontier(handle, num_buckets);

      if ((source_idx >= source_offsets[my_rank]) && (source_idx < source_offsets[my_rank + 1])) {
        vertex_frontier.bucket(bucket_idx_cur)
          .insert(vertices_begin + (source_idx - source_offsets[my_rank]),
                  vertices_begin + (source_idx - source_offsets[my_rank]) + 1);
      }

      auto [distances, sigmas] = detail::brandes_bfs(
        handle, graph_view, edge_weight_view, vertex_frontier, do_expensive_check);
      detail::accumulate_vertex_results(
        handle,
        graph_view,
        edge_weight_view,
        raft::device_span<weight_t>{centralities.data(), centralities.size()},
        std::move(distances),
        std::move(sigmas),
        include_endpoints,
        do_expensive_check);
    }
  } else {
    // Single-GPU: Use parallel multisource_bfs with adaptive batching
    printf("[DEBUG] Running MULTISOURCE version (single-GPU mode) with adaptive batching\n");

    while (processed_sources < num_sources) {
      size_t current_batch_size = memory_manager.get_batch_size();
      size_t remaining_sources  = num_sources - processed_sources;
      size_t actual_batch_size  = std::min(current_batch_size, remaining_sources);

      printf("[Attempt] Processing %zu sources (processed: %zu/%zu, remaining: %zu)\n",
             actual_batch_size,
             processed_sources,
             num_sources,
             remaining_sources);
      printf("[Stage: %s]\n",
             (memory_manager.get_stage() == MemoryStage::RAMP_UP)       ? "RAMP_UP"
             : (memory_manager.get_stage() == MemoryStage::OSCILLATION) ? "OSCILLATION"
             : (memory_manager.get_stage() == MemoryStage::FINE_TUNE)   ? "FINE_TUNE"
                                                                        : "CONVERGED");

      bool batch_success          = true;
      MemoryInfo mem_before_batch = MemoryInfo::get_device_memory();

      try {
        // Create iterators for current batch
        auto batch_begin = vertices_begin + processed_sources;
        auto batch_end   = vertices_begin + processed_sources + actual_batch_size;

        auto [distances_2d, sigmas_2d] = detail::multisource_bfs(
          handle, graph_view, edge_weight_view, batch_begin, batch_end, do_expensive_check);

        // Use parallel multisource backward pass for better performance
        detail::multisource_backward_pass(
          handle,
          graph_view,
          edge_weight_view,
          raft::device_span<weight_t>{centralities.data(), centralities.size()},
          std::move(distances_2d),
          std::move(sigmas_2d),
          batch_begin,
          batch_end,
          include_endpoints,
          do_expensive_check);

        processed_sources += actual_batch_size;
        batch_success = true;
        batch_number++;  // Only increment on success

      } catch (const std::exception& e) {
        printf("[ERROR] Attempt failed: %s\n", e.what());
        batch_success = false;
      }

      // Update memory manager and cleanup
      MemoryInfo mem_after_batch = MemoryInfo::get_device_memory();
      {
        std::string label =
          batch_success
            ? "After processing " + std::to_string(actual_batch_size) + " sources"
            : "After failed attempt to process " + std::to_string(actual_batch_size) + " sources";
        mem_after_batch.print(label.c_str());
      }

      // Record batch result first (handles OOMs immediately)
      memory_manager.record_batch_result(batch_success);

      // Then update stage logic
      memory_manager.update_batch_size(batch_success, mem_after_batch);

      // Print clean, separated stats
      printf("\n");
      memory_manager.print_stats();
      printf("\n");

      // Cleanup batch memory
      MemoryCleanup::cleanup_batch_memory(handle);

      // Check if we need to throttle due to memory pressure
      double memory_usage =
        static_cast<double>(mem_after_batch.used_memory) / mem_after_batch.total_memory;
      if (memory_usage > 0.95) {
        printf("[WARNING] High memory usage (%.1f%%), forcing cleanup\n", memory_usage * 100.0);
        MemoryCleanup::cleanup_test_memory(handle);
      }
    }

    printf("[Memory] Completed %zu sources in %zu batches\n", processed_sources, batch_number);
  }

  // Final memory cleanup and measurement
  MemoryCleanup::cleanup_test_memory(handle);
  MemoryInfo final_mem = MemoryInfo::get_device_memory();
  final_mem.print("After completion");

  printf("[Memory] Memory change: %.1fGB -> %.1fGB (delta: %.1fGB)\n",
         initial_mem.used_memory / (1024.0 * 1024.0 * 1024.0),
         final_mem.used_memory / (1024.0 * 1024.0 * 1024.0),
         (final_mem.used_memory - initial_mem.used_memory) / (1024.0 * 1024.0 * 1024.0));

  std::optional<weight_t> scale_nonsource{std::nullopt};
  std::optional<weight_t> scale_source{std::nullopt};

  weight_t num_vertices = static_cast<weight_t>(graph_view.number_of_vertices());
  if (!include_endpoints) num_vertices = num_vertices - 1;

  if ((static_cast<edge_t>(num_sources) == num_vertices) || include_endpoints) {
    if (normalized) {
      scale_nonsource = static_cast<weight_t>(num_sources * (num_vertices - 1));
    } else if (graph_view.is_symmetric()) {
      scale_nonsource =
        static_cast<weight_t>(num_sources * 2) / static_cast<weight_t>(num_vertices);
    } else {
      scale_nonsource = static_cast<weight_t>(num_sources) / static_cast<weight_t>(num_vertices);
    }

    scale_source = scale_nonsource;
  } else if (normalized) {
    scale_nonsource = static_cast<weight_t>(num_sources) * (num_vertices - 1);
    scale_source    = static_cast<weight_t>(num_sources - 1) * (num_vertices - 1);
  } else {
    scale_nonsource = static_cast<weight_t>(num_sources) / num_vertices;
    scale_source    = static_cast<weight_t>(num_sources - 1) / num_vertices;

    if (graph_view.is_symmetric()) {
      *scale_nonsource *= 2;
      *scale_source *= 2;
    }
  }

  if (scale_nonsource) {
    auto iter = thrust::make_zip_iterator(
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
      centralities.begin());

    thrust::transform(
      handle.get_thrust_policy(),
      iter,
      iter + centralities.size(),
      centralities.begin(),
      [nonsource = *scale_nonsource,
       source    = *scale_source,
       vertices_begin,
       vertices_end] __device__(auto t) {
        vertex_t v          = cuda::std::get<0>(t);
        weight_t centrality = cuda::std::get<1>(t);

        return (thrust::find(thrust::seq, vertices_begin, vertices_end, v) == vertices_end)
                 ? centrality / nonsource
                 : centrality / source;
      });
  }

  return centralities;
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename VertexIterator>
edge_property_t<edge_t, weight_t> edge_betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  VertexIterator vertices_begin,
  VertexIterator vertices_end,
  bool const normalized,
  bool const do_expensive_check)
{
  //
  // Betweenness Centrality algorithm based on the Brandes Algorithm (2001)
  //
  if (do_expensive_check) {
    auto vertex_partition =
      vertex_partition_device_view_t<vertex_t, multi_gpu>(graph_view.local_vertex_partition_view());
    auto num_invalid_vertices =
      thrust::count_if(handle.get_thrust_policy(),
                       vertices_begin,
                       vertices_end,
                       [vertex_partition] __device__(auto val) {
                         return !(vertex_partition.is_valid_vertex(val) &&
                                  vertex_partition.in_local_vertex_partition_range_nocheck(val));
                       });
    if constexpr (multi_gpu) {
      num_invalid_vertices = host_scalar_allreduce(
        handle.get_comms(), num_invalid_vertices, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(num_invalid_vertices == 0,
                    "Invalid input argument: sources have invalid vertex IDs.");
  }

  edge_property_t<edge_t, weight_t> centralities(handle, graph_view);

  if (graph_view.has_edge_mask()) {
    auto unmasked_graph_view = graph_view;
    unmasked_graph_view.clear_edge_mask();
    fill_edge_property(
      handle, unmasked_graph_view, centralities.mutable_view(), weight_t{0}, do_expensive_check);
  } else {
    fill_edge_property(
      handle, graph_view, centralities.mutable_view(), weight_t{0}, do_expensive_check);
  }

  size_t num_sources = cuda::std::distance(vertices_begin, vertices_end);
  std::vector<size_t> source_offsets{{0, num_sources}};
  int my_rank = 0;

  if constexpr (multi_gpu) {
    auto source_counts =
      host_scalar_allgather(handle.get_comms(), num_sources, handle.get_stream());

    num_sources = std::accumulate(source_counts.begin(), source_counts.end(), 0);
    source_offsets.resize(source_counts.size() + 1);
    source_offsets[0] = 0;
    std::inclusive_scan(source_counts.begin(), source_counts.end(), source_offsets.begin() + 1);
    my_rank = handle.get_comms().get_rank();
  }

  //
  // FIXME: This could be more efficient using something akin to the
  // technique in WCC.  Take the entire set of sources, insert them into
  // a tagged frontier (tagging each source with itself).  Then we can
  // expand from multiple sources concurrently. The challenge is managing
  // the memory explosion.
  //
  for (size_t source_idx = 0; source_idx < num_sources; ++source_idx) {
    //
    //  BFS
    //
    constexpr size_t bucket_idx_cur = 0;
    constexpr size_t num_buckets    = 2;

    vertex_frontier_t<vertex_t, void, multi_gpu, true> vertex_frontier(handle, num_buckets);

    if ((source_idx >= source_offsets[my_rank]) && (source_idx < source_offsets[my_rank + 1])) {
      vertex_frontier.bucket(bucket_idx_cur)
        .insert(vertices_begin + (source_idx - source_offsets[my_rank]),
                vertices_begin + (source_idx - source_offsets[my_rank]) + 1);
    }

    //
    //  Now we need to do modified BFS
    //
    // FIXME:  This has an inefficiency in early iterations, as it doesn't have enough work to
    //         keep the GPUs busy.  But we can't run too many at once or we will run out of
    //         memory. Need to investigate options to improve this performance
    auto [distances, sigmas] =
      brandes_bfs(handle, graph_view, edge_weight_view, vertex_frontier, do_expensive_check);
    accumulate_edge_results(handle,
                            graph_view,
                            edge_weight_view,
                            centralities.mutable_view(),
                            std::move(distances),
                            std::move(sigmas),
                            do_expensive_check);
  }

  std::optional<weight_t> scale_factor{std::nullopt};

  if (normalized) {
    weight_t n   = static_cast<weight_t>(graph_view.number_of_vertices());
    scale_factor = n * (n - 1);
  } else if (graph_view.is_symmetric()) {
    scale_factor = weight_t{2};
  }

  if (scale_factor) {
    if (graph_view.number_of_vertices() > 1) {
      if (static_cast<vertex_t>(num_sources) < graph_view.number_of_vertices()) {
        (*scale_factor) *= static_cast<weight_t>(num_sources) /
                           static_cast<weight_t>(graph_view.number_of_vertices());
      }

      auto firsts         = centralities.view().value_firsts();
      auto counts         = centralities.view().edge_counts();
      auto mutable_firsts = centralities.mutable_view().value_firsts();
      for (size_t k = 0; k < counts.size(); k++) {
        thrust::transform(
          handle.get_thrust_policy(),
          firsts[k],
          firsts[k] + counts[k],
          mutable_firsts[k],
          [sf = *scale_factor] __device__(auto centrality) { return centrality / sf; });
      }
    }
  }

  return centralities;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<vertex_t const>> vertices,
  bool const normalized,
  bool const include_endpoints,
  bool const do_expensive_check)
{
  if (vertices) {
    return detail::betweenness_centrality(handle,
                                          graph_view,
                                          edge_weight_view,
                                          vertices->begin(),
                                          vertices->end(),
                                          normalized,
                                          include_endpoints,
                                          do_expensive_check);
  } else {
    return detail::betweenness_centrality(
      handle,
      graph_view,
      edge_weight_view,
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
      normalized,
      include_endpoints,
      do_expensive_check);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
edge_property_t<edge_t, weight_t> edge_betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<vertex_t const>> vertices,
  bool const normalized,
  bool const do_expensive_check)
{
  if (vertices) {
    return detail::edge_betweenness_centrality(handle,
                                               graph_view,
                                               edge_weight_view,
                                               vertices->begin(),
                                               vertices->end(),
                                               normalized,
                                               do_expensive_check);
  } else {
    return detail::edge_betweenness_centrality(
      handle,
      graph_view,
      edge_weight_view,
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
      normalized,
      do_expensive_check);
  }
}

}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/misc_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_wrappers/fill.hpp>
#include <cugraph/utilities/thrust_wrappers/scan.hpp>
#include <cugraph/utilities/thrust_wrappers/scatter.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

template <typename edge_t, typename T>
struct check_invalid_t {
  edge_t invalid_idx{};

  __device__ bool operator()(cuda::std::tuple<edge_t, T> pair) const
  {
    return cuda::std::get<0>(pair) == invalid_idx;
  }
};

template <typename edge_t>
struct count_valids_t {
  raft::device_span<edge_t const> sample_local_nbr_indices{};
  size_t K{};
  edge_t invalid_idx{};

  __device__ int32_t operator()(size_t i) const
  {
    auto first = sample_local_nbr_indices.begin() + i * K;
    return static_cast<int32_t>(
      cuda::std::distance(first, thrust::find(thrust::seq, first, first + K, invalid_idx)));
  }
};

struct count_t {
  raft::device_span<int32_t> sample_counts{};

  __device__ size_t operator()(size_t key_idx) const
  {
    cuda::atomic_ref<int32_t, cuda::thread_scope_device> counter(sample_counts[key_idx]);
    return counter.fetch_add(int32_t{1}, cuda::std::memory_order_relaxed);
  }
};

template <bool use_invalid_value>
struct return_value_compute_offset_t {
  raft::device_span<size_t const> sample_key_indices{};
  raft::device_span<int32_t const> sample_intra_partition_displacements{};
  std::conditional_t<use_invalid_value, size_t, raft::device_span<size_t const>>
    K_or_sample_offsets{};

  __device__ size_t operator()(size_t i) const
  {
    auto key_idx = sample_key_indices[i];
    size_t key_start_offset{};
    if constexpr (use_invalid_value) {
      key_start_offset = key_idx * K_or_sample_offsets;
    } else {
      key_start_offset = K_or_sample_offsets[key_idx];
    }
    return key_start_offset + static_cast<size_t>(sample_intra_partition_displacements[i]);
  }
};

/**
 * @brief Shuffle randomly selected & transformed results and compute sample_offsets.
 *
 * Template dependence is limited to @p edge_t and output type @p T.
 */
template <typename edge_t, typename T>
std::tuple<std::optional<rmm::device_uvector<size_t>>, dataframe_buffer_type_t<T>>
finalize_random_select_output(raft::handle_t const& handle,
                              int minor_comm_size,
                              rmm::device_uvector<edge_t>& sample_local_nbr_indices,
                              dataframe_buffer_type_t<T>& sample_e_op_results,
                              std::optional<rmm::device_uvector<size_t>>& sample_key_indices,
                              raft::host_span<size_t const> local_key_list_sample_counts,
                              size_t key_list_size,
                              size_t K_sum,
                              std::optional<T> const& invalid_value);

}  // namespace detail
}  // namespace CUGRAPH_EXPORT cugraph

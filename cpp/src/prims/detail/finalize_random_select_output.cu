/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Explicit instantiations for cugraph/prims/detail/finalize_random_select_output.cuh.
 */

#include "finalize_random_select_output_impl.cuh"

#include <cugraph/export.hpp>

#include <cuda/std/tuple>

#include <cstdint>

namespace cugraph {
namespace detail {

namespace {

using finalize_output_i32_v32_v32_t = cuda::std::tuple<std::int32_t, std::int32_t>;
using finalize_output_i32_v32_v32_e32_t =
  cuda::std::tuple<std::int32_t, std::int32_t, std::int32_t>;
using finalize_output_i32_v32_f32_t = cuda::std::tuple<std::int32_t, float>;
using finalize_output_i32_v32_f64_t = cuda::std::tuple<std::int32_t, double>;

using finalize_output_i64_v64_v64_t = cuda::std::tuple<std::int64_t, std::int64_t>;
using finalize_output_i64_v64_v64_e64_t =
  cuda::std::tuple<std::int64_t, std::int64_t, std::int64_t>;
using finalize_output_i64_v64_f32_t = cuda::std::tuple<std::int64_t, float>;
using finalize_output_i64_v64_f64_t = cuda::std::tuple<std::int64_t, double>;

#define CUGRAPH_FINALIZE_RANDOM_SELECT_OUTPUT_INST(edge_t, output_t)                          \
  template CUGRAPH_EXPORT                                                                     \
    std::tuple<std::optional<rmm::device_uvector<size_t>>, dataframe_buffer_type_t<output_t>> \
    finalize_random_select_output<edge_t, output_t>(                                          \
      raft::handle_t const& handle,                                                           \
      int minor_comm_size,                                                                    \
      rmm::device_uvector<edge_t>& sample_local_nbr_indices,                                  \
      dataframe_buffer_type_t<output_t>& sample_e_op_results,                                 \
      std::optional<rmm::device_uvector<size_t>>& sample_key_indices,                         \
      raft::host_span<size_t const> local_key_list_sample_counts,                             \
      size_t key_list_size,                                                                   \
      size_t K_sum,                                                                           \
      std::optional<output_t> const& invalid_value)

}  // namespace

CUGRAPH_FINALIZE_RANDOM_SELECT_OUTPUT_INST(std::int32_t, finalize_output_i32_v32_v32_t);
CUGRAPH_FINALIZE_RANDOM_SELECT_OUTPUT_INST(std::int32_t, finalize_output_i32_v32_v32_e32_t);
CUGRAPH_FINALIZE_RANDOM_SELECT_OUTPUT_INST(std::int32_t, finalize_output_i32_v32_f32_t);
CUGRAPH_FINALIZE_RANDOM_SELECT_OUTPUT_INST(std::int32_t, finalize_output_i32_v32_f64_t);
CUGRAPH_FINALIZE_RANDOM_SELECT_OUTPUT_INST(std::int32_t, std::int32_t);

CUGRAPH_FINALIZE_RANDOM_SELECT_OUTPUT_INST(std::int64_t, finalize_output_i64_v64_v64_t);
CUGRAPH_FINALIZE_RANDOM_SELECT_OUTPUT_INST(std::int64_t, finalize_output_i64_v64_v64_e64_t);
CUGRAPH_FINALIZE_RANDOM_SELECT_OUTPUT_INST(std::int64_t, finalize_output_i64_v64_f32_t);
CUGRAPH_FINALIZE_RANDOM_SELECT_OUTPUT_INST(std::int64_t, finalize_output_i64_v64_f64_t);
CUGRAPH_FINALIZE_RANDOM_SELECT_OUTPUT_INST(std::int64_t, std::int64_t);

#undef CUGRAPH_FINALIZE_RANDOM_SELECT_OUTPUT_INST

}  // namespace detail
}  // namespace cugraph

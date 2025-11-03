/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "path_retrieval.cuh"

#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/path_retrieval.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace cugraph {

template void get_traversed_cost<int32_t, float>(raft::handle_t const& handle,
                                                 int32_t const* vertices,
                                                 int32_t const* preds,
                                                 float const* info_weights,
                                                 float* out,
                                                 int32_t stop_vertex,
                                                 int32_t num_vertices);

template void get_traversed_cost<int32_t, double>(raft::handle_t const& handle,
                                                  int32_t const* vertices,
                                                  int32_t const* preds,
                                                  double const* info_weights,
                                                  double* out,
                                                  int32_t stop_vertex,
                                                  int32_t num_vertices);

}  // namespace cugraph

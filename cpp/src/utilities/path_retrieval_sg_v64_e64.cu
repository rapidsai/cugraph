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

template void get_traversed_cost<int64_t, float>(raft::handle_t const& handle,
                                                 int64_t const* vertices,
                                                 int64_t const* preds,
                                                 float const* info_weights,
                                                 float* out,
                                                 int64_t stop_vertex,
                                                 int64_t num_vertices);

template void get_traversed_cost<int64_t, double>(raft::handle_t const& handle,
                                                  int64_t const* vertices,
                                                  int64_t const* preds,
                                                  double const* info_weights,
                                                  double* out,
                                                  int64_t stop_vertex,
                                                  int64_t num_vertices);
}  // namespace cugraph

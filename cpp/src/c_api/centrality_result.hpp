/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "c_api/array.hpp"

namespace cugraph {
namespace c_api {

struct cugraph_centrality_result_t {
  cugraph_type_erased_device_array_t* vertex_ids_{};
  cugraph_type_erased_device_array_t* values_{};
  size_t num_iterations_{0};
  bool converged_{false};
};

struct cugraph_edge_centrality_result_t {
  cugraph_type_erased_device_array_t* src_ids_{};
  cugraph_type_erased_device_array_t* dst_ids_{};
  cugraph_type_erased_device_array_t* edge_ids_{};
  cugraph_type_erased_device_array_t* values_{};
};

}  // namespace c_api
}  // namespace cugraph

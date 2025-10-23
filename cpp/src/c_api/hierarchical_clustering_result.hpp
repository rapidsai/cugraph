/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "c_api/array.hpp"

#include <cugraph_c/algorithms.h>

namespace cugraph {
namespace c_api {

struct cugraph_hierarchical_clustering_result_t {
  double modularity{0};
  cugraph_type_erased_device_array_t* vertices_{nullptr};
  cugraph_type_erased_device_array_t* clusters_{nullptr};
};

}  // namespace c_api
}  // namespace cugraph

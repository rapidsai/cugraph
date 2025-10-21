/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "c_api/array.hpp"

namespace cugraph {
namespace c_api {

struct cugraph_paths_result_t {
  cugraph_type_erased_device_array_t* vertex_ids_;
  cugraph_type_erased_device_array_t* distances_;
  cugraph_type_erased_device_array_t* predecessors_;
};

}  // namespace c_api
}  // namespace cugraph

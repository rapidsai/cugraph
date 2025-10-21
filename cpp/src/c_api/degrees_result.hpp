/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "c_api/array.hpp"

namespace cugraph {
namespace c_api {

struct cugraph_degrees_result_t {
  bool is_symmetric{false};
  cugraph_type_erased_device_array_t* vertex_ids_{};
  cugraph_type_erased_device_array_t* in_degrees_{};
  cugraph_type_erased_device_array_t* out_degrees_{};
};

}  // namespace c_api
}  // namespace cugraph

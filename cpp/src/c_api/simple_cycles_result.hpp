/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "c_api/array.hpp"

namespace cugraph {
namespace c_api {

struct cugraph_simple_cycles_result_t {
  cugraph_type_erased_device_array_t* cycle_vertices_{nullptr};
  cugraph_type_erased_device_array_t* cycle_offsets_{nullptr};
};

}  // namespace c_api
}  // namespace cugraph

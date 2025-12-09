/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "c_api/array.hpp"

namespace cugraph {
namespace c_api {

struct cugraph_core_result_t {
  cugraph_type_erased_device_array_t* vertex_ids_{};
  cugraph_type_erased_device_array_t* core_numbers_{};
};

struct cugraph_k_core_result_t {
  cugraph_type_erased_device_array_t* src_vertices_{};
  cugraph_type_erased_device_array_t* dst_vertices_{};
  cugraph_type_erased_device_array_t* weights_{};
};

}  // namespace c_api
}  // namespace cugraph

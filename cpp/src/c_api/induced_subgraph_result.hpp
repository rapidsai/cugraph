/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "c_api/array.hpp"

namespace cugraph {
namespace c_api {

struct cugraph_induced_subgraph_result_t {
  cugraph_type_erased_device_array_t* src_{};
  cugraph_type_erased_device_array_t* dst_{};
  cugraph_type_erased_device_array_t* wgt_{};
  cugraph_type_erased_device_array_t* edge_ids_{};
  cugraph_type_erased_device_array_t* edge_type_ids_{};
  cugraph_type_erased_device_array_t* subgraph_offsets_{};
};

}  // namespace c_api
}  // namespace cugraph

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "c_api/array.hpp"

namespace cugraph {
namespace c_api {

struct cugraph_vertex_pairs_t {
  cugraph_type_erased_device_array_t* first_;
  cugraph_type_erased_device_array_t* second_;
};

}  // namespace c_api
}  // namespace cugraph

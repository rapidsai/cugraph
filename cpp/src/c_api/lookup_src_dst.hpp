/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "c_api/array.hpp"
#include "c_api/error.hpp"

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>

#include <memory>

namespace cugraph {
namespace c_api {

struct cugraph_lookup_container_t {
  cugraph_data_type_id_t edge_type_;
  cugraph_data_type_id_t edge_type_id_type_;
  cugraph_data_type_id_t vertex_type_;

  void* lookup_container_;
};

struct cugraph_lookup_result_t {
  cugraph_type_erased_device_array_t* srcs_{nullptr};
  cugraph_type_erased_device_array_t* dsts_{nullptr};
};

}  // namespace c_api
}  // namespace cugraph

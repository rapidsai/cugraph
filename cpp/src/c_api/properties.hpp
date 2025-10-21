/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph_c/properties.h>

namespace cugraph {
namespace c_api {

typedef struct {
  cugraph_data_type_id_t property_type_;
  void* vertex_property_;
} cugraph_vertex_property_t;

typedef struct {
  cugraph_data_type_id_t property_type_;
  void* edge_property_;
} cugraph_edge_property_t;

typedef struct {
  cugraph_data_type_id_t property_type_;
  void* vertex_property_;
} cugraph_vertex_property_view_t;

typedef struct {
  cugraph_data_type_id_t property_type_;
  void* edge_property_;
} cugraph_edge_property_view_t;

}  // namespace c_api
}  // namespace cugraph

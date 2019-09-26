/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * ---------------------------------------------------------------------------*
 * @brief Core Number implementation
 *
 * @file core_number.cu
 * --------------------------------------------------------------------------*/

#include <cugraph.h>
#include "utilities/error_utils.h"
#include <Hornet.hpp>
#include <Static/CoreNumber/CoreNumber.cuh>
#include <rmm_utils.h>
#include <nvgraph_gdf.h>

gdf_error core_number_impl(gdf_graph *graph,
                          int *core_number) {
  using HornetGraph = hornet::gpu::HornetStatic<int>;
  using HornetInit  = hornet::HornetInit<int>;
  using CoreNumber  = hornets_nest::CoreNumberStatic;
  HornetInit init(graph->numberOfVertices, graph->adjList->indices->size,
      static_cast<int*>(graph->adjList->offsets->data),
      static_cast<int*>(graph->adjList->indices->data));
  HornetGraph hnt(init, hornet::DeviceType::DEVICE);
  CoreNumber cn(hnt, core_number);
  cn.run();
  return GDF_SUCCESS;
}

gdf_error gdf_core_number(gdf_graph *graph,
                          gdf_column *core_number) {
  GDF_REQUIRE(graph->adjList != nullptr || graph->edgeList != nullptr, GDF_INVALID_API_CALL);
  gdf_error err = gdf_add_adj_list(graph);
  if (err != GDF_SUCCESS)
    return err;
  GDF_REQUIRE(graph->adjList->offsets->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
  GDF_REQUIRE(graph->adjList->indices->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
  GDF_REQUIRE(core_number->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
  GDF_REQUIRE(core_number->size == graph->numberOfVertices, GDF_COLUMN_SIZE_MISMATCH);

  return core_number_impl(graph, static_cast<int*>(core_number->data));
}

struct LessThan {
  int k;

  LessThan(int _k) : k(_k) {}

  __host__ __device__
    bool operator()(const int core) {
      return core < k;
    }
};

gdf_error gdf_k_core(gdf_graph *in_graph,
                     int k,
                     gdf_column *vertex_id,
                     gdf_column *core_number,
                     gdf_graph *out_graph) {
  GDF_REQUIRE(out_graph != nullptr && in_graph != nullptr, GDF_INVALID_API_CALL);
  GDF_REQUIRE(in_graph->adjList != nullptr || in_graph->edgeList != nullptr, GDF_INVALID_API_CALL);
  gdf_error err = gdf_add_adj_list(in_graph);
  if (err != GDF_SUCCESS)
    return err;
  GDF_REQUIRE(in_graph->adjList->offsets->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
  GDF_REQUIRE(in_graph->adjList->indices->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
  GDF_REQUIRE((vertex_id == nullptr) == (core_number == nullptr), GDF_INVALID_API_CALL);
  if (vertex_id != nullptr) {
    GDF_REQUIRE(vertex_id->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
    GDF_REQUIRE(core_number->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
    GDF_REQUIRE(core_number->size == vertex_id->size, GDF_INVALID_API_CALL);
  }

  cudaStream_t stream {nullptr};
  gdf_size_type nV = in_graph->numberOfVertices;

  rmm::device_vector<int> vertexIdentifier;
  rmm::device_vector<int> coreNumber;

  int * vertex_identifier_ptr = nullptr;
  int * core_number_ptr = nullptr;
  gdf_size_type vLen = 0;

  if (vertex_id == nullptr) {
    vertexIdentifier.resize(nV);
    thrust::sequence(
        rmm::exec_policy(stream)->on(stream),
        vertexIdentifier.begin(), vertexIdentifier.end());
    vertex_identifier_ptr = vertexIdentifier.data().get();
    vLen = nV;

    coreNumber.resize(nV);
    core_number_ptr = coreNumber.data().get();
    err = core_number_impl(in_graph, core_number_ptr);
    GDF_REQUIRE(err, GDF_SUCCESS);
  } else {
    core_number_ptr = static_cast<int*>(core_number->data);
    vertex_identifier_ptr = static_cast<int*>(vertex_id->data);
    vLen = vertex_id->size;
  }

  auto cnPtr = thrust::device_pointer_cast(core_number_ptr);
  if (k == -1) {
    k = thrust::reduce(
        rmm::exec_policy(stream)->on(stream),
        cnPtr, cnPtr + vLen, -1, thrust::maximum<int>());
  }

  rmm::device_vector<int> filteredVertices(vLen);
  auto vInPtr = thrust::device_pointer_cast(vertex_identifier_ptr);
  auto ptr = thrust::remove_copy_if(
      rmm::exec_policy(stream)->on(stream),
      vInPtr, vInPtr + vLen,
      cnPtr,
      filteredVertices.begin(),
      LessThan(k));
  filteredVertices.resize(ptr - filteredVertices.begin());

  gdf_column vertices;
  gdf_column_view(&vertices,
                  static_cast<void*>(filteredVertices.data().get()),
                  nullptr,
                  filteredVertices.size(),
                  GDF_INT32);
  return gdf_extract_subgraph_vertex_nvgraph(in_graph, &vertices, out_graph);
}

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
/** ---------------------------------------------------------------------------*
 * @brief Wrapper functions for Nvgraph
 *
 * @file nvgraph_gdf.cu
 * ---------------------------------------------------------------------------**/

#include <cugraph.h>
#include <nvgraph/nvgraph.h>
#include <thrust/device_vector.h>
#include <ctime>
#include "utilities/error_utils.h"
#include "converters/nvgraph.cuh"
#include <rmm_utils.h>
#include "utilities/graph_utils.cuh"
#include "converters/COOtoCSR.cuh"

namespace {
struct prg {
  __host__ __device__
  float operator()(n){
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(0.0, 1.0);
    rng.discard(n);
    return dist(rng);
  }
};

template <typename IdxT>
struct permutation_functor{
  IdxT* permutation;
  permutation_functor(IdxT* p):permutation(p){}
  __host__ __device__
  IdxT operator()(IdxT in){
    return permutation[in];
  }
};

template<typename IdxT, typename ValT>
cugraph::Graph* permute_graph(cugraph::Graph* graph, IdxT* permutation) {
  // Get the source indices from the offsets
  IdxT* src_indices;
  IdxT nnz = graph->adjList->indices->size;
  ALLOC_TRY(src_indices, sizeof(IdxT) * graph->adjList->indices->size, nullptr);
  cugraph::detail::offsets_to_indices((IdxT*) graph->adjList->indices->data,
                                      graph->adjList->offsets->size - 1,
                                      src_indices);
  // Permute the src_indices
  permutation_functor<IdxT>pf(permutation);
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    src_indices,
                    src_indices + nnz,
                    src_indices,
                    pf);

  // Copy the indices and values
  IdxT* dest_indices;
  ALLOC_TRY(dest_indices, sizeof(IdxT) * nnz, nullptr);
  ValT* values;
  ALLOC_TRY(values, sizeof(ValT) * nnz, nullptr);
  thrust::copy(rmm::exec_policy(nullptr)->on(nullptr),
               (IdxT*) graph->adjList->indices->data,
               (IdxT*) graph->adjList->indices->data + nnz,
               dest_indices);
  thrust::copy(rmm::exec_policy(nullptr)->on(nullptr),
               (ValT*) graph->adjList->edge_data->data,
               (ValT*) graph->adjList->edge_data->data + nnz,
               values);

  // Permute the destination indices
  thrust::transform(rmm::exec_policy(nullptr)->on(nullptr),
                    dest_indices,
                    dest_indices + nnz,
                    dest_indices,
                    pf);

  // Call COO2CSR to get the new adjacency
  CSR_result_weighted<IdxT, ValT> new_csr;
  ConvertCOOtoCSR_weighted(src_indices, dest_indices, values, (int64_t)nnz, new_csr);

  // Construct the result graph
  cugraph::Graph* result = new cugraph::Graph;
  result->adjList = new gdf_adj_list;
  result->adjList->offsets = new gdf_column;
  result->adjList->indices = new gdf_column;
  result->adjList->edge_data = new gdf_column;
  result->adjList->ownership = 0;

  gdf_column_view(result->adjList->offsets,
                  new_csr.rowOffsets,
                  nullptr,
                  new_csr.size + 1,
                  graph->adjList->offsets->dtype);
  gdf_column_view(result->adjList->indices,
                  new_csr.colIndices,
                  nullptr,
                  nnz,
                  graph->adjList->offsets->dtype);
  gdf_column_view(result->adjList->edge_data,
                  new_csr.edgeWeights,
                  nullptr,
                  nnz,
                  graph->adjList->edge_data->dtype);

  ALLOC_FREE_TRY(src_indices, nullptr);
  ALLOC_FREE_TRY(dest_indices, nullptr);
  ALLOC_FREE_TRY(values, nullptr);

  return result;
}

template <typename IdxT>
IdxT* get_permutation_vector(IdxT size, IdxT seed) {
  IdxT* output_vector;
  ALLOC_TRY(output_vector, sizeof(IdxT) * size, nullptr);
  float* randoms;
  ALLOC_TRY(randoms, sizeof(float) * size, nullptr);

  thrust::counting_iterator<uint32_t> index(seed);
  thrust::transform(rmm::exec_policy(nullptr)->on(nullptr), index, index + size, randoms, prg());
  thrust::sequence(rmm::exec_policy(nullptr)->on(nullptr), output_vector, output_vector + size, 0);
  thrust::sort_by_key(rmm::exec_policy(nullptr)->on(nullptr), randoms, randoms + size, output_vector);

  ALLOC_FREE_TRY(randoms, nullptr);
  return output_vector;
}

template <typename IdxT, typename ValT>
void ecg_impl(cugraph::Graph* graph,
              double min_weight,
              int ensemble_size,
              gdf_column *ecg_parts) {
  IdxT size = graph->adjList->offsets->size - 1;
  IdxT* offsets = (IdxT*)graph->adjList->offsets->data;
  IdxT* indices = (IdxT*)graph->adjList->indices->data;
  ValT* weights = (ValT*)graph->adjList->edge_data->data;

  for (int i = 0; i < ensemble_size; i++) {
    // Take random permutation of the graph
    IdxT* permutation = getPermutationVector(size, size * i);

    // Run Louvain clustering on the random permutation

    // For each edge in the graph determine whether the endpoints are in the same partition

    // Keep a sum for each edge of the total number of times its endpoints are in the same partition
  }

  // Set weights = min_weight + (1 - min-weight)*sum/ensemble_size

  // Run Louvain on the original graph using the computed weights
}
} // anonymous namespace


namespace cugraph {
void ecg(Graph* graph,
         double min_weight,
         int ensemble_size,
         gdf_column *ecg_parts) {
  CUGRAPH_EXPECTS(graph != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(ecg_parts != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(graph->adjList != nullptr, "Graph must have adjacency list");
  CUGRAPH_EXPECTS(graph->adjList->edge_data != nullptr, "Graph must have weights");
  CUGRAPH_EXPECTS(graph->adfList->offsets->dtype == ecg_parts->dtype, "Output type must match index type!");

  // determine the index type and value type of the graph
  // Call the appropriate templated instance of the implementation
  switch (graph->adjList->offsets->dtype) {
    case GDF_INT32: {
      switch (graph->adjList->edge_data) {
        case GDF_FLOAT32: {
          ecg_impl<int32_t, float>(graph, min_weight, ensemble_size, ecg_parts);
          break;
        }
        case GDF_FLOAT64: {
          ecg_impl<int32_t, double>(graph, min_weight, ensemble_size, ecg_parts);
          break;
        }
      }
      break;
    }
    case GDF_INT64: {
      switch (graph->adjList->edge_data) {
        case GDF_FLOAT32: {
          ecg_impl<int64_t, float>(graph, min_weight, ensemble_size, ecg_parts);
          break;
        }
        case GDF_FLOAT64: {
          ecg_impl<int64_t, double>(graph, min_weight, ensemble_size, ecg_parts);
          break;
        }
      }
    }
  }
}
} // cugraph namespace

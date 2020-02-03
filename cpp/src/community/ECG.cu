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
#include <thrust/random.h>
#include <ctime>
#include "utilities/error_utils.h"
#include <rmm_utils.h>
#include "utilities/graph_utils.cuh"
#include <converters/permute_graph.cuh>

namespace {
template<typename IndexType>
__device__ IndexType binsearch_maxle(const IndexType *vec,
                                      const IndexType val,
                                      IndexType low,
                                      IndexType high) {
  while (true) {
    if (low == high)
      return low; //we know it exists
    if ((low + 1) == high)
      return (vec[high] <= val) ? high : low;

    IndexType mid = low + (high - low) / 2;

    if (vec[mid] > val)
      high = mid - 1;
    else
      low = mid;
  }
}

template<typename IdxT, typename ValT>
__global__ void match_check_kernel(IdxT size,
                                   IdxT num_verts,
                                   IdxT* offsets,
                                   IdxT* indices,
                                   IdxT* permutation,
                                   IdxT* parts,
                                   ValT* weights) {
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < size) {
    IdxT source = binsearch_maxle(offsets, tid, (IdxT)0, num_verts);
    IdxT dest = indices[tid];
    if (parts[permutation[source]] == parts[permutation[dest]])
      weights[tid] += 1;
    tid += gridDim.x * blockDim.x;
  }
}

struct prg {
  __host__ __device__
  float operator()(int n){
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(0.0, 1.0);
    rng.discard(n);
    return dist(rng);
  }
};

template<typename ValT>
struct update_functor{
  ValT min_value;
  ValT ensemble_size;
  update_functor(ValT minv, ValT es):min_value(minv), ensemble_size(es){}
  __host__ __device__
  ValT operator()(ValT input) {
    return min_value + (1 - min_value)*(input / ensemble_size);
  }
};

/**
 * Computes a random permutation vector of length size. A permutation vector of length n
 * contains all values [0..n-1] exactly once.
 * @param size The length of the permutation vector to generate
 * @param seed A seed value for the random number generator, the generator will discard this many
 * values before using values. Calling this method with the same seed will result in the same
 * permutation vector.
 * @return A pointer to memory containing the requested permutation vector. The caller is
 * responsible for freeing the allocated memory using ALLOC_FREE_TRY().
 */
template <typename IdxT>
IdxT* get_permutation_vector(IdxT size, IdxT seed) {
  IdxT* output_vector;
  ALLOC_TRY(&output_vector, sizeof(IdxT) * size, nullptr);
  float* randoms;
  ALLOC_TRY(&randoms, sizeof(float) * size, nullptr);

  thrust::counting_iterator<uint32_t> index(seed);
  thrust::transform(rmm::exec_policy(nullptr)->on(nullptr), index, index + size, randoms, prg());
  thrust::sequence(rmm::exec_policy(nullptr)->on(nullptr), output_vector, output_vector + size, 0);
  thrust::sort_by_key(rmm::exec_policy(nullptr)->on(nullptr), randoms, randoms + size, output_vector);

  ALLOC_FREE_TRY(randoms, nullptr);

  return output_vector;
}


} // anonymous namespace

namespace cugraph {

template<typename IdxT, typename ValT>
void ecg(cugraph::Graph* graph,
              ValT min_weight,
              size_t ensemble_size,
              IdxT* ecg_parts) {
  CUGRAPH_EXPECTS(graph != nullptr, "Invalid API parameter: Graph object is NULL");
  CUGRAPH_EXPECTS(ecg_parts != nullptr, "Invalid API parameter: ecg_parts is NULL");
  CUGRAPH_EXPECTS(graph->adjList != nullptr, "Invalid API parameter: Graph must have adjacency list");
  CUGRAPH_EXPECTS(graph->adjList->edge_data != nullptr, "Invalid API parameter: Graph must have weights");

  IdxT size = graph->adjList->offsets->size - 1;
  IdxT nnz = graph->adjList->indices->size;
  IdxT* offsets = (IdxT*) graph->adjList->offsets->data;
  IdxT* indices = (IdxT*) graph->adjList->indices->data;
  ValT* ecg_weights;
  ALLOC_TRY(&ecg_weights, sizeof(ValT) * nnz, nullptr);
  thrust::fill(rmm::exec_policy(nullptr)->on(nullptr),
               ecg_weights,
               ecg_weights + nnz,
               0.0);
  // Iterate over each member of the ensemble
  for (size_t i = 0; i < ensemble_size; i++) {
    // Take random permutation of the graph
    IdxT* permutation = get_permutation_vector(size, (IdxT)(size * i));
    cugraph::Graph* permuted = detail::permute_graph<IdxT, ValT>(graph, permutation);

    // Run Louvain clustering on the random permutation
    IdxT* parts;
    ALLOC_TRY(&parts, sizeof(IdxT) * size, nullptr);
    ValT final_modularity;
    IdxT num_level;
    cugraph::louvain(permuted, &final_modularity, &num_level, parts, 1);

    // For each edge in the graph determine whether the endpoints are in the same partition
    // Keep a sum for each edge of the total number of times its endpoints are in the same partition
    dim3 grid, block;
    block.x = 512;
    grid.x = min((IdxT) CUDA_MAX_BLOCKS, (nnz / 512 + 1));
    match_check_kernel<<<grid, block, 0, nullptr>>>(nnz,
                                                    size,
                                                    offsets,
                                                    indices,
                                                    permutation,
                                                    parts,
                                                    ecg_weights);

    // Clean up temporary allocations
    delete permuted;
    ALLOC_FREE_TRY(parts, nullptr);
    ALLOC_FREE_TRY(permutation, nullptr);
  }

  // Set weights = min_weight + (1 - min-weight)*sum/ensemble_size
  update_functor<ValT> uf(min_weight, ensemble_size);
  thrust::transform(rmm::exec_policy(nullptr)->on(nullptr), ecg_weights, ecg_weights + nnz, ecg_weights, uf);

  // Run Louvain on the original graph using the computed weights
  cugraph::Graph* result = new cugraph::Graph;
  result->adjList = new cugraph::gdf_adj_list;
  result->adjList->offsets = new gdf_column;
  result->adjList->indices = new gdf_column;
  result->adjList->edge_data = new gdf_column;
  result->adjList->ownership = 0;
  gdf_column_view(result->adjList->offsets,
                  offsets,
                  nullptr,
                  graph->adjList->offsets->size,
                  graph->adjList->offsets->dtype);
  gdf_column_view(result->adjList->indices,
                  indices,
                  nullptr,
                  graph->adjList->indices->size,
                  graph->adjList->indices->dtype);
  gdf_column_view(result->adjList->edge_data,
                  ecg_weights,
                  nullptr,
                  graph->adjList->edge_data->size,
                  graph->adjList->edge_data->dtype);
  ValT final_modularity;
  IdxT num_level;
  cugraph::louvain(result, &final_modularity, &num_level, ecg_parts, 100);

  // Cleaning up temporary allocations
  delete result;
  ALLOC_FREE_TRY(ecg_weights, nullptr);
}

// Explicit template instantiations.
template void ecg<int32_t, float>(cugraph::Graph* graph,
                                  float min_weight,
                                  size_t ensemble_size,
                                  int32_t* ecg_parts);
template void ecg<int32_t, double>(cugraph::Graph* graph,
                                   double min_weight,
                                   size_t ensemble_size,
                                   int32_t* ecg_parts);
template void ecg<int64_t, float>(cugraph::Graph* graph,
                                  float min_weight,
                                  size_t ensemble_size,
                                  int64_t* ecg_parts);
template void ecg<int64_t, double>(cugraph::Graph* graph,
                                   double min_weight,
                                   size_t ensemble_size,
                                   int64_t* ecg_parts);

} // cugraph namespace

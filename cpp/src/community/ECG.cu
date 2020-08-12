/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <algorithms.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/random.h>
#include <converters/permute_graph.cuh>
#include <ctime>
#include <utilities/error.hpp>
#include "utilities/graph_utils.cuh"

namespace {
template <typename IndexType>
__device__ IndexType
binsearch_maxle(const IndexType *vec, const IndexType val, IndexType low, IndexType high)
{
  while (true) {
    if (low == high) return low;  // we know it exists
    if ((low + 1) == high) return (vec[high] <= val) ? high : low;

    IndexType mid = low + (high - low) / 2;

    if (vec[mid] > val)
      high = mid - 1;
    else
      low = mid;
  }
}

template <typename IdxT, typename ValT>
__global__ void match_check_kernel(IdxT size,
                                   IdxT num_verts,
                                   IdxT *offsets,
                                   IdxT *indices,
                                   IdxT *permutation,
                                   IdxT *parts,
                                   ValT *weights)
{
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < size) {
    IdxT source = binsearch_maxle(offsets, tid, (IdxT)0, num_verts);
    IdxT dest   = indices[tid];
    if (parts[permutation[source]] == parts[permutation[dest]]) weights[tid] += 1;
    tid += gridDim.x * blockDim.x;
  }
}

struct prg {
  __host__ __device__ float operator()(int n)
  {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(0.0, 1.0);
    rng.discard(n);
    return dist(rng);
  }
};

template <typename ValT>
struct update_functor {
  ValT min_value;
  ValT ensemble_size;
  update_functor(ValT minv, ValT es) : min_value(minv), ensemble_size(es) {}
  __host__ __device__ ValT operator()(ValT input)
  {
    return min_value + (1 - min_value) * (input / ensemble_size);
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
template <typename T>
void get_permutation_vector(T size, T seed, T *permutation, cudaStream_t stream)
{
  rmm::device_vector<float> randoms_v(size);

  thrust::counting_iterator<uint32_t> index(seed);
  thrust::transform(
    rmm::exec_policy(stream)->on(stream), index, index + size, randoms_v.begin(), prg());
  thrust::sequence(rmm::exec_policy(stream)->on(stream), permutation, permutation + size, 0);
  thrust::sort_by_key(
    rmm::exec_policy(stream)->on(stream), randoms_v.begin(), randoms_v.end(), permutation);
}

}  // anonymous namespace

namespace cugraph {

template <typename vertex_t, typename edge_t, typename weight_t>
void ecg(GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
         weight_t min_weight,
         vertex_t ensemble_size,
         vertex_t *ecg_parts)
{
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "API error, louvain expects a weighted graph");
  CUGRAPH_EXPECTS(ecg_parts != nullptr, "Invalid API parameter: ecg_parts is NULL");

  cudaStream_t stream{0};

  rmm::device_vector<weight_t> ecg_weights_v(graph.edge_data,
                                             graph.edge_data + graph.number_of_edges);

  vertex_t size{graph.number_of_vertices};
  vertex_t seed{1};

  auto permuted_graph = std::make_unique<GraphCSR<vertex_t, edge_t, weight_t>>(
    size, graph.number_of_edges, graph.has_data());

  // Iterate over each member of the ensemble
  for (vertex_t i = 0; i < ensemble_size; i++) {
    // Take random permutation of the graph
    rmm::device_vector<vertex_t> permutation_v(size);
    vertex_t *d_permutation = permutation_v.data().get();

    get_permutation_vector(size, seed, d_permutation, stream);
    seed += size;

    detail::permute_graph<vertex_t, edge_t, weight_t>(graph, d_permutation, permuted_graph->view());

    // Run one level of Louvain clustering on the random permutation
    rmm::device_vector<vertex_t> parts_v(size);
    vertex_t *d_parts = parts_v.data().get();

    weight_t final_modularity;
    vertex_t num_level;

    cugraph::louvain(permuted_graph->view(), &final_modularity, &num_level, d_parts, 1);

    // For each edge in the graph determine whether the endpoints are in the same partition
    // Keep a sum for each edge of the total number of times its endpoints are in the same partition
    dim3 grid, block;
    block.x = 512;
    grid.x  = min(vertex_t{CUDA_MAX_BLOCKS}, (graph.number_of_edges / 512 + 1));
    match_check_kernel<<<grid, block, 0, stream>>>(graph.number_of_edges,
                                                   graph.number_of_vertices,
                                                   graph.offsets,
                                                   graph.indices,
                                                   permutation_v.data().get(),
                                                   d_parts,
                                                   ecg_weights_v.data().get());
  }

  // Set weights = min_weight + (1 - min-weight)*sum/ensemble_size
  update_functor<weight_t> uf(min_weight, ensemble_size);
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    ecg_weights_v.data().get(),
                    ecg_weights_v.data().get() + graph.number_of_edges,
                    ecg_weights_v.data().get(),
                    uf);

  // Run Louvain on the original graph using the computed weights
  // (pass max_level = 100 for a "full run")
  GraphCSRView<vertex_t, edge_t, weight_t> louvain_graph;
  louvain_graph.indices            = graph.indices;
  louvain_graph.offsets            = graph.offsets;
  louvain_graph.edge_data          = ecg_weights_v.data().get();
  louvain_graph.number_of_vertices = graph.number_of_vertices;
  louvain_graph.number_of_edges    = graph.number_of_edges;

  weight_t final_modularity;
  vertex_t num_level;
  cugraph::louvain(louvain_graph, &final_modularity, &num_level, ecg_parts, 100);
}

// Explicit template instantiations.
template void ecg<int32_t, int32_t, float>(GraphCSRView<int32_t, int32_t, float> const &graph,
                                           float min_weight,
                                           int32_t ensemble_size,
                                           int32_t *ecg_parts);
template void ecg<int32_t, int32_t, double>(GraphCSRView<int32_t, int32_t, double> const &graph,
                                            double min_weight,
                                            int32_t ensemble_size,
                                            int32_t *ecg_parts);
}  // namespace cugraph

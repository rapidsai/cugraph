/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <community/legacy/louvain.cuh>
#include <cugraph/algorithms.hpp>
#include <cugraph/utilities/error.hpp>
#include <utilities/graph_utils.cuh>

#include <rmm/exec_policy.hpp>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <ctime>

namespace {
template <typename IndexType>
__device__ IndexType
binsearch_maxle(const IndexType* vec, const IndexType val, IndexType low, IndexType high)
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

// FIXME: This shouldn't need to be a custom kernel, this
//        seems like it should just be a thrust::transform
template <typename IdxT, typename ValT>
__global__ void match_check_kernel(
  IdxT size, IdxT num_verts, IdxT* offsets, IdxT* indices, IdxT* parts, ValT* weights)
{
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < size) {
    IdxT source = binsearch_maxle(offsets, tid, (IdxT)0, num_verts);
    IdxT dest   = indices[tid];
    if (parts[source] == parts[dest]) weights[tid] += 1;
    tid += gridDim.x * blockDim.x;
  }
}

struct prg {
  __device__ float operator()(int n)
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
void get_permutation_vector(T size, T seed, T* permutation, rmm::cuda_stream_view stream_view)
{
  rmm::device_uvector<float> randoms_v(size, stream_view);

  thrust::counting_iterator<uint32_t> index(seed);
  thrust::transform(rmm::exec_policy(stream_view), index, index + size, randoms_v.begin(), prg());
  thrust::sequence(rmm::exec_policy(stream_view), permutation, permutation + size, 0);
  thrust::sort_by_key(
    rmm::exec_policy(stream_view), randoms_v.begin(), randoms_v.end(), permutation);
}

template <typename graph_type>
class EcgLouvain : public cugraph::legacy::Louvain<graph_type> {
 public:
  using graph_t  = graph_type;
  using vertex_t = typename graph_type::vertex_type;
  using edge_t   = typename graph_type::edge_type;
  using weight_t = typename graph_type::weight_type;

  EcgLouvain(raft::handle_t const& handle, graph_type const& graph, vertex_t seed)
    : cugraph::legacy::Louvain<graph_type>(handle, graph), seed_(seed)
  {
  }

  void initialize_dendrogram_level(vertex_t num_vertices) override
  {
    this->dendrogram_->add_level(0, num_vertices, this->handle_.get_stream());

    get_permutation_vector(
      num_vertices, seed_, this->dendrogram_->current_level_begin(), this->handle_.get_stream());
  }

 private:
  vertex_t seed_;
};

}  // anonymous namespace

namespace cugraph {

template <typename vertex_t, typename edge_t, typename weight_t>
void ecg(raft::handle_t const& handle,
         legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
         weight_t min_weight,
         vertex_t ensemble_size,
         vertex_t* clustering)
{
  using graph_type = legacy::GraphCSRView<vertex_t, edge_t, weight_t>;

  CUGRAPH_EXPECTS(graph.edge_data != nullptr,
                  "Invalid input argument: ecg expects a weighted graph");
  CUGRAPH_EXPECTS(clustering != nullptr,
                  "Invalid input argument: clustering is NULL, should be a device pointer to "
                  "memory for storing the result");

  rmm::device_uvector<weight_t> ecg_weights_v(graph.number_of_edges, handle.get_stream());

  thrust::copy(handle.get_thrust_policy(),
               graph.edge_data,
               graph.edge_data + graph.number_of_edges,
               ecg_weights_v.data());

  vertex_t size{graph.number_of_vertices};

  // FIXME:  This seed should be a parameter
  vertex_t seed{1};

  // Iterate over each member of the ensemble
  for (vertex_t i = 0; i < ensemble_size; i++) {
    EcgLouvain<graph_type> runner(handle, graph, seed);
    seed += size;

    weight_t wt = runner(size_t{1}, weight_t{1});

    // For each edge in the graph determine whether the endpoints are in the same partition
    // Keep a sum for each edge of the total number of times its endpoints are in the same partition
    dim3 grid, block;
    block.x = 512;
    grid.x  = min(vertex_t{CUDA_MAX_BLOCKS}, (graph.number_of_edges / 512 + 1));
    match_check_kernel<<<grid, block, 0, handle.get_stream()>>>(
      graph.number_of_edges,
      graph.number_of_vertices,
      graph.offsets,
      graph.indices,
      runner.get_dendrogram().get_level_ptr_nocheck(0),
      ecg_weights_v.data());
  }

  // Set weights = min_weight + (1 - min-weight)*sum/ensemble_size
  update_functor<weight_t> uf(min_weight, ensemble_size);
  thrust::transform(handle.get_thrust_policy(),
                    ecg_weights_v.begin(),
                    ecg_weights_v.end(),
                    ecg_weights_v.begin(),
                    uf);

  // Run Louvain on the original graph using the computed weights
  // (pass max_level = 100 for a "full run")
  legacy::GraphCSRView<vertex_t, edge_t, weight_t> louvain_graph;
  louvain_graph.indices            = graph.indices;
  louvain_graph.offsets            = graph.offsets;
  louvain_graph.edge_data          = ecg_weights_v.data();
  louvain_graph.number_of_vertices = graph.number_of_vertices;
  louvain_graph.number_of_edges    = graph.number_of_edges;

  cugraph::louvain(handle, louvain_graph, clustering, size_t{100});
}

// Explicit template instantiations.
template void ecg<int32_t, int32_t, float>(
  raft::handle_t const&,
  legacy::GraphCSRView<int32_t, int32_t, float> const& graph,
  float min_weight,
  int32_t ensemble_size,
  int32_t* clustering);
template void ecg<int32_t, int32_t, double>(
  raft::handle_t const&,
  legacy::GraphCSRView<int32_t, int32_t, double> const& graph,
  double min_weight,
  int32_t ensemble_size,
  int32_t* clustering);
}  // namespace cugraph

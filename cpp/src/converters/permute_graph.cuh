/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include "converters/COOtoCSR.cuh"
#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/error.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <utilities/graph_utils.cuh>

namespace cugraph {
namespace detail {

template <typename IdxT>
struct permutation_functor {
  IdxT const* permutation;
  permutation_functor(IdxT const* p) : permutation(p) {}
  __host__ __device__ IdxT operator()(IdxT in) const { return permutation[in]; }
};

/**
 * This function takes a graph and a permutation vector and permutes the
 * graph according to the permutation vector. So each vertex id i becomes
 * vertex id permutation[i] in the permuted graph.
 *
 * @param graph The graph to permute.
 * @param permutation The permutation vector to use, must be a valid permutation
 * i.e. contains all values 0-n exactly once.
 * @param result View of the resulting graph... note this should be pre allocated
 *               and number_of_vertices and number_of_edges should be set
 * @return The permuted graph.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void permute_graph(legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                   vertex_t const* permutation,
                   legacy::GraphCSRView<vertex_t, edge_t, weight_t> result,
                   cudaStream_t stream = 0)
{
  //  Create a COO out of the CSR
  rmm::device_vector<vertex_t> src_vertices_v(graph.number_of_edges);
  rmm::device_vector<vertex_t> dst_vertices_v(graph.number_of_edges);
  rmm::device_vector<weight_t> weights_v(graph.number_of_edges);

  vertex_t* d_src     = src_vertices_v.data().get();
  vertex_t* d_dst     = dst_vertices_v.data().get();
  weight_t* d_weights = weights_v.data().get();

  graph.get_source_indices(d_src);

  if (graph.has_data())
    thrust::copy(rmm::exec_policy(stream),
                 graph.edge_data,
                 graph.edge_data + graph.number_of_edges,
                 d_weights);

  // Permute the src_indices
  permutation_functor<vertex_t> pf(permutation);
  thrust::transform(rmm::exec_policy(stream), d_src, d_src + graph.number_of_edges, d_src, pf);

  // Permute the destination indices
  thrust::transform(
    rmm::exec_policy(stream), graph.indices, graph.indices + graph.number_of_edges, d_dst, pf);

  legacy::GraphCOOView<vertex_t, edge_t, weight_t> graph_coo;

  graph_coo.number_of_vertices = graph.number_of_vertices;
  graph_coo.number_of_edges    = graph.number_of_edges;
  graph_coo.src_indices        = d_src;
  graph_coo.dst_indices        = d_dst;

  if (graph.has_data()) {
    graph_coo.edge_data = d_weights;
  } else {
    graph_coo.edge_data = nullptr;
  }

  cugraph::coo_to_csr_inplace(graph_coo, result);
}

}  // namespace detail
}  // namespace cugraph

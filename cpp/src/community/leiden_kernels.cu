/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <graph.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <community/louvain_kernels.hpp>
#include <utilities/graph_utils.cuh>

//#define TIMING

#ifdef TIMING
#include <utilities/high_res_timer.hpp>
#endif

#include <converters/COOtoCSR.cuh>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t update_clustering_by_delta_modularity_constrained(
  weight_t total_edge_weight,
  weight_t resolution,
  GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
  rmm::device_vector<vertex_t> const &src_indices,
  rmm::device_vector<weight_t> const &vertex_weights,
  rmm::device_vector<weight_t> &cluster_weights,
  rmm::device_vector<vertex_t> &cluster,
  rmm::device_vector<vertex_t> &constraint,
  cudaStream_t stream)
{
  rmm::device_vector<vertex_t> next_cluster(cluster);
  rmm::device_vector<weight_t> delta_Q(graph.number_of_edges);
  rmm::device_vector<vertex_t> cluster_hash(graph.number_of_edges);
  rmm::device_vector<weight_t> old_cluster_sum(graph.number_of_vertices);

  weight_t *d_delta_Q           = delta_Q.data().get();
  vertex_t *d_constraint        = constraint.data().get();
  vertex_t const *d_src_indices = src_indices.data().get();
  vertex_t const *d_dst_indices = graph.indices;

  weight_t new_Q = modularity(total_edge_weight, resolution, graph, cluster.data().get(), stream);

  weight_t cur_Q = new_Q - 1;

  // To avoid the potential of having two vertices swap clusters
  // we will only allow vertices to move up (true) or down (false)
  // during each iteration of the loop
  bool up_down = true;

  while (new_Q > (cur_Q + 0.0001)) {
    cur_Q = new_Q;

    compute_delta_modularity(total_edge_weight,
                             resolution,
                             graph,
                             src_indices,
                             vertex_weights,
                             cluster_weights,
                             cluster,
                             cluster_hash,
                             delta_Q,
                             old_cluster_sum,
                             stream);

    // Filter out positive delta_Q values for nodes not in the same constraint group
    thrust::for_each(
      rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(graph.number_of_edges),
      [d_src_indices, d_dst_indices, d_constraint, d_delta_Q] __device__(vertex_t i) {
        vertex_t start_cluster = d_constraint[d_src_indices[i]];
        vertex_t end_cluster   = d_constraint[d_dst_indices[i]];
        if (start_cluster != end_cluster) d_delta_Q[i] = weight_t{0.0};
      });

    assign_nodes(graph,
                 delta_Q,
                 cluster_hash,
                 src_indices,
                 next_cluster,
                 vertex_weights,
                 cluster_weights,
                 up_down,
                 stream);

    up_down = !up_down;

    new_Q = modularity(total_edge_weight, resolution, graph, next_cluster.data().get(), stream);

    if (new_Q > cur_Q) { thrust::copy(rmm::exec_policy(stream)->on(stream), next_cluster.begin(), next_cluster.end(), cluster.begin()); }
  }

  return cur_Q;
}

template float update_clustering_by_delta_modularity_constrained(
  float,
  float,
  GraphCSRView<int32_t, int32_t, float> const &,
  rmm::device_vector<int32_t> const &,
  rmm::device_vector<float> const &,
  rmm::device_vector<float> &,
  rmm::device_vector<int32_t> &,
  rmm::device_vector<int32_t> &,
  cudaStream_t);

template double update_clustering_by_delta_modularity_constrained(
  double,
  double,
  GraphCSRView<int32_t, int32_t, double> const &,
  rmm::device_vector<int32_t> const &,
  rmm::device_vector<double> const &,
  rmm::device_vector<double> &,
  rmm::device_vector<int32_t> &,
  rmm::device_vector<int32_t> &,
  cudaStream_t);

template <typename vertex_t, typename edge_t, typename weight_t>
void leiden(GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
            weight_t &final_modularity,
            int &num_level,
            vertex_t *cluster_vec,
            int max_level,
            weight_t resolution,
            cudaStream_t stream)
{
#ifdef TIMING
  HighResTimer hr_timer;
#endif

  num_level = 0;

  //
  //  Vectors to create a copy of the graph
  //
  rmm::device_vector<edge_t> offsets_v(graph.offsets, graph.offsets + graph.number_of_vertices + 1);
  rmm::device_vector<vertex_t> indices_v(graph.indices, graph.indices + graph.number_of_edges);
  rmm::device_vector<weight_t> weights_v(graph.edge_data, graph.edge_data + graph.number_of_edges);
  rmm::device_vector<vertex_t> src_indices_v(graph.number_of_edges);

  //
  //  Weights and clustering across iterations of algorithm
  //
  rmm::device_vector<weight_t> vertex_weights_v(graph.number_of_vertices);
  rmm::device_vector<weight_t> cluster_weights_v(graph.number_of_vertices);
  rmm::device_vector<vertex_t> cluster_v(graph.number_of_vertices);

  //
  //  Temporaries used within kernels.  Each iteration uses less
  //  of this memory
  //
  rmm::device_vector<vertex_t> tmp_arr_v(graph.number_of_vertices);
  rmm::device_vector<vertex_t> cluster_inverse_v(graph.number_of_vertices);

  weight_t total_edge_weight =
    thrust::reduce(rmm::exec_policy(stream)->on(stream), weights_v.begin(), weights_v.end());
  weight_t best_modularity = -1;

  //
  //  Initialize every cluster to reference each vertex to itself
  //
  thrust::sequence(rmm::exec_policy(stream)->on(stream), cluster_v.begin(), cluster_v.end());
  thrust::copy(rmm::exec_policy(stream)->on(stream), cluster_v.begin(), cluster_v.end(), cluster_vec);

  //
  //  Our copy of the graph.  Each iteration of the outer loop will
  //  shrink this copy of the graph.
  //
  GraphCSRView<vertex_t, edge_t, weight_t> current_graph(offsets_v.data().get(),
                                                         indices_v.data().get(),
                                                         weights_v.data().get(),
                                                         graph.number_of_vertices,
                                                         graph.number_of_edges);

  current_graph.get_source_indices(src_indices_v.data().get());

  while (num_level < max_level) {
    //
    //  Sum the weights of all edges departing a vertex.  This is
    //  loop invariant, so we'll compute it here.
    //
    //  Cluster weights are equivalent to vertex weights with this initial
    //  graph
    //
#ifdef TIMING
    hr_timer.start("init");
#endif

    cugraph::detail::compute_vertex_sums(current_graph, vertex_weights_v, stream);
    thrust::copy(rmm::exec_policy(stream)->on(stream), vertex_weights_v.begin(), vertex_weights_v.end(), cluster_weights_v.begin());

#ifdef TIMING
    hr_timer.stop();

    hr_timer.start("update_clustering");
#endif

    weight_t new_Q = update_clustering_by_delta_modularity(total_edge_weight,
                                                           resolution,
                                                           current_graph,
                                                           src_indices_v,
                                                           vertex_weights_v,
                                                           cluster_weights_v,
                                                           cluster_v,
                                                           stream);

    // After finding the initial unconstrained partition we use that partitioning as the constraint
    // for the second round.
    rmm::device_vector<vertex_t> constraint(graph.number_of_vertices);
    thrust::copy(rmm::exec_policy(stream)->on(stream), cluster_v.begin(), cluster_v.end(), constraint.begin());
    new_Q = update_clustering_by_delta_modularity_constrained(total_edge_weight,
                                                              resolution,
                                                              current_graph,
                                                              src_indices_v,
                                                              vertex_weights_v,
                                                              cluster_weights_v,
                                                              cluster_v,
                                                              constraint,
                                                              stream);

#ifdef TIMING
    hr_timer.stop();
#endif

    if (new_Q <= best_modularity) { break; }

    best_modularity = new_Q;

#ifdef TIMING
    hr_timer.start("shrinking graph");
#endif

    // renumber the clusters to the range 0..(num_clusters-1)
    vertex_t num_clusters = renumber_clusters(
      graph.number_of_vertices, cluster_v, tmp_arr_v, cluster_inverse_v, cluster_vec, stream);
    cluster_weights_v.resize(num_clusters);

    // shrink our graph to represent the graph of supervertices
    generate_superverticies_graph(current_graph, src_indices_v, num_clusters, cluster_v, stream);

    // assign each new vertex to its own cluster
    thrust::sequence(rmm::exec_policy(stream)->on(stream), cluster_v.begin(), cluster_v.end());

#ifdef TIMING
    hr_timer.stop();
#endif

    num_level++;
  }

#ifdef TIMING
  hr_timer.display(std::cout);
#endif

  final_modularity = best_modularity;
}

template void leiden(GraphCSRView<int32_t, int32_t, float> const &,
                     float &,
                     int &,
                     int32_t *,
                     int,
                     float,
                     cudaStream_t);
template void leiden(GraphCSRView<int32_t, int32_t, double> const &,
                     double &,
                     int &,
                     int32_t *,
                     int,
                     double,
                     cudaStream_t);

}  // namespace detail
}  // namespace cugraph

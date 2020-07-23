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

#include <utilities/graph_utils.cuh>

#ifdef TIMING
#include <utilities/high_res_timer.hpp>
#endif

#include <converters/COOtoCSR.cuh>

namespace cugraph {
namespace detail {

namespace {  // anonym.
constexpr int BLOCK_SIZE_1D = 64;
}

template <typename vertex_t, typename edge_t, typename weight_t>
__global__  // __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
  void
  compute_vertex_sums(vertex_t n_vertex,
                      edge_t const *offsets,
                      weight_t const *weights,
                      weight_t *output)
{
  int src = blockDim.x * blockIdx.x + threadIdx.x;

  if ((src < n_vertex)) {
    weight_t sum{0.0};

    for (int i = offsets[src]; i < offsets[src + 1]; ++i) { sum += weights[i]; }

    output[src] = sum;
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t modularity(weight_t m2,
                    GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
                    vertex_t const *d_cluster,
                    cudaStream_t stream)
{
  vertex_t n_verts = graph.number_of_vertices;

  rmm::device_vector<weight_t> inc(n_verts, weight_t{0.0});
  rmm::device_vector<weight_t> deg(n_verts, weight_t{0.0});

  edge_t const *d_offsets   = graph.offsets;
  vertex_t const *d_indices = graph.indices;
  weight_t const *d_weights = graph.edge_data;
  weight_t *d_inc           = inc.data().get();
  weight_t *d_deg           = deg.data().get();

  thrust::for_each(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(graph.number_of_vertices),
    [d_inc, d_deg, d_offsets, d_indices, d_weights, d_cluster] __device__(vertex_t v) {
      vertex_t community = d_cluster[v];
      weight_t increase{0.0};
      weight_t degree{0.0};

      for (edge_t loc = d_offsets[v]; loc < d_offsets[v + 1]; ++loc) {
        vertex_t neighbor = d_indices[loc];
        degree += d_weights[loc];
        if (d_cluster[neighbor] == community) { increase += d_weights[loc] / 2; }
      }

      if (degree > weight_t{0.0}) atomicAdd(d_deg + community, degree);

      if (increase > weight_t{0.0}) atomicAdd(d_inc + community, increase);
    });

  weight_t Q = thrust::transform_reduce(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(graph.number_of_vertices),
    [d_deg, d_inc, m2] __device__(vertex_t community) {
#ifdef DEBUG
      printf("  d_inc[%d] = %g, d_deg = %g, return = %g\n",
             community,
             d_inc[community],
             d_deg[community],
             ((2 * d_inc[community] / m2) - pow(d_deg[community] / m2, 2)));
#endif

      return (2 * d_inc[community] / m2) - pow(d_deg[community] / m2, 2);
    },
    weight_t{0.0},
    thrust::plus<weight_t>());
  return Q;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void generate_superverticies_graph(cugraph::GraphCSRView<vertex_t, edge_t, weight_t> &current_graph,
                                   rmm::device_vector<vertex_t> &src_indices_v,
                                   vertex_t new_number_of_vertices,
                                   rmm::device_vector<vertex_t> &cluster_v,
                                   cudaStream_t stream)
{
  rmm::device_vector<vertex_t> new_src_v(current_graph.number_of_edges);
  rmm::device_vector<vertex_t> new_dst_v(current_graph.number_of_edges);
  rmm::device_vector<weight_t> new_weight_v(current_graph.number_of_edges);

  vertex_t *d_old_src    = src_indices_v.data().get();
  vertex_t *d_old_dst    = current_graph.indices;
  weight_t *d_old_weight = current_graph.edge_data;
  vertex_t *d_new_src    = new_src_v.data().get();
  vertex_t *d_new_dst    = new_dst_v.data().get();
  vertex_t *d_clusters   = cluster_v.data().get();
  weight_t *d_new_weight = new_weight_v.data().get();

  //
  //  Renumber the COO
  //
  thrust::for_each(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<edge_t>(0),
    thrust::make_counting_iterator<edge_t>(current_graph.number_of_edges),
    [d_old_src, d_old_dst, d_new_src, d_new_dst, d_clusters, d_new_weight, d_old_weight] __device__(
      edge_t e) {
      d_new_src[e]    = d_clusters[d_old_src[e]];
      d_new_dst[e]    = d_clusters[d_old_dst[e]];
      d_new_weight[e] = d_old_weight[e];
    });

  thrust::stable_sort_by_key(
    rmm::exec_policy(stream)->on(stream),
    d_new_dst,
    d_new_dst + current_graph.number_of_edges,
    thrust::make_zip_iterator(thrust::make_tuple(d_new_src, d_new_weight)));
  thrust::stable_sort_by_key(
    rmm::exec_policy(stream)->on(stream),
    d_new_src,
    d_new_src + current_graph.number_of_edges,
    thrust::make_zip_iterator(thrust::make_tuple(d_new_dst, d_new_weight)));

  //
  //  Now we reduce by key to combine the weights of duplicate
  //  edges.
  //
  auto start     = thrust::make_zip_iterator(thrust::make_tuple(d_new_src, d_new_dst));
  auto new_start = thrust::make_zip_iterator(thrust::make_tuple(d_old_src, d_old_dst));
  auto new_end   = thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                                       start,
                                       start + current_graph.number_of_edges,
                                       d_new_weight,
                                       new_start,
                                       d_old_weight,
                                       thrust::equal_to<thrust::tuple<vertex_t, vertex_t>>(),
                                       thrust::plus<weight_t>());

  current_graph.number_of_edges    = thrust::distance(new_start, new_end.first);
  current_graph.number_of_vertices = new_number_of_vertices;

  detail::fill_offset(d_old_src,
                      current_graph.offsets,
                      new_number_of_vertices,
                      current_graph.number_of_edges,
                      stream);
  CHECK_CUDA(stream);

  src_indices_v.resize(current_graph.number_of_edges);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void compute_vertex_sums(GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
                         rmm::device_vector<weight_t> &sums,
                         cudaStream_t stream)
{
  dim3 block_size_1d =
    dim3((graph.number_of_vertices + BLOCK_SIZE_1D * 4 - 1) / BLOCK_SIZE_1D * 4, 1, 1);
  dim3 grid_size_1d = dim3(BLOCK_SIZE_1D * 4, 1, 1);

  compute_vertex_sums<vertex_t, edge_t, weight_t><<<block_size_1d, grid_size_1d>>>(
    graph.number_of_vertices, graph.offsets, graph.edge_data, sums.data().get());
}

template <typename vertex_t>
vertex_t renumber_clusters(vertex_t graph_num_vertices,
                           rmm::device_vector<vertex_t> &cluster,
                           rmm::device_vector<vertex_t> &temp_array,
                           rmm::device_vector<vertex_t> &cluster_inverse,
                           vertex_t *cluster_vec,
                           cudaStream_t stream)
{
  //
  //  Now we're going to renumber the clusters from 0 to (k-1), where k is the number of
  //  clusters in this level of the dendogram.
  //
  thrust::copy(cluster.begin(), cluster.end(), temp_array.begin());
  thrust::sort(temp_array.begin(), temp_array.end());
  auto tmp_end = thrust::unique(temp_array.begin(), temp_array.end());

  vertex_t old_num_clusters = cluster.size();
  vertex_t new_num_clusters = thrust::distance(temp_array.begin(), tmp_end);

  cluster.resize(new_num_clusters);
  temp_array.resize(new_num_clusters);

  thrust::fill(cluster_inverse.begin(), cluster_inverse.end(), vertex_t{-1});

  vertex_t *d_tmp_array       = temp_array.data().get();
  vertex_t *d_cluster_inverse = cluster_inverse.data().get();
  vertex_t *d_cluster         = cluster.data().get();

  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator<vertex_t>(0),
                   thrust::make_counting_iterator<vertex_t>(new_num_clusters),
                   [d_tmp_array, d_cluster_inverse] __device__(vertex_t i) {
                     d_cluster_inverse[d_tmp_array[i]] = i;
                   });

  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator<vertex_t>(0),
                   thrust::make_counting_iterator<vertex_t>(old_num_clusters),
                   [d_cluster, d_cluster_inverse] __device__(vertex_t i) {
                     d_cluster[i] = d_cluster_inverse[d_cluster[i]];
                   });

  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator<vertex_t>(0),
                   thrust::make_counting_iterator<vertex_t>(graph_num_vertices),
                   [cluster_vec, d_cluster] __device__(vertex_t i) {
                     cluster_vec[i] = d_cluster[cluster_vec[i]];
                   });

  return new_num_clusters;
}

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t update_clustering_by_delta_modularity(
  weight_t m2,
  GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
  rmm::device_vector<vertex_t> const &src_indices,
  rmm::device_vector<weight_t> const &vertex_weights,
  rmm::device_vector<weight_t> &cluster_weights,
  rmm::device_vector<vertex_t> &cluster,
  cudaStream_t stream)
{
  rmm::device_vector<vertex_t> next_cluster(cluster);
  rmm::device_vector<weight_t> old_cluster_sum(graph.number_of_vertices);
  rmm::device_vector<weight_t> delta_Q(graph.number_of_edges);
  rmm::device_vector<vertex_t> cluster_hash(graph.number_of_edges);
  rmm::device_vector<weight_t> cluster_hash_sum(graph.number_of_edges, weight_t{0.0});

  vertex_t *d_cluster_hash         = cluster_hash.data().get();
  weight_t *d_cluster_hash_sum     = cluster_hash_sum.data().get();
  vertex_t *d_cluster              = cluster.data().get();
  vertex_t const *d_src_indices    = src_indices.data().get();
  vertex_t *d_dst_indices          = graph.indices;
  edge_t *d_offsets                = graph.offsets;
  weight_t *d_weights              = graph.edge_data;
  weight_t const *d_vertex_weights = vertex_weights.data().get();
  weight_t *d_cluster_weights      = cluster_weights.data().get();
  weight_t *d_delta_Q              = delta_Q.data().get();
  weight_t *d_old_cluster_sum      = old_cluster_sum.data().get();

  weight_t new_Q = modularity<vertex_t, edge_t, weight_t>(m2, graph, cluster.data().get(), stream);

  weight_t cur_Q = new_Q - 1;

  // To avoid the potential of having two vertices swap clusters
  // we will only allow vertices to move up (true) or down (false)
  // during each iteration of the loop
  bool up_down = true;

  while (new_Q > (cur_Q + 0.0001)) {
    cur_Q = new_Q;

    thrust::fill(cluster_hash.begin(), cluster_hash.end(), vertex_t{-1});
    thrust::fill(cluster_hash_sum.begin(), cluster_hash_sum.end(), weight_t{0.0});
    thrust::fill(old_cluster_sum.begin(), old_cluster_sum.end(), weight_t{0.0});

    //
    // For each source vertex, we're going to build a hash
    // table to the destination cluster ids.  We can use
    // the offsets ranges to define the bounds of the hash
    // table.
    //
    thrust::for_each(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator<edge_t>(0),
                     thrust::make_counting_iterator<edge_t>(graph.number_of_edges),
                     [d_src_indices,
                      d_dst_indices,
                      d_cluster,
                      d_offsets,
                      d_cluster_hash,
                      d_cluster_hash_sum,
                      d_weights,
                      d_old_cluster_sum] __device__(edge_t loc) {
                       vertex_t src = d_src_indices[loc];
                       vertex_t dst = d_dst_indices[loc];

                       if (src != dst) {
                         vertex_t old_cluster = d_cluster[src];
                         vertex_t new_cluster = d_cluster[dst];
                         edge_t hash_base     = d_offsets[src];
                         edge_t n_edges       = d_offsets[src + 1] - hash_base;

                         int h         = (new_cluster % n_edges);
                         edge_t offset = hash_base + h;
                         while (d_cluster_hash[offset] != new_cluster) {
                           if (d_cluster_hash[offset] == -1) {
                             atomicCAS(d_cluster_hash + offset, -1, new_cluster);
                           } else {
                             h      = (h + 1) % n_edges;
                             offset = hash_base + h;
                           }
                         }

                         atomicAdd(d_cluster_hash_sum + offset, d_weights[loc]);

                         if (old_cluster == new_cluster)
                           atomicAdd(d_old_cluster_sum + src, d_weights[loc]);
                       }
                     });

    thrust::for_each(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator<edge_t>(0),
                     thrust::make_counting_iterator<edge_t>(graph.number_of_edges),
                     [m2,
                      d_cluster_hash,
                      d_src_indices,
                      d_cluster,
                      d_vertex_weights,
                      d_delta_Q,
                      d_cluster_hash_sum,
                      d_old_cluster_sum,
                      d_cluster_weights] __device__(edge_t loc) {
                       vertex_t new_cluster = d_cluster_hash[loc];
                       if (new_cluster >= 0) {
                         vertex_t src         = d_src_indices[loc];
                         vertex_t old_cluster = d_cluster[src];
                         weight_t degc_totw   = d_vertex_weights[src] / m2;

                         d_delta_Q[loc] =
                           d_cluster_hash_sum[loc] - degc_totw * d_cluster_weights[new_cluster] -
                           (d_old_cluster_sum[src] -
                            (degc_totw * (d_cluster_weights[old_cluster] - d_vertex_weights[src])));

#ifdef DEBUG
                         printf("src = %d, new cluster = %d, d_delta_Q[%d] = %g\n",
                                src,
                                new_cluster,
                                loc,
                                d_delta_Q[loc]);
#endif
                       } else {
                         d_delta_Q[loc] = weight_t{0.0};
                       }
                     });

    auto cluster_reduce_iterator =
      thrust::make_zip_iterator(thrust::make_tuple(d_cluster_hash, d_delta_Q));

    rmm::device_vector<vertex_t> temp_vertices(graph.number_of_vertices);
    rmm::device_vector<vertex_t> temp_cluster(graph.number_of_vertices, vertex_t{-1});
    rmm::device_vector<weight_t> temp_delta_Q(graph.number_of_vertices, weight_t{0.0});

    auto output_edge_iterator2 = thrust::make_zip_iterator(
      thrust::make_tuple(temp_cluster.data().get(), temp_delta_Q.data().get()));

    auto cluster_reduce_end =
      thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                            d_src_indices,
                            d_src_indices + graph.number_of_edges,
                            cluster_reduce_iterator,
                            temp_vertices.data().get(),
                            output_edge_iterator2,
                            thrust::equal_to<vertex_t>(),
                            [] __device__(auto pair1, auto pair2) {
                              if (thrust::get<1>(pair1) > thrust::get<1>(pair2))
                                return pair1;
                              else
                                return pair2;
                            });

    vertex_t final_size = thrust::distance(temp_vertices.data().get(), cluster_reduce_end.first);

    vertex_t *d_temp_vertices = temp_vertices.data().get();
    vertex_t *d_temp_clusters = temp_cluster.data().get();
    vertex_t *d_next_cluster  = next_cluster.data().get();
    weight_t *d_temp_delta_Q  = temp_delta_Q.data().get();

    thrust::for_each(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator<vertex_t>(0),
                     thrust::make_counting_iterator<vertex_t>(final_size),
                     [d_temp_delta_Q,
                      up_down,
                      d_next_cluster,
                      d_temp_vertices,
                      d_vertex_weights,
                      d_temp_clusters,
                      d_cluster_weights] __device__(vertex_t id) {
                       if ((d_temp_clusters[id] >= 0) && (d_temp_delta_Q[id] > weight_t{0.0})) {
                         vertex_t new_cluster = d_temp_clusters[id];
                         vertex_t old_cluster = d_next_cluster[d_temp_vertices[id]];

                         if ((new_cluster > old_cluster) == up_down) {
#ifdef DEBUG
                           printf(
                             "%s moving vertex %d from cluster %d to cluster %d - deltaQ = %g\n",
                             (up_down ? "up" : "down"),
                             d_temp_vertices[id],
                             d_next_cluster[d_temp_vertices[id]],
                             d_temp_clusters[id],
                             d_temp_delta_Q[id]);
#endif

                           weight_t src_weight = d_vertex_weights[d_temp_vertices[id]];
                           d_next_cluster[d_temp_vertices[id]] = d_temp_clusters[id];

                           atomicAdd(d_cluster_weights + new_cluster, src_weight);
                           atomicAdd(d_cluster_weights + old_cluster, -src_weight);
                         }
                       }
                     });

    up_down = !up_down;

    new_Q = modularity<vertex_t, edge_t, weight_t>(m2, graph, next_cluster.data().get(), stream);

    if (new_Q > cur_Q) { thrust::copy(next_cluster.begin(), next_cluster.end(), cluster.begin()); }
  }

  return cur_Q;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void louvain(GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
             weight_t *final_modularity,
             int *num_level,
             vertex_t *cluster_vec,
             int max_iter,
             cudaStream_t stream)
{
#ifdef TIMING
  HighResTimer hr_timer;
#endif

  *num_level = 0;

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

  weight_t m2 =
    thrust::reduce(rmm::exec_policy(stream)->on(stream), weights_v.begin(), weights_v.end());
  weight_t best_modularity = -1;

  //
  //  Initialize every cluster to reference each vertex to itself
  //
  thrust::sequence(rmm::exec_policy(stream)->on(stream), cluster_v.begin(), cluster_v.end());
  thrust::copy(cluster_v.begin(), cluster_v.end(), cluster_vec);

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

  while (true) {
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
    thrust::copy(vertex_weights_v.begin(), vertex_weights_v.end(), cluster_weights_v.begin());

#ifdef TIMING
    hr_timer.stop();

    hr_timer.start("update_clustering");
#endif

    weight_t new_Q = update_clustering_by_delta_modularity(
      m2, current_graph, src_indices_v, vertex_weights_v, cluster_weights_v, cluster_v, stream);

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
  }

#ifdef TIMING
  hr_timer.display(std::cout);
#endif

  *final_modularity = best_modularity;
}

template void louvain(
  GraphCSRView<int32_t, int32_t, float> const &, float *, int *, int32_t *, int, cudaStream_t);
template void louvain(
  GraphCSRView<int32_t, int32_t, double> const &, double *, int *, int32_t *, int, cudaStream_t);

}  // namespace detail
}  // namespace cugraph

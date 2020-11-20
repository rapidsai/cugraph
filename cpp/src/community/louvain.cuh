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
#pragma once

#include <graph.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <converters/COOtoCSR.cuh>
#include <utilities/graph_utils.cuh>

//#define TIMING

#ifdef TIMING
#include <utilities/high_res_timer.hpp>
#endif

namespace cugraph {

template <typename graph_type>
class Louvain {
 public:
  using graph_t  = graph_type;
  using vertex_t = typename graph_type::vertex_type;
  using edge_t   = typename graph_type::edge_type;
  using weight_t = typename graph_type::weight_type;

  Louvain(raft::handle_t const &handle, graph_type const &graph)
    :
#ifdef TIMING
      hr_timer_(),
#endif
      handle_(handle),

      // FIXME:  Don't really need to copy here but would need
      //         to change the logic to populate this properly
      //         in generate_superverticies_graph.
      //
      offsets_v_(graph.offsets, graph.offsets + graph.number_of_vertices + 1),
      indices_v_(graph.indices, graph.indices + graph.number_of_edges),
      weights_v_(graph.edge_data, graph.edge_data + graph.number_of_edges),
      src_indices_v_(graph.number_of_edges),
      vertex_weights_v_(graph.number_of_vertices),
      cluster_weights_v_(graph.number_of_vertices),
      cluster_v_(graph.number_of_vertices),
      tmp_arr_v_(graph.number_of_vertices),
      cluster_inverse_v_(graph.number_of_vertices),
      number_of_vertices_(graph.number_of_vertices),
      number_of_edges_(graph.number_of_edges),
      stream_(handle.get_stream())
  {
  }

  weight_t modularity(weight_t total_edge_weight,
                      weight_t resolution,
                      graph_t const &graph,
                      vertex_t const *d_cluster)
  {
    vertex_t n_verts = graph.number_of_vertices;

    rmm::device_vector<weight_t> inc(n_verts, weight_t{0.0});
    rmm::device_vector<weight_t> deg(n_verts, weight_t{0.0});

    edge_t const *d_offsets   = graph.offsets;
    vertex_t const *d_indices = graph.indices;
    weight_t const *d_weights = graph.edge_data;
    weight_t *d_inc           = inc.data().get();
    weight_t *d_deg           = deg.data().get();

    // FIXME:  Already have weighted degree computed in main loop,
    //         could pass that in rather than computing d_deg... which
    //         would save an atomicAdd (synchronization)
    //
    thrust::for_each(
      rmm::exec_policy(stream_)->on(stream_),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(graph.number_of_vertices),
      [d_inc, d_deg, d_offsets, d_indices, d_weights, d_cluster] __device__(vertex_t v) {
        vertex_t community = d_cluster[v];
        weight_t increase{0.0};
        weight_t degree{0.0};

        for (edge_t loc = d_offsets[v]; loc < d_offsets[v + 1]; ++loc) {
          vertex_t neighbor = d_indices[loc];
          degree += d_weights[loc];
          if (d_cluster[neighbor] == community) { increase += d_weights[loc]; }
        }

        if (degree > weight_t{0.0}) atomicAdd(d_deg + community, degree);
        if (increase > weight_t{0.0}) atomicAdd(d_inc + community, increase);
      });

    weight_t Q = thrust::transform_reduce(
      rmm::exec_policy(stream_)->on(stream_),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(graph.number_of_vertices),
      [d_deg, d_inc, total_edge_weight, resolution] __device__(vertex_t community) {
        return ((d_inc[community] / total_edge_weight) - resolution *
                                                           (d_deg[community] * d_deg[community]) /
                                                           (total_edge_weight * total_edge_weight));
      },
      weight_t{0.0},
      thrust::plus<weight_t>());

    return Q;
  }

  virtual std::pair<size_t, weight_t> operator()(vertex_t *d_cluster_vec,
                                                 size_t max_level,
                                                 weight_t resolution)
  {
    size_t num_level{0};

    weight_t total_edge_weight =
      thrust::reduce(rmm::exec_policy(stream_)->on(stream_), weights_v_.begin(), weights_v_.end());

    weight_t best_modularity = weight_t{-1};

    //
    //  Initialize every cluster to reference each vertex to itself
    //
    thrust::sequence(rmm::exec_policy(stream_)->on(stream_), cluster_v_.begin(), cluster_v_.end());
    thrust::copy(
      rmm::exec_policy(stream_)->on(stream_), cluster_v_.begin(), cluster_v_.end(), d_cluster_vec);

    //
    //  Our copy of the graph.  Each iteration of the outer loop will
    //  shrink this copy of the graph.
    //
    GraphCSRView<vertex_t, edge_t, weight_t> current_graph(offsets_v_.data().get(),
                                                           indices_v_.data().get(),
                                                           weights_v_.data().get(),
                                                           number_of_vertices_,
                                                           number_of_edges_);

    current_graph.get_source_indices(src_indices_v_.data().get());

    while (num_level < max_level) {
      compute_vertex_and_cluster_weights(current_graph);

      weight_t new_Q = update_clustering(total_edge_weight, resolution, current_graph);

      if (new_Q <= best_modularity) { break; }

      best_modularity = new_Q;

      shrink_graph(current_graph, d_cluster_vec);

      num_level++;
    }

    timer_display(std::cout);

    return std::make_pair(num_level, best_modularity);
  }

 protected:
  void timer_start(std::string const &region)
  {
#ifdef TIMING
    hr_timer_.start(region);
#endif
  }

  void timer_stop(cudaStream_t stream)
  {
#ifdef TIMING
    CUDA_TRY(cudaStreamSynchronize(stream));
    hr_timer_.stop();
#endif
  }

  void timer_display(std::ostream &os)
  {
#ifdef TIMING
    hr_timer_.display(os);
#endif
  }

 public:
  void compute_vertex_and_cluster_weights(graph_type const &graph)
  {
    timer_start("compute_vertex_and_cluster_weights");

    edge_t const *d_offsets     = graph.offsets;
    vertex_t const *d_indices   = graph.indices;
    weight_t const *d_weights   = graph.edge_data;
    weight_t *d_vertex_weights  = vertex_weights_v_.data().get();
    weight_t *d_cluster_weights = cluster_weights_v_.data().get();

    //
    // MNMG:  copy_v_transform_reduce_out_nbr, then copy
    //
    thrust::for_each(
      rmm::exec_policy(stream_)->on(stream_),
      thrust::make_counting_iterator<edge_t>(0),
      thrust::make_counting_iterator<edge_t>(graph.number_of_vertices),
      [d_offsets, d_indices, d_weights, d_vertex_weights, d_cluster_weights] __device__(
        vertex_t src) {
        weight_t sum =
          thrust::reduce(thrust::seq, d_weights + d_offsets[src], d_weights + d_offsets[src + 1]);

        d_vertex_weights[src]  = sum;
        d_cluster_weights[src] = sum;
      });

    timer_stop(stream_);
  }

  virtual weight_t update_clustering(weight_t total_edge_weight,
                                     weight_t resolution,
                                     graph_type const &graph)
  {
    timer_start("update_clustering");

    //
    // MNMG: This is the hard one, see writeup
    //
    rmm::device_vector<vertex_t> next_cluster_v(cluster_v_);
    rmm::device_vector<weight_t> delta_Q_v(graph.number_of_edges);
    rmm::device_vector<vertex_t> cluster_hash_v(graph.number_of_edges);
    rmm::device_vector<weight_t> old_cluster_sum_v(graph.number_of_vertices);

    vertex_t *d_cluster_hash         = cluster_hash_v.data().get();
    vertex_t *d_cluster              = cluster_v_.data().get();
    weight_t const *d_vertex_weights = vertex_weights_v_.data().get();
    weight_t *d_cluster_weights      = cluster_weights_v_.data().get();
    weight_t *d_delta_Q              = delta_Q_v.data().get();

    weight_t new_Q = modularity(total_edge_weight, resolution, graph, cluster_v_.data().get());

    weight_t cur_Q = new_Q - 1;

    // To avoid the potential of having two vertices swap clusters
    // we will only allow vertices to move up (true) or down (false)
    // during each iteration of the loop
    bool up_down = true;

    while (new_Q > (cur_Q + 0.0001)) {
      cur_Q = new_Q;

      compute_delta_modularity(
        total_edge_weight, resolution, graph, cluster_hash_v, old_cluster_sum_v, delta_Q_v);

      assign_nodes(graph, cluster_hash_v, next_cluster_v, delta_Q_v, up_down);

      up_down = !up_down;

      new_Q = modularity(total_edge_weight, resolution, graph, next_cluster_v.data().get());

      if (new_Q > cur_Q) {
        thrust::copy(rmm::exec_policy(stream_)->on(stream_),
                     next_cluster_v.begin(),
                     next_cluster_v.end(),
                     cluster_v_.begin());
      }
    }

    timer_stop(stream_);
    return cur_Q;
  }

  void compute_delta_modularity(weight_t total_edge_weight,
                                weight_t resolution,
                                graph_type const &graph,
                                rmm::device_vector<vertex_t> &cluster_hash_v,
                                rmm::device_vector<weight_t> &old_cluster_sum_v,
                                rmm::device_vector<weight_t> &delta_Q_v)
  {
    vertex_t const *d_src_indices     = src_indices_v_.data().get();
    vertex_t const *d_dst_indices     = graph.indices;
    edge_t const *d_offsets           = graph.offsets;
    weight_t const *d_weights         = graph.edge_data;
    vertex_t const *d_cluster         = cluster_v_.data().get();
    weight_t const *d_vertex_weights  = vertex_weights_v_.data().get();
    weight_t const *d_cluster_weights = cluster_weights_v_.data().get();

    vertex_t *d_cluster_hash    = cluster_hash_v.data().get();
    weight_t *d_delta_Q         = delta_Q_v.data().get();
    weight_t *d_old_cluster_sum = old_cluster_sum_v.data().get();
    weight_t *d_new_cluster_sum = d_delta_Q;

    thrust::fill(cluster_hash_v.begin(), cluster_hash_v.end(), vertex_t{-1});
    thrust::fill(delta_Q_v.begin(), delta_Q_v.end(), weight_t{0.0});
    thrust::fill(old_cluster_sum_v.begin(), old_cluster_sum_v.end(), weight_t{0.0});

    // MNMG:  New technique using reduce_by_key.  Would require a segmented sort
    //        or a pair of sorts on each node, so probably slower than what's here.
    //        This might still be faster even in MNMG...
    //
    //
    // FIXME:  Eventually this should use cuCollections concurrent map
    //         implementation, but that won't be available for a while.
    //
    // For each source vertex, we're going to build a hash
    // table to the destination cluster ids.  We can use
    // the offsets ranges to define the bounds of the hash
    // table.
    //
    thrust::for_each(rmm::exec_policy(stream_)->on(stream_),
                     thrust::make_counting_iterator<edge_t>(0),
                     thrust::make_counting_iterator<edge_t>(graph.number_of_edges),
                     [d_src_indices,
                      d_dst_indices,
                      d_cluster,
                      d_offsets,
                      d_cluster_hash,
                      d_new_cluster_sum,
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

                         atomicAdd(d_new_cluster_sum + offset, d_weights[loc]);

                         if (old_cluster == new_cluster)
                           atomicAdd(d_old_cluster_sum + src, d_weights[loc]);
                       }
                     });

    thrust::for_each(
      rmm::exec_policy(stream_)->on(stream_),
      thrust::make_counting_iterator<edge_t>(0),
      thrust::make_counting_iterator<edge_t>(graph.number_of_edges),
      [total_edge_weight,
       resolution,
       d_cluster_hash,
       d_src_indices,
       d_cluster,
       d_vertex_weights,
       d_delta_Q,
       d_new_cluster_sum,
       d_old_cluster_sum,
       d_cluster_weights] __device__(edge_t loc) {
        vertex_t new_cluster = d_cluster_hash[loc];
        if (new_cluster >= 0) {
          vertex_t src         = d_src_indices[loc];
          vertex_t old_cluster = d_cluster[src];
          weight_t k_k         = d_vertex_weights[src];
          weight_t a_old       = d_cluster_weights[old_cluster];
          weight_t a_new       = d_cluster_weights[new_cluster];

          // NOTE: d_delta_Q and d_new_cluster_sum are aliases
          //       for same device array to save memory
          d_delta_Q[loc] =
            2 * (((d_new_cluster_sum[loc] - d_old_cluster_sum[src]) / total_edge_weight) -
                 resolution * (a_new * k_k - a_old * k_k + k_k * k_k) /
                   (total_edge_weight * total_edge_weight));
        } else {
          d_delta_Q[loc] = weight_t{0.0};
        }
      });
  }

  void assign_nodes(graph_type const &graph,
                    rmm::device_vector<vertex_t> &cluster_hash_v,
                    rmm::device_vector<vertex_t> &next_cluster_v,
                    rmm::device_vector<weight_t> &delta_Q_v,
                    bool up_down)
  {
    rmm::device_vector<vertex_t> temp_vertices_v(graph.number_of_vertices);
    rmm::device_vector<vertex_t> temp_cluster_v(graph.number_of_vertices, vertex_t{-1});
    rmm::device_vector<weight_t> temp_delta_Q_v(graph.number_of_vertices, weight_t{0.0});

    weight_t *d_delta_Q              = delta_Q_v.data().get();
    vertex_t *d_next_cluster         = next_cluster_v.data().get();
    vertex_t *d_cluster_hash         = cluster_hash_v.data().get();
    weight_t const *d_vertex_weights = vertex_weights_v_.data().get();
    weight_t *d_cluster_weights      = cluster_weights_v_.data().get();

    auto cluster_reduce_iterator =
      thrust::make_zip_iterator(thrust::make_tuple(d_cluster_hash, d_delta_Q));

    auto output_edge_iterator2 = thrust::make_zip_iterator(
      thrust::make_tuple(temp_cluster_v.data().get(), temp_delta_Q_v.data().get()));

    auto cluster_reduce_end =
      thrust::reduce_by_key(rmm::exec_policy(stream_)->on(stream_),
                            src_indices_v_.begin(),
                            src_indices_v_.end(),
                            cluster_reduce_iterator,
                            temp_vertices_v.data().get(),
                            output_edge_iterator2,
                            thrust::equal_to<vertex_t>(),
                            [] __device__(auto pair1, auto pair2) {
                              if (thrust::get<1>(pair1) > thrust::get<1>(pair2))
                                return pair1;
                              else if ((thrust::get<1>(pair1) == thrust::get<1>(pair2)) &&
                                       (thrust::get<0>(pair1) < thrust::get<0>(pair2)))
                                return pair1;
                              else
                                return pair2;
                            });

    vertex_t final_size = thrust::distance(temp_vertices_v.data().get(), cluster_reduce_end.first);

    vertex_t *d_temp_vertices = temp_vertices_v.data().get();
    vertex_t *d_temp_clusters = temp_cluster_v.data().get();
    weight_t *d_temp_delta_Q  = temp_delta_Q_v.data().get();

    thrust::for_each(rmm::exec_policy(stream_)->on(stream_),
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
                           weight_t src_weight = d_vertex_weights[d_temp_vertices[id]];
                           d_next_cluster[d_temp_vertices[id]] = d_temp_clusters[id];

                           atomicAdd(d_cluster_weights + new_cluster, src_weight);
                           atomicAdd(d_cluster_weights + old_cluster, -src_weight);
                         }
                       }
                     });
  }

  void shrink_graph(graph_t &graph, vertex_t *d_cluster_vec)
  {
    timer_start("shrinking graph");

    // renumber the clusters to the range 0..(num_clusters-1)
    vertex_t num_clusters = renumber_clusters(d_cluster_vec);
    cluster_weights_v_.resize(num_clusters);

    // shrink our graph to represent the graph of supervertices
    generate_superverticies_graph(graph, num_clusters);

    // assign each new vertex to its own cluster
    thrust::sequence(rmm::exec_policy(stream_)->on(stream_), cluster_v_.begin(), cluster_v_.end());

    timer_stop(stream_);
  }

  vertex_t renumber_clusters(vertex_t *d_cluster_vec)
  {
    vertex_t *d_tmp_array       = tmp_arr_v_.data().get();
    vertex_t *d_cluster_inverse = cluster_inverse_v_.data().get();
    vertex_t *d_cluster         = cluster_v_.data().get();

    vertex_t old_num_clusters = cluster_v_.size();

    //
    //  New technique.  Initialize cluster_inverse_v_ to 0
    //
    thrust::fill(cluster_inverse_v_.begin(), cluster_inverse_v_.end(), vertex_t{0});

    //
    // Iterate over every element c in cluster_v_ and set cluster_inverse_v to 1
    //
    auto first_1 = thrust::make_constant_iterator<vertex_t>(1);
    auto last_1  = first_1 + old_num_clusters;

    thrust::scatter(rmm::exec_policy(stream_)->on(stream_),
                    first_1,
                    last_1,
                    cluster_v_.begin(),
                    cluster_inverse_v_.begin());

    //
    // Now we'll copy all of the clusters that have a value of 1 into a temporary array
    //
    auto copy_end = thrust::copy_if(
      rmm::exec_policy(stream_)->on(stream_),
      thrust::make_counting_iterator<vertex_t>(0),
      thrust::make_counting_iterator<vertex_t>(old_num_clusters),
      tmp_arr_v_.begin(),
      [d_cluster_inverse] __device__(const vertex_t idx) { return d_cluster_inverse[idx] == 1; });

    vertex_t new_num_clusters = thrust::distance(tmp_arr_v_.begin(), copy_end);
    tmp_arr_v_.resize(new_num_clusters);

    //
    // Now we can set each value in cluster_inverse of a cluster to its index
    //
    thrust::for_each(rmm::exec_policy(stream_)->on(stream_),
                     thrust::make_counting_iterator<vertex_t>(0),
                     thrust::make_counting_iterator<vertex_t>(new_num_clusters),
                     [d_cluster_inverse, d_tmp_array] __device__(const vertex_t idx) {
                       d_cluster_inverse[d_tmp_array[idx]] = idx;
                     });

    thrust::for_each(rmm::exec_policy(stream_)->on(stream_),
                     thrust::make_counting_iterator<vertex_t>(0),
                     thrust::make_counting_iterator<vertex_t>(old_num_clusters),
                     [d_cluster, d_cluster_inverse] __device__(vertex_t i) {
                       d_cluster[i] = d_cluster_inverse[d_cluster[i]];
                     });

    thrust::for_each(rmm::exec_policy(stream_)->on(stream_),
                     thrust::make_counting_iterator<vertex_t>(0),
                     thrust::make_counting_iterator<vertex_t>(number_of_vertices_),
                     [d_cluster_vec, d_cluster] __device__(vertex_t i) {
                       d_cluster_vec[i] = d_cluster[d_cluster_vec[i]];
                     });

    cluster_inverse_v_.resize(new_num_clusters);
    cluster_v_.resize(new_num_clusters);

    return new_num_clusters;
  }

  void generate_superverticies_graph(graph_t &graph, vertex_t num_clusters)
  {
    rmm::device_vector<vertex_t> new_src_v(graph.number_of_edges);
    rmm::device_vector<vertex_t> new_dst_v(graph.number_of_edges);
    rmm::device_vector<weight_t> new_weight_v(graph.number_of_edges);

    vertex_t *d_old_src    = src_indices_v_.data().get();
    vertex_t *d_old_dst    = graph.indices;
    weight_t *d_old_weight = graph.edge_data;
    vertex_t *d_new_src    = new_src_v.data().get();
    vertex_t *d_new_dst    = new_dst_v.data().get();
    vertex_t *d_clusters   = cluster_v_.data().get();
    weight_t *d_new_weight = new_weight_v.data().get();

    //
    //  Renumber the COO
    //
    thrust::for_each(rmm::exec_policy(stream_)->on(stream_),
                     thrust::make_counting_iterator<edge_t>(0),
                     thrust::make_counting_iterator<edge_t>(graph.number_of_edges),
                     [d_old_src,
                      d_old_dst,
                      d_old_weight,
                      d_new_src,
                      d_new_dst,
                      d_new_weight,
                      d_clusters] __device__(edge_t e) {
                       d_new_src[e]    = d_clusters[d_old_src[e]];
                       d_new_dst[e]    = d_clusters[d_old_dst[e]];
                       d_new_weight[e] = d_old_weight[e];
                     });

    thrust::stable_sort_by_key(
      rmm::exec_policy(stream_)->on(stream_),
      d_new_dst,
      d_new_dst + graph.number_of_edges,
      thrust::make_zip_iterator(thrust::make_tuple(d_new_src, d_new_weight)));
    thrust::stable_sort_by_key(
      rmm::exec_policy(stream_)->on(stream_),
      d_new_src,
      d_new_src + graph.number_of_edges,
      thrust::make_zip_iterator(thrust::make_tuple(d_new_dst, d_new_weight)));

    //
    //  Now we reduce by key to combine the weights of duplicate
    //  edges.
    //
    auto start     = thrust::make_zip_iterator(thrust::make_tuple(d_new_src, d_new_dst));
    auto new_start = thrust::make_zip_iterator(thrust::make_tuple(d_old_src, d_old_dst));
    auto new_end   = thrust::reduce_by_key(rmm::exec_policy(stream_)->on(stream_),
                                         start,
                                         start + graph.number_of_edges,
                                         d_new_weight,
                                         new_start,
                                         d_old_weight,
                                         thrust::equal_to<thrust::tuple<vertex_t, vertex_t>>(),
                                         thrust::plus<weight_t>());

    graph.number_of_edges    = thrust::distance(new_start, new_end.first);
    graph.number_of_vertices = num_clusters;

    detail::fill_offset(d_old_src, graph.offsets, num_clusters, graph.number_of_edges, stream_);
    CHECK_CUDA(stream_);

    src_indices_v_.resize(graph.number_of_edges);
    indices_v_.resize(graph.number_of_edges);
    weights_v_.resize(graph.number_of_edges);
  }

 protected:
  raft::handle_t const &handle_;
  vertex_t number_of_vertices_;
  edge_t number_of_edges_;
  cudaStream_t stream_;

  //
  //  Copy of graph
  //
  rmm::device_vector<edge_t> offsets_v_;
  rmm::device_vector<vertex_t> indices_v_;
  rmm::device_vector<weight_t> weights_v_;
  rmm::device_vector<vertex_t> src_indices_v_;

  //
  //  Weights and clustering across iterations of algorithm
  //
  rmm::device_vector<weight_t> vertex_weights_v_;
  rmm::device_vector<weight_t> cluster_weights_v_;
  rmm::device_vector<vertex_t> cluster_v_;

  //
  //  Temporaries used within kernels.  Each iteration uses less
  //  of this memory
  //
  rmm::device_vector<vertex_t> tmp_arr_v_;
  rmm::device_vector<vertex_t> cluster_inverse_v_;

#ifdef TIMING
  HighResTimer hr_timer_;
#endif
};

}  // namespace cugraph

/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

//#define TIMING

#include <cugraph/legacy/graph.hpp>

#include <converters/legacy/COOtoCSR.cuh>
#include <utilities/graph_utils.cuh>

#include <cugraph/dendrogram.hpp>
#ifdef TIMING
#include <cugraph/utilities/high_res_timer.hpp>
#endif

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace cugraph {
namespace legacy {

template <typename graph_type>
class Louvain {
 public:
  using graph_t  = graph_type;
  using vertex_t = typename graph_type::vertex_type;
  using edge_t   = typename graph_type::edge_type;
  using weight_t = typename graph_type::weight_type;

  Louvain(raft::handle_t const& handle, graph_type const& graph)
    :
#ifdef TIMING
      hr_timer_(),
#endif
      handle_(handle),
      dendrogram_(std::make_unique<Dendrogram<vertex_t>>()),

      // FIXME:  Don't really need to copy here but would need
      //         to change the logic to populate this properly
      //         in generate_superverticies_graph.
      //
      offsets_v_(graph.number_of_vertices + 1, handle.get_stream()),
      indices_v_(graph.number_of_edges, handle.get_stream()),
      weights_v_(graph.number_of_edges, handle.get_stream()),
      src_indices_v_(graph.number_of_edges, handle.get_stream()),
      vertex_weights_v_(graph.number_of_vertices, handle.get_stream()),
      cluster_weights_v_(graph.number_of_vertices, handle.get_stream()),
      tmp_arr_v_(graph.number_of_vertices, handle.get_stream()),
      cluster_inverse_v_(graph.number_of_vertices, handle.get_stream()),
      number_of_vertices_(graph.number_of_vertices),
      number_of_edges_(graph.number_of_edges)
  {
    thrust::copy(handle.get_thrust_policy(),
                 graph.offsets,
                 graph.offsets + graph.number_of_vertices + 1,
                 offsets_v_.begin());

    thrust::copy(handle.get_thrust_policy(),
                 graph.indices,
                 graph.indices + graph.number_of_edges,
                 indices_v_.begin());

    thrust::copy(handle.get_thrust_policy(),
                 graph.edge_data,
                 graph.edge_data + graph.number_of_edges,
                 weights_v_.begin());
  }

  virtual ~Louvain() {}

  weight_t modularity(weight_t total_edge_weight,
                      weight_t resolution,
                      graph_t const& graph,
                      vertex_t const* d_cluster)
  {
    vertex_t n_verts = graph.number_of_vertices;

    rmm::device_uvector<weight_t> inc(n_verts, handle_.get_stream());
    rmm::device_uvector<weight_t> deg(n_verts, handle_.get_stream());

    thrust::fill(handle_.get_thrust_policy(), inc.begin(), inc.end(), weight_t{0.0});
    thrust::fill(handle_.get_thrust_policy(), deg.begin(), deg.end(), weight_t{0.0});

    // FIXME:  Already have weighted degree computed in main loop,
    //         could pass that in rather than computing d_deg... which
    //         would save an atomicAdd (synchronization)
    //
    thrust::for_each(handle_.get_thrust_policy(),
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(graph.number_of_vertices),
                     [d_inc     = inc.data(),
                      d_deg     = deg.data(),
                      d_offsets = graph.offsets,
                      d_indices = graph.indices,
                      d_weights = graph.edge_data,
                      d_cluster] __device__(vertex_t v) {
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
      handle_.get_thrust_policy(),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(graph.number_of_vertices),
      [d_deg = deg.data(), d_inc = inc.data(), total_edge_weight, resolution] __device__(
        vertex_t community) {
        return ((d_inc[community] / total_edge_weight) - resolution *
                                                           (d_deg[community] * d_deg[community]) /
                                                           (total_edge_weight * total_edge_weight));
      },
      weight_t{0.0},
      thrust::plus<weight_t>());

    return Q;
  }

  Dendrogram<vertex_t> const& get_dendrogram() const { return *dendrogram_; }

  Dendrogram<vertex_t>& get_dendrogram() { return *dendrogram_; }

  std::unique_ptr<Dendrogram<vertex_t>> move_dendrogram() { return std::move(dendrogram_); }

  virtual weight_t operator()(size_t max_level, weight_t resolution)
  {
    weight_t total_edge_weight =
      thrust::reduce(handle_.get_thrust_policy(), weights_v_.begin(), weights_v_.end());

    weight_t best_modularity = weight_t{-1};

    //
    //  Our copy of the graph.  Each iteration of the outer loop will
    //  shrink this copy of the graph.
    //
    legacy::GraphCSRView<vertex_t, edge_t, weight_t> current_graph(offsets_v_.data(),
                                                                   indices_v_.data(),
                                                                   weights_v_.data(),
                                                                   number_of_vertices_,
                                                                   number_of_edges_);

    current_graph.get_source_indices(src_indices_v_.data());

    while (dendrogram_->num_levels() < max_level) {
      //
      //  Initialize every cluster to reference each vertex to itself
      //
      initialize_dendrogram_level(current_graph.number_of_vertices);

      compute_vertex_and_cluster_weights(current_graph);

      weight_t new_Q = update_clustering(total_edge_weight, resolution, current_graph);

      if (new_Q <= best_modularity) { break; }

      best_modularity = new_Q;

      shrink_graph(current_graph);
    }

    timer_display_and_clear(std::cout);

    return best_modularity;
  }

 protected:
  void timer_start(std::string const& region)
  {
#ifdef TIMING
    hr_timer_.start(region);
#endif
  }

  void timer_stop(rmm::cuda_stream_view stream_view)
  {
#ifdef TIMING
    stream_view.synchronize();
    hr_timer_.stop();
#endif
  }

  void timer_display_and_clear(std::ostream& os)
  {
#ifdef TIMING
    hr_timer_.display_and_clear(os);
#endif
  }

  virtual void initialize_dendrogram_level(vertex_t num_vertices)
  {
    dendrogram_->add_level(0, num_vertices, handle_.get_stream());

    thrust::sequence(handle_.get_thrust_policy(),
                     dendrogram_->current_level_begin(),
                     dendrogram_->current_level_end());
  }

 public:
  void compute_vertex_and_cluster_weights(graph_type const& graph)
  {
    timer_start("compute_vertex_and_cluster_weights");

    edge_t const* d_offsets     = graph.offsets;
    vertex_t const* d_indices   = graph.indices;
    weight_t const* d_weights   = graph.edge_data;
    weight_t* d_vertex_weights  = vertex_weights_v_.data();
    weight_t* d_cluster_weights = cluster_weights_v_.data();

    //
    // MNMG:  per_v_transform_reduce_outgoing_e, then copy
    //
    thrust::for_each(
      handle_.get_thrust_policy(),
      thrust::make_counting_iterator<edge_t>(0),
      thrust::make_counting_iterator<edge_t>(graph.number_of_vertices),
      [d_offsets, d_indices, d_weights, d_vertex_weights, d_cluster_weights] __device__(
        vertex_t src) {
        weight_t sum =
          thrust::reduce(thrust::seq, d_weights + d_offsets[src], d_weights + d_offsets[src + 1]);

        d_vertex_weights[src]  = sum;
        d_cluster_weights[src] = sum;
      });

    timer_stop(handle_.get_stream());
  }

  virtual weight_t update_clustering(weight_t total_edge_weight,
                                     weight_t resolution,
                                     graph_type const& graph)
  {
    timer_start("update_clustering");

    rmm::device_uvector<vertex_t> next_cluster_v(dendrogram_->current_level_size(),
                                                 handle_.get_stream());
    rmm::device_uvector<weight_t> delta_Q_v(graph.number_of_edges, handle_.get_stream());
    rmm::device_uvector<vertex_t> cluster_hash_v(graph.number_of_edges, handle_.get_stream());
    rmm::device_uvector<weight_t> old_cluster_sum_v(graph.number_of_vertices, handle_.get_stream());

    vertex_t* d_cluster              = dendrogram_->current_level_begin();
    weight_t const* d_vertex_weights = vertex_weights_v_.data();
    weight_t* d_cluster_weights      = cluster_weights_v_.data();
    weight_t* d_delta_Q              = delta_Q_v.data();

    thrust::copy(handle_.get_thrust_policy(),
                 dendrogram_->current_level_begin(),
                 dendrogram_->current_level_end(),
                 next_cluster_v.data());

    weight_t new_Q =
      modularity(total_edge_weight, resolution, graph, dendrogram_->current_level_begin());

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

      new_Q = modularity(total_edge_weight, resolution, graph, next_cluster_v.data());

      if (new_Q > cur_Q) {
        thrust::copy(handle_.get_thrust_policy(),
                     next_cluster_v.begin(),
                     next_cluster_v.end(),
                     dendrogram_->current_level_begin());
      }
    }

    timer_stop(handle_.get_stream());
    return cur_Q;
  }

  void compute_delta_modularity(weight_t total_edge_weight,
                                weight_t resolution,
                                graph_type const& graph,
                                rmm::device_uvector<vertex_t>& cluster_hash_v,
                                rmm::device_uvector<weight_t>& old_cluster_sum_v,
                                rmm::device_uvector<weight_t>& delta_Q_v)
  {
    edge_t const* d_offsets           = graph.offsets;
    weight_t const* d_weights         = graph.edge_data;
    vertex_t const* d_cluster         = dendrogram_->current_level_begin();
    weight_t const* d_vertex_weights  = vertex_weights_v_.data();
    weight_t const* d_cluster_weights = cluster_weights_v_.data();

    vertex_t* d_cluster_hash    = cluster_hash_v.data();
    weight_t* d_delta_Q         = delta_Q_v.data();
    weight_t* d_old_cluster_sum = old_cluster_sum_v.data();
    weight_t* d_new_cluster_sum = d_delta_Q;

    thrust::fill(
      handle_.get_thrust_policy(), cluster_hash_v.begin(), cluster_hash_v.end(), vertex_t{-1});
    thrust::fill(handle_.get_thrust_policy(), delta_Q_v.begin(), delta_Q_v.end(), weight_t{0.0});
    thrust::fill(handle_.get_thrust_policy(),
                 old_cluster_sum_v.begin(),
                 old_cluster_sum_v.end(),
                 weight_t{0.0});

    thrust::for_each(handle_.get_thrust_policy(),
                     thrust::make_counting_iterator<edge_t>(0),
                     thrust::make_counting_iterator<edge_t>(graph.number_of_edges),
                     [d_src_indices = src_indices_v_.data(),
                      d_dst_indices = graph.indices,
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
      handle_.get_thrust_policy(),
      thrust::make_counting_iterator<edge_t>(0),
      thrust::make_counting_iterator<edge_t>(graph.number_of_edges),
      [total_edge_weight,
       resolution,
       d_cluster_hash,
       d_src_indices = src_indices_v_.data(),
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

  void assign_nodes(graph_type const& graph,
                    rmm::device_uvector<vertex_t>& cluster_hash_v,
                    rmm::device_uvector<vertex_t>& next_cluster_v,
                    rmm::device_uvector<weight_t>& delta_Q_v,
                    bool up_down)
  {
    rmm::device_uvector<vertex_t> temp_vertices_v(graph.number_of_vertices, handle_.get_stream());
    rmm::device_uvector<vertex_t> temp_cluster_v(graph.number_of_vertices, handle_.get_stream());
    rmm::device_uvector<weight_t> temp_delta_Q_v(graph.number_of_vertices, handle_.get_stream());

    thrust::fill(
      handle_.get_thrust_policy(), temp_cluster_v.begin(), temp_cluster_v.end(), vertex_t{-1});

    thrust::fill(
      handle_.get_thrust_policy(), temp_delta_Q_v.begin(), temp_delta_Q_v.end(), weight_t{0});

    auto cluster_reduce_iterator =
      thrust::make_zip_iterator(thrust::make_tuple(cluster_hash_v.begin(), delta_Q_v.begin()));

    auto output_edge_iterator2 =
      thrust::make_zip_iterator(thrust::make_tuple(temp_cluster_v.begin(), temp_delta_Q_v.begin()));

    auto cluster_reduce_end =
      thrust::reduce_by_key(handle_.get_thrust_policy(),
                            src_indices_v_.begin(),
                            src_indices_v_.end(),
                            cluster_reduce_iterator,
                            temp_vertices_v.data(),
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

    vertex_t final_size = thrust::distance(temp_vertices_v.data(), cluster_reduce_end.first);

    thrust::for_each(handle_.get_thrust_policy(),
                     thrust::make_counting_iterator<vertex_t>(0),
                     thrust::make_counting_iterator<vertex_t>(final_size),
                     [up_down,
                      d_temp_delta_Q    = temp_delta_Q_v.data(),
                      d_next_cluster    = next_cluster_v.data(),
                      d_temp_vertices   = temp_vertices_v.data(),
                      d_vertex_weights  = vertex_weights_v_.data(),
                      d_temp_clusters   = temp_cluster_v.data(),
                      d_cluster_weights = cluster_weights_v_.data()] __device__(vertex_t id) {
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

  void shrink_graph(graph_t& graph)
  {
    timer_start("shrinking graph");

    // renumber the clusters to the range 0..(num_clusters-1)
    vertex_t num_clusters = renumber_clusters();
    cluster_weights_v_.resize(num_clusters, handle_.get_stream());

    // shrink our graph to represent the graph of supervertices
    generate_superverticies_graph(graph, num_clusters);

    timer_stop(handle_.get_stream());
  }

  vertex_t renumber_clusters()
  {
    vertex_t* d_tmp_array       = tmp_arr_v_.data();
    vertex_t* d_cluster_inverse = cluster_inverse_v_.data();
    vertex_t* d_cluster         = dendrogram_->current_level_begin();

    vertex_t old_num_clusters = dendrogram_->current_level_size();

    //
    //  New technique.  Initialize cluster_inverse_v_ to 0
    //
    thrust::fill(handle_.get_thrust_policy(),
                 cluster_inverse_v_.begin(),
                 cluster_inverse_v_.end(),
                 vertex_t{0});

    //
    // Iterate over every element c in the current clustering and set cluster_inverse_v to 1
    //
    auto first_1 = thrust::make_constant_iterator<vertex_t>(1);
    auto last_1  = first_1 + old_num_clusters;

    thrust::scatter(handle_.get_thrust_policy(),
                    first_1,
                    last_1,
                    dendrogram_->current_level_begin(),
                    cluster_inverse_v_.begin());

    //
    // Now we'll copy all of the clusters that have a value of 1 into a temporary array
    //
    auto copy_end = thrust::copy_if(
      handle_.get_thrust_policy(),
      thrust::make_counting_iterator<vertex_t>(0),
      thrust::make_counting_iterator<vertex_t>(old_num_clusters),
      tmp_arr_v_.begin(),
      [d_cluster_inverse] __device__(const vertex_t idx) { return d_cluster_inverse[idx] == 1; });

    vertex_t new_num_clusters = thrust::distance(tmp_arr_v_.begin(), copy_end);
    tmp_arr_v_.resize(new_num_clusters, handle_.get_stream());

    //
    // Now we can set each value in cluster_inverse of a cluster to its index
    //
    thrust::for_each(handle_.get_thrust_policy(),
                     thrust::make_counting_iterator<vertex_t>(0),
                     thrust::make_counting_iterator<vertex_t>(new_num_clusters),
                     [d_cluster_inverse, d_tmp_array] __device__(const vertex_t idx) {
                       d_cluster_inverse[d_tmp_array[idx]] = idx;
                     });

    thrust::for_each(handle_.get_thrust_policy(),
                     thrust::make_counting_iterator<vertex_t>(0),
                     thrust::make_counting_iterator<vertex_t>(old_num_clusters),
                     [d_cluster, d_cluster_inverse] __device__(vertex_t i) {
                       d_cluster[i] = d_cluster_inverse[d_cluster[i]];
                     });

    cluster_inverse_v_.resize(new_num_clusters, handle_.get_stream());

    return new_num_clusters;
  }

  void generate_superverticies_graph(graph_t& graph, vertex_t num_clusters)
  {
    rmm::device_uvector<vertex_t> new_src_v(graph.number_of_edges, handle_.get_stream());
    rmm::device_uvector<vertex_t> new_dst_v(graph.number_of_edges, handle_.get_stream());
    rmm::device_uvector<weight_t> new_weight_v(graph.number_of_edges, handle_.get_stream());

    //
    //  Renumber the COO
    //
    thrust::for_each(handle_.get_thrust_policy(),
                     thrust::make_counting_iterator<edge_t>(0),
                     thrust::make_counting_iterator<edge_t>(graph.number_of_edges),
                     [d_old_src    = src_indices_v_.data(),
                      d_old_dst    = graph.indices,
                      d_old_weight = graph.edge_data,
                      d_new_src    = new_src_v.data(),
                      d_new_dst    = new_dst_v.data(),
                      d_new_weight = new_weight_v.data(),
                      d_clusters   = dendrogram_->current_level_begin()] __device__(edge_t e) {
                       d_new_src[e]    = d_clusters[d_old_src[e]];
                       d_new_dst[e]    = d_clusters[d_old_dst[e]];
                       d_new_weight[e] = d_old_weight[e];
                     });

    thrust::stable_sort_by_key(
      handle_.get_thrust_policy(),
      new_dst_v.begin(),
      new_dst_v.end(),
      thrust::make_zip_iterator(thrust::make_tuple(new_src_v.begin(), new_weight_v.begin())));
    thrust::stable_sort_by_key(
      handle_.get_thrust_policy(),
      new_src_v.begin(),
      new_src_v.end(),
      thrust::make_zip_iterator(thrust::make_tuple(new_dst_v.begin(), new_weight_v.begin())));

    //
    //  Now we reduce by key to combine the weights of duplicate
    //  edges.
    //
    auto start =
      thrust::make_zip_iterator(thrust::make_tuple(new_src_v.begin(), new_dst_v.begin()));
    auto new_start =
      thrust::make_zip_iterator(thrust::make_tuple(src_indices_v_.data(), graph.indices));
    auto new_end = thrust::reduce_by_key(handle_.get_thrust_policy(),
                                         start,
                                         start + graph.number_of_edges,
                                         new_weight_v.begin(),
                                         new_start,
                                         graph.edge_data,
                                         thrust::equal_to<thrust::tuple<vertex_t, vertex_t>>(),
                                         thrust::plus<weight_t>());

    graph.number_of_edges    = thrust::distance(new_start, new_end.first);
    graph.number_of_vertices = num_clusters;

    detail::fill_offset(src_indices_v_.data(),
                        graph.offsets,
                        num_clusters,
                        graph.number_of_edges,
                        handle_.get_stream());

    src_indices_v_.resize(graph.number_of_edges, handle_.get_stream());
    indices_v_.resize(graph.number_of_edges, handle_.get_stream());
    weights_v_.resize(graph.number_of_edges, handle_.get_stream());
  }

 protected:
  raft::handle_t const& handle_;
  vertex_t number_of_vertices_;
  edge_t number_of_edges_;

  std::unique_ptr<Dendrogram<vertex_t>> dendrogram_;

  //
  //  Copy of graph
  //
  rmm::device_uvector<edge_t> offsets_v_;
  rmm::device_uvector<vertex_t> indices_v_;
  rmm::device_uvector<weight_t> weights_v_;
  rmm::device_uvector<vertex_t> src_indices_v_;

  //
  //  Weights and clustering across iterations of algorithm
  //
  rmm::device_uvector<weight_t> vertex_weights_v_;
  rmm::device_uvector<weight_t> cluster_weights_v_;

  //
  //  Temporaries used within kernels.  Each iteration uses less
  //  of this memory
  //
  rmm::device_uvector<vertex_t> tmp_arr_v_;
  rmm::device_uvector<vertex_t> cluster_inverse_v_;

#ifdef TIMING
  HighResTimer hr_timer_;
#endif
};

}  // namespace legacy
}  // namespace cugraph

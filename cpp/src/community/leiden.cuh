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

#include <community/louvain.cuh>

namespace cugraph {

template <typename graph_type>
class Leiden : public Louvain<graph_type> {
 public:
  using graph_t  = graph_type;
  using vertex_t = typename graph_type::vertex_type;
  using edge_t   = typename graph_type::edge_type;
  using weight_t = typename graph_type::weight_type;

  Leiden(graph_type const &graph, cudaStream_t stream)
    : Louvain<graph_type>(graph, stream), constraint_v_(graph.number_of_vertices)
  {
  }

  weight_t update_clustering_constrained(weight_t total_edge_weight,
                                         weight_t resolution,
                                         graph_type const &graph)
  {
    this->timer_start("update_clustering_constrained");

    rmm::device_vector<vertex_t> next_cluster_v(this->cluster_v_);
    rmm::device_vector<weight_t> delta_Q_v(graph.number_of_edges);
    rmm::device_vector<vertex_t> cluster_hash_v(graph.number_of_edges);
    rmm::device_vector<weight_t> old_cluster_sum_v(graph.number_of_vertices);

    vertex_t const *d_src_indices    = this->src_indices_v_.data().get();
    vertex_t const *d_dst_indices    = graph.indices;
    vertex_t *d_cluster_hash         = cluster_hash_v.data().get();
    vertex_t *d_cluster              = this->cluster_v_.data().get();
    weight_t const *d_vertex_weights = this->vertex_weights_v_.data().get();
    weight_t *d_cluster_weights      = this->cluster_weights_v_.data().get();
    weight_t *d_delta_Q              = delta_Q_v.data().get();
    vertex_t *d_constraint           = constraint_v_.data().get();

    weight_t new_Q =
      this->modularity(total_edge_weight, resolution, graph, this->cluster_v_.data().get());

    weight_t cur_Q = new_Q - 1;

    // To avoid the potential of having two vertices swap clusters
    // we will only allow vertices to move up (true) or down (false)
    // during each iteration of the loop
    bool up_down = true;

    while (new_Q > (cur_Q + 0.0001)) {
      cur_Q = new_Q;

      this->compute_delta_modularity(
        total_edge_weight, resolution, graph, cluster_hash_v, old_cluster_sum_v, delta_Q_v);

      // Filter out positive delta_Q values for nodes not in the same constraint group
      thrust::for_each(
        rmm::exec_policy(this->stream_)->on(this->stream_),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(graph.number_of_edges),
        [d_src_indices, d_dst_indices, d_constraint, d_delta_Q] __device__(vertex_t i) {
          vertex_t start_cluster = d_constraint[d_src_indices[i]];
          vertex_t end_cluster   = d_constraint[d_dst_indices[i]];
          if (start_cluster != end_cluster) d_delta_Q[i] = weight_t{0.0};
        });

      this->assign_nodes(graph, cluster_hash_v, next_cluster_v, delta_Q_v, up_down);

      up_down = !up_down;

      new_Q = this->modularity(total_edge_weight, resolution, graph, next_cluster_v.data().get());

      if (new_Q > cur_Q) {
        thrust::copy(rmm::exec_policy(this->stream_)->on(this->stream_),
                     next_cluster_v.begin(),
                     next_cluster_v.end(),
                     this->cluster_v_.begin());
      }
    }

    this->timer_stop(this->stream_);
    return cur_Q;
  }

  std::pair<int, weight_t> compute(vertex_t *d_cluster_vec, int max_level, weight_t resolution)
  {
    int num_level{0};

    weight_t total_edge_weight = thrust::reduce(rmm::exec_policy(this->stream_)->on(this->stream_),
                                                this->weights_v_.begin(),
                                                this->weights_v_.end());

    weight_t best_modularity = weight_t{-1};

    //
    //  Initialize every cluster to reference each vertex to itself
    //
    thrust::sequence(rmm::exec_policy(this->stream_)->on(this->stream_),
                     this->cluster_v_.begin(),
                     this->cluster_v_.end());
    thrust::copy(rmm::exec_policy(this->stream_)->on(this->stream_),
                 this->cluster_v_.begin(),
                 this->cluster_v_.end(),
                 d_cluster_vec);

    //
    //  Our copy of the graph.  Each iteration of the outer loop will
    //  shrink this copy of the graph.
    //
    GraphCSRView<vertex_t, edge_t, weight_t> current_graph(this->offsets_v_.data().get(),
                                                           this->indices_v_.data().get(),
                                                           this->weights_v_.data().get(),
                                                           this->number_of_vertices_,
                                                           this->number_of_edges_);

    current_graph.get_source_indices(this->src_indices_v_.data().get());

    while (num_level < max_level) {
      this->compute_vertex_and_cluster_weights(current_graph);

      weight_t new_Q = this->update_clustering(total_edge_weight, resolution, current_graph);

      thrust::copy(rmm::exec_policy(this->stream_)->on(this->stream_),
                   this->cluster_v_.begin(),
                   this->cluster_v_.end(),
                   constraint_v_.begin());

      new_Q = update_clustering_constrained(total_edge_weight, resolution, current_graph);

      if (new_Q <= best_modularity) { break; }

      best_modularity = new_Q;

      this->shrink_graph(current_graph, d_cluster_vec);

      num_level++;
    }

    this->timer_display(std::cout);

    return std::make_pair(num_level, best_modularity);
  }

 private:
  rmm::device_vector<vertex_t> constraint_v_;
};

}  // namespace cugraph

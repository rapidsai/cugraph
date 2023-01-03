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

#include <community/legacy/louvain.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

namespace cugraph {
namespace legacy {

template <typename graph_type>
class Leiden : public Louvain<graph_type> {
 public:
  using graph_t  = graph_type;
  using vertex_t = typename graph_type::vertex_type;
  using edge_t   = typename graph_type::edge_type;
  using weight_t = typename graph_type::weight_type;

  Leiden(raft::handle_t const& handle, graph_type const& graph)
    : Louvain<graph_type>(handle, graph),
      constraint_v_(graph.number_of_vertices, handle.get_stream())
  {
  }

  weight_t update_clustering_constrained(weight_t total_edge_weight,
                                         weight_t resolution,
                                         graph_type const& graph)
  {
    this->timer_start("update_clustering_constrained");

    rmm::device_uvector<vertex_t> next_cluster_v(this->dendrogram_->current_level_size(),
                                                 this->handle_.get_stream());
    rmm::device_uvector<weight_t> delta_Q_v(graph.number_of_edges, this->handle_.get_stream());
    rmm::device_uvector<vertex_t> cluster_hash_v(graph.number_of_edges, this->handle_.get_stream());
    rmm::device_uvector<weight_t> old_cluster_sum_v(graph.number_of_vertices,
                                                    this->handle_.get_stream());

    vertex_t const* d_src_indices    = this->src_indices_v_.data();
    vertex_t const* d_dst_indices    = graph.indices;
    vertex_t* d_cluster_hash         = cluster_hash_v.data();
    vertex_t* d_cluster              = this->dendrogram_->current_level_begin();
    weight_t const* d_vertex_weights = this->vertex_weights_v_.data();
    weight_t* d_cluster_weights      = this->cluster_weights_v_.data();
    weight_t* d_delta_Q              = delta_Q_v.data();
    vertex_t* d_constraint           = constraint_v_.data();

    thrust::copy(this->handle_.get_thrust_policy(),
                 this->dendrogram_->current_level_begin(),
                 this->dendrogram_->current_level_end(),
                 next_cluster_v.data());

    weight_t new_Q = this->modularity(
      total_edge_weight, resolution, graph, this->dendrogram_->current_level_begin());

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
        this->handle_.get_thrust_policy(),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(graph.number_of_edges),
        [d_src_indices, d_dst_indices, d_constraint, d_delta_Q] __device__(vertex_t i) {
          vertex_t start_cluster = d_constraint[d_src_indices[i]];
          vertex_t end_cluster   = d_constraint[d_dst_indices[i]];
          if (start_cluster != end_cluster) d_delta_Q[i] = weight_t{0.0};
        });

      this->assign_nodes(graph, cluster_hash_v, next_cluster_v, delta_Q_v, up_down);

      up_down = !up_down;

      new_Q = this->modularity(total_edge_weight, resolution, graph, next_cluster_v.data());

      if (new_Q > cur_Q) {
        thrust::copy(this->handle_.get_thrust_policy(),
                     next_cluster_v.begin(),
                     next_cluster_v.end(),
                     this->dendrogram_->current_level_begin());
      }
    }

    this->timer_stop(this->handle_.get_stream());
    return cur_Q;
  }

  weight_t operator()(size_t max_level, weight_t resolution) override
  {
    size_t num_level{0};

    weight_t total_edge_weight = thrust::reduce(
      this->handle_.get_thrust_policy(), this->weights_v_.begin(), this->weights_v_.end());

    weight_t best_modularity = weight_t{-1};

    //
    //  Our copy of the graph.  Each iteration of the outer loop will
    //  shrink this copy of the graph.
    //
    legacy::GraphCSRView<vertex_t, edge_t, weight_t> current_graph(this->offsets_v_.data(),
                                                                   this->indices_v_.data(),
                                                                   this->weights_v_.data(),
                                                                   this->number_of_vertices_,
                                                                   this->number_of_edges_);

    current_graph.get_source_indices(this->src_indices_v_.data());

    while (num_level < max_level) {
      //
      //  Initialize every cluster to reference each vertex to itself
      //
      this->dendrogram_->add_level(0, current_graph.number_of_vertices, this->handle_.get_stream());

      thrust::sequence(this->handle_.get_thrust_policy(),
                       this->dendrogram_->current_level_begin(),
                       this->dendrogram_->current_level_end());

      this->compute_vertex_and_cluster_weights(current_graph);

      weight_t new_Q = this->update_clustering(total_edge_weight, resolution, current_graph);

      new_Q = update_clustering_constrained(total_edge_weight, resolution, current_graph);

      if (new_Q <= best_modularity) { break; }

      best_modularity = new_Q;

      this->shrink_graph(current_graph);

      num_level++;
    }

    this->timer_display_and_clear(std::cout);

    return best_modularity;
  }

 private:
  rmm::device_uvector<vertex_t> constraint_v_;
};

}  // namespace legacy
}  // namespace cugraph

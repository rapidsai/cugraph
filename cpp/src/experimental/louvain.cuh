/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <community/dendrogram.cuh>

#include <experimental/graph.hpp>
#include <experimental/graph_functions.hpp>

#include <patterns/copy_to_adj_matrix_row_col.cuh>
#include <patterns/copy_v_transform_reduce_in_out_nbr.cuh>
#include <patterns/copy_v_transform_reduce_key_aggregated_out_nbr.cuh>
#include <patterns/transform_reduce_by_adj_matrix_row_col_key_e.cuh>
#include <patterns/transform_reduce_e.cuh>
#include <patterns/transform_reduce_v.cuh>

//#define TIMING

#ifdef TIMING
#include <utilities/high_res_timer.hpp>
#endif

namespace cugraph {
namespace experimental {

template <typename graph_view_type>
class Louvain {
 public:
  using graph_view_t = graph_view_type;
  using vertex_t     = typename graph_view_t::vertex_type;
  using edge_t       = typename graph_view_t::edge_type;
  using weight_t     = typename graph_view_t::weight_type;
  using graph_t      = experimental::graph_t<vertex_t,
                                        edge_t,
                                        weight_t,
                                        graph_view_t::is_adj_matrix_transposed,
                                        graph_view_t::is_multi_gpu>;

  Louvain(raft::handle_t const &handle, graph_view_t const &graph_view)
    :
#ifdef TIMING
      hr_timer_(),
#endif
      handle_(handle),
      dendrogram_(std::make_unique<Dendrogram<vertex_t>>()),
      current_graph_view_(graph_view),
      local_num_vertices_(graph_view.get_number_of_local_vertices()),
      vertex_weights_v_(graph_view.get_number_of_local_vertices(), handle.get_stream()),
      cluster_weights_v_(graph_view.get_number_of_local_vertices(), handle.get_stream()),
      src_vertex_weights_cache_v_(0, handle.get_stream()),
      src_cluster_weights_cache_v_(0, handle.get_stream()),
      dst_cluster_weights_cache_v_(0, handle.get_stream()),
      src_cluster_cache_v_(0, handle.get_stream()),
      dst_cluster_cache_v_(0, handle.get_stream()),
      stream_(handle.get_stream())
  {
    if (graph_view_t::is_multi_gpu) {
      rank_           = handle.get_comms().get_rank();
      base_vertex_id_ = graph_view.get_local_vertex_first();
    }
  }

  Dendrogram<vertex_t> const &get_dendrogram() const { return *dendrogram_; }

  Dendrogram<vertex_t> &get_dendrogram() { return *dendrogram_; }

  std::unique_ptr<Dendrogram<vertex_t>> move_dendrogram() { return dendrogram_; }

  virtual weight_t operator()(size_t max_level, weight_t resolution)
  {
    weight_t best_modularity = weight_t{-1};

    weight_t total_edge_weight = experimental::transform_reduce_e(
      handle_,
      current_graph_view_,
      thrust::make_constant_iterator(0),
      thrust::make_constant_iterator(0),
      [] __device__(auto, auto, weight_t wt, auto, auto) { return wt; },
      weight_t{0});

    while (dendrogram_->num_levels() < max_level) {
      //
      //  Initialize every cluster to reference each vertex to itself
      //
      initialize_dendrogram_level(current_graph_view_.get_number_of_local_vertices());

      compute_vertex_and_cluster_weights();

      weight_t new_Q = update_clustering(total_edge_weight, resolution);

      if (new_Q <= best_modularity) { break; }

      best_modularity = new_Q;

      shrink_graph();
    }

    timer_display(std::cout);

    return best_modularity;
  }

 protected:
  void timer_start(std::string const &region)
  {
#ifdef TIMING
    if (rank_ == 0) hr_timer_.start(region);
#endif
  }

  void timer_stop(cudaStream_t stream)
  {
#ifdef TIMING
    if (rank_ == 0) {
      CUDA_TRY(cudaStreamSynchronize(stream));
      hr_timer_.stop();
    }
#endif
  }

  void timer_display(std::ostream &os)
  {
#ifdef TIMING
    if (rank_ == 0) hr_timer_.display(os);
#endif
  }

 protected:
  void initialize_dendrogram_level(vertex_t num_vertices)
  {
    dendrogram_->add_level(num_vertices);

    thrust::sequence(rmm::exec_policy(stream_)->on(stream_),
                     dendrogram_->current_level_begin(),
                     dendrogram_->current_level_end(),
                     base_vertex_id_);
  }

 public:
  weight_t modularity(weight_t total_edge_weight, weight_t resolution)
  {
    weight_t sum_degree_squared = experimental::transform_reduce_v(
      handle_,
      current_graph_view_,
      cluster_weights_v_.begin(),
      [] __device__(weight_t p) { return p * p; },
      weight_t{0});

    weight_t sum_internal = experimental::transform_reduce_e(
      handle_,
      current_graph_view_,
      d_src_cluster_cache_,
      d_dst_cluster_cache_,
      [] __device__(auto src, auto dst, weight_t wt, auto src_cluster, auto nbr_cluster) {
        if (src_cluster == nbr_cluster) {
          return wt;
        } else {
          return weight_t{0};
        }
      },
      weight_t{0});

    weight_t Q = sum_internal / total_edge_weight -
                 (resolution * sum_degree_squared) / (total_edge_weight * total_edge_weight);

    return Q;
  }

  void compute_vertex_and_cluster_weights()
  {
    timer_start("compute_vertex_and_cluster_weights");

    //
    // TODO: Once PR 1394 is merged, this can be replaced by:
    //    vertex_weights_v_ = current_graph_view_.compute_out_weight_sums(handle_);
    //
    experimental::copy_v_transform_reduce_out_nbr(
      handle_,
      current_graph_view_,
      thrust::make_constant_iterator(0),
      thrust::make_constant_iterator(0),
      [] __device__(auto src, auto, auto wt, auto, auto) { return wt; },
      weight_t{0},
      vertex_weights_v_.begin());

    thrust::copy(rmm::exec_policy(stream_)->on(stream_),
                 vertex_weights_v_.begin(),
                 vertex_weights_v_.end(),
                 cluster_weights_v_.begin());

    d_src_vertex_weights_cache_ =
      cache_src_vertex_properties(vertex_weights_v_, src_vertex_weights_cache_v_);

    std::tie(d_src_cluster_weights_cache_, d_dst_cluster_weights_cache_) = cache_vertex_properties(
      cluster_weights_v_, src_cluster_weights_cache_v_, dst_cluster_weights_cache_v_);

    timer_stop(stream_);
  }

  template <typename T>
  T *cache_src_vertex_properties(rmm::device_uvector<T> &input, rmm::device_uvector<T> &src_cache_v)
  {
    if (graph_view_t::is_multi_gpu) {
      src_cache_v.resize(current_graph_view_.get_number_of_local_adj_matrix_partition_rows(),
                         stream_);
      copy_to_adj_matrix_row(handle_, current_graph_view_, input.begin(), src_cache_v.begin());
      return src_cache_v.begin();
    } else {
      return input.begin();
    }
  }

  template <typename T>
  T *cache_dst_vertex_properties(rmm::device_uvector<T> &input, rmm::device_uvector<T> &dst_cache_v)
  {
    if (graph_view_t::is_multi_gpu) {
      dst_cache_v.resize(current_graph_view_.get_number_of_local_adj_matrix_partition_cols(),
                         stream_);
      copy_to_adj_matrix_col(handle_, current_graph_view_, input.begin(), dst_cache_v.begin());
      return dst_cache_v.begin();
    } else {
      return input.begin();
    }
  }

  template <typename T>
  std::tuple<T *, T *> cache_vertex_properties(rmm::device_uvector<T> &input,
                                               rmm::device_uvector<T> &src_cache_v,
                                               rmm::device_uvector<T> &dst_cache_v)
  {
    auto src = cache_src_vertex_properties(input, src_cache_v);
    auto dst = cache_dst_vertex_properties(input, dst_cache_v);

    return std::make_tuple(src, dst);
  }

  virtual weight_t update_clustering(weight_t total_edge_weight, weight_t resolution)
  {
    timer_start("update_clustering");

    rmm::device_uvector<vertex_t> next_cluster_v(dendrogram_->current_level_size(), stream_);

    raft::copy(next_cluster_v.begin(),
               dendrogram_->current_level_begin(),
               dendrogram_->current_level_size(),
               stream_);

    std::tie(d_src_cluster_cache_, d_dst_cluster_cache_) =
      cache_vertex_properties(next_cluster_v, src_cluster_cache_v_, dst_cluster_cache_v_);

    weight_t new_Q = modularity(total_edge_weight, resolution);
    weight_t cur_Q = new_Q - 1;

    // To avoid the potential of having two vertices swap clusters
    // we will only allow vertices to move up (true) or down (false)
    // during each iteration of the loop
    bool up_down = true;

    while (new_Q > (cur_Q + 0.0001)) {
      cur_Q = new_Q;

      update_by_delta_modularity(total_edge_weight, resolution, next_cluster_v, up_down);

      up_down = !up_down;

      new_Q = modularity(total_edge_weight, resolution);

      if (new_Q > cur_Q) {
        thrust::copy(rmm::exec_policy(stream_)->on(stream_),
                     next_cluster_v.begin(),
                     next_cluster_v.end(),
                     dendrogram_->current_level_begin());
      }
    }

    timer_stop(stream_);
    return cur_Q;
  }

  void update_by_delta_modularity(weight_t total_edge_weight,
                                  weight_t resolution,
                                  rmm::device_uvector<vertex_t> &next_cluster_v,
                                  bool up_down)
  {
    rmm::device_uvector<weight_t> old_cluster_sum_v(local_num_vertices_, stream_);
    rmm::device_uvector<weight_t> cluster_subtract_v(local_num_vertices_, stream_);

    rmm::device_uvector<vertex_t> tmp_cluster_keys_v(0, stream_);
    rmm::device_uvector<weight_t> tmp_cluster_weights_v(0, stream_);

    experimental::copy_v_transform_reduce_out_nbr(
      handle_,
      current_graph_view_,
      d_src_cluster_cache_,
      d_dst_cluster_cache_,
      [] __device__(auto src, auto dst, auto wt, auto src_cluster, auto nbr_cluster) {
        if ((src != dst) && (src_cluster == nbr_cluster)) {
          return wt;
        } else
          return weight_t{0};
      },
      weight_t{0},
      old_cluster_sum_v.begin());

    experimental::copy_v_transform_reduce_out_nbr(
      handle_,
      current_graph_view_,
      d_src_cluster_cache_,
      d_dst_cluster_cache_,
      [] __device__(auto src, auto dst, auto wt, auto src_cluster, auto nbr_cluster) {
        return (src == dst) ? wt : weight_t{0};
      },
      weight_t{0},
      cluster_subtract_v.begin());

    auto output_buffer =
      cugraph::experimental::allocate_dataframe_buffer<thrust::tuple<vertex_t, weight_t>>(
        local_num_vertices_, stream_);

    copy_v_transform_reduce_key_aggregated_out_nbr(
      handle_,
      current_graph_view_,
      thrust::make_zip_iterator(thrust::make_tuple(old_cluster_sum_v.begin(),
                                                   d_src_vertex_weights_cache_,
                                                   cluster_subtract_v.begin(),
                                                   d_src_cluster_cache_)),

      d_dst_cluster_cache_,
      thrust::make_counting_iterator<vertex_t>(base_vertex_id_),
      thrust::make_counting_iterator<vertex_t>(base_vertex_id_ + local_num_vertices_),
      d_dst_cluster_weights_cache_,
      [base_vertex_id        = base_vertex_id_,
       d_src_cluster_weights = d_src_cluster_weights_cache_,
       total_edge_weight,
       resolution] __device__(auto src,
                              auto neighbor_cluster,
                              auto new_cluster_sum,
                              auto src_info,
                              auto a_new) {
        auto old_cluster_sum  = thrust::get<0>(src_info);
        auto k_k              = thrust::get<1>(src_info);
        auto cluster_subtract = thrust::get<2>(src_info);
        auto src_cluster      = thrust::get<3>(src_info);
        auto a_old            = d_src_cluster_weights[src_cluster];

        if (src_cluster == neighbor_cluster) new_cluster_sum -= cluster_subtract;

        weight_t delta_modularity = 2 * (((new_cluster_sum - old_cluster_sum) / total_edge_weight) -
                                         resolution * (a_new * k_k - a_old * k_k + k_k * k_k) /
                                           (total_edge_weight * total_edge_weight));

        return thrust::make_tuple(neighbor_cluster, delta_modularity);
      },
      [] __device__(auto p1, auto p2) {
        return (thrust::get<1>(p1) < thrust::get<1>(p2)) ? p2 : p1;
      },
      thrust::make_tuple(vertex_t{-1}, weight_t{0}),
      cugraph::experimental::get_dataframe_buffer_begin<thrust::tuple<vertex_t, weight_t>>(
        output_buffer));

    thrust::transform(
      rmm::exec_policy(stream_)->on(stream_),
      next_cluster_v.begin(),
      next_cluster_v.end(),
      cugraph::experimental::get_dataframe_buffer_begin<thrust::tuple<vertex_t, weight_t>>(
        output_buffer),
      next_cluster_v.begin(),
      [up_down] __device__(vertex_t old_cluster, auto p) {
        vertex_t new_cluster      = thrust::get<0>(p);
        weight_t delta_modularity = thrust::get<1>(p);

        return (delta_modularity > weight_t{0})
                 ? (((new_cluster > old_cluster) != up_down) ? old_cluster : new_cluster)
                 : old_cluster;
      });

    std::tie(d_src_cluster_cache_, d_dst_cluster_cache_) =
      cache_vertex_properties(next_cluster_v, src_cluster_cache_v_, dst_cluster_cache_v_);

    std::tie(tmp_cluster_keys_v, tmp_cluster_weights_v) =
      cugraph::experimental::transform_reduce_by_adj_matrix_row_key_e(
        handle_,
        current_graph_view_,
        thrust::make_constant_iterator(0),
        thrust::make_constant_iterator(0),
        d_src_cluster_cache_,
        [] __device__(auto src, auto, auto wt, auto, auto) { return wt; },
        weight_t{0});

    thrust::fill(rmm::exec_policy(stream_)->on(stream_),
                 cluster_weights_v_.begin(),
                 cluster_weights_v_.end(),
                 weight_t{0});

    thrust::scatter(rmm::exec_policy(stream_)->on(stream_),
                    tmp_cluster_weights_v.begin(),
                    tmp_cluster_weights_v.end(),
                    tmp_cluster_keys_v.begin(),
                    cluster_weights_v_.begin());

    std::tie(d_src_cluster_weights_cache_, d_dst_cluster_weights_cache_) = cache_vertex_properties(
      cluster_weights_v_, src_cluster_weights_cache_v_, dst_cluster_weights_cache_v_);
  }

  void shrink_graph()
  {
    timer_start("shrinking graph");

    rmm::device_uvector<vertex_t> numbering_map(0, stream_);

    std::tie(current_graph_, numbering_map) =
      coarsen_graph(handle_, current_graph_view_, dendrogram_->current_level_begin());

    current_graph_view_ = current_graph_->view();

    local_num_vertices_ = current_graph_view_.get_number_of_local_vertices();
    base_vertex_id_     = current_graph_view_.get_local_vertex_first();

    rmm::device_uvector<vertex_t> numbering_indices(numbering_map.size(), stream_);
    thrust::sequence(rmm::exec_policy(stream_)->on(stream_),
                     numbering_indices.begin(),
                     numbering_indices.end(),
                     base_vertex_id_);

    relabel<vertex_t, graph_view_t::is_multi_gpu>(
      handle_,
      std::make_tuple(static_cast<vertex_t const *>(numbering_map.begin()),
                      static_cast<vertex_t const *>(numbering_indices.begin())),
      local_num_vertices_,
      dendrogram_->current_level_begin(),
      dendrogram_->current_level_size());

    timer_stop(stream_);
  }

 protected:
  raft::handle_t const &handle_;
  cudaStream_t stream_;

  std::unique_ptr<Dendrogram<vertex_t>> dendrogram_;

  vertex_t local_num_vertices_;
  vertex_t base_vertex_id_{0};
  int rank_{0};

  //
  //  Initially we run on the input graph view,
  //  but as we shrink the graph we'll keep the
  //  current graph here
  //
  std::unique_ptr<graph_t> current_graph_{};
  graph_view_t current_graph_view_;

  rmm::device_uvector<weight_t> vertex_weights_v_;
  rmm::device_uvector<weight_t> cluster_weights_v_;
  rmm::device_uvector<weight_t> src_vertex_weights_cache_v_;
  rmm::device_uvector<weight_t> src_cluster_weights_cache_v_;
  rmm::device_uvector<weight_t> dst_cluster_weights_cache_v_;
  rmm::device_uvector<vertex_t> src_cluster_cache_v_;
  rmm::device_uvector<vertex_t> dst_cluster_cache_v_;

  weight_t *d_src_vertex_weights_cache_;
  weight_t *d_src_cluster_weights_cache_;
  weight_t *d_dst_cluster_weights_cache_;
  vertex_t *d_src_cluster_cache_;
  vertex_t *d_dst_cluster_cache_;

#ifdef TIMING
  HighResTimer hr_timer_;
#endif
};

}  // namespace experimental
}  // namespace cugraph

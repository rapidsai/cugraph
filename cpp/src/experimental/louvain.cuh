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

#include <cugraph/dendrogram.hpp>

#include <cugraph/experimental/graph.hpp>
#include <cugraph/experimental/graph_functions.hpp>

#include <cugraph/patterns/copy_to_adj_matrix_row_col.cuh>
#include <cugraph/patterns/copy_v_transform_reduce_in_out_nbr.cuh>
#include <cugraph/patterns/copy_v_transform_reduce_key_aggregated_out_nbr.cuh>
#include <cugraph/patterns/transform_reduce_by_adj_matrix_row_col_key_e.cuh>
#include <cugraph/patterns/transform_reduce_e.cuh>
#include <cugraph/patterns/transform_reduce_v.cuh>
#include <cugraph/utilities/collect_comm.cuh>

#include <thrust/binary_search.h>
#include <thrust/transform_reduce.h>

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
      cluster_keys_v_(graph_view.get_number_of_local_vertices(), handle.get_stream()),
      cluster_weights_v_(graph_view.get_number_of_local_vertices(), handle.get_stream()),
      vertex_weights_v_(graph_view.get_number_of_local_vertices(), handle.get_stream()),
      src_vertex_weights_cache_v_(0, handle.get_stream()),
      src_cluster_cache_v_(0, handle.get_stream()),
      dst_cluster_cache_v_(0, handle.get_stream())
  {
  }

  Dendrogram<vertex_t> const &get_dendrogram() const { return *dendrogram_; }

  Dendrogram<vertex_t> &get_dendrogram() { return *dendrogram_; }

  std::unique_ptr<Dendrogram<vertex_t>> move_dendrogram() { return std::move(dendrogram_); }

  virtual weight_t operator()(size_t max_level, weight_t resolution)
  {
    weight_t best_modularity = weight_t{-1};

    weight_t total_edge_weight = experimental::transform_reduce_e(
      handle_,
      current_graph_view_,
      thrust::make_constant_iterator(0),
      thrust::make_constant_iterator(0),
      [] __device__(auto src, auto dst, weight_t wt, auto, auto) { return wt; },
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
    if (graph_view_t::is_multi_gpu) {
      if (handle.get_comms().get_rank() == 0) hr_timer_.start(region);
    } else {
      hr_timer_.start(region);
    }
#endif
  }

  void timer_stop(cudaStream_t stream)
  {
#ifdef TIMING
    if (graph_view_t::is_multi_gpu) {
      if (handle.get_comms().get_rank() == 0) {
        CUDA_TRY(cudaStreamSynchronize(stream));
        hr_timer_.stop();
      }
    } else {
      CUDA_TRY(cudaStreamSynchronize(stream));
      hr_timer_.stop();
    }
#endif
  }

  void timer_display(std::ostream &os)
  {
#ifdef TIMING
    if (graph_view_t::is_multi_gpu) {
      if (handle.get_comms().get_rank() == 0) hr_timer_.display(os);
    } else {
      hr_timer_.display(os);
    }
#endif
  }

 protected:
  void initialize_dendrogram_level(vertex_t num_vertices)
  {
    dendrogram_->add_level(
      current_graph_view_.get_local_vertex_first(), num_vertices, handle_.get_stream());

    thrust::sequence(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
                     dendrogram_->current_level_begin(),
                     dendrogram_->current_level_end(),
                     current_graph_view_.get_local_vertex_first());
  }

 public:
  weight_t modularity(weight_t total_edge_weight, weight_t resolution)
  {
    weight_t sum_degree_squared = thrust::transform_reduce(
      rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
      cluster_weights_v_.begin(),
      cluster_weights_v_.end(),
      [] __device__(weight_t p) { return p * p; },
      weight_t{0},
      thrust::plus<weight_t>());

    if (graph_t::is_multi_gpu) {
      sum_degree_squared =
        host_scalar_allreduce(handle_.get_comms(), sum_degree_squared, handle_.get_stream());
    }

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

    vertex_weights_v_ = current_graph_view_.compute_out_weight_sums(handle_);
    cluster_keys_v_.resize(vertex_weights_v_.size(), handle_.get_stream());
    cluster_weights_v_.resize(vertex_weights_v_.size(), handle_.get_stream());

    thrust::sequence(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
                     cluster_keys_v_.begin(),
                     cluster_keys_v_.end(),
                     current_graph_view_.get_local_vertex_first());

    raft::copy(cluster_weights_v_.begin(),
               vertex_weights_v_.begin(),
               vertex_weights_v_.size(),
               handle_.get_stream());

    d_src_vertex_weights_cache_ =
      cache_src_vertex_properties(vertex_weights_v_, src_vertex_weights_cache_v_);

    if (graph_view_t::is_multi_gpu) {
      auto const comm_size = handle_.get_comms().get_size();
      rmm::device_uvector<vertex_t> rx_keys_v(0, handle_.get_stream());
      rmm::device_uvector<weight_t> rx_weights_v(0, handle_.get_stream());

      auto pair_first = thrust::make_zip_iterator(
        thrust::make_tuple(cluster_keys_v_.begin(), cluster_weights_v_.begin()));

      std::forward_as_tuple(std::tie(rx_keys_v, rx_weights_v), std::ignore) =
        groupby_gpuid_and_shuffle_values(
          handle_.get_comms(),
          pair_first,
          pair_first + current_graph_view_.get_number_of_local_vertices(),
          [key_func =
             cugraph::experimental::detail::compute_gpu_id_from_vertex_t<vertex_t>{
               comm_size}] __device__(auto val) { return key_func(thrust::get<0>(val)); },
          handle_.get_stream());

      cluster_keys_v_    = std::move(rx_keys_v);
      cluster_weights_v_ = std::move(rx_weights_v);
    }

    timer_stop(handle_.get_stream());
  }

  template <typename T>
  T *cache_src_vertex_properties(rmm::device_uvector<T> &input, rmm::device_uvector<T> &src_cache_v)
  {
    if (graph_view_t::is_multi_gpu) {
      src_cache_v.resize(current_graph_view_.get_number_of_local_adj_matrix_partition_rows(),
                         handle_.get_stream());
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
                         handle_.get_stream());
      copy_to_adj_matrix_col(handle_, current_graph_view_, input.begin(), dst_cache_v.begin());
      return dst_cache_v.begin();
    } else {
      return input.begin();
    }
  }

  virtual weight_t update_clustering(weight_t total_edge_weight, weight_t resolution)
  {
    timer_start("update_clustering");

    rmm::device_uvector<vertex_t> next_cluster_v(dendrogram_->current_level_size(),
                                                 handle_.get_stream());

    raft::copy(next_cluster_v.begin(),
               dendrogram_->current_level_begin(),
               dendrogram_->current_level_size(),
               handle_.get_stream());

    d_src_cluster_cache_ = cache_src_vertex_properties(next_cluster_v, src_cluster_cache_v_);
    d_dst_cluster_cache_ = cache_dst_vertex_properties(next_cluster_v, dst_cluster_cache_v_);

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
        raft::copy(dendrogram_->current_level_begin(),
                   next_cluster_v.begin(),
                   next_cluster_v.size(),
                   handle_.get_stream());
      }
    }

    timer_stop(handle_.get_stream());
    return cur_Q;
  }

  void compute_cluster_sum_and_subtract(rmm::device_uvector<weight_t> &old_cluster_sum_v,
                                        rmm::device_uvector<weight_t> &cluster_subtract_v)
  {
    auto output_buffer =
      cugraph::experimental::allocate_dataframe_buffer<thrust::tuple<weight_t, weight_t>>(
        current_graph_view_.get_number_of_local_vertices(), handle_.get_stream());

    experimental::copy_v_transform_reduce_out_nbr(
      handle_,
      current_graph_view_,
      d_src_cluster_cache_,
      d_dst_cluster_cache_,
      [] __device__(auto src, auto dst, auto wt, auto src_cluster, auto nbr_cluster) {
        weight_t subtract{0};
        weight_t sum{0};

        if (src == dst)
          subtract = wt;
        else if (src_cluster == nbr_cluster)
          sum = wt;

        return thrust::make_tuple(subtract, sum);
      },
      thrust::make_tuple(weight_t{0}, weight_t{0}),
      cugraph::experimental::get_dataframe_buffer_begin<thrust::tuple<weight_t, weight_t>>(
        output_buffer));

    thrust::transform(
      rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
      cugraph::experimental::get_dataframe_buffer_begin<thrust::tuple<weight_t, weight_t>>(
        output_buffer),
      cugraph::experimental::get_dataframe_buffer_begin<thrust::tuple<weight_t, weight_t>>(
        output_buffer) +
        current_graph_view_.get_number_of_local_vertices(),
      old_cluster_sum_v.begin(),
      [] __device__(auto p) { return thrust::get<1>(p); });

    thrust::transform(
      rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
      cugraph::experimental::get_dataframe_buffer_begin<thrust::tuple<weight_t, weight_t>>(
        output_buffer),
      cugraph::experimental::get_dataframe_buffer_begin<thrust::tuple<weight_t, weight_t>>(
        output_buffer) +
        current_graph_view_.get_number_of_local_vertices(),
      cluster_subtract_v.begin(),
      [] __device__(auto p) { return thrust::get<0>(p); });
  }

  void update_by_delta_modularity(weight_t total_edge_weight,
                                  weight_t resolution,
                                  rmm::device_uvector<vertex_t> &next_cluster_v,
                                  bool up_down)
  {
#ifdef CUCO_STATIC_MAP_DEFINED
    rmm::device_uvector<weight_t> old_cluster_sum_v(
      current_graph_view_.get_number_of_local_vertices(), handle_.get_stream());
    rmm::device_uvector<weight_t> cluster_subtract_v(
      current_graph_view_.get_number_of_local_vertices(), handle_.get_stream());
    rmm::device_uvector<weight_t> src_cluster_weights_v(next_cluster_v.size(),
                                                        handle_.get_stream());

    compute_cluster_sum_and_subtract(old_cluster_sum_v, cluster_subtract_v);

    auto output_buffer =
      cugraph::experimental::allocate_dataframe_buffer<thrust::tuple<vertex_t, weight_t>>(
        current_graph_view_.get_number_of_local_vertices(), handle_.get_stream());

    vertex_t *map_key_first;
    vertex_t *map_key_last;
    weight_t *map_value_first;

    if (graph_t::is_multi_gpu) {
      cugraph::experimental::detail::compute_gpu_id_from_vertex_t<vertex_t> vertex_to_gpu_id_op{
        handle_.get_comms().get_size()};

      src_cluster_weights_v = cugraph::experimental::collect_values_for_keys(
        handle_.get_comms(),
        cluster_keys_v_.begin(),
        cluster_keys_v_.end(),
        cluster_weights_v_.data(),
        d_src_cluster_cache_,
        d_src_cluster_cache_ + src_cluster_cache_v_.size(),
        vertex_to_gpu_id_op,
        handle_.get_stream());

      map_key_first   = cluster_keys_v_.begin();
      map_key_last    = cluster_keys_v_.end();
      map_value_first = cluster_weights_v_.begin();
    } else {
      thrust::sort_by_key(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
                          cluster_keys_v_.begin(),
                          cluster_keys_v_.end(),
                          cluster_weights_v_.begin());

      thrust::transform(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
                        next_cluster_v.begin(),
                        next_cluster_v.end(),
                        src_cluster_weights_v.begin(),
                        [d_cluster_weights = cluster_weights_v_.data(),
                         d_cluster_keys    = cluster_keys_v_.data(),
                         num_clusters      = cluster_keys_v_.size()] __device__(vertex_t cluster) {
                          auto pos = thrust::lower_bound(
                            thrust::seq, d_cluster_keys, d_cluster_keys + num_clusters, cluster);
                          return d_cluster_weights[pos - d_cluster_keys];
                        });

      map_key_first   = d_src_cluster_cache_;
      map_key_last    = d_src_cluster_cache_ + src_cluster_weights_v.size();
      map_value_first = src_cluster_weights_v.begin();
    }

    rmm::device_uvector<weight_t> src_old_cluster_sum_v(
      current_graph_view_.get_number_of_local_adj_matrix_partition_rows(), handle_.get_stream());
    rmm::device_uvector<weight_t> src_cluster_subtract_v(
      current_graph_view_.get_number_of_local_adj_matrix_partition_rows(), handle_.get_stream());
    copy_to_adj_matrix_row(
      handle_, current_graph_view_, old_cluster_sum_v.begin(), src_old_cluster_sum_v.begin());
    copy_to_adj_matrix_row(
      handle_, current_graph_view_, cluster_subtract_v.begin(), src_cluster_subtract_v.begin());

    copy_v_transform_reduce_key_aggregated_out_nbr(
      handle_,
      current_graph_view_,
      thrust::make_zip_iterator(thrust::make_tuple(src_old_cluster_sum_v.begin(),
                                                   d_src_vertex_weights_cache_,
                                                   src_cluster_subtract_v.begin(),
                                                   d_src_cluster_cache_,
                                                   src_cluster_weights_v.begin())),

      d_dst_cluster_cache_,
      map_key_first,
      map_key_last,
      map_value_first,
      [total_edge_weight, resolution] __device__(
        auto src, auto neighbor_cluster, auto new_cluster_sum, auto src_info, auto a_new) {
        auto old_cluster_sum  = thrust::get<0>(src_info);
        auto k_k              = thrust::get<1>(src_info);
        auto cluster_subtract = thrust::get<2>(src_info);
        auto src_cluster      = thrust::get<3>(src_info);
        auto a_old            = thrust::get<4>(src_info);

        if (src_cluster == neighbor_cluster) new_cluster_sum -= cluster_subtract;

        weight_t delta_modularity = 2 * (((new_cluster_sum - old_cluster_sum) / total_edge_weight) -
                                         resolution * (a_new * k_k - a_old * k_k + k_k * k_k) /
                                           (total_edge_weight * total_edge_weight));

        return thrust::make_tuple(neighbor_cluster, delta_modularity);
      },
      [] __device__(auto p1, auto p2) {
        auto id1 = thrust::get<0>(p1);
        auto id2 = thrust::get<0>(p2);
        auto wt1 = thrust::get<1>(p1);
        auto wt2 = thrust::get<1>(p2);

        return (wt1 < wt2) ? p2 : ((wt1 > wt2) ? p1 : ((id1 < id2) ? p1 : p2));
      },
      thrust::make_tuple(vertex_t{-1}, weight_t{0}),
      cugraph::experimental::get_dataframe_buffer_begin<thrust::tuple<vertex_t, weight_t>>(
        output_buffer));

    thrust::transform(
      rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
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

    d_src_cluster_cache_ = cache_src_vertex_properties(next_cluster_v, src_cluster_cache_v_);
    d_dst_cluster_cache_ = cache_dst_vertex_properties(next_cluster_v, dst_cluster_cache_v_);

    std::tie(cluster_keys_v_, cluster_weights_v_) =
      cugraph::experimental::transform_reduce_by_adj_matrix_row_key_e(
        handle_,
        current_graph_view_,
        thrust::make_constant_iterator(0),
        thrust::make_constant_iterator(0),
        d_src_cluster_cache_,
        [] __device__(auto src, auto dst, auto wt, auto x, auto y) { return wt; },
        weight_t{0});
#endif
  }

  void shrink_graph()
  {
    timer_start("shrinking graph");

    rmm::device_uvector<vertex_t> numbering_map(0, handle_.get_stream());

    std::tie(current_graph_, numbering_map) =
      coarsen_graph(handle_, current_graph_view_, dendrogram_->current_level_begin());

    current_graph_view_ = current_graph_->view();

    rmm::device_uvector<vertex_t> numbering_indices(numbering_map.size(), handle_.get_stream());
    thrust::sequence(rmm::exec_policy(handle_.get_stream())->on(handle_.get_stream()),
                     numbering_indices.begin(),
                     numbering_indices.end(),
                     current_graph_view_.get_local_vertex_first());

    relabel<vertex_t, graph_view_t::is_multi_gpu>(
      handle_,
      std::make_tuple(static_cast<vertex_t const *>(numbering_map.begin()),
                      static_cast<vertex_t const *>(numbering_indices.begin())),
      current_graph_view_.get_number_of_local_vertices(),
      dendrogram_->current_level_begin(),
      dendrogram_->current_level_size(),
      false);

    timer_stop(handle_.get_stream());
  }

 protected:
  raft::handle_t const &handle_;

  std::unique_ptr<Dendrogram<vertex_t>> dendrogram_;

  //
  //  Initially we run on the input graph view,
  //  but as we shrink the graph we'll keep the
  //  current graph here
  //
  std::unique_ptr<graph_t> current_graph_{};
  graph_view_t current_graph_view_;

  rmm::device_uvector<weight_t> vertex_weights_v_;
  rmm::device_uvector<weight_t> src_vertex_weights_cache_v_;
  rmm::device_uvector<vertex_t> src_cluster_cache_v_;
  rmm::device_uvector<vertex_t> dst_cluster_cache_v_;
  rmm::device_uvector<vertex_t> cluster_keys_v_;
  rmm::device_uvector<weight_t> cluster_weights_v_;

  weight_t *d_src_vertex_weights_cache_;
  vertex_t *d_src_cluster_cache_;
  vertex_t *d_dst_cluster_cache_;

#ifdef TIMING
  HighResTimer hr_timer_;
#endif
};

}  // namespace experimental
}  // namespace cugraph

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

#include <cugraph/dendrogram.hpp>

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>

#include <cugraph/prims/copy_v_transform_reduce_in_out_nbr.cuh>
#include <cugraph/prims/copy_v_transform_reduce_key_aggregated_out_nbr.cuh>
#include <cugraph/prims/edge_partition_src_dst_property.cuh>
#include <cugraph/prims/transform_reduce_by_src_dst_key_e.cuh>
#include <cugraph/prims/transform_reduce_e.cuh>
#include <cugraph/prims/transform_reduce_v.cuh>
#include <cugraph/prims/update_edge_partition_src_dst_property.cuh>
#include <cugraph/utilities/collect_comm.cuh>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/transform_reduce.h>

//#define TIMING

#ifdef TIMING
#include <utilities/high_res_timer.hpp>
#endif

namespace cugraph {

namespace detail {

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t>
struct key_aggregated_edge_op_t {
  weight_t total_edge_weight{};
  weight_t resolution{};
  __device__ auto operator()(
    vertex_t src,
    vertex_t neighbor_cluster,
    weight_t new_cluster_sum,
    thrust::tuple<weight_t, vertex_t, weight_t, weight_t, weight_t> src_info,
    weight_t a_new) const
  {
    auto k_k              = thrust::get<0>(src_info);
    auto src_cluster      = thrust::get<1>(src_info);
    auto a_old            = thrust::get<2>(src_info);
    auto old_cluster_sum  = thrust::get<3>(src_info);
    auto cluster_subtract = thrust::get<4>(src_info);

    if (src_cluster == neighbor_cluster) new_cluster_sum -= cluster_subtract;

    weight_t delta_modularity = 2 * (((new_cluster_sum - old_cluster_sum) / total_edge_weight) -
                                     resolution * (a_new * k_k - a_old * k_k + k_k * k_k) /
                                       (total_edge_weight * total_edge_weight));

    return thrust::make_tuple(neighbor_cluster, delta_modularity);
  }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t>
struct reduce_op_t {
  __device__ auto operator()(thrust::tuple<vertex_t, weight_t> p0,
                             thrust::tuple<vertex_t, weight_t> p1) const
  {
    auto id0 = thrust::get<0>(p0);
    auto id1 = thrust::get<0>(p1);
    auto wt0 = thrust::get<1>(p0);
    auto wt1 = thrust::get<1>(p1);

    return (wt0 < wt1) ? p1 : ((wt0 > wt1) ? p0 : ((id0 < id1) ? p0 : p1));
  }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t>
struct cluster_update_op_t {
  bool up_down{};
  __device__ auto operator()(vertex_t old_cluster, thrust::tuple<vertex_t, weight_t> p) const
  {
    vertex_t new_cluster      = thrust::get<0>(p);
    weight_t delta_modularity = thrust::get<1>(p);

    return (delta_modularity > weight_t{0})
             ? (((new_cluster > old_cluster) != up_down) ? old_cluster : new_cluster)
             : old_cluster;
  }
};

// a workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
template <typename vertex_t, typename weight_t>
struct return_edge_weight_t {
  __device__ auto operator()(
    vertex_t, vertex_t, weight_t w, thrust::nullopt_t, thrust::nullopt_t) const
  {
    return w;
  }
};

}  // namespace detail

template <typename graph_view_type>
class Louvain {
 public:
  using graph_view_t = graph_view_type;
  using vertex_t     = typename graph_view_t::vertex_type;
  using edge_t       = typename graph_view_t::edge_type;
  using weight_t     = typename graph_view_t::weight_type;
  using graph_t      = graph_t<vertex_t,
                          edge_t,
                          weight_t,
                          graph_view_t::is_storage_transposed,
                          graph_view_t::is_multi_gpu>;

  static_assert(!graph_view_t::is_storage_transposed);

  Louvain(raft::handle_t const& handle, graph_view_t const& graph_view)
    :
#ifdef TIMING
      hr_timer_(),
#endif
      handle_(handle),
      dendrogram_(std::make_unique<Dendrogram<vertex_t>>()),
      current_graph_(handle),
      current_graph_view_(graph_view),
      cluster_keys_v_(0, handle.get_stream()),
      cluster_weights_v_(0, handle.get_stream()),
      vertex_weights_v_(0, handle.get_stream()),
      src_vertex_weights_cache_(handle),
      next_clusters_v_(0, handle.get_stream()),
      src_clusters_cache_(handle),
      dst_clusters_cache_(handle)
  {
  }

  Dendrogram<vertex_t> const& get_dendrogram() const { return *dendrogram_; }

  Dendrogram<vertex_t>& get_dendrogram() { return *dendrogram_; }

  std::unique_ptr<Dendrogram<vertex_t>> move_dendrogram() { return std::move(dendrogram_); }

  virtual weight_t operator()(size_t max_level, weight_t resolution)
  {
    weight_t best_modularity = weight_t{-1};

    weight_t total_edge_weight = transform_reduce_e(
      handle_,
      current_graph_view_,
      dummy_property_t<vertex_t>{}.device_view(),
      dummy_property_t<vertex_t>{}.device_view(),
      [] __device__(auto, auto, weight_t wt, auto, auto) { return wt; },
      weight_t{0});

    while (dendrogram_->num_levels() < max_level) {
      //
      //  Initialize every cluster to reference each vertex to itself
      //
      initialize_dendrogram_level();

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
  void timer_start(std::string const& region)
  {
#ifdef TIMING
    if (graph_view_t::is_multi_gpu) {
      if (handle.get_comms().get_rank() == 0) hr_timer_.start(region);
    } else {
      hr_timer_.start(region);
    }
#endif
  }

  void timer_stop(rmm::cuda_stream_view stream_view)
  {
#ifdef TIMING
    if (graph_view_t::is_multi_gpu) {
      if (handle.get_comms().get_rank() == 0) {
        stream_view.synchronize();
        hr_timer_.stop();
      }
    } else {
      stream_view.synchronize();
      hr_timer_.stop();
    }
#endif
  }

  void timer_display(std::ostream& os)
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
  void initialize_dendrogram_level()
  {
    dendrogram_->add_level(current_graph_view_.local_vertex_partition_range_first(),
                           current_graph_view_.local_vertex_partition_range_size(),
                           handle_.get_stream());

    thrust::sequence(handle_.get_thrust_policy(),
                     dendrogram_->current_level_begin(),
                     dendrogram_->current_level_end(),
                     current_graph_view_.local_vertex_partition_range_first());
  }

 public:
  weight_t modularity(weight_t total_edge_weight, weight_t resolution) const
  {
    weight_t sum_degree_squared = thrust::transform_reduce(
      handle_.get_thrust_policy(),
      cluster_weights_v_.begin(),
      cluster_weights_v_.end(),
      [] __device__(weight_t p) { return p * p; },
      weight_t{0},
      thrust::plus<weight_t>());

    if (graph_view_t::is_multi_gpu) {
      sum_degree_squared = host_scalar_allreduce(
        handle_.get_comms(), sum_degree_squared, raft::comms::op_t::SUM, handle_.get_stream());
    }

    weight_t sum_internal = transform_reduce_e(
      handle_,
      current_graph_view_,
      graph_view_t::is_multi_gpu
        ? src_clusters_cache_.device_view()
        : detail::edge_partition_major_property_device_view_t<vertex_t, vertex_t const*>(
            next_clusters_v_.begin()),
      graph_view_t::is_multi_gpu
        ? dst_clusters_cache_.device_view()
        : detail::edge_partition_minor_property_device_view_t<vertex_t, vertex_t const*>(
            next_clusters_v_.begin()),
      [] __device__(auto, auto, weight_t wt, auto src_cluster, auto nbr_cluster) {
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

    thrust::sequence(handle_.get_thrust_policy(),
                     cluster_keys_v_.begin(),
                     cluster_keys_v_.end(),
                     current_graph_view_.local_vertex_partition_range_first());

    raft::copy(cluster_weights_v_.begin(),
               vertex_weights_v_.begin(),
               vertex_weights_v_.size(),
               handle_.get_stream());

    if constexpr (graph_view_t::is_multi_gpu) {
      auto const comm_size = handle_.get_comms().get_size();
      rmm::device_uvector<vertex_t> rx_keys_v(0, handle_.get_stream());
      rmm::device_uvector<weight_t> rx_weights_v(0, handle_.get_stream());

      auto pair_first = thrust::make_zip_iterator(
        thrust::make_tuple(cluster_keys_v_.begin(), cluster_weights_v_.begin()));

      std::forward_as_tuple(std::tie(rx_keys_v, rx_weights_v), std::ignore) =
        groupby_gpu_id_and_shuffle_values(
          handle_.get_comms(),
          pair_first,
          pair_first + current_graph_view_.local_vertex_partition_range_size(),
          [key_func =
             cugraph::detail::compute_gpu_id_from_vertex_t<vertex_t>{
               comm_size}] __device__(auto val) { return key_func(thrust::get<0>(val)); },
          handle_.get_stream());

      cluster_keys_v_    = std::move(rx_keys_v);
      cluster_weights_v_ = std::move(rx_weights_v);
    }

    if constexpr (graph_view_t::is_multi_gpu) {
      src_vertex_weights_cache_ =
        edge_partition_src_property_t<graph_view_t, weight_t>(handle_, current_graph_view_);
      update_edge_partition_src_property(
        handle_, current_graph_view_, vertex_weights_v_.begin(), src_vertex_weights_cache_);
      vertex_weights_v_.resize(0, handle_.get_stream());
      vertex_weights_v_.shrink_to_fit(handle_.get_stream());
    }

    timer_stop(handle_.get_stream());
  }

  virtual weight_t update_clustering(weight_t total_edge_weight, weight_t resolution)
  {
    timer_start("update_clustering");

    next_clusters_v_ =
      rmm::device_uvector<vertex_t>(dendrogram_->current_level_size(), handle_.get_stream());

    raft::copy(next_clusters_v_.begin(),
               dendrogram_->current_level_begin(),
               dendrogram_->current_level_size(),
               handle_.get_stream());

    if constexpr (graph_view_t::is_multi_gpu) {
      src_clusters_cache_ =
        edge_partition_src_property_t<graph_view_t, vertex_t>(handle_, current_graph_view_);
      update_edge_partition_src_property(
        handle_, current_graph_view_, next_clusters_v_.begin(), src_clusters_cache_);
      dst_clusters_cache_ =
        edge_partition_dst_property_t<graph_view_t, vertex_t>(handle_, current_graph_view_);
      update_edge_partition_dst_property(
        handle_, current_graph_view_, next_clusters_v_.begin(), dst_clusters_cache_);
    }

    weight_t new_Q = modularity(total_edge_weight, resolution);
    weight_t cur_Q = new_Q - 1;

    // To avoid the potential of having two vertices swap clusters
    // we will only allow vertices to move up (true) or down (false)
    // during each iteration of the loop
    bool up_down = true;

    while (new_Q > (cur_Q + 0.0001)) {
      cur_Q = new_Q;

      update_by_delta_modularity(total_edge_weight, resolution, next_clusters_v_, up_down);

      up_down = !up_down;

      new_Q = modularity(total_edge_weight, resolution);

      if (new_Q > cur_Q) {
        raft::copy(dendrogram_->current_level_begin(),
                   next_clusters_v_.begin(),
                   next_clusters_v_.size(),
                   handle_.get_stream());
      }
    }

    timer_stop(handle_.get_stream());
    return cur_Q;
  }

  std::tuple<rmm::device_uvector<weight_t>, rmm::device_uvector<weight_t>>
  compute_cluster_sum_and_subtract() const
  {
    rmm::device_uvector<weight_t> old_cluster_sum_v(
      current_graph_view_.local_vertex_partition_range_size(), handle_.get_stream());
    rmm::device_uvector<weight_t> cluster_subtract_v(
      current_graph_view_.local_vertex_partition_range_size(), handle_.get_stream());

    copy_v_transform_reduce_out_nbr(
      handle_,
      current_graph_view_,
      graph_view_t::is_multi_gpu
        ? src_clusters_cache_.device_view()
        : detail::edge_partition_major_property_device_view_t<vertex_t, vertex_t const*>(
            next_clusters_v_.data()),
      graph_view_t::is_multi_gpu
        ? dst_clusters_cache_.device_view()
        : detail::edge_partition_minor_property_device_view_t<vertex_t, vertex_t const*>(
            next_clusters_v_.data()),
      [] __device__(auto src, auto dst, auto wt, auto src_cluster, auto nbr_cluster) {
        weight_t sum{0};
        weight_t subtract{0};

        if (src == dst)
          subtract = wt;
        else if (src_cluster == nbr_cluster)
          sum = wt;

        return thrust::make_tuple(sum, subtract);
      },
      thrust::make_tuple(weight_t{0}, weight_t{0}),
      thrust::make_zip_iterator(
        thrust::make_tuple(old_cluster_sum_v.begin(), cluster_subtract_v.begin())));

    return std::make_tuple(std::move(old_cluster_sum_v), std::move(cluster_subtract_v));
  }

  void update_by_delta_modularity(weight_t total_edge_weight,
                                  weight_t resolution,
                                  rmm::device_uvector<vertex_t>& next_clusters_v_,
                                  bool up_down)
  {
    rmm::device_uvector<weight_t> vertex_cluster_weights_v(0, handle_.get_stream());
    edge_partition_src_property_t<graph_view_t, weight_t> src_cluster_weights(handle_);
    if constexpr (graph_view_t::is_multi_gpu) {
      cugraph::detail::compute_gpu_id_from_vertex_t<vertex_t> vertex_to_gpu_id_op{
        handle_.get_comms().get_size()};

      vertex_cluster_weights_v = cugraph::collect_values_for_keys(handle_.get_comms(),
                                                                  cluster_keys_v_.begin(),
                                                                  cluster_keys_v_.end(),
                                                                  cluster_weights_v_.data(),
                                                                  next_clusters_v_.begin(),
                                                                  next_clusters_v_.end(),
                                                                  vertex_to_gpu_id_op,
                                                                  handle_.get_stream());

      src_cluster_weights =
        edge_partition_src_property_t<graph_view_t, weight_t>(handle_, current_graph_view_);
      update_edge_partition_src_property(
        handle_, current_graph_view_, vertex_cluster_weights_v.begin(), src_cluster_weights);
      vertex_cluster_weights_v.resize(0, handle_.get_stream());
      vertex_cluster_weights_v.shrink_to_fit(handle_.get_stream());
    } else {
      thrust::sort_by_key(handle_.get_thrust_policy(),
                          cluster_keys_v_.begin(),
                          cluster_keys_v_.end(),
                          cluster_weights_v_.begin());

      vertex_cluster_weights_v.resize(next_clusters_v_.size(), handle_.get_stream());
      thrust::transform(handle_.get_thrust_policy(),
                        next_clusters_v_.begin(),
                        next_clusters_v_.end(),
                        vertex_cluster_weights_v.begin(),
                        [d_cluster_weights = cluster_weights_v_.data(),
                         d_cluster_keys    = cluster_keys_v_.data(),
                         num_clusters      = cluster_keys_v_.size()] __device__(vertex_t cluster) {
                          auto pos = thrust::lower_bound(
                            thrust::seq, d_cluster_keys, d_cluster_keys + num_clusters, cluster);
                          return d_cluster_weights[pos - d_cluster_keys];
                        });
    }

    auto [old_cluster_sum_v, cluster_subtract_v] = compute_cluster_sum_and_subtract();

    edge_partition_src_property_t<graph_view_t, thrust::tuple<weight_t, weight_t>>
      src_old_cluster_sum_subtract_pairs(handle_);
    if constexpr (graph_view_t::is_multi_gpu) {
      src_old_cluster_sum_subtract_pairs =
        edge_partition_src_property_t<graph_view_t, thrust::tuple<weight_t, weight_t>>(
          handle_, current_graph_view_);
      update_edge_partition_src_property(handle_,
                                         current_graph_view_,
                                         thrust::make_zip_iterator(thrust::make_tuple(
                                           old_cluster_sum_v.begin(), cluster_subtract_v.begin())),
                                         src_old_cluster_sum_subtract_pairs);
      old_cluster_sum_v.resize(0, handle_.get_stream());
      old_cluster_sum_v.shrink_to_fit(handle_.get_stream());
      cluster_subtract_v.resize(0, handle_.get_stream());
      cluster_subtract_v.shrink_to_fit(handle_.get_stream());
    }

    auto output_buffer = allocate_dataframe_buffer<thrust::tuple<vertex_t, weight_t>>(
      current_graph_view_.local_vertex_partition_range_size(), handle_.get_stream());

    auto cluster_old_sum_subtract_pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(old_cluster_sum_v.cbegin(), cluster_subtract_v.cbegin()));
    auto zipped_src_device_view =
      graph_view_t::is_multi_gpu
        ? device_view_concat(src_vertex_weights_cache_.device_view(),
                             src_clusters_cache_.device_view(),
                             src_cluster_weights.device_view(),
                             src_old_cluster_sum_subtract_pairs.device_view())
        : device_view_concat(
            detail::edge_partition_major_property_device_view_t<vertex_t, weight_t const*>(
              vertex_weights_v_.data()),
            detail::edge_partition_major_property_device_view_t<vertex_t, vertex_t const*>(
              next_clusters_v_.data()),
            detail::edge_partition_major_property_device_view_t<vertex_t, weight_t const*>(
              vertex_cluster_weights_v.data()),
            detail::edge_partition_major_property_device_view_t<
              vertex_t,
              decltype(cluster_old_sum_subtract_pair_first)>(cluster_old_sum_subtract_pair_first));

    copy_v_transform_reduce_key_aggregated_out_nbr(
      handle_,
      current_graph_view_,
      zipped_src_device_view,
      graph_view_t::is_multi_gpu
        ? dst_clusters_cache_.device_view()
        : detail::edge_partition_minor_property_device_view_t<vertex_t, vertex_t const*>(
            next_clusters_v_.data()),
      cluster_keys_v_.begin(),
      cluster_keys_v_.end(),
      cluster_weights_v_.begin(),
      detail::key_aggregated_edge_op_t<vertex_t, weight_t>{total_edge_weight, resolution},
      detail::reduce_op_t<vertex_t, weight_t>{},
      thrust::make_tuple(vertex_t{-1}, weight_t{0}),
      cugraph::get_dataframe_buffer_begin(output_buffer));

    thrust::transform(handle_.get_thrust_policy(),
                      next_clusters_v_.begin(),
                      next_clusters_v_.end(),
                      cugraph::get_dataframe_buffer_begin(output_buffer),
                      next_clusters_v_.begin(),
                      detail::cluster_update_op_t<vertex_t, weight_t>{up_down});

    if constexpr (graph_view_t::is_multi_gpu) {
      update_edge_partition_src_property(
        handle_, current_graph_view_, next_clusters_v_.begin(), src_clusters_cache_);
      update_edge_partition_dst_property(
        handle_, current_graph_view_, next_clusters_v_.begin(), dst_clusters_cache_);
    }

    std::tie(cluster_keys_v_, cluster_weights_v_) = cugraph::transform_reduce_by_src_key_e(
      handle_,
      current_graph_view_,
      dummy_property_t<vertex_t>{}.device_view(),
      dummy_property_t<vertex_t>{}.device_view(),
      graph_view_t::is_multi_gpu
        ? src_clusters_cache_.device_view()
        : detail::edge_partition_major_property_device_view_t<vertex_t, vertex_t const*>(
            next_clusters_v_.data()),
      detail::return_edge_weight_t<vertex_t, weight_t>{},
      weight_t{0});
  }

  void shrink_graph()
  {
    timer_start("shrinking graph");

    cluster_keys_v_.resize(0, handle_.get_stream());
    cluster_weights_v_.resize(0, handle_.get_stream());
    vertex_weights_v_.resize(0, handle_.get_stream());
    next_clusters_v_.resize(0, handle_.get_stream());
    cluster_keys_v_.shrink_to_fit(handle_.get_stream());
    cluster_weights_v_.shrink_to_fit(handle_.get_stream());
    vertex_weights_v_.shrink_to_fit(handle_.get_stream());
    next_clusters_v_.shrink_to_fit(handle_.get_stream());
    src_vertex_weights_cache_.clear(handle_);
    src_clusters_cache_.clear(handle_);
    dst_clusters_cache_.clear(handle_);

    rmm::device_uvector<vertex_t> numbering_map(0, handle_.get_stream());

    std::tie(current_graph_, numbering_map) =
      coarsen_graph(handle_, current_graph_view_, dendrogram_->current_level_begin());

    current_graph_view_ = current_graph_.view();

    rmm::device_uvector<vertex_t> numbering_indices(numbering_map.size(), handle_.get_stream());
    thrust::sequence(handle_.get_thrust_policy(),
                     numbering_indices.begin(),
                     numbering_indices.end(),
                     current_graph_view_.local_vertex_partition_range_first());

    relabel<vertex_t, graph_view_t::is_multi_gpu>(
      handle_,
      std::make_tuple(static_cast<vertex_t const*>(numbering_map.begin()),
                      static_cast<vertex_t const*>(numbering_indices.begin())),
      current_graph_view_.local_vertex_partition_range_size(),
      dendrogram_->current_level_begin(),
      dendrogram_->current_level_size(),
      false);

    timer_stop(handle_.get_stream());
  }

 protected:
  raft::handle_t const& handle_;

  std::unique_ptr<Dendrogram<vertex_t>> dendrogram_;

  //
  //  Initially we run on the input graph view,
  //  but as we shrink the graph we'll keep the
  //  current graph here
  //
  graph_t current_graph_;
  graph_view_t current_graph_view_;

  rmm::device_uvector<vertex_t> cluster_keys_v_;
  rmm::device_uvector<weight_t> cluster_weights_v_;

  rmm::device_uvector<weight_t> vertex_weights_v_;
  edge_partition_src_property_t<graph_view_t, weight_t>
    src_vertex_weights_cache_;  // src cache for vertex_weights_v_

  rmm::device_uvector<vertex_t> next_clusters_v_;
  edge_partition_src_property_t<graph_view_t, vertex_t>
    src_clusters_cache_;  // src cache for next_clusters_v_
  edge_partition_dst_property_t<graph_view_t, vertex_t>
    dst_clusters_cache_;  // dst cache for next_clusters_v_

#ifdef TIMING
  HighResTimer hr_timer_;
#endif
};

}  // namespace cugraph

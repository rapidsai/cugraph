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

#include <thrust/binary_search.h>

#include <experimental/graph.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <compute_partition.cuh>
#include <experimental/shuffle.cuh>
#include <utilities/graph_utils.cuh>

#include <raft/device_atomics.cuh>

#include <experimental/graph_functions.hpp>
#include <patterns/copy_to_adj_matrix_row_col.cuh>
#include <patterns/copy_v_transform_reduce_in_out_nbr.cuh>
#include <patterns/transform_reduce_e.cuh>
#include <patterns/transform_reduce_v.cuh>

#include <experimental/include_cuco_static_map.cuh>

#include <community/dendrogram.cuh>

#include <numeric>

//#define TIMING

#ifdef TIMING
#include <utilities/high_res_timer.hpp>
#endif

namespace cugraph {
namespace experimental {

namespace detail {

#ifdef CUCO_STATIC_MAP_DEFINED
template <typename data_t>
struct create_cuco_pair_t {
  cuco::pair_type<data_t, data_t> __device__ operator()(data_t data)
  {
    cuco::pair_type<data_t, data_t> tmp;
    tmp.first  = data;
    tmp.second = data_t{0};
    return tmp;
  }
};
#endif

//
// These classes should allow cuco::static_map to generate hash tables of
// different configurations.
//

//
//  Compare edges based on src[e] and dst[e] matching
//
template <typename data_t, typename sentinel_t>
class src_dst_equality_comparator_t {
 public:
  src_dst_equality_comparator_t(rmm::device_vector<data_t> const &src,
                                rmm::device_vector<data_t> const &dst,
                                sentinel_t sentinel_value)
    : d_src_{src.data().get()}, d_dst_{dst.data().get()}, sentinel_value_(sentinel_value)
  {
  }

  src_dst_equality_comparator_t(data_t const *d_src, data_t const *d_dst, sentinel_t sentinel_value)
    : d_src_{d_src}, d_dst_{d_dst}, sentinel_value_(sentinel_value)
  {
  }

  template <typename idx_type>
  __device__ bool operator()(idx_type lhs_index, idx_type rhs_index) const noexcept
  {
    return (lhs_index != sentinel_value_) && (rhs_index != sentinel_value_) &&
           (d_src_[lhs_index] == d_src_[rhs_index]) && (d_dst_[lhs_index] == d_dst_[rhs_index]);
  }

 private:
  data_t const *d_src_;
  data_t const *d_dst_;
  sentinel_t sentinel_value_;
};

//
//  Hash edges based src[e] and dst[e]
//
template <typename data_t>
class src_dst_hasher_t {
 public:
  src_dst_hasher_t(rmm::device_vector<data_t> const &src, rmm::device_vector<data_t> const &dst)
    : d_src_{src.data().get()}, d_dst_{dst.data().get()}
  {
  }

  src_dst_hasher_t(data_t const *d_src, data_t const *d_dst) : d_src_{d_src}, d_dst_{d_dst} {}

  template <typename idx_type>
  __device__ auto operator()(idx_type index) const
  {
    cuco::detail::MurmurHash3_32<data_t> hasher;

    auto h_src = hasher(d_src_[index]);
    auto h_dst = hasher(d_dst_[index]);

    /*
     * Combine the source hash and the dest hash into a single hash value
     *
     * Taken from the Boost hash_combine function
     * https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
     */
    h_src ^= h_dst + 0x9e3779b9 + (h_src << 6) + (h_src >> 2);

    return h_src;
  }

 private:
  data_t const *d_src_;
  data_t const *d_dst_;
};

//
//  Compare edges based on src[e] and cluster[dst[e]] matching
//
template <typename data_t, typename sentinel_t>
class src_cluster_equality_comparator_t {
 public:
  src_cluster_equality_comparator_t(rmm::device_vector<data_t> const &src,
                                    rmm::device_vector<data_t> const &dst,
                                    rmm::device_vector<data_t> const &dst_cluster_cache,
                                    data_t base_dst_id,
                                    sentinel_t sentinel_value)
    : d_src_{src.data().get()},
      d_dst_{dst.data().get()},
      d_dst_cluster_{dst_cluster_cache.data().get()},
      base_dst_id_(base_dst_id),
      sentinel_value_(sentinel_value)
  {
  }

  src_cluster_equality_comparator_t(data_t const *d_src,
                                    data_t const *d_dst,
                                    data_t const *d_dst_cluster_cache,
                                    data_t base_dst_id,
                                    sentinel_t sentinel_value)
    : d_src_{d_src},
      d_dst_{d_dst},
      d_dst_cluster_{d_dst_cluster_cache},
      base_dst_id_(base_dst_id),
      sentinel_value_(sentinel_value)
  {
  }

  __device__ bool operator()(sentinel_t lhs_index, sentinel_t rhs_index) const noexcept
  {
    return (lhs_index != sentinel_value_) && (rhs_index != sentinel_value_) &&
           (d_src_[lhs_index] == d_src_[rhs_index]) &&
           (d_dst_cluster_[d_dst_[lhs_index] - base_dst_id_] ==
            d_dst_cluster_[d_dst_[rhs_index] - base_dst_id_]);
  }

 private:
  data_t const *d_src_;
  data_t const *d_dst_;
  data_t const *d_dst_cluster_;
  data_t base_dst_id_;
  sentinel_t sentinel_value_;
};

//
//  Hash edges based src[e] and cluster[dst[e]]
//
template <typename data_t>
class src_cluster_hasher_t {
 public:
  src_cluster_hasher_t(rmm::device_vector<data_t> const &src,
                       rmm::device_vector<data_t> const &dst,
                       rmm::device_vector<data_t> const &dst_cluster_cache,
                       data_t base_dst_id)
    : d_src_{src.data().get()},
      d_dst_{dst.data().get()},
      d_dst_cluster_{dst_cluster_cache.data().get()},
      base_dst_id_(base_dst_id)
  {
  }

  src_cluster_hasher_t(data_t const *d_src,
                       data_t const *d_dst,
                       data_t const *d_dst_cluster_cache,
                       data_t base_dst_id)
    : d_src_{d_src}, d_dst_{d_dst}, d_dst_cluster_{d_dst_cluster_cache}, base_dst_id_(base_dst_id)
  {
  }

  template <typename idx_type>
  __device__ auto operator()(idx_type index) const
  {
    cuco::detail::MurmurHash3_32<data_t> hasher;

    auto h_src     = hasher(d_src_[index]);
    auto h_cluster = hasher(d_dst_cluster_[d_dst_[index] - base_dst_id_]);

    /*
     * Combine the source hash and the cluster hash into a single hash value
     *
     * Taken from the Boost hash_combine function
     * https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
     */
    h_src ^= h_cluster + 0x9e3779b9 + (h_src << 6) + (h_src >> 2);

    return h_src;
  }

 private:
  data_t const *d_src_;
  data_t const *d_dst_;
  data_t const *d_dst_cluster_;
  data_t base_dst_id_;
};

//
// Skip edges where src[e] == dst[e]
//
template <typename data_t>
class skip_edge_t {
 public:
  skip_edge_t(rmm::device_vector<data_t> const &src, rmm::device_vector<data_t> const &dst)
    : d_src_{src.data().get()}, d_dst_{dst.data().get()}
  {
  }

  skip_edge_t(data_t const *src, data_t const *dst) : d_src_{src}, d_dst_{dst} {}

  template <typename idx_type>
  __device__ auto operator()(idx_type index) const
  {
    return d_src_[index] == d_dst_[index];
  }

 private:
  data_t const *d_src_;
  data_t const *d_dst_;
};

template <typename vertex_t, typename data_t>
struct lookup_by_vertex_id {
 public:
  lookup_by_vertex_id(data_t const *d_array, vertex_t const *d_vertices, vertex_t base_vertex_id)
    : d_array_(d_array), d_vertices_(d_vertices), base_vertex_id_(base_vertex_id)
  {
  }

  template <typename edge_t>
  data_t operator() __device__(edge_t edge_id) const
  {
    return d_array_[d_vertices_[edge_id] - base_vertex_id_];
  }

 private:
  data_t const *d_array_;
  vertex_t const *d_vertices_;
  vertex_t base_vertex_id_;
};

template <typename vector_t, typename iterator_t, typename function_t>
vector_t remove_elements_from_vector(vector_t const &input_v,
                                     iterator_t iterator_begin,
                                     iterator_t iterator_end,
                                     function_t function,
                                     cudaStream_t stream)
{
  vector_t temp_v(input_v.size());

  auto last = thrust::copy_if(
    rmm::exec_policy(stream)->on(stream), iterator_begin, iterator_end, temp_v.begin(), function);

  temp_v.resize(thrust::distance(temp_v.begin(), last));

  return temp_v;
}

template <typename vector_t, typename function_t>
vector_t remove_elements_from_vector(vector_t const &input_v,
                                     function_t function,
                                     cudaStream_t stream)
{
  return remove_elements_from_vector(input_v, input_v.begin(), input_v.end(), function, stream);
}

// FIXME:  This should be a generic utility.  The one in cython.cu
//         is very close to this
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool transposed,
          bool multi_gpu,
          typename view_t,
          std::enable_if_t<multi_gpu> * = nullptr>
std::unique_ptr<experimental::graph_t<vertex_t, edge_t, weight_t, transposed, multi_gpu>>
create_graph(raft::handle_t const &handle,
             rmm::device_vector<vertex_t> const &src_v,
             rmm::device_vector<vertex_t> const &dst_v,
             rmm::device_vector<weight_t> const &weight_v,
             std::size_t num_local_verts,
             experimental::graph_properties_t graph_props,
             view_t const &view)
{
  std::vector<experimental::edgelist_t<vertex_t, edge_t, weight_t>> edgelist(
    {{src_v.data().get(),
      dst_v.data().get(),
      weight_v.data().get(),
      static_cast<edge_t>(src_v.size())}});

  return std::make_unique<experimental::graph_t<vertex_t, edge_t, weight_t, transposed, multi_gpu>>(
    handle,
    edgelist,
    view.get_partition(),
    num_local_verts,
    src_v.size(),
    graph_props,
    false,
    false);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool transposed,
          bool multi_gpu,
          typename view_t,
          std::enable_if_t<!multi_gpu> * = nullptr>
std::unique_ptr<experimental::graph_t<vertex_t, edge_t, weight_t, transposed, multi_gpu>>
create_graph(raft::handle_t const &handle,
             rmm::device_vector<vertex_t> const &src_v,
             rmm::device_vector<vertex_t> const &dst_v,
             rmm::device_vector<weight_t> const &weight_v,
             std::size_t num_local_verts,
             experimental::graph_properties_t graph_props,
             view_t const &view)
{
  experimental::edgelist_t<vertex_t, edge_t, weight_t> edgelist{
    src_v.data().get(),
    dst_v.data().get(),
    weight_v.data().get(),
    static_cast<vertex_t>(src_v.size())};

  return std::make_unique<experimental::graph_t<vertex_t, edge_t, weight_t, transposed, multi_gpu>>(
    handle, edgelist, num_local_verts, graph_props, false, false);
}

}  // namespace detail

//
// FIXME:  Ultimately, this would be cleaner and more efficient if we did the following:
//
//   1) Create an object that does a single level Louvain computation on an input graph
//      (no graph contraction)
//   2) Create an object that does graph contraction
//   3) Create Louvain to use these objects in sequence to compute the aggregate result.
//
//  In MNMG-world, the graph contraction step is going to create another graph that likely
//  fits efficiently in a smaller number of GPUs (eventually one).  Decomposing the algorithm
//  as above would allow us to eventually run the single GPU version of single level Louvain
//  on the contracted graphs - which should be more efficient.
//
// FIXME: We should return the dendrogram and let the python layer clean it up (or have a
//  separate C++ function to flatten the dendrogram).  There are customers that might
//  like the dendrogram and the implementation would be a bit cleaner if we did the
//  collapsing as a separate step
//
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
      compute_partition_(graph_view),
      local_num_vertices_(graph_view.get_number_of_local_vertices()),
      local_num_rows_(graph_view.get_number_of_local_adj_matrix_partition_rows()),
      local_num_cols_(graph_view.get_number_of_local_adj_matrix_partition_cols()),
      local_num_edges_(graph_view.get_number_of_edges()),
      vertex_weights_v_(graph_view.get_number_of_local_vertices()),
      cluster_weights_v_(graph_view.get_number_of_local_vertices()),
      number_of_vertices_(graph_view.get_number_of_local_vertices()),
      stream_(handle.get_stream())
  {
    if (graph_view_t::is_multi_gpu) {
      rank_               = handle.get_comms().get_rank();
      base_vertex_id_     = graph_view.get_local_vertex_first();
      base_src_vertex_id_ = graph_view.get_local_adj_matrix_partition_row_first(0);
      base_dst_vertex_id_ = graph_view.get_local_adj_matrix_partition_col_first(0);

      local_num_edges_ = thrust::transform_reduce(
        thrust::host,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(
          graph_view.get_number_of_local_adj_matrix_partitions()),
        [&graph_view](auto indx) {
          return graph_view.get_number_of_local_adj_matrix_partition_edges(indx);
        },
        size_t{0},
        thrust::plus<size_t>());

      CUDA_TRY(cudaStreamSynchronize(stream_));
    }

    src_indices_v_.resize(local_num_edges_);

    cugraph::detail::offsets_to_indices(
      current_graph_view_.offsets(), local_num_rows_, src_indices_v_.data().get());

    if (base_src_vertex_id_ > 0) {
      thrust::transform(rmm::exec_policy(stream_)->on(stream_),
                        src_indices_v_.begin(),
                        src_indices_v_.end(),
                        thrust::make_constant_iterator(base_src_vertex_id_),
                        src_indices_v_.begin(),
                        thrust::plus<vertex_t>());
    }
  }

  Dendrogram<vertex_t> &get_dendrogram() const { return *dendrogram_; }

  std::unique_ptr<Dendrogram<vertex_t>> move_dendrogram() { return dendrogram_; }

  virtual weight_t operator()(size_t max_level, weight_t resolution)
  {
    weight_t best_modularity = weight_t{-1};

#ifdef CUCO_STATIC_MAP_DEFINED
    weight_t total_edge_weight;
    total_edge_weight = experimental::transform_reduce_e(
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
#endif

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
      src_cluster_cache_v_.begin(),
      dst_cluster_cache_v_.begin(),
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

    cache_vertex_properties(
      vertex_weights_v_.begin(), src_vertex_weights_cache_v_, dst_vertex_weights_cache_v_);

    cache_vertex_properties(
      cluster_weights_v_.begin(), src_cluster_weights_cache_v_, dst_cluster_weights_cache_v_);

    timer_stop(stream_);
  }

  template <typename iterator_t, typename T>
  void cache_vertex_properties(iterator_t const &local_input_iterator,
                               rmm::device_vector<T> &src_cache_v,
                               rmm::device_vector<T> &dst_cache_v,
                               bool src = true,
                               bool dst = true)
  {
    if (src) {
      src_cache_v.resize(current_graph_view_.get_number_of_local_adj_matrix_partition_rows());
      copy_to_adj_matrix_row(
        handle_, current_graph_view_, local_input_iterator, src_cache_v.begin());
    }

    if (dst) {
      dst_cache_v.resize(current_graph_view_.get_number_of_local_adj_matrix_partition_cols());
      copy_to_adj_matrix_col(
        handle_, current_graph_view_, local_input_iterator, dst_cache_v.begin());
    }
  }

#ifdef CUCO_STATIC_MAP_DEFINED
  virtual weight_t update_clustering(weight_t total_edge_weight, weight_t resolution)
  {
    timer_start("update_clustering");

    rmm::device_vector<vertex_t> next_cluster_v(dendrogram_->current_level_begin(),
                                                dendrogram_->current_level_end());

    cache_vertex_properties(next_cluster_v.begin(), src_cluster_cache_v_, dst_cluster_cache_v_);

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

      cache_vertex_properties(next_cluster_v.begin(), src_cluster_cache_v_, dst_cluster_cache_v_);

      new_Q = modularity(total_edge_weight, resolution);

      if (new_Q > cur_Q) {
        thrust::copy(rmm::exec_policy(stream_)->on(stream_),
                     next_cluster_v.begin(),
                     next_cluster_v.end(),
                     dendrogram_->current_level_begin());
      }
    }

    // cache the final clustering locally on each cpu
    cache_vertex_properties(
      dendrogram_->current_level_begin(), src_cluster_cache_v_, dst_cluster_cache_v_);

    timer_stop(stream_);
    return cur_Q;
  }

  void update_by_delta_modularity(weight_t total_edge_weight,
                                  weight_t resolution,
                                  rmm::device_vector<vertex_t> &next_cluster_v,
                                  bool up_down)
  {
    rmm::device_vector<weight_t> old_cluster_sum_v(local_num_vertices_);
    rmm::device_vector<weight_t> src_old_cluster_sum_cache_v;

    experimental::copy_v_transform_reduce_out_nbr(
      handle_,
      current_graph_view_,
      src_cluster_cache_v_.begin(),
      dst_cluster_cache_v_.begin(),
      [] __device__(auto src, auto dst, auto wt, auto src_cluster, auto nbr_cluster) {
        if ((src != dst) && (src_cluster == nbr_cluster)) {
          return wt;
        } else
          return weight_t{0};
      },
      weight_t{0},
      old_cluster_sum_v.begin());

    cache_vertex_properties(
      old_cluster_sum_v.begin(), src_old_cluster_sum_cache_v, empty_cache_weight_v_, true, false);

    detail::src_cluster_equality_comparator_t<vertex_t, edge_t> compare(
      src_indices_v_.data().get(),
      current_graph_view_.indices(),
      dst_cluster_cache_v_.data().get(),
      base_dst_vertex_id_,
      std::numeric_limits<edge_t>::max());
    detail::src_cluster_hasher_t<vertex_t> hasher(src_indices_v_.data().get(),
                                                  current_graph_view_.indices(),
                                                  dst_cluster_cache_v_.data().get(),
                                                  base_dst_vertex_id_);
    detail::skip_edge_t<vertex_t> skip_edge(src_indices_v_.data().get(),
                                            current_graph_view_.indices());

    //
    //  Group edges that lead from same source to same neighboring cluster together
    //  local_cluster_edge_ids_v will contain edge ids of unique pairs of (src,nbr_cluster).
    //  If multiple edges exist, one edge id will be chosen (by a parallel race).
    //  nbr_weights_v will contain the combined weight of all of the edges that connect
    //  that pair.
    //
    rmm::device_vector<edge_t> local_cluster_edge_ids_v;
    rmm::device_vector<weight_t> nbr_weights_v;

    //
    //  Perform this combining on the local edges
    //
    std::tie(local_cluster_edge_ids_v, nbr_weights_v) = combine_local_src_nbr_cluster_weights(
      hasher, compare, skip_edge, current_graph_view_.weights(), local_num_edges_);

    //
    //  In order to compute delta_Q for a given src/nbr_cluster pair, I need the following
    //  information:
    //       src
    //       old_cluster - the cluster that src is currently assigned to
    //       nbr_cluster
    //       sum of edges going to new cluster
    //       vertex weight of the src vertex
    //       sum of edges going to old cluster
    //       cluster_weights of old cluster
    //       cluster_weights of nbr_cluster
    //
    //  Each GPU has locally cached:
    //       The sum of edges going to the old cluster (computed from
    //           experimental::copy_v_transform_reduce_out_nbr call above.
    //       old_cluster
    //       nbr_cluster
    //       vertex weight of src vertex
    //       partial sum of edges going to the new cluster (in nbr_weights)
    //
    //  So the plan is to take the tuple:
    //      (src, old_cluster, src_vertex_weight, old_cluster_sum, nbr_cluster, nbr_weights)
    //  and shuffle it around the cluster so that they arrive at the GPU where the pair
    //  (old_cluster, new_cluster) would be assigned.  Then we can aggregate this information
    //  and compute the delta_Q values.
    //

    //
    //  Define the communication pattern, we're going to send detail
    //  for edge i to the GPU that is responsible for the vertex
    //  pair (cluster[src[i]], cluster[dst[i]])
    //
    auto communication_schedule = thrust::make_transform_iterator(
      local_cluster_edge_ids_v.begin(),
      [d_edge_device_view = compute_partition_.edge_device_view(),
       d_src_indices      = src_indices_v_.data().get(),
       d_src_cluster      = src_cluster_cache_v_.data().get(),
       d_dst_indices      = current_graph_view_.indices(),
       d_dst_cluster      = dst_cluster_cache_v_.data().get(),
       base_src_vertex_id = base_src_vertex_id_,
       base_dst_vertex_id = base_dst_vertex_id_] __device__(edge_t edge_id) {
        return d_edge_device_view(d_src_cluster[d_src_indices[edge_id] - base_src_vertex_id],
                                  d_dst_cluster[d_dst_indices[edge_id] - base_dst_vertex_id]);
      });

    // FIXME:  This should really be a variable_shuffle of a tuple, for time
    //         reasons I'm just doing 6 independent shuffles.
    //
    rmm::device_vector<weight_t> ocs_v = variable_shuffle<graph_view_t::is_multi_gpu, weight_t>(
      handle_,
      local_cluster_edge_ids_v.size(),
      thrust::make_transform_iterator(
        local_cluster_edge_ids_v.begin(),
        detail::lookup_by_vertex_id<vertex_t, weight_t>(src_old_cluster_sum_cache_v.data().get(),
                                                        src_indices_v_.data().get(),
                                                        base_src_vertex_id_)),
      communication_schedule);

    rmm::device_vector<vertex_t> src_cluster_v =
      variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
        handle_,
        local_cluster_edge_ids_v.size(),
        thrust::make_transform_iterator(
          local_cluster_edge_ids_v.begin(),
          detail::lookup_by_vertex_id<vertex_t, vertex_t>(
            src_cluster_cache_v_.data().get(), src_indices_v_.data().get(), base_src_vertex_id_)),
        communication_schedule);

    rmm::device_vector<weight_t> src_vertex_weight_v =
      variable_shuffle<graph_view_t::is_multi_gpu, weight_t>(
        handle_,
        local_cluster_edge_ids_v.size(),
        thrust::make_transform_iterator(
          local_cluster_edge_ids_v.begin(),
          detail::lookup_by_vertex_id<vertex_t, weight_t>(src_vertex_weights_cache_v_.data().get(),
                                                          src_indices_v_.data().get(),
                                                          base_src_vertex_id_)),
        communication_schedule);

    rmm::device_vector<vertex_t> src_v = variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
      handle_,
      local_cluster_edge_ids_v.size(),
      thrust::make_permutation_iterator(src_indices_v_.begin(), local_cluster_edge_ids_v.begin()),
      communication_schedule);

    rmm::device_vector<vertex_t> nbr_cluster_v =
      variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
        handle_,
        local_cluster_edge_ids_v.size(),
        thrust::make_transform_iterator(
          local_cluster_edge_ids_v.begin(),
          detail::lookup_by_vertex_id<vertex_t, vertex_t>(
            dst_cluster_cache_v_.data().get(), current_graph_view_.indices(), base_dst_vertex_id_)),
        communication_schedule);

    nbr_weights_v = variable_shuffle<graph_view_t::is_multi_gpu, weight_t>(
      handle_, nbr_weights_v.size(), nbr_weights_v.begin(), communication_schedule);

    //
    //  At this point, src_v, nbr_cluster_v and nbr_weights_v have been
    //  shuffled to the correct GPU.  We can now compute the final
    //  value of delta_Q for each neigboring cluster
    //
    //  Again, we'll combine edges that connect the same source to the same
    //  neighboring cluster and sum their weights.
    //
    detail::src_dst_equality_comparator_t<vertex_t, vertex_t> compare2(
      src_v, nbr_cluster_v, std::numeric_limits<vertex_t>::max());
    detail::src_dst_hasher_t<vertex_t> hasher2(src_v, nbr_cluster_v);

    auto skip_edge2 = [] __device__(auto) { return false; };

    std::tie(local_cluster_edge_ids_v, nbr_weights_v) = combine_local_src_nbr_cluster_weights(
      hasher2, compare2, skip_edge2, nbr_weights_v.data().get(), src_v.size());

    //
    //  Now local_cluster_edge_ids_v contains the edge ids of the src id/dest
    //  cluster id pairs, and nbr_weights_v contains the weight of edges
    //  going to that cluster id
    //
    //  Now we can compute (locally) each delta_Q value
    //
    auto iter = thrust::make_zip_iterator(
      thrust::make_tuple(local_cluster_edge_ids_v.begin(), nbr_weights_v.begin()));

    thrust::transform(rmm::exec_policy(stream_)->on(stream_),
                      iter,
                      iter + local_cluster_edge_ids_v.size(),
                      nbr_weights_v.begin(),
                      [total_edge_weight,
                       resolution,
                       d_src                 = src_v.data().get(),
                       d_src_cluster         = src_cluster_v.data().get(),
                       d_nbr_cluster         = nbr_cluster_v.data().get(),
                       d_src_vertex_weights  = src_vertex_weight_v.data().get(),
                       d_src_cluster_weights = src_cluster_weights_cache_v_.data().get(),
                       d_dst_cluster_weights = dst_cluster_weights_cache_v_.data().get(),
                       d_ocs                 = ocs_v.data().get(),
                       base_src_vertex_id    = base_src_vertex_id_,
                       base_dst_vertex_id    = base_dst_vertex_id_] __device__(auto tuple) {
                        edge_t edge_id           = thrust::get<0>(tuple);
                        vertex_t nbr_cluster     = d_nbr_cluster[edge_id];
                        weight_t new_cluster_sum = thrust::get<1>(tuple);
                        vertex_t old_cluster     = d_src_cluster[edge_id];
                        weight_t k_k             = d_src_vertex_weights[edge_id];
                        weight_t old_cluster_sum = d_ocs[edge_id];

                        weight_t a_old = d_src_cluster_weights[old_cluster - base_src_vertex_id];
                        weight_t a_new = d_dst_cluster_weights[nbr_cluster - base_dst_vertex_id];

                        return 2 * (((new_cluster_sum - old_cluster_sum) / total_edge_weight) -
                                    resolution * (a_new * k_k - a_old * k_k + k_k * k_k) /
                                      (total_edge_weight * total_edge_weight));
                      });

    //
    //  Pick the largest delta_Q value for each vertex on this gpu.
    //  Then we will shuffle back to the gpu by vertex id
    //
    rmm::device_vector<vertex_t> final_src_v(local_cluster_edge_ids_v.size());
    rmm::device_vector<vertex_t> final_nbr_cluster_v(local_cluster_edge_ids_v.size());
    rmm::device_vector<weight_t> final_nbr_weights_v(local_cluster_edge_ids_v.size());

    auto final_input_iter = thrust::make_zip_iterator(thrust::make_tuple(
      thrust::make_permutation_iterator(src_v.begin(), local_cluster_edge_ids_v.begin()),
      thrust::make_permutation_iterator(nbr_cluster_v.begin(), local_cluster_edge_ids_v.begin()),
      nbr_weights_v.begin()));

    auto final_output_iter = thrust::make_zip_iterator(thrust::make_tuple(
      final_src_v.begin(), final_nbr_cluster_v.begin(), final_nbr_weights_v.begin()));

    auto final_output_pos =
      thrust::copy_if(rmm::exec_policy(stream_)->on(stream_),
                      final_input_iter,
                      final_input_iter + local_cluster_edge_ids_v.size(),
                      final_output_iter,
                      [] __device__(auto p) { return (thrust::get<2>(p) > weight_t{0}); });

    final_src_v.resize(thrust::distance(final_output_iter, final_output_pos));
    final_nbr_cluster_v.resize(thrust::distance(final_output_iter, final_output_pos));
    final_nbr_weights_v.resize(thrust::distance(final_output_iter, final_output_pos));

    //
    // Sort the results, pick the largest version
    //
    thrust::sort(rmm::exec_policy(stream_)->on(stream_),
                 thrust::make_zip_iterator(thrust::make_tuple(
                   final_src_v.begin(), final_nbr_weights_v.begin(), final_nbr_cluster_v.begin())),
                 thrust::make_zip_iterator(thrust::make_tuple(
                   final_src_v.end(), final_nbr_weights_v.end(), final_nbr_cluster_v.begin())),
                 [] __device__(auto left, auto right) {
                   if (thrust::get<0>(left) < thrust::get<0>(right)) return true;
                   if (thrust::get<0>(left) > thrust::get<0>(right)) return false;
                   if (thrust::get<1>(left) > thrust::get<1>(right)) return true;
                   if (thrust::get<1>(left) < thrust::get<1>(right)) return false;
                   return (thrust::get<2>(left) < thrust::get<2>(right));
                 });

    //
    //  Now that we're sorted the first entry for each src value is the largest.
    //
    local_cluster_edge_ids_v.resize(final_src_v.size());

    thrust::transform(rmm::exec_policy(stream_)->on(stream_),
                      thrust::make_counting_iterator<edge_t>(0),
                      thrust::make_counting_iterator<edge_t>(final_src_v.size()),
                      local_cluster_edge_ids_v.begin(),
                      [sentinel = std::numeric_limits<edge_t>::max(),
                       d_src    = final_src_v.data().get()] __device__(edge_t edge_id) {
                        if (edge_id == 0) { return edge_id; }

                        if (d_src[edge_id - 1] != d_src[edge_id]) { return edge_id; }

                        return sentinel;
                      });

    local_cluster_edge_ids_v = detail::remove_elements_from_vector(
      local_cluster_edge_ids_v,
      [sentinel = std::numeric_limits<edge_t>::max()] __device__(auto edge_id) {
        return (edge_id != sentinel);
      },
      stream_);

    final_nbr_cluster_v = variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
      handle_,
      local_cluster_edge_ids_v.size(),
      thrust::make_permutation_iterator(final_nbr_cluster_v.begin(),
                                        local_cluster_edge_ids_v.begin()),
      thrust::make_transform_iterator(
        thrust::make_permutation_iterator(final_src_v.begin(), local_cluster_edge_ids_v.begin()),
        [d_vertex_device_view = compute_partition_.vertex_device_view()] __device__(vertex_t v) {
          return d_vertex_device_view(v);
        }));

    final_nbr_weights_v = variable_shuffle<graph_view_t::is_multi_gpu, weight_t>(
      handle_,
      local_cluster_edge_ids_v.size(),
      thrust::make_permutation_iterator(final_nbr_weights_v.begin(),
                                        local_cluster_edge_ids_v.begin()),
      thrust::make_transform_iterator(
        thrust::make_permutation_iterator(final_src_v.begin(), local_cluster_edge_ids_v.begin()),
        [d_vertex_device_view = compute_partition_.vertex_device_view()] __device__(vertex_t v) {
          return d_vertex_device_view(v);
        }));

    final_src_v = variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
      handle_,
      local_cluster_edge_ids_v.size(),
      thrust::make_permutation_iterator(final_src_v.begin(), local_cluster_edge_ids_v.begin()),
      thrust::make_transform_iterator(
        thrust::make_permutation_iterator(final_src_v.begin(), local_cluster_edge_ids_v.begin()),
        [d_vertex_device_view = compute_partition_.vertex_device_view()] __device__(vertex_t v) {
          return d_vertex_device_view(v);
        }));

    //
    //  At this point...
    //     final_src_v contains the source indices
    //     final_nbr_cluster_v contains the neighboring clusters
    //     final_nbr_weights_v contains delta_Q for moving src to the neighboring
    //
    //  They have been shuffled to the gpus responsible for their source vertex
    //
    //  FIXME:  Think about how this should work.
    //          I think Leiden is broken.  I don't think that the code we have
    //          actually does anything.  For now I'm going to ignore Leiden in
    //          MNMG, we can reconsider this later.
    //
    //  If we ignore Leiden, I'd like to think about whether the reduction
    //  should occur now...
    //

    //
    // Sort the results, pick the largest version
    //
    thrust::sort(rmm::exec_policy(stream_)->on(stream_),
                 thrust::make_zip_iterator(thrust::make_tuple(
                   final_src_v.begin(), final_nbr_weights_v.begin(), final_nbr_cluster_v.begin())),
                 thrust::make_zip_iterator(thrust::make_tuple(
                   final_src_v.end(), final_nbr_weights_v.end(), final_nbr_cluster_v.begin())),
                 [] __device__(auto left, auto right) {
                   if (thrust::get<0>(left) < thrust::get<0>(right)) return true;
                   if (thrust::get<0>(left) > thrust::get<0>(right)) return false;
                   if (thrust::get<1>(left) > thrust::get<1>(right)) return true;
                   if (thrust::get<1>(left) < thrust::get<1>(right)) return false;
                   return (thrust::get<2>(left) < thrust::get<2>(right));
                 });

    //
    //  Now that we're sorted (ascending), the last entry for each src value is the largest.
    //
    local_cluster_edge_ids_v.resize(final_src_v.size());

    thrust::transform(rmm::exec_policy(stream_)->on(stream_),
                      thrust::make_counting_iterator<edge_t>(0),
                      thrust::make_counting_iterator<edge_t>(final_src_v.size()),
                      local_cluster_edge_ids_v.begin(),
                      [sentinel = std::numeric_limits<edge_t>::max(),
                       d_src    = final_src_v.data().get()] __device__(edge_t edge_id) {
                        if (edge_id == 0) { return edge_id; }

                        if (d_src[edge_id - 1] != d_src[edge_id]) { return edge_id; }

                        return sentinel;
                      });

    local_cluster_edge_ids_v = detail::remove_elements_from_vector(
      local_cluster_edge_ids_v,
      [sentinel = std::numeric_limits<edge_t>::max()] __device__(auto edge_id) {
        return (edge_id != sentinel);
      },
      stream_);

    rmm::device_vector<weight_t> cluster_increase_v(final_src_v.size());
    rmm::device_vector<weight_t> cluster_decrease_v(final_src_v.size());
    rmm::device_vector<vertex_t> old_cluster_v(final_src_v.size());

    //
    //   Then we can, on each gpu, do a local assignment for all of the
    //   vertices assigned to that gpu using the up_down logic
    //
    local_cluster_edge_ids_v = detail::remove_elements_from_vector(
      local_cluster_edge_ids_v,
      local_cluster_edge_ids_v.begin(),
      local_cluster_edge_ids_v.end(),
      [d_final_src         = final_src_v.data().get(),
       d_final_nbr_cluster = final_nbr_cluster_v.data().get(),
       d_final_nbr_weights = final_nbr_weights_v.data().get(),
       d_cluster_increase  = cluster_increase_v.data().get(),
       d_cluster_decrease  = cluster_decrease_v.data().get(),
       d_vertex_weights    = src_vertex_weights_cache_v_.data().get(),
       d_next_cluster      = next_cluster_v.data().get(),
       d_old_cluster       = old_cluster_v.data().get(),
       base_vertex_id      = base_vertex_id_,
       base_src_vertex_id  = base_src_vertex_id_,
       up_down] __device__(edge_t idx) {
        vertex_t src         = d_final_src[idx];
        vertex_t new_cluster = d_final_nbr_cluster[idx];
        vertex_t old_cluster = d_next_cluster[src - base_vertex_id];
        weight_t src_weight  = d_vertex_weights[src - base_src_vertex_id];

        if (d_final_nbr_weights[idx] <= weight_t{0}) return false;
        if (new_cluster == old_cluster) return false;
        if ((new_cluster > old_cluster) != up_down) return false;

        d_next_cluster[src - base_vertex_id] = new_cluster;
        d_cluster_increase[idx]              = src_weight;
        d_cluster_decrease[idx]              = src_weight;
        d_old_cluster[idx]                   = old_cluster;
        return true;
      },
      stream_);

    cluster_increase_v = variable_shuffle<graph_view_t::is_multi_gpu, weight_t>(
      handle_,
      local_cluster_edge_ids_v.size(),
      thrust::make_permutation_iterator(cluster_increase_v.begin(),
                                        local_cluster_edge_ids_v.begin()),
      thrust::make_transform_iterator(
        thrust::make_permutation_iterator(final_nbr_cluster_v.begin(),
                                          local_cluster_edge_ids_v.begin()),
        [d_vertex_device_view = compute_partition_.vertex_device_view()] __device__(vertex_t v) {
          return d_vertex_device_view(v);
        }));

    final_nbr_cluster_v = variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
      handle_,
      local_cluster_edge_ids_v.size(),
      thrust::make_permutation_iterator(final_nbr_cluster_v.begin(),
                                        local_cluster_edge_ids_v.begin()),
      thrust::make_transform_iterator(
        thrust::make_permutation_iterator(final_nbr_cluster_v.begin(),
                                          local_cluster_edge_ids_v.begin()),
        [d_vertex_device_view = compute_partition_.vertex_device_view()] __device__(vertex_t v) {
          return d_vertex_device_view(v);
        }));

    cluster_decrease_v = variable_shuffle<graph_view_t::is_multi_gpu, weight_t>(
      handle_,
      local_cluster_edge_ids_v.size(),
      thrust::make_permutation_iterator(cluster_decrease_v.begin(),
                                        local_cluster_edge_ids_v.begin()),
      thrust::make_transform_iterator(
        thrust::make_permutation_iterator(old_cluster_v.begin(), local_cluster_edge_ids_v.begin()),
        [d_vertex_device_view = compute_partition_.vertex_device_view()] __device__(vertex_t v) {
          return d_vertex_device_view(v);
        }));

    old_cluster_v = variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
      handle_,
      local_cluster_edge_ids_v.size(),
      thrust::make_permutation_iterator(old_cluster_v.begin(), local_cluster_edge_ids_v.begin()),
      thrust::make_transform_iterator(
        thrust::make_permutation_iterator(old_cluster_v.begin(), local_cluster_edge_ids_v.begin()),
        [d_vertex_device_view = compute_partition_.vertex_device_view()] __device__(vertex_t v) {
          return d_vertex_device_view(v);
        }));

    thrust::for_each(rmm::exec_policy(stream_)->on(stream_),
                     thrust::make_zip_iterator(
                       thrust::make_tuple(final_nbr_cluster_v.begin(), cluster_increase_v.begin())),
                     thrust::make_zip_iterator(
                       thrust::make_tuple(final_nbr_cluster_v.end(), cluster_increase_v.end())),
                     [d_cluster_weights = cluster_weights_v_.data().get(),
                      base_vertex_id    = base_vertex_id_] __device__(auto p) {
                       vertex_t cluster_id = thrust::get<0>(p);
                       weight_t weight     = thrust::get<1>(p);

                       atomicAdd(d_cluster_weights + cluster_id - base_vertex_id, weight);
                     });

    thrust::for_each(
      rmm::exec_policy(stream_)->on(stream_),
      thrust::make_zip_iterator(
        thrust::make_tuple(old_cluster_v.begin(), cluster_decrease_v.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(old_cluster_v.end(), cluster_decrease_v.end())),
      [d_cluster_weights = cluster_weights_v_.data().get(),
       base_vertex_id    = base_vertex_id_] __device__(auto p) {
        vertex_t cluster_id = thrust::get<0>(p);
        weight_t weight     = thrust::get<1>(p);

        atomicAdd(d_cluster_weights + cluster_id - base_vertex_id, -weight);
      });

    cache_vertex_properties(
      cluster_weights_v_.begin(), src_cluster_weights_cache_v_, dst_cluster_weights_cache_v_);
  }

  template <typename hash_t, typename compare_t, typename skip_edge_t, typename count_t>
  std::pair<rmm::device_vector<count_t>, rmm::device_vector<weight_t>>
  combine_local_src_nbr_cluster_weights(hash_t hasher,
                                        compare_t compare,
                                        skip_edge_t skip_edge,
                                        weight_t const *d_weights,
                                        count_t num_weights)
  {
    rmm::device_vector<count_t> relevant_edges_v;
    rmm::device_vector<weight_t> relevant_edge_weights_v;

    if (num_weights > 0) {
      std::size_t capacity{static_cast<std::size_t>(num_weights / 0.7)};

      cuco::static_map<count_t, count_t> hash_map(
        capacity, std::numeric_limits<count_t>::max(), count_t{0});
      detail::create_cuco_pair_t<count_t> create_cuco_pair;

      CUDA_TRY(cudaStreamSynchronize(stream_));

      hash_map.insert(thrust::make_transform_iterator(thrust::make_counting_iterator<count_t>(0),
                                                      create_cuco_pair),
                      thrust::make_transform_iterator(
                        thrust::make_counting_iterator<count_t>(num_weights), create_cuco_pair),
                      hasher,
                      compare);

      CUDA_TRY(cudaStreamSynchronize(stream_));

      relevant_edges_v.resize(num_weights);

      relevant_edges_v = detail::remove_elements_from_vector(
        relevant_edges_v,
        thrust::make_counting_iterator<count_t>(0),
        thrust::make_counting_iterator<count_t>(num_weights),
        [d_hash_map = hash_map.get_device_view(), hasher, compare] __device__(count_t idx) {
          auto pos = d_hash_map.find(idx, hasher, compare);
          return (pos->first == idx);
        },
        stream_);

      thrust::for_each_n(
        rmm::exec_policy(stream_)->on(stream_),
        thrust::make_counting_iterator<count_t>(0),
        relevant_edges_v.size(),
        [d_hash_map = hash_map.get_device_view(),
         hasher,
         compare,
         d_relevant_edges = relevant_edges_v.data().get()] __device__(count_t idx) mutable {
          count_t edge_id = d_relevant_edges[idx];
          auto pos        = d_hash_map.find(edge_id, hasher, compare);
          pos->second.store(idx);
        });

      relevant_edge_weights_v.resize(relevant_edges_v.size());
      thrust::fill(rmm::exec_policy(stream_)->on(stream_),
                   relevant_edge_weights_v.begin(),
                   relevant_edge_weights_v.end(),
                   weight_t{0});

      thrust::for_each_n(
        rmm::exec_policy(stream_)->on(stream_),
        thrust::make_counting_iterator<count_t>(0),
        num_weights,
        [d_hash_map = hash_map.get_device_view(),
         hasher,
         compare,
         skip_edge,
         d_relevant_edge_weights = relevant_edge_weights_v.data().get(),
         d_weights] __device__(count_t idx) {
          if (!skip_edge(idx)) {
            auto pos = d_hash_map.find(idx, hasher, compare);
            if (pos != d_hash_map.end()) {
              atomicAdd(d_relevant_edge_weights + pos->second.load(cuda::std::memory_order_relaxed),
                        d_weights[idx]);
            }
          }
        });
    }

    return std::make_pair(relevant_edges_v, relevant_edge_weights_v);
  }
#endif

  void shrink_graph()
  {
    timer_start("shrinking graph");

    rmm::device_uvector<vertex_t> numbering_map(0, stream_);

    std::tie(current_graph_, numbering_map) =
      coarsen_graph(handle_, current_graph_view_, dendrogram_->current_level_begin());

    current_graph_view_ = current_graph_->view();

    local_num_vertices_ = current_graph_view_.get_number_of_local_vertices();
    local_num_rows_     = current_graph_view_.get_number_of_local_adj_matrix_partition_rows();
    local_num_cols_     = current_graph_view_.get_number_of_local_adj_matrix_partition_cols();
    base_vertex_id_     = current_graph_view_.get_local_vertex_first();

    local_num_edges_ = thrust::transform_reduce(
      thrust::host,
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(
        current_graph_view_.get_number_of_local_adj_matrix_partitions()),
      [this](auto indx) {
        return current_graph_view_.get_number_of_local_adj_matrix_partition_edges(indx);
      },
      size_t{0},
      thrust::plus<size_t>());

    src_indices_v_.resize(local_num_edges_);

    cugraph::detail::offsets_to_indices(
      current_graph_view_.offsets(), local_num_rows_, src_indices_v_.data().get());

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

  vertex_t number_of_vertices_;
  vertex_t base_vertex_id_{0};
  vertex_t base_src_vertex_id_{0};
  vertex_t base_dst_vertex_id_{0};
  int rank_{0};

  vertex_t local_num_vertices_;
  vertex_t local_num_rows_;
  vertex_t local_num_cols_;
  edge_t local_num_edges_;

  //
  //  Copy of graph
  //
  std::unique_ptr<graph_t> current_graph_{};
  graph_view_t current_graph_view_;

  //
  //  For partitioning
  //
  detail::compute_partition_t<graph_view_t> compute_partition_;

  rmm::device_vector<vertex_t> src_indices_v_;

  //
  //  Weights and clustering across iterations of algorithm
  //
  rmm::device_vector<weight_t> vertex_weights_v_;
  rmm::device_vector<weight_t> src_vertex_weights_cache_v_{};
  rmm::device_vector<weight_t> dst_vertex_weights_cache_v_{};

  rmm::device_vector<weight_t> cluster_weights_v_;
  rmm::device_vector<weight_t> src_cluster_weights_cache_v_{};
  rmm::device_vector<weight_t> dst_cluster_weights_cache_v_{};

  rmm::device_vector<vertex_t> src_cluster_cache_v_{};
  rmm::device_vector<vertex_t> dst_cluster_cache_v_{};

  rmm::device_vector<weight_t> empty_cache_weight_v_{};

#ifdef TIMING
  HighResTimer hr_timer_;
#endif
};  // namespace experimental

}  // namespace experimental
}  // namespace cugraph

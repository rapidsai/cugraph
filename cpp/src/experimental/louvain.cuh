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

#include <thrust/binary_search.h>

#include <experimental/graph.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <compute_partition.cuh>
#include <cuco/static_map.cuh>
#include <experimental/shuffle.cuh>
#include <utilities/graph_utils.cuh>

#include <raft/device_atomics.cuh>

#include <patterns/copy_to_adj_matrix_row_col.cuh>
#include <patterns/copy_v_transform_reduce_in_out_nbr.cuh>
#include <patterns/transform_reduce_e.cuh>
#include <patterns/transform_reduce_v.cuh>

//#define TIMING
#define DEBUG

#ifdef TIMING
#include <utilities/high_res_timer.hpp>
#endif

namespace cugraph {
namespace experimental {

//
// TODO: Delete these print_v functions after debugging is finished
//
template <typename T>
void print_v(const char *label, rmm::device_vector<T> const &vector_v)
{
  std::cout << label << "(" << vector_v.size() << "): ";
  thrust::copy(vector_v.begin(), vector_v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

void print_v(const char *label, const int32_t *ptr, int32_t size)
{
  printf("%s(%d): ", label, size);
  thrust::for_each_n(
    rmm::exec_policy(0)->on(0), thrust::make_counting_iterator(0), 1, [ptr, size] __device__(auto) {
      for (int32_t i = 0; i < size; ++i) printf("%d ", ptr[i]);
    });

  printf("\n");
}

void print_v(const char *label, const int64_t *ptr, int32_t size)
{
  printf("%s(%d): ", label, size);
  thrust::for_each_n(
    rmm::exec_policy(0)->on(0), thrust::make_counting_iterator(0), 1, [ptr, size] __device__(auto) {
      for (int32_t i = 0; i < size; ++i) printf("%ld ", ptr[i]);
    });

  printf("\n");
}

void print_v(const char *label, const float *ptr, int32_t size)
{
  printf("%s(%d): ", label, size);
  thrust::for_each_n(
    rmm::exec_policy(0)->on(0), thrust::make_counting_iterator(0), 1, [ptr, size] __device__(auto) {
      for (int32_t i = 0; i < size; ++i) printf("%g ", ptr[i]);
    });

  printf("\n");
}

void print_v(const char *label, const double *ptr, int32_t size)
{
  printf("%s(%d): ", label, size);
  thrust::for_each_n(
    rmm::exec_policy(0)->on(0), thrust::make_counting_iterator(0), 1, [ptr, size] __device__(auto) {
      for (int32_t i = 0; i < size; ++i) printf("%g ", ptr[i]);
    });

  printf("\n");
}

namespace detail {

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

//
// These classes should allow cuco::static_map to generate hash tables of
// different configurations.
//

//
//  Compare edges based on src[e] and dst[e] matching
//
template <typename data_t>
class src_dst_equality_comparator_t {
 public:
  src_dst_equality_comparator_t(rmm::device_vector<data_t> const &src,
                                rmm::device_vector<data_t> const &dst)
    : d_src_{src.data().get()}, d_dst_{dst.data().get()}
  {
  }

  src_dst_equality_comparator_t(data_t const *d_src, data_t const *d_dst)
    : d_src_{d_src}, d_dst_{d_dst}
  {
  }

  template <typename idx_type>
  __device__ bool operator()(idx_type lhs_index, idx_type rhs_index) const noexcept
  {
    return (d_src_[lhs_index] == d_src_[rhs_index]) && (d_dst_[lhs_index] == d_dst_[rhs_index]);
  }

 private:
  data_t const *d_src_;
  data_t const *d_dst_;
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
    MurmurHash3_32<data_t> hasher;

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
template <typename data_t>
class src_cluster_equality_comparator_t {
 public:
  src_cluster_equality_comparator_t(rmm::device_vector<data_t> const &src,
                                    rmm::device_vector<data_t> const &dst,
                                    rmm::device_vector<data_t> const &dst_cluster_cache)
    : d_src_{src.data().get()},
      d_dst_{dst.data().get()},
      d_dst_cluster_{dst_cluster_cache.data().get()}
  {
  }

  src_cluster_equality_comparator_t(data_t const *d_src,
                                    data_t const *d_dst,
                                    data_t const *d_dst_cluster_cache)
    : d_src_{d_src}, d_dst_{d_dst}, d_dst_cluster_{d_dst_cluster_cache}
  {
  }

  template <typename idx_type>
  __device__ bool operator()(idx_type lhs_index, idx_type rhs_index) const noexcept
  {
    return (d_src_[lhs_index] == d_src_[rhs_index]) &&
           (d_dst_cluster_[d_dst_[lhs_index]] == d_dst_cluster_[d_dst_[rhs_index]]);
  }

 private:
  data_t const *d_src_;
  data_t const *d_dst_;
  data_t const *d_dst_cluster_;
};

//
//  Hash edges based src[e] and cluster[dst[e]]
//
template <typename data_t>
class src_cluster_hasher_t {
 public:
  src_cluster_hasher_t(rmm::device_vector<data_t> const &src,
                       rmm::device_vector<data_t> const &dst,
                       rmm::device_vector<data_t> const &dst_cluster_cache)
    : d_src_{src.data().get()},
      d_dst_{dst.data().get()},
      d_dst_cluster_{dst_cluster_cache.data().get()}
  {
  }

  src_cluster_hasher_t(data_t const *d_src, data_t const *d_dst, data_t const *d_dst_cluster_cache)
    : d_src_{d_src}, d_dst_{d_dst}, d_dst_cluster_{d_dst_cluster_cache}
  {
  }

  template <typename idx_type>
  __device__ auto operator()(idx_type index) const
  {
    MurmurHash3_32<data_t> hasher;

    auto h_src     = hasher(d_src_[index]);
    auto h_cluster = hasher(d_dst_cluster_[d_dst_[index]]);

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
#ifdef DEBUG
    if (d_src_[index] == d_dst_[index]) printf("skipping edge %d\n", (int)index);
#endif
    return d_src_[index] == d_dst_[index];
  }

 private:
  data_t const *d_src_;
  data_t const *d_dst_;
};

}  // namespace detail

namespace detail {

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
      static_cast<vertex_t>(num_local_verts)}});

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

  const vertex_t VERTEX_MAX = std::numeric_limits<vertex_t>::max();

  Louvain(raft::handle_t const &handle, graph_view_t const &graph_view)
    :
#ifdef TIMING
      hr_timer_(),
#endif
      handle_(handle),
      current_graph_view_(graph_view),
      compute_partition_(graph_view),
      local_num_vertices_(graph_view.get_number_of_local_vertices()),
      local_num_rows_(graph_view.get_number_of_local_adj_matrix_partition_rows()),
      local_num_cols_(graph_view.get_number_of_local_adj_matrix_partition_cols()),
      local_num_edges_(graph_view.get_number_of_edges()),
      vertex_weights_v_(graph_view.get_number_of_local_vertices()),
      cluster_weights_v_(graph_view.get_number_of_local_vertices()),
      cluster_v_(graph_view.get_number_of_local_vertices()),
      number_of_vertices_(graph_view.get_number_of_local_vertices()),
      src_indices_v_(graph_view.get_number_of_edges()),
      stream_(handle.get_stream())
  {
    cugraph::detail::offsets_to_indices(
      current_graph_view_.offsets(), local_num_vertices_, src_indices_v_.data().get());

    if (graph_view_t::is_multi_gpu) {
      rank_               = handle.get_comms().get_rank();
      base_vertex_id_     = graph_view.get_local_vertex_first();
      base_src_vertex_id_ = graph_view.get_local_adj_matrix_partition_row_first(rank_);
      base_dst_vertex_id_ = graph_view.get_local_adj_matrix_partition_col_first(rank_);
    }
  }

  virtual std::pair<size_t, weight_t> operator()(vertex_t *d_cluster_vec,
                                                 size_t max_level,
                                                 weight_t resolution)
  {
    size_t num_level{0};

    std::cout << "computing total_edge_weight" << std::endl;

    weight_t total_edge_weight;
    total_edge_weight = experimental::transform_reduce_e(
      handle_,
      current_graph_view_,
      thrust::make_constant_iterator(0),
      thrust::make_constant_iterator(0),
      [] __device__(auto, auto, weight_t wt, auto, auto) { return wt; },
      weight_t{0});

    weight_t best_modularity = weight_t{-1};

    std::cout << "total_edge_weight = " << total_edge_weight << std::endl;

    //
    //  Initialize every cluster to reference each vertex to itself
    //
    thrust::sequence(rmm::exec_policy(stream_)->on(stream_),
                     cluster_v_.begin(),
                     cluster_v_.end(),
                     base_vertex_id_);
    thrust::copy(
      rmm::exec_policy(stream_)->on(stream_), cluster_v_.begin(), cluster_v_.end(), d_cluster_vec);

    while (num_level < max_level) {
      compute_vertex_and_cluster_weights();

      weight_t new_Q = update_clustering(total_edge_weight, resolution);

#ifdef DEBUG
      std::cout << "new_Q = " << new_Q << std::endl;
#endif

      if (new_Q <= best_modularity) { break; }

      best_modularity = new_Q;

      shrink_graph(d_cluster_vec);

      num_level++;
    }

    timer_display(std::cout);

#ifdef DEBUG
    print_v("resulting cluster", d_cluster_vec, number_of_vertices_);
#endif

    return std::make_pair(num_level, best_modularity);
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

 public:
  // TODO CHECK:  Should be right, as long as
  //
  //    src_cluster_cache_v_
  //    dst_cluster_cache_v_
  //
  // are populated properly
  //
  weight_t modularity(weight_t total_edge_weight, weight_t resolution)
  {
    weight_t Q = -resolution / (total_edge_weight * total_edge_weight) *
                 experimental::transform_reduce_v(
                   handle_,
                   current_graph_view_,
                   cluster_weights_v_.begin(),
                   [] __device__(auto p) { return p * p; },
                   weight_t{0});

#if 0
    std::cout << "** Q (subtraction) = " << Q << std::endl;

    print_v("weights", vertex_weights_v_);
    print_v("src_cluster_cache", src_cluster_cache_v_);
    print_v("dst_cluster_cache", dst_cluster_cache_v_);
#endif

    rmm::device_vector<weight_t> increase_v(local_num_vertices_, weight_t{0.0});

    experimental::copy_v_transform_reduce_out_nbr(
      handle_,
      current_graph_view_,
      src_cluster_cache_v_.begin(),
      dst_cluster_cache_v_.begin(),
      [] __device__(auto src, auto, auto weight, auto src_cluster, auto nbr_cluster) {
        if (src_cluster == nbr_cluster) {
          return weight;
        } else
          return weight_t{0};
      },
      weight_t{0},
      increase_v.begin());

#if 0
    CHECK_CUDA(stream_);
    print_v("increase", increase_v);
#endif

    Q += (experimental::transform_reduce_v(
            handle_,
            current_graph_view_,
            increase_v.begin(),
            [] __device__(auto p) { return p; },
            weight_t{0}) /
          total_edge_weight);

#if 0
    std::cout << "** Q (final) = " << Q << std::endl;
#endif

    return Q;
  }

  //
  // TODO CHECK:  Should be right, should populate
  //       vertex_weights_v_
  //       *_vertex_weights_cache_v
  //       cluster_weights_v_
  //       *_cluster_weights_cache_v
  //
  void compute_vertex_and_cluster_weights()
  {
    timer_start("compute_vertex_and_cluster_weights");

    experimental::copy_v_transform_reduce_out_nbr(
      handle_,
      current_graph_view_,
      thrust::make_constant_iterator(0),
      thrust::make_constant_iterator(0),
      [] __device__(auto src, auto, auto wt, auto, auto) {
        printf("src = %d, wt = %g\n", (int)src, wt);
        return wt;
      },
      weight_t{0},
      vertex_weights_v_.begin());

    thrust::copy(rmm::exec_policy(stream_)->on(stream_),
                 vertex_weights_v_.begin(),
                 vertex_weights_v_.end(),
                 cluster_weights_v_.begin());

    cache_vertex_properties(
      vertex_weights_v_, src_vertex_weights_cache_v_, dst_vertex_weights_cache_v_);

    cache_vertex_properties(
      cluster_weights_v_, src_cluster_weights_cache_v_, dst_cluster_weights_cache_v_);

    print_v("vertex_weights_v_", vertex_weights_v_);
    print_v("cluster_weights_v_", cluster_weights_v_);

    timer_stop(stream_);
  }

  //
  // TODO CHECK:  Should be correct, verify that copy_to_adj_matrix_row
  //              and copy_to_adj_matrix_col work.  Verify that
  //              src_cache_v and dst_cache_v are sized properly
  //
  // FIXME:  Consider returning d_src_cache and d_dst_cache
  //         (as a pair).  This would be a nice optimization
  //         for single GPU, as we wouldn't need to make 3 copies
  //         of the data, could return a pair of device pointers to
  //         local_input_v.
  //
  template <typename T>
  void cache_vertex_properties(rmm::device_vector<T> const &local_input_v,
                               rmm::device_vector<T> &src_cache_v,
                               rmm::device_vector<T> &dst_cache_v)
  {
    src_cache_v.resize(current_graph_view_.get_number_of_local_adj_matrix_partition_rows());
    dst_cache_v.resize(current_graph_view_.get_number_of_local_adj_matrix_partition_cols());

    copy_to_adj_matrix_row(
      handle_, current_graph_view_, local_input_v.begin(), src_cache_v.begin());
    copy_to_adj_matrix_col(
      handle_, current_graph_view_, local_input_v.begin(), dst_cache_v.begin());
  }

  //
  // TODO CHECK:  Should be correct, calls other functions.
  //              Does populate src_cluster_cache_v, dst_cluster_cache_v_ multiple
  //              times and will update cluster_v_ if a better clustering is
  //              identified
  //
  virtual weight_t update_clustering(weight_t total_edge_weight, weight_t resolution)
  {
    timer_start("update_clustering");

    rmm::device_vector<vertex_t> next_cluster_v(cluster_v_);

    cache_vertex_properties(next_cluster_v, src_cluster_cache_v_, dst_cluster_cache_v_);

    weight_t new_Q = modularity(total_edge_weight, resolution);
    weight_t cur_Q = new_Q - 1;

#ifdef DEBUG
    std::cout << "update_clustering, new_Q = " << new_Q << std::endl;
#endif

    // To avoid the potential of having two vertices swap clusters
    // we will only allow vertices to move up (true) or down (false)
    // during each iteration of the loop
    bool up_down = true;

    while (new_Q > (cur_Q + 0.0001)) {
      cur_Q = new_Q;

      update_by_delta_modularity(total_edge_weight, resolution, next_cluster_v, up_down);

      up_down = !up_down;

      cache_vertex_properties(next_cluster_v, src_cluster_cache_v_, dst_cluster_cache_v_);

      new_Q = modularity(total_edge_weight, resolution);

#ifdef DEBUG
      std::cout << "new_Q = " << new_Q << std::endl;
#endif

      if (new_Q > cur_Q) {
        thrust::copy(rmm::exec_policy(stream_)->on(stream_),
                     next_cluster_v.begin(),
                     next_cluster_v.end(),
                     cluster_v_.begin());
      }
    }

    // cache the final clustering locally on each cpu
    cache_vertex_properties(cluster_v_, src_cluster_cache_v_, dst_cluster_cache_v_);

#ifdef DEBUG
    print_v("dst_cluster_cache_v_", dst_cluster_cache_v_);
#endif

    timer_stop(stream_);
    return cur_Q;
  }

  //
  // TODO CHECK:  This is the big one to debug in MNMG
  //
  void update_by_delta_modularity(weight_t total_edge_weight,
                                  weight_t resolution,
                                  rmm::device_vector<vertex_t> &next_cluster_v,
                                  bool up_down)
  {
    rmm::device_vector<weight_t> old_cluster_sum_v(local_num_vertices_);

    experimental::copy_v_transform_reduce_out_nbr(
      handle_,
      current_graph_view_,
      src_cluster_cache_v_.begin(),
      dst_cluster_cache_v_.begin(),
      [] __device__(auto src, auto dst, auto wt, auto src_cluster, auto nbr_cluster) {
        if ((src != dst) && (src_cluster == nbr_cluster))
          return wt;
        else
          return weight_t{0};
      },
      weight_t{0},
      old_cluster_sum_v.begin());

    rmm::device_vector<weight_t> src_old_cluster_sum_v(local_num_rows_);
    rmm::device_vector<weight_t> dst_old_cluster_sum_v(local_num_cols_);

    cache_vertex_properties(old_cluster_sum_v, src_old_cluster_sum_v, dst_old_cluster_sum_v);

#ifdef DEBUG
    std::cout << "after cache_vertex_properties" << std::endl;
#endif

    vertex_t const *d_src_indices       = src_indices_v_.data().get();
    vertex_t const *d_dst_indices       = current_graph_view_.indices();
    weight_t const *d_weights           = current_graph_view_.weights();
    vertex_t const *d_dst_cluster_cache = dst_cluster_cache_v_.data().get();

    detail::src_cluster_equality_comparator_t<vertex_t> compare(
      d_src_indices, d_dst_indices, d_dst_cluster_cache);
    detail::src_cluster_hasher_t<vertex_t> hasher(
      d_src_indices, d_dst_indices, d_dst_cluster_cache);
    detail::skip_edge_t<vertex_t> skip_edge(d_src_indices, d_dst_indices);

    //
    //  Group edges that lead from same source to same neighboring cluster together
    //
    rmm::device_vector<edge_t> local_cluster_edge_ids_v;
    rmm::device_vector<weight_t> nbr_weights_v;

#ifdef DEBUG
    print_v("src_indices", src_indices_v_);
    print_v("dst_indices", d_dst_indices, local_num_edges_);
    print_v("nbr_weights", nbr_weights_v);
#endif

    std::tie(local_cluster_edge_ids_v, nbr_weights_v) =
      local_src_dest_weights(hasher, compare, skip_edge, d_weights, local_num_edges_);

    //
    //  Now we will gather the relevant source ids, neighboring clusters and
    //  accumulated weights together and shuffle them to the desired GPU
    //

    auto d_dst_cluster = dst_cluster_cache_v_.data().get();

    //
    //  src values can simply be gathered as they are stored in src_indices_v
    //
    edge_t new_edge_size = local_cluster_edge_ids_v.size();

    rmm::device_vector<vertex_t> src_v;

#ifdef DEBUG
    print_v("src_indices", src_indices_v_);
#endif

    auto d_edge_device_view = compute_partition_.edge_device_view();

    src_v = variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
      handle_,
      new_edge_size,
      thrust::make_permutation_iterator(src_indices_v_.begin(), local_cluster_edge_ids_v.begin()),
      thrust::make_transform_iterator(
        local_cluster_edge_ids_v.begin(),
        [d_edge_device_view, d_src_indices, d_dst_indices, d_dst_cluster] __device__(
          edge_t edge_id) {
          return d_edge_device_view(d_src_indices[edge_id], d_dst_cluster[d_dst_indices[edge_id]]);
        }));

#ifdef DEBUG
    print_v("src", src_v);
#endif

    //
    //  neighboring cluster id must be transformed
    //
    rmm::device_vector<vertex_t> nbr_cluster_v;

    vertex_t base_dst_vertex_id = base_dst_vertex_id_;

    nbr_cluster_v = variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
      handle_,
      new_edge_size,
      thrust::make_transform_iterator(
        local_cluster_edge_ids_v.begin(),
        [d_dst_cluster, d_dst_indices, base_dst_vertex_id] __device__(auto edge_id) {
          return d_dst_cluster[d_dst_indices[edge_id] - base_dst_vertex_id];
        }),
      thrust::make_transform_iterator(
        local_cluster_edge_ids_v.begin(),
        [d_edge_device_view, d_src_indices, d_dst_indices, d_dst_cluster] __device__(
          edge_t edge_id) {
          return d_edge_device_view(d_src_indices[edge_id], d_dst_cluster[d_dst_indices[edge_id]]);
        }));

    //
    //   neighboring weights were already organized in the call
    //
    nbr_weights_v = variable_shuffle<graph_view_t::is_multi_gpu, weight_t>(
      handle_,
      nbr_weights_v.size(),
      nbr_weights_v.begin(),
      thrust::make_transform_iterator(
        local_cluster_edge_ids_v.begin(),
        [d_edge_device_view, d_src_indices, d_dst_indices, d_dst_cluster] __device__(
          edge_t edge_id) {
          return d_edge_device_view(d_src_indices[edge_id], d_dst_cluster[d_dst_indices[edge_id]]);
        }));

    //
    //  At this point, src_v, nbr_cluster_v and nbr_weights_v have been
    //  shuffled to the correct GPU.  We can now compute the final
    //  value of delta_Q for each neigboring cluster
    //
    //  Again, we'll combine edges that connect the same source to the same
    //  neighboring cluster and sum their weights.
    //
    detail::src_dst_equality_comparator_t<vertex_t> compare2(src_v, nbr_cluster_v);
    detail::src_dst_hasher_t<vertex_t> hasher2(src_v, nbr_cluster_v);

    auto skip_edge2 = [] __device__(auto) { return false; };

#ifdef DEBUG
    print_v("src", src_v);
    print_v("nbr_cluster", nbr_cluster_v);
#endif

    std::tie(local_cluster_edge_ids_v, nbr_weights_v) = local_src_dest_weights(
      hasher2, compare2, skip_edge2, nbr_weights_v.data().get(), src_v.size());

    //
    //  Now local_cluster_edge_ids_v contains the edge ids of the src id/dest
    //  cluster id pairs, and nbr_weights_v contains the weight of edges
    //  going to that cluster id
    //
    //  Now we can compute (locally) each delta_Q value
    //
    auto d_src                    = src_v.data();
    auto d_nbr_cluster            = nbr_cluster_v.data().get();
    auto d_nbr_weights            = nbr_weights_v.data().get();
    auto d_local_cluster_edge_ids = local_cluster_edge_ids_v.data();
    auto d_src_cluster            = src_cluster_cache_v_.data();
    auto d_src_vertex_weights     = src_vertex_weights_cache_v_.data();
    auto d_src_old_cluster_sum    = src_old_cluster_sum_v.data();

    auto d_cluster_weights = cluster_weights_v_.data();  // TODO: not right

#ifdef DEBUG
    std::cout << "compute delta_Q" << std::endl;
    print_v("nbr_weights", nbr_weights_v);
#endif

    auto iter = thrust::make_zip_iterator(
      thrust::make_tuple(src_v.begin(), nbr_cluster_v.begin(), nbr_weights_v.begin()));

    thrust::transform(rmm::exec_policy(stream_)->on(stream_),
                      iter,
                      iter + src_v.size(),
                      nbr_weights_v.begin(),
                      [total_edge_weight,
                       resolution,
                       d_src_cluster,
                       d_src_vertex_weights,
                       d_src_old_cluster_sum,
                       d_cluster_weights] __device__(auto tuple) {
                        vertex_t src             = thrust::get<0>(tuple);
                        vertex_t nbr_cluster     = thrust::get<1>(tuple);
                        weight_t new_cluster_sum = thrust::get<2>(tuple);

                        vertex_t old_cluster     = d_src_cluster[src];
                        weight_t k_k             = d_src_vertex_weights[src];
                        weight_t old_cluster_sum = d_src_old_cluster_sum[src];

                        // TODO: How are these distributed
                        weight_t a_old = d_cluster_weights[old_cluster];
                        weight_t a_new = d_cluster_weights[nbr_cluster];

                        weight_t res =
                          2 * (((new_cluster_sum - old_cluster_sum) / total_edge_weight) -
                               resolution * (a_new * k_k - a_old * k_k + k_k * k_k) /
                                 (total_edge_weight * total_edge_weight));

#ifdef DEBUG
                        // if (res > 0) {
                        printf(
                          "  (%d, %d, %g) ncs = %g, ocs = %g, a_new = %g, k_k = %g, a_old = "
                          "%g, "
                          "total_edge_weight = %g\n",
                          (int)src,
                          (int)nbr_cluster,
                          res,
                          new_cluster_sum,
                          old_cluster_sum,
                          a_new,
                          k_k,
                          a_old,
                          total_edge_weight);
      //}
#endif

                        return 2 * (((new_cluster_sum - old_cluster_sum) / total_edge_weight) -
                                    resolution * (a_new * k_k - a_old * k_k + k_k * k_k) /
                                      (total_edge_weight * total_edge_weight));
                      });

#ifdef DEBUG
    std::cout << "computed delta_Q" << std::endl;
#endif

    auto num_nbr_weights = src_v.size();

#ifdef DEBUG
#if 1
    printf("after loop...\n");
    thrust::for_each_n(
      rmm::exec_policy(stream_)->on(stream_),
      thrust::make_counting_iterator<std::size_t>(0),
      1,
      [d_src, d_nbr_cluster, d_nbr_weights, num_nbr_weights] __device__(auto idx) {
        for (std::size_t i = 0; i < num_nbr_weights; ++i)
          if (d_nbr_weights[i] > 0)
            printf(" %ld: (%d, %d, %g)\n", i, (int) d_src.get()[i], (int) d_nbr_cluster[i], d_nbr_weights[i]);
      });
#endif
#endif

    //
    //  At this point...
    //     src_v contains the source indices
    //     nbr_cluster_v contains the neighboring clusters
    //     nbr_weights_v contains delta_Q for moving src to the neighboring
    //     cluster local_cluster_edge_ids_v contains the edge ids
    //
    //  TODO:  Think about how this should work.
    //         I think Leiden is broken.  I don't think that the code we have
    //         actually does anything.  For now I'm going to ignore Leiden in
    //         MNMG, we can reconsider this later.
    //
    //  If we ignore Leiden, I'd like to think about whether the reduction
    //  should occur now...
    //

    // Approach
    //   Create an array of edge ids representing the best delta_Q.
    //   Initialize to MAX = numeric_limits<edge_t>::max()
    //
    //   Iterate over all local_cluster_edge_ids_v elements.
    //     1) If the nbr_weights_v for this element is <= 0, skip this edge
    //     id 2) If the element in the best delta_Q array is MAX, then CAS
    //     this edge
    //        id into the space.  If I won the race then I'm done with this
    //        edge id
    //     3) At this point, there's an element (not me) in the best delta_Q
    //     array.
    //        check to see (no lock) if I am smaller than that element.  If
    //        so I am done.  If not, CAS this edge id into the space.  If I
    //        win the race then I'm done with this edge id.  If I lose the
    //        race then another thread updated it, repeat step 3.
    //
    //   The consequence of this will be an array (best delta_Q) that
    //   contains the edge id (relative to src_v, nbr_cluster_v,
    //   nbr_weights_v) of the highest delta_Q score.
    //
    //   We will then shuffle these to the proper GPU for the source vertex
    //   where we repeat this process to compute the final best delta_Q
    //   score.
    //
    const std::size_t MAX = std::numeric_limits<std::size_t>::max();
    rmm::device_vector<std::size_t> best_delta_Q_v(local_num_rows_, MAX);

    auto d_best_delta_Q = best_delta_Q_v.data();

    thrust::for_each_n(
      rmm::exec_policy(stream_)->on(stream_),
      thrust::make_counting_iterator<std::size_t>(0),
      src_v.size(),
      [d_nbr_weights, d_src, d_nbr_cluster, d_best_delta_Q, MAX] __device__(std::size_t idx) {
        weight_t weight = d_nbr_weights[idx];
        if (weight > weight_t{0}) {
          vertex_t src        = d_src[idx];
          std::size_t current = d_best_delta_Q[src];

          if (current == MAX) {
            if (atomicCAS(d_best_delta_Q.get() + src, current, idx) == current) return;
          }

          while (weight > d_nbr_weights[current]) {
            if (atomicCAS(d_best_delta_Q.get() + src, current, idx) == current) return;

            current = d_best_delta_Q[src];
          }
        }
      });

#ifdef DEBUG
    std::cout << "computed best_delta_Q" << std::endl;
    print_v("best_delta_Q", best_delta_Q_v);
#endif

    //
    //  Next, gather and shuffle
    //
    //  We're going to create an array of size graph.number_of_vertices
    //  of the best neighbor, and another array of the same size of the
    //  corresponding delta_Q.  Then we'll do a fixed shuffle of these
    //  elements.
    //
    rmm::device_vector<vertex_t> best_nbr_cluster_id_v(local_num_rows_);
    rmm::device_vector<vertex_t> best_delta_Q_value_v(local_num_rows_);

    //
    //  FIXME: Would a transform iterator be better?
    //
    auto d_best_nbr_cluster_id = best_nbr_cluster_id_v.data();
    auto d_best_delta_Q_value  = best_delta_Q_value_v.data();
    d_best_delta_Q             = best_delta_Q_v.data();

    auto VMAX = VERTEX_MAX;

    thrust::for_each_n(rmm::exec_policy(stream_)->on(stream_),
                       thrust::make_counting_iterator<vertex_t>(0),
                       local_num_rows_,
                       [VMAX,
                        MAX,
                        d_nbr_cluster,
                        d_best_delta_Q,
                        d_best_nbr_cluster_id,
                        d_best_delta_Q_value] __device__(vertex_t v) {
                         if (d_best_delta_Q[v] == MAX) {
                           d_best_nbr_cluster_id[v] = VMAX;
                         } else {
                           d_best_nbr_cluster_id[v] = d_nbr_cluster[d_best_delta_Q[v]];
                           d_best_delta_Q_value[v]  = d_nbr_cluster[d_best_delta_Q[v]];
                         }
                       });

    best_nbr_cluster_id_v = variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
      handle_,
      local_num_rows_,
      best_nbr_cluster_id_v.begin(),
      // TODO:  this is wrong... need to iterate over source vertex range
      // and use a vertex partitioning lambda
      thrust::make_constant_iterator(0));

    best_delta_Q_value_v = variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
      handle_,
      local_num_rows_,
      best_delta_Q_value_v.begin(),
      // TODO:  this is wrong... need to iterate over source vertex range
      // and use a vertex partitioning lambda
      thrust::make_constant_iterator(0));

    d_best_delta_Q_value  = best_delta_Q_value_v.data();
    d_best_nbr_cluster_id = best_nbr_cluster_id_v.data();
    vertex_t num_vertices = local_num_vertices_;

    // TODO:  How to discover this...
    int num_gpu_rows = 1;

    thrust::for_each_n(
      rmm::exec_policy(stream_)->on(stream_),
      thrust::make_counting_iterator<vertex_t>(0),
      num_vertices,
      [d_best_delta_Q_value, d_best_nbr_cluster_id, num_gpu_rows, num_vertices] __device__(
        auto idx) {
        for (auto pos = idx + num_vertices; pos < num_vertices * num_gpu_rows;
             pos += num_vertices) {
          if (d_best_delta_Q_value[pos] > d_best_delta_Q_value[idx]) {
            d_best_nbr_cluster_id[idx] = d_best_nbr_cluster_id[pos];
            d_best_delta_Q_value[idx]  = d_best_delta_Q_value[pos];
          }
        }
      });

    // TODO:  A whole bunch of things in here (this whole function) are
    // probably
    //        wrong because we'll only have a subset of vertices.  The actual
    //        vertex id will be a local offset plus a base offset

    auto d_next_cluster   = next_cluster_v.data();
    auto d_vertex_weights = vertex_weights_v_.data();

#ifdef DEBUG
    print_v("cluster_weights", cluster_weights_v_);
#endif

    //
    //   Then we can, on each gpu, do a local assignment for all of the
    //   vertices assigned to that gpu using the up_down logic
    //
    thrust::for_each_n(rmm::exec_policy(stream_)->on(stream_),
                       thrust::make_counting_iterator<vertex_t>(0),
                       local_num_vertices_,
                       [d_best_nbr_cluster_id,
                        d_best_delta_Q_value,
                        VMAX,
                        up_down,
                        d_next_cluster,
                        d_vertex_weights,
                        d_cluster_weights] __device__(vertex_t idx) {
#ifdef DEBUG
                         printf("best = %d, max = %d\n", (int) d_best_nbr_cluster_id.get()[idx], (int) VMAX);
#endif
                         if (d_best_nbr_cluster_id[idx] != VMAX) {
                           vertex_t new_cluster = d_best_nbr_cluster_id[idx];
                           vertex_t old_cluster = d_next_cluster[idx];

                           if ((new_cluster > old_cluster) == up_down) {
#ifdef DEBUG
                             printf("moving vertex %d from cluster %d to cluster %d\n",
                                    (int)idx,
                                    (int)old_cluster,
                                    (int)new_cluster);
#endif
                             weight_t src_weight = d_vertex_weights[idx];
                             d_next_cluster[idx] = new_cluster;

                             // TODO:  These might be remote...
                             atomicAdd(d_cluster_weights.get() + new_cluster, src_weight);
                             atomicAdd(d_cluster_weights.get() + old_cluster, -src_weight);
                           }
                         }
                       });

#ifdef DEBUG
    print_v("next_cluster", next_cluster_v);
    print_v("cluster_weights", cluster_weights_v_);
#endif
  }

  //
  // TODO CHECK:   This function should be OK, it operates only on local data.
  //               Could validate that d_weights/num_weights are proper size
  //
  template <typename hash_t, typename compare_t, typename skip_edge_t, typename count_t>
  std::pair<rmm::device_vector<count_t>, rmm::device_vector<weight_t>> local_src_dest_weights(
    hash_t hasher,
    compare_t compare,
    skip_edge_t skip_edge,
    weight_t const *d_weights,
    count_t num_weights)
  {
    std::size_t capacity{static_cast<std::size_t>(num_weights / 0.7)};

    cuco::static_map<count_t, count_t> hash_map(capacity, VERTEX_MAX, count_t{0});

    detail::create_cuco_pair_t<count_t> create_cuco_pair;

    hash_map.insert(
      thrust::make_transform_iterator(thrust::make_counting_iterator<count_t>(0), create_cuco_pair),
      thrust::make_transform_iterator(thrust::make_counting_iterator<count_t>(num_weights),
                                      create_cuco_pair),
      hasher,
      compare);

    auto d_hash_map = hash_map.get_device_view();

    rmm::device_vector<count_t> relevant_edges_v(num_weights);

    auto edge_end = thrust::copy_if(rmm::exec_policy(stream_)->on(stream_),
                                    thrust::make_counting_iterator<count_t>(0),
                                    thrust::make_counting_iterator<count_t>(num_weights),
                                    relevant_edges_v.begin(),
                                    [d_hash_map, hasher, compare] __device__(count_t idx) {
                                      auto pos = d_hash_map.find(idx, hasher, compare);
                                      return (pos->first == idx);
                                    });

    edge_t new_edge_size = thrust::distance(relevant_edges_v.begin(), edge_end);
    relevant_edges_v.resize(new_edge_size);

    auto d_relevant_edges = relevant_edges_v.data();

    thrust::for_each_n(
      rmm::exec_policy(stream_)->on(stream_),
      thrust::make_counting_iterator<count_t>(0),
      new_edge_size,
      [d_hash_map, hasher, compare, d_relevant_edges] __device__(count_t idx) mutable {
        count_t edge_id = d_relevant_edges[idx];
        auto pos        = d_hash_map.find(edge_id, hasher, compare);
#ifdef DEBUG
        printf("setting value for key %d to %d\n", (int)edge_id, (int)idx);
#endif
        pos->second.store(idx);
      });

    rmm::device_vector<weight_t> relevant_edge_weights_v(new_edge_size, weight_t{0});
    auto d_relevant_edge_weights = relevant_edge_weights_v.data();

    const count_t MAX = std::numeric_limits<count_t>::max();

    thrust::for_each_n(
      rmm::exec_policy(stream_)->on(stream_),
      thrust::make_counting_iterator<count_t>(0),
      num_weights,
      [d_hash_map, hasher, compare, skip_edge, d_relevant_edge_weights, d_weights, MAX] __device__(
        count_t idx) {
        if (!skip_edge(idx)) {
          auto pos = d_hash_map.find(idx, hasher, compare);
          if (pos->first != MAX) {
#ifdef DEBUG
            printf("idx = %d, key = %d, offsets = %d, weight = %g\n",
                   (int)idx,
                   (int)pos->first,
                   (int)pos->second.load(),
                   d_weights[idx]);
#endif
            atomicAdd(d_relevant_edge_weights.get() + pos->second.load(), d_weights[idx]);
          }
        }
      });

    return std::make_pair(relevant_edges_v, relevant_edge_weights_v);
  }

  //
  // TODO CHECK:  Should be OK, calls other functions to do most of the work.
  //              Reinitializes cluster_v_ at the end which leaves the
  //              cached values as old, but they are refreshed before the next
  //              code references them.
  //
  void shrink_graph(vertex_t *d_cluster_vec)
  {
    timer_start("shrinking graph");

    std::size_t capacity{static_cast<std::size_t>((local_num_rows_ + local_num_cols_) / 0.7)};

#ifdef DEBUG
    std::cout << "creating hash map, VERTEX_MAX = " << VERTEX_MAX << std::endl;
#endif

    cuco::static_map<vertex_t, vertex_t> hash_map(capacity, VERTEX_MAX, VERTEX_MAX);

#ifdef DEBUG
    std::cout << "in shrink_graph" << std::endl;
#endif

    // renumber the clusters to the range 0..(num_clusters-1)
    renumber_clusters(hash_map);

    renumber_result(hash_map, d_cluster_vec);

    // shrink our graph to represent the graph of supervertices
    generate_supervertices_graph(hash_map);

    // assign each new vertex to its own cluster
    //  MNMG:  This can be done locally with no communication required
    thrust::sequence(rmm::exec_policy(stream_)->on(stream_), cluster_v_.begin(), cluster_v_.end());

    timer_stop(stream_);
  }

  //
  // TODO CHECK:  There's another check indicator inside this function.
  //              Seems OK, although should be carefully examined
  //
  void renumber_clusters(cuco::static_map<vertex_t, vertex_t> &hash_map)
  {
    rmm::device_vector<vertex_t> cluster_inverse_v(local_num_vertices_, vertex_t{0});

#ifdef DEBUG
    print_v("src_indices", src_indices_v_);
    print_v("dst_indices", current_graph_view_.indices(), src_indices_v_.size());
    print_v("weights", current_graph_view_.weights(), src_indices_v_.size());
#endif

    //
    // FIXME:  Faster to iterate from graph_.get_vertex_partition_first()
    //         to graph_.get_vertex_partition_last()?  That would potentially
    //         result in adding a cluster that isn't used on this GPU,
    //         although I don't think it would break the result in any way.
    //
    //         This would also eliminate this use of src_indices_v_.
    //
    auto d_src_cluster_cache = src_cluster_cache_v_.data();
    auto d_dst_cluster_cache = dst_cluster_cache_v_.data();

    vertex_t base_src_vertex_id = base_src_vertex_id_;
    vertex_t base_dst_vertex_id = base_dst_vertex_id_;

    auto it_src = thrust::make_transform_iterator(
      src_indices_v_.begin(), [base_src_vertex_id, d_src_cluster_cache] __device__(auto idx) {
        return detail::create_cuco_pair_t<vertex_t>()(
          d_src_cluster_cache[idx - base_src_vertex_id]);
      });

    auto it_dst = thrust::make_transform_iterator(
      current_graph_view_.indices(),
      [base_dst_vertex_id, d_dst_cluster_cache] __device__(auto idx) {
        return detail::create_cuco_pair_t<vertex_t>()(
          d_dst_cluster_cache[idx - base_dst_vertex_id]);
      });

    hash_map.insert(it_src, it_src + local_num_edges_);
    hash_map.insert(it_dst, it_dst + local_num_edges_);

#ifdef DEBUG
    std::cout << "inserted pairs into hash map" << std::endl;
#endif

    // Now I need to get the keys into an array and shuffle them
    rmm::device_vector<vertex_t> used_cluster_ids_v(hash_map.get_size());

    auto d_hash_map = hash_map.get_device_view();

    vertex_t VMAX = VERTEX_MAX;

#ifdef DEBUG
    std::cout << "call copy_if" << std::endl;
#endif

    auto transform_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator<std::size_t>(0), [d_hash_map] __device__(std::size_t idx) {
        return d_hash_map.begin_slot()[idx].first.load();
      });

    auto transform_end =
      thrust::copy_if(rmm::exec_policy(stream_)->on(stream_),
                      transform_iter,
                      transform_iter + hash_map.get_capacity(),
                      used_cluster_ids_v.begin(),
                      [VMAX] __device__(vertex_t cluster) { return cluster != VMAX; });

#ifdef DEBUG
    std::cout << "distance returns: " << thrust::distance(used_cluster_ids_v.begin(), transform_end)
              << std::endl;
#endif

    used_cluster_ids_v.resize(thrust::distance(used_cluster_ids_v.begin(), transform_end));

#ifdef DEBUG
    std::cout << "populated used_cluster_ids" << std::endl;
    print_v("used_cluster_ids", used_cluster_ids_v);
#endif

    auto d_vertex_device_view = compute_partition_.vertex_device_view();

    auto partition_cluster_ids_iter = thrust::make_transform_iterator(
      used_cluster_ids_v.begin(),
      [d_vertex_device_view] __device__(vertex_t v) { return d_vertex_device_view(v); });

    //
    //   TODO CHECK:  This logic seems right, but need to be tested carefully in MNMG
    //
    rmm::device_vector<std::size_t> original_gpus_v;
    rmm::device_vector<vertex_t> my_cluster_ids_v =
      variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
        handle_, used_cluster_ids_v.size(), used_cluster_ids_v.begin(), partition_cluster_ids_iter);

    if (graph_view_t::is_multi_gpu) {
      original_gpus_v = variable_shuffle<graph_view_t::is_multi_gpu, std::size_t>(
        handle_,
        used_cluster_ids_v.size(),
        thrust::make_constant_iterator<std::size_t>(rank_),
        partition_cluster_ids_iter);
    }

#ifdef DEBUG
    std::cout << "my_cluster_ids shuffle done" << std::endl;
    print_v("my_cluster_ids", my_cluster_ids_v);
#endif

    //
    //   Now my_cluster_ids contains the cluster ids that this gpu is
    //   responsible for. I'm going to set cluster_inverse_v to one
    //   for each cluster in this list.
    //
    auto base_vertex_id    = base_vertex_id_;
    auto d_cluster_inverse = cluster_inverse_v.data().get();

    thrust::for_each(rmm::exec_policy(stream_)->on(stream_),
                     my_cluster_ids_v.begin(),
                     my_cluster_ids_v.end(),
                     [base_vertex_id, d_cluster_inverse] __device__(vertex_t cluster) {
                       d_cluster_inverse[cluster - base_vertex_id] = 1;
                     });

    rmm::device_vector<vertex_t> my_cluster_ids_deduped_v(my_cluster_ids_v.size());
    auto copy_end = thrust::copy_if(
      rmm::exec_policy(stream_)->on(stream_),
      thrust::make_counting_iterator<vertex_t>(0),
      thrust::make_counting_iterator<vertex_t>(cluster_inverse_v.size()),
      my_cluster_ids_deduped_v.begin(),
      [d_cluster_inverse] __device__(vertex_t idx) { return d_cluster_inverse[idx] == 1; });

    my_cluster_ids_deduped_v.resize(thrust::distance(my_cluster_ids_deduped_v.begin(), copy_end));

#ifdef DEBUG
    print_v("my_cluster_ids_deduped", my_cluster_ids_deduped_v);
#endif

    //
    //  Need to gather everything to be able to compute base addresses
    //
    vertex_t base_address{0};

    if (graph_view_t::is_multi_gpu) {
      int num_gpus{1};
      rmm::device_vector<std::size_t> sizes_v(num_gpus + 1, my_cluster_ids_deduped_v.size());

      handle_.get_comms().allgather(
        sizes_v.data().get() + num_gpus, sizes_v.data().get(), num_gpus, stream_);

      base_address = thrust::reduce(rmm::exec_policy(stream_)->on(stream_),
                                    sizes_v.begin(),
                                    sizes_v.begin() + rank_,
                                    vertex_t{0});
    }

    //
    //  Now let's update cluster_inverse_v to contain
    //  the mapping of old cluster id to new vertex id
    //
    auto d_my_cluster_ids_deduped = my_cluster_ids_deduped_v.data();

    thrust::fill(cluster_inverse_v.begin(), cluster_inverse_v.end(), VERTEX_MAX);

    thrust::for_each_n(
      rmm::exec_policy(stream_)->on(stream_),
      thrust::make_counting_iterator<std::size_t>(0),
      my_cluster_ids_deduped_v.size(),
      [base_address, base_vertex_id, d_my_cluster_ids_deduped, d_cluster_inverse] __device__(
        auto idx) { d_cluster_inverse[d_my_cluster_ids_deduped[idx]] = idx + base_address; });

#ifdef DEBUG
    print_v("d_cluster_inverse", cluster_inverse_v);
#endif

    //
    //  Now I need to shuffle back to original gpus the
    //  subset of my mapping that is required
    //
    rmm::device_vector<vertex_t> new_vertex_ids_v =
      variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
        handle_,
        my_cluster_ids_v.size(),
        thrust::make_transform_iterator(my_cluster_ids_v.begin(),
                                        [d_cluster_inverse, base_vertex_id] __device__(auto v) {
                                          return d_cluster_inverse[v - base_vertex_id];
                                        }),
        original_gpus_v.begin());

    if (graph_view_t::is_multi_gpu) {
      my_cluster_ids_v = variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
        handle_, my_cluster_ids_v.size(), my_cluster_ids_v.begin(), original_gpus_v.begin());
    }

#ifdef DEBUG
    print_v("my_cluster_ids", my_cluster_ids_v);
    print_v("new_vertex_ids", new_vertex_ids_v);
#endif

    //
    //  Now update the hash map with the new vertex id
    //
    thrust::for_each_n(rmm::exec_policy(stream_)->on(stream_),
                       thrust::make_zip_iterator(
                         thrust::make_tuple(my_cluster_ids_v.begin(), new_vertex_ids_v.begin())),
                       my_cluster_ids_v.size(),
                       [d_hash_map] __device__(auto p) mutable {
                         auto pos = d_hash_map.find(thrust::get<0>(p));
                         pos->second.store(thrust::get<1>(p));
                       });

    //
    //  At this point we have a renumbered COO that is
    //  improperly distributed around the cluster, which
    //  will be fixed by generate_supervertices_graph
    //
    vertex_t num_clusters = my_cluster_ids_deduped_v.size();
    cluster_v_.resize(num_clusters);
    cluster_weights_v_.resize(num_clusters);
    vertex_weights_v_.resize(num_clusters);
  }

  //
  // TODO CHECK:  Multi-gpu implementation needs to be verified.
  //
  void renumber_result(cuco::static_map<vertex_t, vertex_t> const &hash_map,
                       vertex_t *d_cluster_vec)
  {
    auto d_dst_cluster = dst_cluster_cache_v_.data();

    if (graph_view_t::is_multi_gpu) {
      //
      // FIXME: Perhaps there's a general purpose function hidden here...
      //        Given a set of vertex_t values, and a distributed set of
      //        vertex properties, go to the proper node and retrieve
      //        the vertex properties and return them to this gpu.
      //
      vertex_t VMAX = VERTEX_MAX;
      std::size_t capacity{static_cast<std::size_t>((local_num_vertices_) / 0.7)};
      cuco::static_map<vertex_t, vertex_t> result_hash_map(capacity, VMAX, VMAX);

      auto cluster_iter = thrust::make_transform_iterator(d_cluster_vec, [] __device__(vertex_t c) {
        return detail::create_cuco_pair_t<vertex_t>()(c);
      });

      result_hash_map.insert(cluster_iter, cluster_iter + local_num_vertices_);

      auto d_cluster         = cluster_v_.data();
      auto d_hash_map        = hash_map.get_device_view();
      auto d_result_hash_map = result_hash_map.get_device_view();

      // TODO:  This could be a reusable function...
      //        It is very similar to function in renumber_clusters
      //
      rmm::device_vector<vertex_t> used_cluster_ids_v(result_hash_map.get_size());

      auto transform_iter =
        thrust::make_transform_iterator(thrust::make_counting_iterator<std::size_t>(0),
                                        [d_result_hash_map] __device__(std::size_t idx) {
                                          return d_result_hash_map.begin_slot()[idx].first.load();
                                        });

      auto transform_end =
        thrust::copy_if(rmm::exec_policy(stream_)->on(stream_),
                        transform_iter,
                        transform_iter + hash_map.get_capacity(),
                        used_cluster_ids_v.begin(),
                        [VMAX] __device__(vertex_t cluster) { return cluster != VMAX; });

#ifdef DEBUG
      std::cout << "distance returns: "
                << thrust::distance(used_cluster_ids_v.begin(), transform_end) << std::endl;
#endif

      used_cluster_ids_v.resize(thrust::distance(used_cluster_ids_v.begin(), transform_end));

#ifdef DEBUG
      std::cout << "populated used_cluster_ids" << std::endl;
      print_v("used_cluster_ids", used_cluster_ids_v);
#endif

      auto d_vertex_device_view = compute_partition_.vertex_device_view();

      auto partition_cluster_ids_iter = thrust::make_transform_iterator(
        used_cluster_ids_v.begin(),
        [d_vertex_device_view] __device__(vertex_t v) { return d_vertex_device_view(v); });

      //
      //   TODO CHECK:  This logic seems right, but need to be tested carefully in MNMG
      //
      rmm::device_vector<vertex_t> old_cluster_ids_v =
        variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(handle_,
                                                               used_cluster_ids_v.size(),
                                                               used_cluster_ids_v.begin(),
                                                               partition_cluster_ids_iter);

      rmm::device_vector<std::size_t> original_gpus_v =
        variable_shuffle<graph_view_t::is_multi_gpu, std::size_t>(
          handle_,
          used_cluster_ids_v.size(),
          thrust::make_constant_iterator<std::size_t>(rank_),
          partition_cluster_ids_iter);

#ifdef DEBUG
      std::cout << "old_cluster_ids shuffle done" << std::endl;
      print_v("old_cluster_ids", old_cluster_ids_v);
#endif

      // Now each GPU has old cluster ids, let's compute new cluster ids
      rmm::device_vector<vertex_t> new_cluster_ids_v(old_cluster_ids_v.size());

      thrust::transform(rmm::exec_policy(stream_)->on(stream_),
                        old_cluster_ids_v.begin(),
                        old_cluster_ids_v.end(),
                        new_cluster_ids_v.begin(),
                        [d_cluster, d_hash_map] __device__(vertex_t cluster_id) {
                          vertex_t c = d_cluster[cluster_id];
                          auto pos   = d_hash_map.find(c);
                          return pos->second.load();
                        });

      // Shuffle everything back
      old_cluster_ids_v = variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
        handle_, old_cluster_ids_v.size(), old_cluster_ids_v.begin(), original_gpus_v.begin());
      new_cluster_ids_v = variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
        handle_, new_cluster_ids_v.size(), new_cluster_ids_v.begin(), original_gpus_v.begin());

      // Update result_hash_map
      thrust::for_each_n(rmm::exec_policy(stream_)->on(stream_),
                         thrust::make_zip_iterator(thrust::make_tuple(old_cluster_ids_v.begin(),
                                                                      new_cluster_ids_v.begin())),
                         old_cluster_ids_v.size(),
                         [d_result_hash_map] __device__(auto pair) mutable {
                           auto pos = d_result_hash_map.find(thrust::get<0>(pair));
                           pos->second.store(thrust::get<1>(pair));
                         });

      thrust::transform(rmm::exec_policy(stream_)->on(stream_),
                        d_cluster_vec,
                        d_cluster_vec + number_of_vertices_,
                        d_cluster_vec,
                        [d_result_hash_map] __device__(vertex_t c) {
                          auto pos = d_result_hash_map.find(c);
                          return pos->second.load();
                        });

    } else {
      auto d_hash_map = hash_map.get_device_view();

      thrust::transform(rmm::exec_policy(stream_)->on(stream_),
                        d_cluster_vec,
                        d_cluster_vec + number_of_vertices_,
                        d_cluster_vec,
                        [d_hash_map, d_dst_cluster] __device__(vertex_t v) {
                          vertex_t c = d_dst_cluster[v];
                          auto pos   = d_hash_map.find(c);
#ifdef DEBUG
                          printf("cluster = %d, pos->first = %d, pos->second = %d\n",
                                 (int)c,
                                 (int)pos->first,
                                 (int)pos->second.load());
#endif

                          return pos->second.load();
                        });
    }

#ifdef DEBUG
    print_v("d_cluster_vec", d_cluster_vec, number_of_vertices_);
#endif
  }

  //
  // TODO CHECK:  Multi-gpu implementation needs compile!
  //
  void generate_supervertices_graph(cuco::static_map<vertex_t, vertex_t> const &hash_map)
  {
    rmm::device_vector<vertex_t> new_src_v(local_num_edges_);
    rmm::device_vector<vertex_t> new_dst_v(local_num_edges_);
    rmm::device_vector<weight_t> new_weight_v(current_graph_view_.weights(),
                                              current_graph_view_.weights() + local_num_edges_);

    auto d_hash_map    = hash_map.get_device_view();
    auto d_src_cluster = src_cluster_cache_v_.data();
    auto d_dst_cluster = dst_cluster_cache_v_.data();

    vertex_t base_src_vertex_id = base_src_vertex_id_;
    vertex_t base_dst_vertex_id = base_dst_vertex_id_;

    thrust::transform(rmm::exec_policy(stream_)->on(stream_),
                      src_indices_v_.begin(),
                      src_indices_v_.end(),
                      new_src_v.begin(),
                      [base_src_vertex_id, d_src_cluster, d_hash_map] __device__(vertex_t v) {
                        vertex_t c = d_src_cluster[v - base_src_vertex_id];
                        auto pos   = d_hash_map.find(c);
                        return pos->second.load();
                      });

    thrust::transform(rmm::exec_policy(stream_)->on(stream_),
                      current_graph_view_.indices(),
                      current_graph_view_.indices() + local_num_edges_,
                      new_dst_v.begin(),
                      [base_dst_vertex_id, d_dst_cluster, d_hash_map] __device__(vertex_t v) {
                        vertex_t c = d_dst_cluster[v - base_dst_vertex_id];
                        auto pos   = d_hash_map.find(c);
                        return pos->second.load();
                      });

#ifdef DEBUG
    print_v("new_src", new_src_v);
    print_v("new_dst", new_dst_v);
    print_v("new_weight", new_weight_v);
#endif

    // Combine common edges on local gpu
    std::tie(new_src_v, new_dst_v, new_weight_v) =
      combine_local_edges(new_src_v, new_dst_v, new_weight_v);

#ifdef DEBUG
    print_v("new_src", new_src_v);
    print_v("new_dst", new_dst_v);
    print_v("new_weight", new_weight_v);
#endif

    if (graph_view_t::is_multi_gpu) {
      //
      // Shuffle the data to the proper GPU
      //   FIXME:  This needs some performance exploration.  It is
      //           possible (likely?) that the shrunken graph is
      //           more dense than the original graph.  Perhaps that
      //           changes the dynamic of partitioning efficiently.
      //
      // For now, we're going to keep the partitioning the same,
      // but because we've renumbered to lower numbers, fewer
      // partitions will actually have data.
      //
      rmm::device_vector<int> partition_v(new_src_v.size());

      auto d_edge_device_view = compute_partition_.edge_device_view();

      thrust::transform(
        rmm::exec_policy(stream_)->on(stream_),
        thrust::make_zip_iterator(thrust::make_tuple(new_src_v.begin(), new_dst_v.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(new_src_v.end(), new_dst_v.end())),
        partition_v.begin(),
        [d_edge_device_view] __device__(thrust::tuple<vertex_t, vertex_t> tuple) {
          return d_edge_device_view(thrust::get<0>(tuple), thrust::get<1>(tuple));
        });

      new_src_v = variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
        handle_, partition_v.size(), new_src_v.begin(), partition_v.begin());

      new_dst_v = variable_shuffle<graph_view_t::is_multi_gpu, vertex_t>(
        handle_, partition_v.size(), new_dst_v.begin(), partition_v.begin());

      new_weight_v = variable_shuffle<graph_view_t::is_multi_gpu, weight_t>(
        handle_, partition_v.size(), new_weight_v.begin(), partition_v.begin());

      //
      //  Now everything is on the correct node, again combine like edges
      //
      std::tie(new_src_v, new_dst_v, new_weight_v) =
        combine_local_edges(new_src_v, new_dst_v, new_weight_v);
    }

    //
    //  Now I have a COO of the new graph, distributed according to the
    //  original clustering (eventually this likely fits on one GPU and
    //  everything else is empty).
    //
#if 0
    if (graph_view_t::is_multi_gpu) {
      experimental::partition_t<vertex_t> graph_partition = current_graph_view_.get_partition();

      current_graph_ = std::make_unique<graph_t>(
        handle_,
        {experimental::edgelist_t<vertex_t, edge_t, weight_t>{
            new_src_v.data().get(), new_dst_v.data().get(), new_weight_v.data().get()}},
        graph_partition,
        num_clusters,
        // need to compute (comm reduction)
        new_src_v.size(),
        //graph_.get_graph_properties(),
        graph_properties_t{true, true},
        false,
        false);

      current_graph_view_ = current_graph_->graph_view();

      cugraph::detail::offsets_to_indices<vertex_t>(
        current_graph_view_.offsets(), num_clusters, src_indices_v_.data().get());
    } else {
#endif

#ifdef DEBUG
    std::cout << "make new graph, num_edges = " << new_src_v.size() << std::endl;

    print_v("new_src", new_src_v);
    print_v("new_dst", new_dst_v);
    print_v("new_weight", new_weight_v);
#endif

    current_graph_ =
      detail::create_graph<vertex_t,
                           edge_t,
                           weight_t,
                           graph_t::is_adj_matrix_transposed,
                           graph_t::is_multi_gpu>(handle_,
                                                  new_src_v,
                                                  new_dst_v,
                                                  new_weight_v,
                                                  cluster_v_.size(),
                                                  experimental::graph_properties_t{true, true},
                                                  current_graph_view_);

    current_graph_view_ = current_graph_->view();

    src_indices_v_.resize(new_src_v.size());

    cugraph::detail::offsets_to_indices(current_graph_view_.offsets(),
                                        static_cast<vertex_t>(cluster_v_.size()),
                                        src_indices_v_.data().get());

#if 0
    }
#endif

    local_num_vertices_ = current_graph_view_.get_number_of_local_vertices();
    local_num_rows_     = current_graph_view_.get_number_of_local_adj_matrix_partition_rows();
    local_num_cols_     = current_graph_view_.get_number_of_local_adj_matrix_partition_cols();
    local_num_edges_    = new_src_v.size();

#ifdef DEBUG
    print_v("offsets", current_graph_view_.offsets(), local_num_vertices_);
    print_v("new graph... src_indices", src_indices_v_);
#endif

    CHECK_CUDA(stream_);
  }

  //
  // TODO CHECK:  This function is single GPU only
  //
  std::
    tuple<rmm::device_vector<vertex_t>, rmm::device_vector<vertex_t>, rmm::device_vector<weight_t>>
    combine_local_edges(rmm::device_vector<vertex_t> &src_v,
                        rmm::device_vector<vertex_t> &dst_v,
                        rmm::device_vector<weight_t> &weight_v)
  {
    thrust::stable_sort_by_key(
      rmm::exec_policy(stream_)->on(stream_),
      dst_v.begin(),
      dst_v.end(),
      thrust::make_zip_iterator(thrust::make_tuple(src_v.begin(), weight_v.begin())));
    thrust::stable_sort_by_key(
      rmm::exec_policy(stream_)->on(stream_),
      src_v.begin(),
      src_v.end(),
      thrust::make_zip_iterator(thrust::make_tuple(dst_v.begin(), weight_v.begin())));

#ifdef DEBUG
    std::cout << "after sort" << std::endl;
    print_v("src_v", src_v);
    print_v("dst_v", dst_v);
    print_v("weight_v", weight_v);
#endif

    rmm::device_vector<vertex_t> combined_src_v(src_v.size());
    rmm::device_vector<vertex_t> combined_dst_v(src_v.size());
    rmm::device_vector<weight_t> combined_weight_v(src_v.size());

    //
    //  Now we reduce by key to combine the weights of duplicate
    //  edges.
    //
    auto start = thrust::make_zip_iterator(thrust::make_tuple(src_v.begin(), dst_v.begin()));
    auto new_start =
      thrust::make_zip_iterator(thrust::make_tuple(combined_src_v.begin(), combined_dst_v.begin()));
    auto new_end = thrust::reduce_by_key(rmm::exec_policy(stream_)->on(stream_),
                                         start,
                                         start + src_v.size(),
                                         weight_v.begin(),
                                         new_start,
                                         combined_weight_v.begin(),
                                         thrust::equal_to<thrust::tuple<vertex_t, vertex_t>>(),
                                         thrust::plus<weight_t>());

    auto num_edges = thrust::distance(new_start, new_end.first);

    combined_src_v.resize(num_edges);
    combined_dst_v.resize(num_edges);
    combined_weight_v.resize(num_edges);

    return std::make_tuple(combined_src_v, combined_dst_v, combined_weight_v);
  }

 protected:
  raft::handle_t const &handle_;
  cudaStream_t stream_;

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

  rmm::device_vector<vertex_t> cluster_v_;
  rmm::device_vector<vertex_t> src_cluster_cache_v_{};
  rmm::device_vector<vertex_t> dst_cluster_cache_v_{};

#ifdef TIMING
  HighResTimer hr_timer_;
#endif
};

}  // namespace experimental
}  // namespace cugraph

/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

// Andrei Schaffer, 6/10/19, 7/23/21;
//
#include <cugraph/utilities/error.hpp>

#include <cub/cub.cuh>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <raft/handle.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cstddef>  // for byte_t
#include <iostream>
#include <iterator>
#include <optional>
#include <tuple>
#include <vector>

namespace cugraph {
namespace topology {

/**
 * @brief Check symmetry of CSR adjacency matrix (raw pointers version);
 * Algorithm outline:
 *  flag = true;
 *  for j in rows:
 *    for k in [row_offsets[j]..row_offsets[j+1]):
 *      col_indx = col_indices[k];
 *      flag &= find(j, [col_indices[row_offsets[col_indx]]..col_indices[row_offsets[col_indx+1]]);
 * return flag;
 *
 * @tparam IndexT type of indices for rows and columns
 * @param handle raft handle
 * @param nrows number of vertices
 * @param ptr_r_o CSR row ofssets array
 * @param nnz number of edges
 * @param ptr_c_i CSR column indices array
 */
template <typename IndexT>
bool check_symmetry(raft::handle_t const& handle,
                    IndexT nrows,
                    const IndexT* ptr_r_o,
                    IndexT nnz,
                    const IndexT* ptr_c_i)
{
  using BoolT = bool;
  rmm::device_uvector<BoolT> d_flags(nrows, handle.get_stream());
  thrust::fill_n(handle.get_thrust_policy(), d_flags.begin(), nrows, true);

  BoolT* start_flags = d_flags.data();  // d_flags.begin();
  BoolT* end_flags   = start_flags + nrows;
  BoolT init{1};
  return thrust::transform_reduce(
    handle.get_thrust_policy(),
    start_flags,
    end_flags,
    [ptr_r_o, ptr_c_i, start_flags, nnz] __device__(BoolT & crt_flag) {
      IndexT row_indx = thrust::distance(start_flags, &crt_flag);
      BoolT flag{1};
      for (auto k = ptr_r_o[row_indx]; k < ptr_r_o[row_indx + 1]; ++k) {
        auto col_indx = ptr_c_i[k];
        auto begin    = ptr_c_i + ptr_r_o[col_indx];
        auto end =
          ptr_c_i + ptr_r_o[col_indx + 1];  // end is okay to point beyond last element of ptr_c_i
        auto it = thrust::find(thrust::seq, begin, end, row_indx);
        flag &= (it != end);
      }
      return crt_flag & flag;
    },
    init,
    thrust::logical_and<BoolT>());
}

#ifdef _DEBUG_
/**
 * @brief Sort weights of outgoing edges for each vertex; this requires a segmented (multi-partition
 * parallel partial) sort, rather than complete sort. Required by Biased Random Walks. Caveat: this
 * affects edge numbering. Naive implementation (slow) used only for debugging purposes, as it
 * returns more information.
 *
 * Algorithm outline (version 1):
 * input: num_vertices, num_edges, offsets[],indices[], weights[];
 * output: reordered indices[], weights[];
 *
 * keys[] = sequence(num_edges);
 * segment[] = vectorized-upper-bound(keys, offsets[]); // segment for each key,
 *                                                      // to which key belongs
 * sort-by_key(keys[], zip(indices[], weights[], [segment[], weights[]] (left_index,
 * right_index){
 *                // if left, right indices in the same segment then compare values:
 *                //
 *                if( segment[left_index] == segment[right_index] )
 *                  return weights[left_index] < weights[right_index];
 *                else // "do nothing"
 *                 return (left_index < right_index); // leave things unchanged
 *             });
 *
 * The functor has on-demand instantiation semantics; meaning, the object is type agnostic
 * (so that it can be instantiated by the caller without access to (ro, ci, vals))
 * while operator()(...) deduces its types from arguments and is meant to be called
 * from inside `graph_t` memf's;
 * operator()(...) returns std::tuple rather than thrust::tuple because the later has a bug with
 * structured bindings;
 */
struct thrust_segment_sorter_by_weights_t {
  thrust_segment_sorter_by_weights_t(raft::handle_t const& handle, size_t num_v, size_t num_e)
    : handle_(handle), num_vertices_(num_v), num_edges_(num_e)
  {
  }

  template <typename vertex_t, typename edge_t, typename weight_t>
  std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<edge_t>> operator()(
    edge_t const* ptr_d_offsets_, vertex_t* ptr_d_indices_, weight_t* ptr_d_weights_) const
  {
    rmm::device_uvector<edge_t> d_keys(num_edges_, handle_.get_stream());
    rmm::device_uvector<edge_t> d_segs(num_edges_, handle_.get_stream());

    // cannot use counting iterator, because d_keys gets passed to sort-by-key()
    //
    thrust::sequence(handle.get_thrust_policy(), d_keys.begin(), d_keys.end(), edge_t{0});

    // d_segs = map each key(i.e., edge index), to corresponding
    // segment (i.e., partition = out-going set) index
    //
    thrust::upper_bound(handle.get_thrust_policy(),
                        ptr_d_offsets_,
                        ptr_d_offsets_ + num_vertices_ + 1,
                        d_keys.begin(),
                        d_keys.end(),
                        d_segs.begin());

    thrust::sort_by_key(
      handle.get_thrust_policy(),
      d_keys.begin(),
      d_keys.end(),
      thrust::make_zip_iterator(thrust::make_tuple(ptr_d_indices_, ptr_d_weights_)),
      [ptr_segs = d_segs.data(), ptr_d_vals = ptr_d_weights_] __device__(auto left, auto right) {
        // if both indices in same segment then compare the values:
        //
        if (ptr_segs[left] == ptr_segs[right]) {
          return ptr_d_vals[left] < ptr_d_vals[right];
        } else {  // don't compare them (leave things unchanged)
          return (left < right);
        }
      });

    return std::make_tuple(std::move(d_keys), std::move(d_segs));
  }

 private:
  raft::handle_t const& handle_;
  size_t num_vertices_;
  size_t num_edges_;
};
#endif

/**
 * @brief Sort weights of outgoing edges for each vertex; this requires a segmented (multi-partition
 * parallel partial) sort, rather than complete sort. Required by Biased Random Walks. Caveat: this
 * affects edge numbering.
 *
 * Algorithm outline (version 2):
 * Uses cub::DeviceSegmentedRadixSort:
 * https://nvlabs.github.io/cub/structcub_1_1_device_segmented_radix_sort.html
 *
 * The functor has on-demand instantiation semantics; meaning, the object is type agnostic
 * (so that it can be instantiated by the caller without access to (ro, ci, vals))
 * while operator()(...) deduces its types from arguments and is meant to be called
 * from inside `graph_t` memf's;
 * It also must provide "in-place" semantics for modyfing the CSR array arguments;
 */
struct segment_sorter_by_weights_t {
  segment_sorter_by_weights_t(raft::handle_t const& handle, size_t num_v, size_t num_e)
    : handle_(handle), num_vertices_(num_v), num_edges_(num_e)
  {
  }

  template <typename vertex_t, typename edge_t, typename weight_t>
  std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>> operator()(
    edge_t* ptr_d_offsets, vertex_t* ptr_d_indices, weight_t* ptr_d_weights) const
  {
    // keys are those on which sorting is done; hence, the weights:
    //
    rmm::device_uvector<weight_t> d_vals_out(num_edges_, handle_.get_stream());

    // vals: are they just shuffled as a result of key sorting?
    // no... it seems they participate in the sort...
    //
    rmm::device_uvector<edge_t> d_keys_out(num_edges_, handle_.get_stream());

    // Note: In-place does not work;

    // Determine temporary device storage requirements:
    //
    void* ptr_d_temp_storage{nullptr};
    size_t temp_storage_bytes{0};
    cub::DeviceSegmentedRadixSort::SortPairs(ptr_d_temp_storage,
                                             temp_storage_bytes,
                                             ptr_d_weights,
                                             d_vals_out.data(),  // no in-place;
                                             ptr_d_indices,
                                             d_keys_out.data(),  // no in-place;
                                             num_edges_,
                                             num_vertices_,
                                             ptr_d_offsets,
                                             ptr_d_offsets + 1,
                                             0,
                                             (sizeof(weight_t) << 3),
                                             handle_.get_stream());
    // Allocate temporary storage
    //
    rmm::device_uvector<std::byte> d_temp_storage(temp_storage_bytes, handle_.get_stream());
    ptr_d_temp_storage = d_temp_storage.data();

    // Run sorting operation
    //
    cub::DeviceSegmentedRadixSort::SortPairs(ptr_d_temp_storage,
                                             temp_storage_bytes,
                                             ptr_d_weights,
                                             d_vals_out.data(),  // no in-place;
                                             ptr_d_indices,
                                             d_keys_out.data(),  // no in-place;
                                             num_edges_,
                                             num_vertices_,
                                             ptr_d_offsets,
                                             ptr_d_offsets + 1,
                                             0,
                                             (sizeof(weight_t) << 3),
                                             handle_.get_stream());

    // move data to deliver "in-place" semantics
    //
    return std::make_tuple(std::move(d_keys_out), std::move(d_vals_out));
  }

  template <typename vertex_t, typename edge_t, typename weight_t>
  void operator()(rmm::device_uvector<edge_t>& offsets,
                  rmm::device_uvector<vertex_t>& indices,
                  std::optional<rmm::device_uvector<weight_t>>& weights) const
  {
    CUGRAPH_EXPECTS(weights.has_value(), "Cannot sort un-weighted graph by weights.");

    auto* ptr_d_offsets = offsets.data();
    auto* ptr_d_indices = indices.data();
    auto* ptr_d_weights = weights->data();

    auto [d_keys_out, d_vals_out] = operator()(ptr_d_offsets, ptr_d_indices, ptr_d_weights);

    // move data to deliver "in-place" semantics
    //
    indices  = std::move(d_keys_out);
    *weights = std::move(d_vals_out);
  }

 private:
  raft::handle_t const& handle_;
  size_t num_vertices_;
  size_t num_edges_;
};

/**
 * @brief Check if CSR's weights are segment-sorted, assuming a segment array is given;
 *
 * @tparam edge_t type of edge indices;
 * @tparam edge_t type of weights;
 * @param handle raft handle;
 * @param ptr_d_segs array of segment index for each edge index;
 * @param ptr_d_weights CSR weights array;
 * @param num_edges number of edges;
 */
template <typename edge_t, typename weight_t>
bool check_segmented_sort(raft::handle_t const& handle,
                          edge_t const* ptr_d_segs,
                          weight_t const* ptr_d_weights,
                          size_t num_edges)
{
  auto end = thrust::make_counting_iterator<size_t>(num_edges - 1);

  // search for any adjacent elements in same segments
  // that are _not_ ordered increasingly:
  //
  auto it = thrust::find_if(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator<size_t>(0),
    end,
    [ptr_d_segs, ptr_d_weights] __device__(auto indx) {
      if (ptr_d_segs[indx] == ptr_d_segs[indx + 1]) {  // consecutive indices in same segment
        return (ptr_d_weights[indx] > ptr_d_weights[indx + 1]);
      } else {  // don't compare consecutive indices in different segments
        return false;
      }
    });

  return it == end;
}

/**
 * @brief Check if CSR's weights are segment-sorted, when no segment array is given;
 *
 * @tparam edge_t type of edge indices;
 * @tparam edge_t type of weights;
 * @param handle raft handle;
 * @param ptr_d_offsets CSR offsets array;
 * @param ptr_d_weights CSR weights array;
 * @param num_vertices number of vertices;
 * @param num_edges number of edges;
 */
template <typename edge_t, typename weight_t>
bool check_segmented_sort(raft::handle_t const& handle,
                          edge_t const* ptr_d_offsets,
                          weight_t const* ptr_d_weights,
                          size_t num_vertices,
                          size_t num_edges)
{
  rmm::device_uvector<edge_t> d_keys(num_edges, handle.get_stream());
  rmm::device_uvector<edge_t> d_segs(num_edges, handle.get_stream());

  // cannot use counting iterator, because d_keys gets passed to sort-by-key()
  //
  thrust::sequence(handle.get_thrust_policy(), d_keys.begin(), d_keys.end(), edge_t{0});

  // d_segs = map each key(i.e., edge index), to corresponding
  // segment (i.e., partition = out-going set) index
  //
  thrust::upper_bound(handle.get_thrust_policy(),
                      ptr_d_offsets,
                      ptr_d_offsets + num_vertices + 1,
                      d_keys.begin(),
                      d_keys.end(),
                      d_segs.begin());

  return check_segmented_sort(handle, d_segs.data(), ptr_d_weights, num_edges);
}

}  // namespace topology
}  // namespace cugraph

namespace {  // unnamed namespace for debugging tools:
template <typename T, typename... Args, template <typename, typename...> class Vector>
void print_v(const Vector<T, Args...>& v, std::ostream& os)
{
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(os, ","));  // okay
  os << "\n";
}

template <typename T, typename... Args, template <typename, typename...> class Vector>
void print_v(const Vector<T, Args...>& v,
             typename Vector<T, Args...>::const_iterator pos,
             std::ostream& os)
{
  thrust::copy(v.begin(), pos, std::ostream_iterator<T>(os, ","));  // okay
  os << "\n";
}

template <typename T, typename... Args, template <typename, typename...> class Vector>
void print_v(const Vector<T, Args...>& v, size_t n, std::ostream& os)
{
  thrust::copy_n(v.begin(), n, std::ostream_iterator<T>(os, ","));  // okay
  os << "\n";
}

template <typename T>
void print_v(const T* p_v, size_t n, std::ostream& os)
{
  thrust::copy_n(p_v, n, std::ostream_iterator<T>(os, ","));  // okay
  os << "\n";
}
}  // namespace

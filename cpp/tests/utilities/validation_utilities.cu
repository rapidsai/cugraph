/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "utilities/validation_utilities.hpp"

#include <thrust/count.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

namespace cugraph::test {

template <typename vertex_t, bool multi_gpu>
size_t count_invalid_vertices(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> vertices,
  cugraph::vertex_partition_device_view_t<vertex_t, multi_gpu> const& vertex_partition)
{
  return thrust::count_if(handle.get_thrust_policy(),
                          vertices.begin(),
                          vertices.end(),
                          [vertex_partition] __device__(auto val) {
                            return !(vertex_partition.is_valid_vertex(val) &&
                                     vertex_partition.in_local_vertex_partition_range_nocheck(val));
                          });
}

template <typename vertex_t, bool multi_gpu>
size_t count_duplicate_vertex_pairs_sorted(raft::handle_t const& handle,
                                           raft::device_span<vertex_t const> src,
                                           raft::device_span<vertex_t const> dst)
{
  return thrust::count_if(handle.get_thrust_policy(),
                          thrust::make_counting_iterator<size_t>(1),
                          thrust::make_counting_iterator<size_t>(src.size()),
                          [src, dst] __device__(size_t index) {
                            return (src[index - 1] == src[index]) && (dst[index - 1] == dst[index]);
                          });
}

// FIXME: Resolve this with dataframe_buffer variations in thrust_wrappers.cu
template <typename vertex_t>
void sort(raft::handle_t const& handle,
          raft::device_span<vertex_t> srcs,
          raft::device_span<vertex_t> dsts)
{
  thrust::sort(handle.get_thrust_policy(),
               thrust::make_zip_iterator(srcs.begin(), dsts.begin()),
               thrust::make_zip_iterator(srcs.end(), dsts.end()));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          bool multi_gpu>
size_t count_intersection(raft::handle_t const& handle,
                          raft::device_span<vertex_t const> srcs1,
                          raft::device_span<vertex_t const> dsts1,
                          std::optional<raft::device_span<weight_t const>> wgts1,
                          std::optional<raft::device_span<edge_t const>> edge_ids1,
                          std::optional<raft::device_span<edge_type_t const>> edge_types1,
                          raft::device_span<vertex_t const> srcs2,
                          raft::device_span<vertex_t const> dsts2,
                          std::optional<raft::device_span<weight_t const>> wgts2,
                          std::optional<raft::device_span<edge_t const>> edge_ids2,
                          std::optional<raft::device_span<edge_type_t const>> edge_types2)
{
  // FIXME: Add support for wgts, edgeids and edge_types...
  //    Added to the API for future support.

  auto iter1       = thrust::make_zip_iterator(srcs1.begin(), dsts1.begin());
  auto iter2       = thrust::make_zip_iterator(srcs2.begin(), dsts2.begin());
  auto output_iter = thrust::make_discard_iterator();

  return thrust::distance(output_iter,
                          thrust::set_intersection(handle.get_thrust_policy(),
                                                   iter1,
                                                   iter1 + srcs1.size(),
                                                   iter2,
                                                   iter2 + srcs2.size(),
                                                   output_iter));
#if 0
  // OLD Approach
  return thrust::count_if(
    handle.get_thrust_policy(),
    thrust::make_zip_iterator(src_out.begin(), dst_out.begin()),
    thrust::make_zip_iterator(src_out.end(), dst_out.end()),
    cuda::proclaim_return_type<size_t>(
      [src = raft::device_span<vertex_t const>{graph_src.data(), graph_src.size()},
       dst = raft::device_span<vertex_t const>{graph_dst.data(),
                                               graph_dst.size()}] __device__(auto tuple) {
#if 0
        // FIXME: This fails on rocky linux CUDA 11.8, works on CUDA 12
        return thrust::binary_search(thrust::seq,
                                     thrust::make_zip_iterator(src.begin(), dst.begin()),
                                     thrust::make_zip_iterator(src.end(), dst.end()),
                                     tuple) ? size_t{1} : size_t{0};
#else
        auto lb = thrust::distance(
          src.begin(),
          thrust::lower_bound(thrust::seq, src.begin(), src.end(), thrust::get<0>(tuple)));
        auto ub = thrust::distance(
          src.begin(),
          thrust::upper_bound(thrust::seq, src.begin(), src.end(), thrust::get<0>(tuple)));

        if (src.data()[lb] == thrust::get<0>(tuple)) {
          return thrust::binary_search(
            thrust::seq, dst.begin() + lb, dst.begin() + ub, thrust::get<1>(tuple))
              ? size_t{1}
              : size_t{0};
        } else {
          return size_t{0};
        }
#endif
      }));
#endif
}

// TODO: Split SG from MG?
template size_t count_invalid_vertices(
  raft::handle_t const& handle,
  raft::device_span<int32_t const> vertices,
  cugraph::vertex_partition_device_view_t<int32_t, false> const& vertex_partition);

template size_t count_invalid_vertices(
  raft::handle_t const& handle,
  raft::device_span<int64_t const> vertices,
  cugraph::vertex_partition_device_view_t<int64_t, false> const& vertex_partition);

template size_t count_duplicate_vertex_pairs_sorted<int32_t, false>(
  raft::handle_t const& handle,
  raft::device_span<int32_t const> src,
  raft::device_span<int32_t const> dst);

template size_t count_duplicate_vertex_pairs_sorted<int64_t, false>(
  raft::handle_t const& handle,
  raft::device_span<int64_t const> src,
  raft::device_span<int64_t const> dst);

template void sort(raft::handle_t const& handle,
                   raft::device_span<int32_t> srcs,
                   raft::device_span<int32_t> dsts);
template void sort(raft::handle_t const& handle,
                   raft::device_span<int64_t> srcs,
                   raft::device_span<int64_t> dsts);

template size_t count_intersection<int32_t, int32_t, float, int32_t, false>(
  raft::handle_t const& handle,
  raft::device_span<int32_t const> srcs1,
  raft::device_span<int32_t const> dsts1,
  std::optional<raft::device_span<float const>> wgts1,
  std::optional<raft::device_span<int32_t const>> edge_ids1,
  std::optional<raft::device_span<int32_t const>> edge_types1,
  raft::device_span<int32_t const> srcs2,
  raft::device_span<int32_t const> dsts2,
  std::optional<raft::device_span<float const>> wgts2,
  std::optional<raft::device_span<int32_t const>> edge_ids2,
  std::optional<raft::device_span<int32_t const>> edge_types2);

template size_t count_intersection<int32_t, int64_t, float, int32_t, false>(
  raft::handle_t const& handle,
  raft::device_span<int32_t const> srcs1,
  raft::device_span<int32_t const> dsts1,
  std::optional<raft::device_span<float const>> wgts1,
  std::optional<raft::device_span<int64_t const>> edge_ids1,
  std::optional<raft::device_span<int32_t const>> edge_types1,
  raft::device_span<int32_t const> srcs2,
  raft::device_span<int32_t const> dsts2,
  std::optional<raft::device_span<float const>> wgts2,
  std::optional<raft::device_span<int64_t const>> edge_ids2,
  std::optional<raft::device_span<int32_t const>> edge_types2);

template size_t count_intersection<int64_t, int64_t, float, int32_t, false>(
  raft::handle_t const& handle,
  raft::device_span<int64_t const> srcs1,
  raft::device_span<int64_t const> dsts1,
  std::optional<raft::device_span<float const>> wgts1,
  std::optional<raft::device_span<int64_t const>> edge_ids1,
  std::optional<raft::device_span<int32_t const>> edge_types1,
  raft::device_span<int64_t const> srcs2,
  raft::device_span<int64_t const> dsts2,
  std::optional<raft::device_span<float const>> wgts2,
  std::optional<raft::device_span<int64_t const>> edge_ids2,
  std::optional<raft::device_span<int32_t const>> edge_types2);

template size_t count_intersection<int32_t, int32_t, double, int32_t, false>(
  raft::handle_t const& handle,
  raft::device_span<int32_t const> srcs1,
  raft::device_span<int32_t const> dsts1,
  std::optional<raft::device_span<double const>> wgts1,
  std::optional<raft::device_span<int32_t const>> edge_ids1,
  std::optional<raft::device_span<int32_t const>> edge_types1,
  raft::device_span<int32_t const> srcs2,
  raft::device_span<int32_t const> dsts2,
  std::optional<raft::device_span<double const>> wgts2,
  std::optional<raft::device_span<int32_t const>> edge_ids2,
  std::optional<raft::device_span<int32_t const>> edge_types2);

template size_t count_intersection<int32_t, int64_t, double, int32_t, false>(
  raft::handle_t const& handle,
  raft::device_span<int32_t const> srcs1,
  raft::device_span<int32_t const> dsts1,
  std::optional<raft::device_span<double const>> wgts1,
  std::optional<raft::device_span<int64_t const>> edge_ids1,
  std::optional<raft::device_span<int32_t const>> edge_types1,
  raft::device_span<int32_t const> srcs2,
  raft::device_span<int32_t const> dsts2,
  std::optional<raft::device_span<double const>> wgts2,
  std::optional<raft::device_span<int64_t const>> edge_ids2,
  std::optional<raft::device_span<int32_t const>> edge_types2);

template size_t count_intersection<int64_t, int64_t, double, int32_t, false>(
  raft::handle_t const& handle,
  raft::device_span<int64_t const> srcs1,
  raft::device_span<int64_t const> dsts1,
  std::optional<raft::device_span<double const>> wgts1,
  std::optional<raft::device_span<int64_t const>> edge_ids1,
  std::optional<raft::device_span<int32_t const>> edge_types1,
  raft::device_span<int64_t const> srcs2,
  raft::device_span<int64_t const> dsts2,
  std::optional<raft::device_span<double const>> wgts2,
  std::optional<raft::device_span<int64_t const>> edge_ids2,
  std::optional<raft::device_span<int32_t const>> edge_types2);

}  // namespace cugraph::test

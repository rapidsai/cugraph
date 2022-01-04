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
/*
 * COOtoCSR_kernels.cuh
 *
 *  Created on: Mar 8, 2018
 *      Author: jwyles
 */

#pragma once

#include <algorithm>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <cugraph/utilities/error.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>

#include <cugraph/functions.hpp>

#include <cugraph/legacy/graph.hpp>

namespace cugraph {
namespace detail {

/**
 * @brief     Sort input graph and find the total number of vertices
 *
 * Lexicographically sort a COO view and find the total number of vertices
 *
 * @throws                 cugraph::logic_error when an error occurs.
 *
 * @tparam VT              Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET              Type of edge identifiers. Supported value : int (signed, 32-bit)
 * @tparam WT              Type of edge weights. Supported value : float or double.
 *
 * @param[in] graph        The input graph object
 * @param[in] stream_view  The cuda stream for kernel calls
 *
 * @param[out] result      Total number of vertices
 */
template <typename VT, typename ET, typename WT>
VT sort(legacy::GraphCOOView<VT, ET, WT>& graph, rmm::cuda_stream_view stream_view)
{
  VT max_src_id;
  VT max_dst_id;
  if (graph.has_data()) {
    thrust::stable_sort_by_key(
      rmm::exec_policy(stream_view),
      graph.dst_indices,
      graph.dst_indices + graph.number_of_edges,
      thrust::make_zip_iterator(thrust::make_tuple(graph.src_indices, graph.edge_data)));
    RAFT_CUDA_TRY(cudaMemcpy(
      &max_dst_id, &(graph.dst_indices[graph.number_of_edges - 1]), sizeof(VT), cudaMemcpyDefault));
    thrust::stable_sort_by_key(
      rmm::exec_policy(stream_view),
      graph.src_indices,
      graph.src_indices + graph.number_of_edges,
      thrust::make_zip_iterator(thrust::make_tuple(graph.dst_indices, graph.edge_data)));
    RAFT_CUDA_TRY(cudaMemcpy(
      &max_src_id, &(graph.src_indices[graph.number_of_edges - 1]), sizeof(VT), cudaMemcpyDefault));
  } else {
    thrust::stable_sort_by_key(rmm::exec_policy(stream_view),
                               graph.dst_indices,
                               graph.dst_indices + graph.number_of_edges,
                               graph.src_indices);
    RAFT_CUDA_TRY(cudaMemcpy(
      &max_dst_id, &(graph.dst_indices[graph.number_of_edges - 1]), sizeof(VT), cudaMemcpyDefault));
    thrust::stable_sort_by_key(rmm::exec_policy(stream_view),
                               graph.src_indices,
                               graph.src_indices + graph.number_of_edges,
                               graph.dst_indices);
    RAFT_CUDA_TRY(cudaMemcpy(
      &max_src_id, &(graph.src_indices[graph.number_of_edges - 1]), sizeof(VT), cudaMemcpyDefault));
  }
  return std::max(max_src_id, max_dst_id) + 1;
}

template <typename VT, typename ET>
void fill_offset(VT* source,
                 ET* offsets,
                 VT number_of_vertices,
                 ET number_of_edges,
                 rmm::cuda_stream_view stream_view)
{
  thrust::fill(
    rmm::exec_policy(stream_view), offsets, offsets + number_of_vertices + 1, number_of_edges);
  thrust::for_each(rmm::exec_policy(stream_view),
                   thrust::make_counting_iterator<ET>(1),
                   thrust::make_counting_iterator<ET>(number_of_edges),
                   [source, offsets] __device__(ET index) {
                     VT id = source[index];
                     if (id != source[index - 1]) { offsets[id] = index; }
                   });
  thrust::device_ptr<VT> src = thrust::device_pointer_cast(source);
  thrust::device_ptr<ET> off = thrust::device_pointer_cast(offsets);
  off[src[0]]                = ET{0};

  auto iter = thrust::make_reverse_iterator(offsets + number_of_vertices + 1);
  thrust::inclusive_scan(rmm::exec_policy(stream_view),
                         iter,
                         iter + number_of_vertices + 1,
                         iter,
                         thrust::minimum<ET>());
}

template <typename VT, typename ET>
rmm::device_buffer create_offset(VT* source,
                                 VT number_of_vertices,
                                 ET number_of_edges,
                                 rmm::cuda_stream_view stream_view,
                                 rmm::mr::device_memory_resource* mr)
{
  // Offset array needs an extra element at the end to contain the ending offsets
  // of the last vertex
  rmm::device_buffer offsets_buffer(sizeof(ET) * (number_of_vertices + 1), stream_view, mr);
  ET* offsets = static_cast<ET*>(offsets_buffer.data());

  fill_offset(source, offsets, number_of_vertices, number_of_edges, stream_view);

  return offsets_buffer;
}

}  // namespace detail

template <typename VT, typename ET, typename WT>
std::unique_ptr<legacy::GraphCSR<VT, ET, WT>> coo_to_csr(
  legacy::GraphCOOView<VT, ET, WT> const& graph, rmm::mr::device_memory_resource* mr)
{
  rmm::cuda_stream_view stream_view;

  legacy::GraphCOO<VT, ET, WT> temp_graph(graph, stream_view.value(), mr);
  legacy::GraphCOOView<VT, ET, WT> temp_graph_view = temp_graph.view();
  VT total_vertex_count                            = detail::sort(temp_graph_view, stream_view);
  rmm::device_buffer offsets                       = detail::create_offset(
    temp_graph.src_indices(), total_vertex_count, temp_graph.number_of_edges(), stream_view, mr);
  auto coo_contents = temp_graph.release();
  legacy::GraphSparseContents<VT, ET, WT> csr_contents{
    total_vertex_count,
    coo_contents.number_of_edges,
    std::make_unique<rmm::device_buffer>(std::move(offsets)),
    std::move(coo_contents.dst_indices),
    std::move(coo_contents.edge_data)};

  return std::make_unique<legacy::GraphCSR<VT, ET, WT>>(std::move(csr_contents));
}

template <typename VT, typename ET, typename WT>
void coo_to_csr_inplace(legacy::GraphCOOView<VT, ET, WT>& graph,
                        legacy::GraphCSRView<VT, ET, WT>& result)
{
  rmm::cuda_stream_view stream_view;

  detail::sort(graph, stream_view);
  detail::fill_offset(graph.src_indices,
                      result.offsets,
                      graph.number_of_vertices,
                      graph.number_of_edges,
                      stream_view);

  RAFT_CUDA_TRY(cudaMemcpy(
    result.indices, graph.dst_indices, sizeof(VT) * graph.number_of_edges, cudaMemcpyDefault));
  if (graph.has_data())
    RAFT_CUDA_TRY(cudaMemcpy(
      result.edge_data, graph.edge_data, sizeof(WT) * graph.number_of_edges, cudaMemcpyDefault));
}

// Explicit Instantiation Declarations (EIDecl)
// to attempt decrease in compile time:
//
// EIDecl for uint32_t + float
extern template std::unique_ptr<legacy::GraphCSR<uint32_t, uint32_t, float>>
coo_to_csr<uint32_t, uint32_t, float>(legacy::GraphCOOView<uint32_t, uint32_t, float> const& graph,
                                      rmm::mr::device_memory_resource*);

// EIDecl for uint32_t + double
extern template std::unique_ptr<legacy::GraphCSR<uint32_t, uint32_t, double>>
coo_to_csr<uint32_t, uint32_t, double>(
  legacy::GraphCOOView<uint32_t, uint32_t, double> const& graph, rmm::mr::device_memory_resource*);

// EIDecl for int + float
extern template std::unique_ptr<legacy::GraphCSR<int32_t, int32_t, float>>
coo_to_csr<int32_t, int32_t, float>(legacy::GraphCOOView<int32_t, int32_t, float> const& graph,
                                    rmm::mr::device_memory_resource*);

// EIDecl for int + double
extern template std::unique_ptr<legacy::GraphCSR<int32_t, int32_t, double>>
coo_to_csr<int32_t, int32_t, double>(legacy::GraphCOOView<int32_t, int32_t, double> const& graph,
                                     rmm::mr::device_memory_resource*);

// EIDecl for int64_t + float
extern template std::unique_ptr<legacy::GraphCSR<int64_t, int64_t, float>>
coo_to_csr<int64_t, int64_t, float>(legacy::GraphCOOView<int64_t, int64_t, float> const& graph,
                                    rmm::mr::device_memory_resource*);

// EIDecl for int64_t + double
extern template std::unique_ptr<legacy::GraphCSR<int64_t, int64_t, double>>
coo_to_csr<int64_t, int64_t, double>(legacy::GraphCOOView<int64_t, int64_t, double> const& graph,
                                     rmm::mr::device_memory_resource*);

// in-place versions:
//
// EIDecl for uint32_t + float
extern template void coo_to_csr_inplace<uint32_t, uint32_t, float>(
  legacy::GraphCOOView<uint32_t, uint32_t, float>& graph,
  legacy::GraphCSRView<uint32_t, uint32_t, float>& result);

// EIDecl for uint32_t + double
extern template void coo_to_csr_inplace<uint32_t, uint32_t, double>(
  legacy::GraphCOOView<uint32_t, uint32_t, double>& graph,
  legacy::GraphCSRView<uint32_t, uint32_t, double>& result);

// EIDecl for int + float
extern template void coo_to_csr_inplace<int32_t, int32_t, float>(
  legacy::GraphCOOView<int32_t, int32_t, float>& graph,
  legacy::GraphCSRView<int32_t, int32_t, float>& result);

// EIDecl for int + double
extern template void coo_to_csr_inplace<int32_t, int32_t, double>(
  legacy::GraphCOOView<int32_t, int32_t, double>& graph,
  legacy::GraphCSRView<int32_t, int32_t, double>& result);

// EIDecl for int64_t + float
extern template void coo_to_csr_inplace<int64_t, int64_t, float>(
  legacy::GraphCOOView<int64_t, int64_t, float>& graph,
  legacy::GraphCSRView<int64_t, int64_t, float>& result);

// EIDecl for int64_t + double
extern template void coo_to_csr_inplace<int64_t, int64_t, double>(
  legacy::GraphCOOView<int64_t, int64_t, double>& graph,
  legacy::GraphCSRView<int64_t, int64_t, double>& result);

}  // namespace cugraph

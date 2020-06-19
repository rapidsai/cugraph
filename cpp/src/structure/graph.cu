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

#include <graph.hpp>
#include "utilities/cuda_utils.cuh"
#include "utilities/error.hpp"
#include "utilities/graph_utils.cuh"

namespace {

template <typename vertex_t, typename edge_t>
void degree_from_offsets(vertex_t number_of_vertices,
                         edge_t const *offsets,
                         edge_t *degree,
                         cudaStream_t stream)
{
  // Computes out-degree for x = 0 and x = 2
  thrust::for_each(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<vertex_t>(0),
    thrust::make_counting_iterator<vertex_t>(number_of_vertices),
    [offsets, degree] __device__(vertex_t v) { degree[v] = offsets[v + 1] - offsets[v]; });
}

template <typename vertex_t, typename edge_t>
void degree_from_vertex_ids(const raft::handle_t &handle,
                            vertex_t number_of_vertices,
                            edge_t number_of_edges,
                            vertex_t const *indices,
                            edge_t *degree,
                            cudaStream_t stream)
{
  thrust::for_each(
    rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<edge_t>(0),
    thrust::make_counting_iterator<edge_t>(number_of_edges),
    [indices, degree] __device__(edge_t e) { cugraph::atomicAdd(degree + indices[e], 1); });
  if (handle.comms_initialized()) {
    auto &comm = handle.get_comms();
    comm.allreduce(degree, degree, number_of_vertices, raft::comms::op_t::SUM, stream);
  }
}

}  // namespace

namespace cugraph {
namespace experimental {

template <typename VT, typename ET, typename WT>
void GraphViewBase<VT, ET, WT>::get_vertex_identifiers(VT *identifiers) const
{
  cugraph::detail::sequence<VT>(number_of_vertices, identifiers);
}

template <typename VT, typename ET, typename WT>
void GraphCompressedSparseBaseView<VT, ET, WT>::get_source_indices(VT *src_indices) const
{
  CUGRAPH_EXPECTS(offsets != nullptr, "No graph specified");
  cugraph::detail::offsets_to_indices<VT>(
    offsets, GraphViewBase<VT, ET, WT>::number_of_vertices, src_indices);
}

template <typename VT, typename ET, typename WT>
void GraphCOOView<VT, ET, WT>::degree(ET *degree, DegreeDirection direction) const
{
  //
  // NOTE:  We assume offsets/indices are a CSR.  If a CSC is passed
  //        in then x should be modified to reflect the expected direction.
  //        (e.g. if you have a CSC and you want in-degree (x=1) then pass
  //        the offsets/indices and request an out-degree (x=2))
  //
  cudaStream_t stream{nullptr};

  if (direction != DegreeDirection::IN) {
    if (GraphViewBase<VT, ET, WT>::handle->comms_initialized())  // FIXME retrieve global source
                                                                 // indexing for the allreduce work
      CUGRAPH_FAIL("OPG degree not implemented for OUT degree");
    degree_from_vertex_ids(GraphViewBase<VT, ET, WT>::handle[0],
                           GraphViewBase<VT, ET, WT>::number_of_vertices,
                           GraphViewBase<VT, ET, WT>::number_of_edges,
                           src_indices,
                           degree,
                           stream);
  }

  if (direction != DegreeDirection::OUT) {
    degree_from_vertex_ids(GraphViewBase<VT, ET, WT>::handle[0],
                           GraphViewBase<VT, ET, WT>::number_of_vertices,
                           GraphViewBase<VT, ET, WT>::number_of_edges,
                           dst_indices,
                           degree,
                           stream);
  }
}

template <typename VT, typename ET, typename WT>
void GraphCompressedSparseBaseView<VT, ET, WT>::degree(ET *degree, DegreeDirection direction) const
{
  //
  // NOTE:  We assume offsets/indices are a CSR.  If a CSC is passed
  //        in then x should be modified to reflect the expected direction.
  //        (e.g. if you have a CSC and you want in-degree (x=1) then pass
  //        the offsets/indices and request an out-degree (x=2))
  //
  cudaStream_t stream{nullptr};

  if (direction != DegreeDirection::IN) {
    if (GraphViewBase<VT, ET, WT>::handle->comms_initialized())
      CUGRAPH_FAIL("OPG degree not implemented for OUT degree");  // FIXME retrieve global
                                                                  // source indexing for
                                                                  // the allreduce to work
    degree_from_offsets(GraphViewBase<VT, ET, WT>::number_of_vertices, offsets, degree, stream);
  }

  if (direction != DegreeDirection::OUT) {
    degree_from_vertex_ids(GraphViewBase<VT, ET, WT>::handle[0],
                           GraphViewBase<VT, ET, WT>::number_of_vertices,
                           GraphViewBase<VT, ET, WT>::number_of_edges,
                           indices,
                           degree,
                           stream);
  }
}

// explicit instantiation
template class GraphViewBase<int32_t, int32_t, float>;
template class GraphViewBase<int32_t, int32_t, double>;
template class GraphCOOView<int32_t, int32_t, float>;
template class GraphCOOView<int32_t, int32_t, double>;
template class GraphCompressedSparseBaseView<int32_t, int32_t, float>;
template class GraphCompressedSparseBaseView<int32_t, int32_t, double>;
}  // namespace experimental
}  // namespace cugraph

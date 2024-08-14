/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/device_span.hpp>
#include <raft/util/device_atomics.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

namespace {

template <typename vertex_t, typename edge_t>
void degree_from_offsets(vertex_t number_of_vertices,
                         edge_t const* offsets,
                         edge_t* degree,
                         rmm::cuda_stream_view stream_view)
{
  // Computes out-degree for x = 0 and x = 2
  thrust::for_each(
    rmm::exec_policy(stream_view),
    thrust::make_counting_iterator<vertex_t>(0),
    thrust::make_counting_iterator<vertex_t>(number_of_vertices),
    [offsets, degree] __device__(vertex_t v) { degree[v] = offsets[v + 1] - offsets[v]; });
}

template <typename vertex_t, typename edge_t>
void degree_from_vertex_ids(const raft::handle_t* handle,
                            vertex_t number_of_vertices,
                            edge_t number_of_edges,
                            vertex_t const* indices,
                            edge_t* degree,
                            rmm::cuda_stream_view stream_view)
{
  thrust::for_each(
    rmm::exec_policy(stream_view),
    thrust::make_counting_iterator<edge_t>(0),
    thrust::make_counting_iterator<edge_t>(number_of_edges),
    [indices, degree] __device__(edge_t e) { atomicAdd(degree + indices[e], edge_t{1}); });
  if ((handle != nullptr) && (handle->comms_initialized())) {
    auto& comm = handle->get_comms();
    comm.allreduce(degree, degree, number_of_vertices, raft::comms::op_t::SUM, stream_view.value());
  }
}

}  // namespace

namespace cugraph {
namespace legacy {

template <typename VT, typename ET, typename WT>
void GraphViewBase<VT, ET, WT>::get_vertex_identifiers(VT* identifiers) const
{
  thrust::sequence(thrust::device,
                   thrust::device_pointer_cast(identifiers),
                   thrust::device_pointer_cast(identifiers + number_of_vertices),
                   VT{0});
  RAFT_CHECK_CUDA(nullptr);
}

// FIXME: Need to get rid of this function... still used in python
template <typename VT, typename ET, typename WT>
void GraphCompressedSparseBaseView<VT, ET, WT>::get_source_indices(VT* src_indices) const
{
  CUGRAPH_EXPECTS(offsets != nullptr, "No graph specified");
  rmm::cuda_stream_view stream_view;

  raft::device_span<VT> indices_span(src_indices, GraphViewBase<VT, ET, WT>::number_of_vertices);

  if (indices_span.size() > 0) {
    thrust::fill(rmm::exec_policy(stream_view), indices_span.begin(), indices_span.end(), VT{0});

    thrust::for_each(rmm::exec_policy(stream_view),
                     offsets + 1,
                     offsets + GraphViewBase<VT, ET, WT>::number_of_vertices,
                     [indices_span] __device__(ET offset) {
                       if (offset < static_cast<ET>(indices_span.size())) {
                         cuda::atomic_ref<VT, cuda::thread_scope_device> atomic_counter(
                           indices_span.data()[offset]);
                         atomic_counter.fetch_add(VT{1}, cuda::std::memory_order_relaxed);
                       }
                     });
    thrust::inclusive_scan(rmm::exec_policy(stream_view),
                           indices_span.begin(),
                           indices_span.end(),
                           indices_span.begin());
  }
}

template <typename VT, typename ET, typename WT>
void GraphCOOView<VT, ET, WT>::degree(ET* degree, DegreeDirection direction) const
{
  //
  // NOTE:  We assume offsets/indices are a CSR.  If a CSC is passed
  //        in then x should be modified to reflect the expected direction.
  //        (e.g. if you have a CSC and you want in-degree (x=1) then pass
  //        the offsets/indices and request an out-degree (x=2))
  //
  cudaStream_t stream{nullptr};

  if (direction != DegreeDirection::IN) {
    if ((GraphViewBase<VT, ET, WT>::handle != nullptr) &&
        (GraphViewBase<VT, ET, WT>::handle
           ->comms_initialized()))  // FIXME retrieve global source
                                    // indexing for the allreduce work
    {
      CUGRAPH_FAIL("MG degree not implemented for OUT degree");
    }
    degree_from_vertex_ids(GraphViewBase<VT, ET, WT>::handle,
                           GraphViewBase<VT, ET, WT>::number_of_vertices,
                           GraphViewBase<VT, ET, WT>::number_of_edges,
                           src_indices,
                           degree,
                           stream);
  }

  if (direction != DegreeDirection::OUT) {
    degree_from_vertex_ids(GraphViewBase<VT, ET, WT>::handle,
                           GraphViewBase<VT, ET, WT>::number_of_vertices,
                           GraphViewBase<VT, ET, WT>::number_of_edges,
                           dst_indices,
                           degree,
                           stream);
  }
}

template <typename VT, typename ET, typename WT>
void GraphCompressedSparseBaseView<VT, ET, WT>::degree(ET* degree, DegreeDirection direction) const
{
  //
  // NOTE:  We assume offsets/indices are a CSR.  If a CSC is passed
  //        in then x should be modified to reflect the expected direction.
  //        (e.g. if you have a CSC and you want in-degree (x=1) then pass
  //        the offsets/indices and request an out-degree (x=2))
  //
  rmm::cuda_stream_view stream_view;

  if (direction != DegreeDirection::IN) {
    if ((GraphViewBase<VT, ET, WT>::handle != nullptr) &&
        (GraphViewBase<VT, ET, WT>::handle->comms_initialized())) {
      CUGRAPH_FAIL("MG degree not implemented for OUT degree");  // FIXME retrieve global
                                                                 // source indexing for
                                                                 // the allreduce to work
    }
    degree_from_offsets(
      GraphViewBase<VT, ET, WT>::number_of_vertices, offsets, degree, stream_view);
  }

  if (direction != DegreeDirection::OUT) {
    degree_from_vertex_ids(GraphViewBase<VT, ET, WT>::handle,
                           GraphViewBase<VT, ET, WT>::number_of_vertices,
                           GraphViewBase<VT, ET, WT>::number_of_edges,
                           indices,
                           degree,
                           stream_view);
  }
}

}  // namespace legacy
}  // namespace cugraph

#include <cugraph/legacy/eidir_graph.hpp>

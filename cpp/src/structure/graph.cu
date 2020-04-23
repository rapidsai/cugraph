 /*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

#include <graph.hpp>
#include "utilities/graph_utils.cuh"
#include "utilities/error_utils.h"
#include "utilities/cuda_utils.cuh"



namespace {

template <typename vertex_t, typename edge_t>
void degree_from_offsets(vertex_t number_of_vertices,
                         edge_t const *offsets,
                         edge_t *degree,
                         cudaStream_t stream) {

  // Computes out-degree for x = 0 and x = 2
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator<vertex_t>(0),
                   thrust::make_counting_iterator<vertex_t>(number_of_vertices),
                   [offsets, degree] __device__ (vertex_t v) {
                     degree[v] = offsets[v+1]-offsets[v];
                   });
}

template <typename vertex_t, typename edge_t>
void degree_from_vertex_ids(cugraph::experimental::Comm& comm,
                            vertex_t number_of_vertices,
                            edge_t number_of_edges,
                            vertex_t const *indices,
                            edge_t *degree,
                            cudaStream_t stream) {

  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   thrust::make_counting_iterator<edge_t>(0),
                   thrust::make_counting_iterator<edge_t>(number_of_edges),
                   [indices, degree] __device__ (edge_t e) {
                     cugraph::atomicAdd(degree + indices[e], 1);
                   });
  comm.allreduce(degree, degree, number_of_vertices, cugraph::ReduceOp::SUM);
}

} //namespace anonymous

namespace cugraph {
namespace experimental {


template <typename VT, typename ET, typename WT>
void GraphBase<VT,ET,WT>::get_vertex_identifiers(VT *identifiers) const {
  cugraph::detail::sequence<VT>(number_of_vertices, identifiers);
}

template <typename VT, typename ET, typename WT>
void GraphCompressedSparseBase<VT,ET,WT>::get_source_indices(VT *src_indices) const {
  CUGRAPH_EXPECTS( offsets != nullptr , "No graph specified");
  cugraph::detail::offsets_to_indices<VT>(offsets, GraphBase<VT,ET,WT>::number_of_vertices, src_indices);
}

template <typename VT, typename ET, typename WT>
void GraphCOO<VT,ET,WT>::degree(ET *degree, DegreeDirection direction) const {
  //
  // NOTE:  We assume offsets/indices are a CSR.  If a CSC is passed
  //        in then x should be modified to reflect the expected direction.
  //        (e.g. if you have a CSC and you want in-degree (x=1) then pass
  //        the offsets/indices and request an out-degree (x=2))
  //
  cudaStream_t stream{nullptr};

  if (direction != DegreeDirection::IN) {
    degree_from_vertex_ids(GraphBase<VT,ET,WT>::comm, GraphBase<VT,ET,WT>::number_of_vertices, GraphBase<VT,ET,WT>::number_of_edges, src_indices, degree, stream);
  }

  if (direction != DegreeDirection::OUT) {
    degree_from_vertex_ids(GraphBase<VT,ET,WT>::comm, GraphBase<VT,ET,WT>::number_of_vertices, GraphBase<VT,ET,WT>::number_of_edges, dst_indices, degree, stream);
  }
}

template <typename VT, typename ET, typename WT>
void GraphCompressedSparseBase<VT,ET,WT>::degree(ET *degree, DegreeDirection direction) const {
  //
  // NOTE:  We assume offsets/indices are a CSR.  If a CSC is passed
  //        in then x should be modified to reflect the expected direction.
  //        (e.g. if you have a CSC and you want in-degree (x=1) then pass
  //        the offsets/indices and request an out-degree (x=2))
  //
  cudaStream_t stream{nullptr};

  if (direction != DegreeDirection::IN) {
    degree_from_offsets(GraphBase<VT,ET,WT>::number_of_vertices, offsets, degree, stream);
  }

  if (direction != DegreeDirection::OUT) {
    degree_from_vertex_ids(GraphBase<VT,ET,WT>::comm, GraphBase<VT,ET,WT>::number_of_vertices, GraphBase<VT,ET,WT>::number_of_edges, indices, degree, stream);
  }
}

// explicit instantiation
template class GraphBase<int32_t, int32_t, float>;
template class GraphBase<int32_t, int32_t, double>;
template class GraphCOO<int32_t,int32_t,float>;
template class GraphCOO<int32_t,int32_t,double>;
template class GraphCompressedSparseBase<int32_t,int32_t,float>;
template class GraphCompressedSparseBase<int32_t,int32_t,double>;
}
}

// -*-c++-*-

 /*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

// Graph analytics features


#include <library_types.h>
#include <nvgraph/nvgraph.h>
#include <thrust/device_vector.h>
#include <rmm_utils.h>
#include <utilities/validation.cuh>

#include "types.h"
#include "functions.h"
#include "utilities/graph_utils.cuh"
#include "converters/COOtoCSR.cuh"
#include "utilities/error_utils.h"
#include "converters/renumber.cuh"

namespace cugraph {
int get_device(const void *ptr) {
    cudaPointerAttributes att;
    cudaPointerGetAttributes(&att, ptr);
    return att.device;
}

template <typename VT, typename WT>
void transposed_adj_list_view(Graph<VT, WT> *graph, 
                            const size_t v,
                            const size_t e,
                            VT *offsets,
                            VT *indices,
                            WT *edge_data) {
  //This function returns an error if this graph object has at least one graph
  //representation to prevent a single object storing two different graphs.
  CUGRAPH_EXPECTS( ((graph->edgeList == nullptr) && (graph->adjList == nullptr) &&
    (graph->transposedAdjList == nullptr)), "Invalid API parameter");
  CUGRAPH_EXPECTS( typeid(offsets) == typeid(indices), "Unsupported data type" );
  CUGRAPH_EXPECTS( typeid(offsets) == typeid(int*), "Unsupported data type" );
  CUGRAPH_EXPECTS( (v > 0), "Column is empty");

  graph->transposedAdjList = new adj_list<VT, WT>;
  graph->transposedAdjList->ownership = 0;
  graph->v = v;
  graph->e = e;
  graph->transposedAdjList->offsets = offsets;
  graph->transposedAdjList->indices = indices;


  
  if (!graph->prop)
      graph->prop = new Graph_properties();

  if (edge_data) {
    graph->transposedAdjList->edge_data = edge_data;
    
    bool has_neg_val;
    has_neg_val = cugraph::detail::has_negative_val(graph->transposedAdjList->edge_data, graph->e);
    graph->prop->has_negative_edges =
        (has_neg_val) ? PROP_TRUE : PROP_FALSE;
  } else {
    graph->adjList->edge_data = nullptr;
    graph->prop->has_negative_edges = PROP_FALSE;
  }
}

template <typename VT, typename WT>
void adj_list_view(Graph<VT, WT> *graph, const size_t v,
                   const size_t e, VT *offsets,
                   VT *indices,
                   WT *edge_data) {
  //This function returns an error if this graph object has at least one graph
  //representation to prevent a single object storing two different graphs.
  CUGRAPH_EXPECTS( ((graph->edgeList == nullptr) && (graph->adjList == nullptr) &&
    (graph->transposedAdjList == nullptr)), "Invalid API parameter");
  CUGRAPH_EXPECTS( typeid(offsets) == typeid(indices), "Unsupported data type" );
  CUGRAPH_EXPECTS( typeid(offsets) == typeid(int*), "Unsupported data type" );
  CUGRAPH_EXPECTS( v > 0, "Column is empty");

  graph->adjList = new adj_list<VT, WT>;
  graph->adjList->ownership = 0;
  graph->v = v;
  graph->e = e;
  graph->adjList->offsets = offsets;
  graph->adjList->indices = indices;
  
  if (!graph->prop)
      graph->prop = new Graph_properties();

  if (edge_data) {
    graph->adjList->edge_data = edge_data;
    
    bool has_neg_val;
    has_neg_val = cugraph::detail::has_negative_val(graph->adjList->edge_data, graph->e);
    graph->prop->has_negative_edges =
        (has_neg_val) ? PROP_TRUE : PROP_FALSE;
  } else {
    graph->adjList->edge_data = nullptr;
    graph->prop->has_negative_edges = PROP_FALSE;
  }  
}

template <typename VT, typename WT>
void adj_list<VT,WT>::get_vertex_identifiers(size_t v, VT *identifiers) {
  CUGRAPH_EXPECTS( offsets != nullptr , "Invalid API parameter");
  cugraph::detail::sequence<VT>(v, identifiers);
}

template <typename VT, typename WT>
void adj_list<VT,WT>::get_source_indices (size_t v, VT *src_indices) {
  CUGRAPH_EXPECTS( offsets != nullptr , "Invalid API parameter");
  CUGRAPH_EXPECTS( typeid(src_indices) == typeid(indices), "Unsupported data type" );
  CUGRAPH_EXPECTS( v > 0, "Column is empty");
  
  cugraph::detail::offsets_to_indices<VT>(offsets, v, src_indices);

  
}
template <typename VT, typename WT>
adj_list<VT,WT>::~adj_list() {
  cudaStream_t stream{nullptr};
  if (ownership == 1 ) {
    ALLOC_FREE_TRY(offsets, stream);
    ALLOC_FREE_TRY(indices, stream);
    ALLOC_FREE_TRY(edge_data, stream);
  }
}
template <typename VT, typename WT>
void edge_list_view(Graph<VT, WT> *graph,
                    const size_t e, VT *src_indices,
                    VT *dest_indices, 
                    WT *edge_data) {
  //This function returns an error if this graph object has at least one graph
  //representation to prevent a single object storing two different graphs.

  CUGRAPH_EXPECTS( ((graph->edgeList == nullptr) && (graph->adjList == nullptr) &&
    (graph->transposedAdjList == nullptr)), "Invalid API parameter");
  CUGRAPH_EXPECTS( typeid(src_indices) == typeid(dest_indices), "Unsupported data type" );
  CUGRAPH_EXPECTS( typeid(src_indices) == typeid(int*), "Unsupported data type" );
  CUGRAPH_EXPECTS( e > 0, "Column is empty");

  graph->edgeList = new edge_list<VT, WT>;
  graph->edgeList->ownership = 0;
  graph->e = e;
  graph->edgeList->src_indices = src_indices;
  graph->edgeList->dest_indices = dest_indices;

  if (!graph->prop)
    graph->prop = new Graph_properties();

  if (edge_data) {
    graph->edgeList->edge_data = edge_data;

    bool has_neg_val;
    has_neg_val = cugraph::detail::has_negative_val(graph->edgeList->edge_data, graph->e);
    graph->prop->has_negative_edges = (has_neg_val) ? PROP_TRUE : PROP_FALSE;

  } else {
    graph->edgeList->edge_data = nullptr;
    graph->prop->has_negative_edges = PROP_FALSE;
  }

  cugraph::detail::indexing_check<VT> (
                                graph->edgeList->src_indices, 
                                graph->edgeList->dest_indices, 
                                graph->e);
}
  
template <typename VT, typename WT>
edge_list<VT,WT>::~edge_list() {
  cudaStream_t stream{nullptr};
  if (ownership == 1 ) {
    ALLOC_FREE_TRY(src_indices,stream);
    ALLOC_FREE_TRY(dest_indices,stream);
    ALLOC_FREE_TRY(edge_data,stream);
  }
  else if (ownership == 2 )
  {
    ALLOC_FREE_TRY(src_indices,stream);
  }
}

template <typename VT, typename WT>
void add_adj_list_impl (Graph<VT, WT> *graph) {
    if (graph->adjList == nullptr) {
      CUGRAPH_EXPECTS( graph->edgeList != nullptr , "Invalid API parameter");
      VT nnz = graph->e;
      graph->adjList = new adj_list<VT, WT>;
      graph->adjList->ownership = 1;

    if (graph->edgeList->edge_data!= nullptr) {
      CSR_Result_Weighted<VT,WT> adj_list;
      ConvertCOOtoCSR_weighted(graph->edgeList->src_indices, graph->edgeList->dest_indices, graph->edgeList->edge_data, nnz, adj_list);
      graph->v = adj_list.size;
      graph->adjList->offsets = adj_list.rowOffsets;
      graph->adjList->indices = adj_list.colIndices;
      graph->adjList->edge_data = adj_list.edgeWeights;
    }
    else {
      CSR_Result<VT> adj_list;
      ConvertCOOtoCSR(graph->edgeList->src_indices,graph->edgeList->dest_indices, nnz, adj_list);
      graph->v = adj_list.size;
      graph->adjList->offsets = adj_list.rowOffsets;
      graph->adjList->indices = adj_list.colIndices;
    }
  }
}

template <typename VT, typename WT>
void add_edge_list (Graph<VT, WT> *graph) {
    if (graph->edgeList == nullptr) {
      CUGRAPH_EXPECTS( graph->adjList != nullptr , "Invalid API parameter");
      VT *d_src;
      graph->edgeList = new edge_list<VT, WT>;

      graph->edgeList->ownership = 2;

      cudaStream_t stream{nullptr};

      ALLOC_TRY((void**)&d_src, sizeof(VT) * graph->e, stream);
      cugraph::detail::offsets_to_indices<VT>(graph->adjList->offsets,
                                  graph->v,
                                  d_src);
      graph->edgeList->src_indices = d_src;
      graph->edgeList->dest_indices = graph->adjList->indices;
      if (graph->adjList->edge_data != nullptr) {
        graph->edgeList->edge_data = graph->adjList->edge_data;
      }
  }
}


template <typename VT, typename WT>
void add_transposed_adj_list_impl (Graph<VT, WT> *graph) {
    if (graph->transposedAdjList == nullptr ) {
      CUGRAPH_EXPECTS( graph->edgeList != nullptr , "Invalid API parameter");
      VT nnz = graph->e;
      graph->transposedAdjList = new adj_list<VT, WT>;
      graph->transposedAdjList->ownership = 1;

      if (graph->edgeList->edge_data) {
        CSR_Result_Weighted<VT,WT> adj_list;
        ConvertCOOtoCSR_weighted( graph->edgeList->dest_indices, graph->edgeList->src_indices, graph->edgeList->edge_data, nnz, adj_list);
        graph->v = adj_list.size;
        graph->transposedAdjList->offsets = adj_list.rowOffsets;
        graph->transposedAdjList->indices = adj_list.colIndices;
        graph->transposedAdjList->edge_data = adj_list.edgeWeights;
      }
      else {

        CSR_Result<VT> adj_list;
        ConvertCOOtoCSR(graph->edgeList->dest_indices, graph->edgeList->src_indices, nnz, adj_list);
        graph->v = adj_list.size;
        graph->transposedAdjList->offsets = adj_list.rowOffsets;
        graph->transposedAdjList->indices = adj_list.colIndices;
      }
    }
}

template <typename VT, typename WT>
void add_adj_list(Graph<VT, WT> *graph) {
  if (graph->adjList == nullptr) {
    CUGRAPH_EXPECTS( graph->edgeList != nullptr , "Invalid API parameter");
    CUGRAPH_EXPECTS( typeid(graph->edgeList->src_indices) == typeid(int*), "Unsupported data type" );
    return cugraph::add_adj_list_impl<VT, WT>(graph);
  }
}

template <typename VT, typename WT>
void add_transposed_adj_list(Graph<VT, WT> *graph) {
  if (graph->transposedAdjList == nullptr) {
    // csr -> coo -> csc
    if (graph->edgeList == nullptr)
      cugraph::add_edge_list(graph);
    CUGRAPH_EXPECTS(typeid(graph->edgeList->src_indices) == typeid(int*), "Unsupported data type");
    return cugraph::add_transposed_adj_list_impl<VT,WT>(graph);
  }
}

template <typename VT, typename WT>
void delete_adj_list(Graph<VT, WT> *graph) {
  if (graph->adjList) {
    delete graph->adjList;
  }
  graph->adjList = nullptr;
}

template <typename VT, typename WT>
void delete_edge_list(Graph<VT, WT> *graph) {
  if (graph->edgeList) {
    delete graph->edgeList;
  }
  graph->edgeList = nullptr;
  
}

template <typename VT, typename WT>
void delete_transposed_adj_list(Graph<VT, WT> *graph) {
  if (graph->transposedAdjList) {
    delete graph->transposedAdjList;
  }
  graph->transposedAdjList = nullptr;
  
}

template <typename VT, typename WT>
void number_of_vertices(Graph<VT, WT> *graph) {
  if (graph->v != 0)
    

  //
  //  int32_t implementation for now, since that's all that
  //  is supported elsewhere.
  //
  CUGRAPH_EXPECTS( (graph->edgeList != nullptr), "Invalid API parameter");
  CUGRAPH_EXPECTS( typeid(graph->edgeList->src_indices) == typeid(int*), "Unsupported data type" );

  VT  h_max[2];
  VT *d_max;
  void    *d_temp_storage = nullptr;
  size_t   temp_storage_bytes = 0;
  
  ALLOC_TRY(&d_max, sizeof(VT), nullptr);
  
  //
  //  Compute size of temp storage
  //
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, graph->edgeList->src_indices, d_max, graph->e);

  //
  //  Compute max of src indices and copy to host
  //
  ALLOC_TRY(&d_temp_storage, temp_storage_bytes, nullptr);
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, graph->edgeList->src_indices, d_max, graph->e);

  CUDA_TRY(cudaMemcpy(h_max, d_max, sizeof(VT), cudaMemcpyDeviceToHost));

  //
  //  Compute max of dest indices and copy to host
  //
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, graph->edgeList->dest_indices, d_max, graph->e);
  CUDA_TRY(cudaMemcpy(h_max + 1, d_max, sizeof(VT), cudaMemcpyDeviceToHost));

  ALLOC_FREE_TRY(d_temp_storage, nullptr);
  ALLOC_FREE_TRY(d_max, nullptr);
  
  graph->v = 1 + std::max(h_max[0], h_max[1]);
  
}

// explicit instantiations

//FP 32
template void edge_list_view<int, float> (Graph<int, float> *graph, const size_t e, int *src_indices, int *dest_indices, float *edge_data);
template void adj_list_view <int, float> (Graph<int, float> *graph, const size_t v, const size_t e, int *offsets, int *indices, float *edge_data); 
template void transposed_adj_list_view <int, float> (Graph<int, float> *graph, const size_t v, const size_t e, int *offsets, int *indices, float *edge_data); 
template void add_adj_list<int, float> (Graph<int, float> *graph);
template void add_transposed_adj_list<int, float> (Graph<int, float> *graph);
template void add_edge_list<int, float> (Graph<int, float> *graph);
template void delete_adj_list<int, float> (Graph<int, float> *graph);
template void delete_edge_list<int, float> (Graph<int, float> *graph);
template void delete_transposed_adj_list<int, float> (Graph<int, float> *graph);
template void number_of_vertices<int, float> (Graph<int, float> *graph);
template class edge_list <int, float>;
template class adj_list <int, float>;
template class Graph <int, float>;

//FP 64
template void edge_list_view<int, double> (Graph<int, double> *graph, const size_t e, int *src_indices, int *dest_indices, double *edge_data);
template void adj_list_view <int, double> (Graph<int, double> *graph, const size_t v, const size_t e, int *offsets, int *indices, double *edge_data); 
template void transposed_adj_list_view <int, double> (Graph<int, double> *graph, const size_t v, const size_t e, int *offsets, int *indices, double *edge_data); 
template void add_adj_list<int, double> (Graph<int, double> *graph);
template void add_transposed_adj_list<int, double> (Graph<int, double> *graph);
template void add_edge_list<int, double> (Graph<int, double> *graph);
template void delete_adj_list<int, double> (Graph<int, double> *graph);
template void delete_edge_list<int, double> (Graph<int, double> *graph);
template void delete_transposed_adj_list<int, double> (Graph<int, double> *graph);
template void number_of_vertices<int, double> (Graph<int, double> *graph);
template class edge_list <int, double>;
template class adj_list <int, double>;
template class Graph <int, double>;

// template void renumber_vertices<int> (const int *src, const int *dst, int *src_renumbered, int *dst_renumbered, int *numbering_map);
// template void get_two_hop_neighbors<int> (Graph<int, float> *graph, int *first, int *second);
// template void degree<int, float> (Graph<int, float> *graph, int *degree, int x);
// template void get_two_hop_neighbors<int, double> (Graph<int, double> *graph, int *first, int *second);
// template void degree<int, double> (Graph<int, double> *graph, int *degree, int x);

} //namespace

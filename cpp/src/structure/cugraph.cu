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

#include <cugraph.h>
#include "utilities/graph_utils.cuh"
#include "converters/COOtoCSR.cuh"
#include "utilities/error_utils.h"
#include "converters/renumber.cuh"
#include <library_types.h>
#include <nvgraph/nvgraph.h>
#include <thrust/device_vector.h>
#include "utilities/cusparse_helper.h"
#include <rmm_utils.h>
#include <utilities/validation.cuh>

namespace cugraph {
int get_device(const void *ptr) {
    cudaPointerAttributes att;
    cudaPointerGetAttributes(&att, ptr);
    return att.device;
}

template <typename VT, typename WT>
void transposed_adj_list_view(Graph *graph, 
                            const size_t v,
                            const size_t e,
                            const VT *offsets,
                            const VT *indices,
                            const WT *edge_data) {
  //This function returns an error if this graph object has at least one graph
  //representation to prevent a single object storing two different graphs.
  CUGRAPH_EXPECTS( ((graph->edgeList == nullptr) && (graph->adjList == nullptr) &&
    (graph->transposedAdjList == nullptr)), "Invalid API parameter");
  CUGRAPH_EXPECTS( offsets->null_count == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( indices->null_count == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( (offsets->dtype == indices->dtype), "Unsupported data type" );
  CUGRAPH_EXPECTS( ((offsets->dtype == GDF_INT32)), "Unsupported data type" );
  CUGRAPH_EXPECTS( (offsets->size > 0), "Column is empty");

  graph->transposedAdjList = new adj_list;
  graph->transposedAdjList->ownership = 0;
  graph->v = v;
  graph->e = e;
  graph->transposedAdjList->offsets = offsets;
  graph->transposedAdjList->indices = indices;


  
  if (!graph->prop)
      graph->prop = new Graph_properties();

  if (edge_data) {
    CUGRAPH_EXPECTS(indices->size == edge_data->size, "Column size mismatch");
    graph->transposedAdjList->edge_data = edge_data;
    
    bool has_neg_val;
    
    switch (graph->adjList->edge_data->dtype) {
    case GDF_INT8:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<int8_t *>(graph->transposedAdjList->edge_data->data),
          graph->transposedAdjList->edge_data->size);
      break;
    case GDF_INT16:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<int16_t *>(graph->transposedAdjList->edge_data->data),
          graph->transposedAdjList->edge_data->size);
      break;
    case GDF_INT32:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<int32_t *>(graph->transposedAdjList->edge_data->data),
          graph->transposedAdjList->edge_data->size);
      break;
    case GDF_INT64:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<int64_t *>(graph->transposedAdjList->edge_data->data),
          graph->transposedAdjList->edge_data->size);
      break;
    case GDF_FLOAT32:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<float *>(graph->transposedAdjList->edge_data->data),
          graph->transposedAdjList->edge_data->size);
      break;
    case GDF_FLOAT64:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<double *>(graph->transposedAdjList->edge_data->data),
          graph->transposedAdjList->edge_data->size);
      break;
    default:
      has_neg_val = false;
    }
    graph->prop->has_negative_edges =
        (has_neg_val) ? PROP_TRUE : PROP_FALSE;
  } else {
    graph->adjList->edge_data = nullptr;
    graph->prop->has_negative_edges = PROP_FALSE;
  }
}

template <typename VT, typename WT>
void adj_list_view(Graph *graph, const size_t v,
                   const size_t e, const VT *offsets,
                   const VT *indices,
                   const WT *edge_data) {
  //This function returns an error if this graph object has at least one graph
  //representation to prevent a single object storing two different graphs.
  CUGRAPH_EXPECTS( ((graph->edgeList == nullptr) && (graph->adjList == nullptr) &&
    (graph->transposedAdjList == nullptr)), "Invalid API parameter");
  CUGRAPH_EXPECTS( offsets->null_count == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( indices->null_count == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( (offsets->dtype == indices->dtype), "Unsupported data type" );
  CUGRAPH_EXPECTS( ((offsets->dtype == GDF_INT32)), "Unsupported data type" );
  CUGRAPH_EXPECTS( (offsets->size > 0), "Column is empty");

  graph->adjList = new adj_list;
  graph->adjList->ownership = 0;
  graph->v = v;
  graph->e = e;
  graph->adjList->offsets = offsets;
  graph->adjList->indices = indices;
  
  if (!graph->prop)
      graph->prop = new Graph_properties();

  if (edge_data) {
    CUGRAPH_EXPECTS(indices->size == edge_data->size, "Column size mismatch");
    graph->adjList->edge_data = edge_data;
    
    bool has_neg_val;
    
    switch (graph->adjList->edge_data->dtype) {
    case GDF_INT8:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<int8_t *>(graph->adjList->edge_data->data),
          graph->adjList->edge_data->size);
      break;
    case GDF_INT16:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<int16_t *>(graph->adjList->edge_data->data),
          graph->adjList->edge_data->size);
      break;
    case GDF_INT32:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<int32_t *>(graph->adjList->edge_data->data),
          graph->adjList->edge_data->size);
      break;
    case GDF_INT64:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<int64_t *>(graph->adjList->edge_data->data),
          graph->adjList->edge_data->size);
      break;
    case GDF_FLOAT32:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<float *>(graph->adjList->edge_data->data),
          graph->adjList->edge_data->size);
      break;
    case GDF_FLOAT64:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<double *>(graph->adjList->edge_data->data),
          graph->adjList->edge_data->size);
      break;
    default:
      has_neg_val = false;
    }
    graph->prop->has_negative_edges =
        (has_neg_val) ? PROP_TRUE : PROP_FALSE;
  } else {
    graph->adjList->edge_data = nullptr;
    graph->prop->has_negative_edges = PROP_FALSE;
  }  
}

template <typename VT, typename WT>
void adj_list::get_vertex_identifiers(VT *identifiers) {
  CUGRAPH_EXPECTS( offsets != nullptr , "Invalid API parameter");
  CUGRAPH_EXPECTS( offsets->data != nullptr , "Invalid API parameter");
  cugraph::detail::sequence<int>((int)offsets->size-1, (int*)identifiers->data);

  
}

template <typename VT, typename WT>
void adj_list::get_source_indices (VT *src_indices) {
  CUGRAPH_EXPECTS( offsets != nullptr , "Invalid API parameter");
  CUGRAPH_EXPECTS( offsets->data != nullptr , "Invalid API parameter");
  CUGRAPH_EXPECTS( src_indices->size == indices->size, "Column size mismatch" );
  CUGRAPH_EXPECTS( src_indices->dtype == indices->dtype, "Unsupported data type" );
  CUGRAPH_EXPECTS( src_indices->size > 0, "Column is empty");
  
  cugraph::detail::offsets_to_indices<int>((int*)offsets->data, offsets->size-1, (int*)src_indices->data);

  
}

template <typename VT, typename WT>
void edge_list_view(Graph *graph, const VT *src_indices,
                             const gdf_column *dest_indices, 
                             const WT *edge_data) {
  //This function returns an error if this graph object has at least one graph
  //representation to prevent a single object storing two different graphs.

  CUGRAPH_EXPECTS( ((graph->edgeList == nullptr) && (graph->adjList == nullptr) &&
    (graph->transposedAdjList == nullptr)), "Invalid API parameter");
  CUGRAPH_EXPECTS( src_indices->size == dest_indices->size, "Column size mismatch" );
  CUGRAPH_EXPECTS( src_indices->dtype == dest_indices->dtype, "Unsupported data type" );
  CUGRAPH_EXPECTS( src_indices->dtype == GDF_INT32, "Unsupported data type" );
  CUGRAPH_EXPECTS( src_indices->size > 0, "Column is empty");
  CUGRAPH_EXPECTS( src_indices->null_count == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( dest_indices->null_count == 0 , "Input column has non-zero null count");


  graph->edgeList = new edge_list;
  graph->edgeList->src_indices = new gdf_column;
  graph->edgeList->dest_indices = new gdf_column;
  graph->edgeList->ownership = 0;

  cpy_column_view(src_indices, graph->edgeList->src_indices);
  cpy_column_view(dest_indices, graph->edgeList->dest_indices);

  if (!graph->prop)
    graph->prop = new Graph_properties();

  if (edge_data) {
    CUGRAPH_EXPECTS(src_indices->size == edge_data->size, "Column size mismatch");
    graph->edgeList->edge_data = new gdf_column;
    cpy_column_view(edge_data, graph->edgeList->edge_data);

    bool has_neg_val;

    switch (graph->edgeList->edge_data->dtype) {
    case GDF_INT8:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<int8_t *>(graph->edgeList->edge_data->data),
          graph->edgeList->edge_data->size);
      break;
    case GDF_INT16:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<int16_t *>(graph->edgeList->edge_data->data),
          graph->edgeList->edge_data->size);
      break;
    case GDF_INT32:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<int32_t *>(graph->edgeList->edge_data->data),
          graph->edgeList->edge_data->size);
      break;
    case GDF_INT64:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<int64_t *>(graph->edgeList->edge_data->data),
          graph->edgeList->edge_data->size);
      break;
    case GDF_FLOAT32:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<float *>(graph->edgeList->edge_data->data),
          graph->edgeList->edge_data->size);
      break;
    case GDF_FLOAT64:
      has_neg_val = cugraph::detail::has_negative_val(
          static_cast<double *>(graph->edgeList->edge_data->data),
          graph->edgeList->edge_data->size);
      break;
    default:
      has_neg_val = false;
    }
    graph->prop->has_negative_edges =
        (has_neg_val) ? PROP_TRUE : PROP_FALSE;

  } else {
    graph->edgeList->edge_data = nullptr;
    graph->prop->has_negative_edges = PROP_FALSE;
  }

  cugraph::detail::indexing_check<int> (
                                static_cast<int*>(graph->edgeList->src_indices->data), 
                                static_cast<int*>(graph->edgeList->dest_indices->data), 
                                graph->edgeList->dest_indices->size);
}

template <typename T, typename WT>
void add_adj_list_impl (Graph *graph) {
    if (graph->adjList == nullptr) {
      CUGRAPH_EXPECTS( graph->edgeList != nullptr , "Invalid API parameter");
      int nnz = graph->edgeList->src_indices->size;
      graph->adjList = new adj_list;
      graph->adjList->offsets = new gdf_column;
      graph->adjList->indices = new gdf_column;
      graph->adjList->ownership = 1;

    if (graph->edgeList->edge_data!= nullptr) {
      graph->adjList->edge_data = new gdf_column;

      CSR_Result_Weighted<int,WT> adj_list;
      ConvertCOOtoCSR_weighted((int*)graph->edgeList->src_indices->data, (int*)graph->edgeList->dest_indices->data, (WT*)graph->edgeList->edge_data->data, nnz, adj_list);

      gdf_column_view(graph->adjList->offsets, adj_list.rowOffsets,
                            nullptr, adj_list.size+1, graph->edgeList->src_indices->dtype);
      gdf_column_view(graph->adjList->indices, adj_list.colIndices,
                            nullptr, adj_list.nnz, graph->edgeList->src_indices->dtype);
      gdf_column_view(graph->adjList->edge_data, adj_list.edgeWeights,
                          nullptr, adj_list.nnz, graph->edgeList->edge_data->dtype);
    }
    else {
      CSR_Result<int> adj_list;
      ConvertCOOtoCSR((int*)graph->edgeList->src_indices->data,(int*)graph->edgeList->dest_indices->data, nnz, adj_list);
      gdf_column_view(graph->adjList->offsets, adj_list.rowOffsets,
                            nullptr, adj_list.size+1, graph->edgeList->src_indices->dtype);
      gdf_column_view(graph->adjList->indices, adj_list.colIndices,
                            nullptr, adj_list.nnz, graph->edgeList->src_indices->dtype);
    }
  }
}

template <typename VT, typename WT>
void add_edge_list (Graph *graph) {
    if (graph->edgeList == nullptr) {
      CUGRAPH_EXPECTS( graph->adjList != nullptr , "Invalid API parameter");
      int *d_src;
      graph->edgeList = new edge_list;
      graph->edgeList->src_indices = new gdf_column;
      graph->edgeList->dest_indices = new gdf_column;
      graph->edgeList->ownership = 2;

      cudaStream_t stream{nullptr};
      ALLOC_TRY((void**)&d_src, sizeof(int) * graph->adjList->indices->size, stream);

      cugraph::detail::offsets_to_indices<int>((int*)graph->adjList->offsets->data,
                                  graph->adjList->offsets->size-1,
                                  (int*)d_src);

      gdf_column_view(graph->edgeList->src_indices, d_src,
                      nullptr, graph->adjList->indices->size, graph->adjList->indices->dtype);
      cpy_column_view(graph->adjList->indices, graph->edgeList->dest_indices);

      if (graph->adjList->edge_data != nullptr) {
        graph->edgeList->edge_data = new gdf_column;
        cpy_column_view(graph->adjList->edge_data, graph->edgeList->edge_data);
      }
  }
  
}


template <typename VT, typename WT>
void add_transposed_adj_list_impl (Graph *graph) {
    if (graph->transposedAdjList == nullptr ) {
      CUGRAPH_EXPECTS( graph->edgeList != nullptr , "Invalid API parameter");
      int nnz = graph->edgeList->src_indices->size;
      graph->transposedAdjList = new adj_list;
      graph->transposedAdjList->offsets = new gdf_column;
      graph->transposedAdjList->indices = new gdf_column;
      graph->transposedAdjList->ownership = 1;

      if (graph->edgeList->edge_data) {
        graph->transposedAdjList->edge_data = new gdf_column;
        CSR_Result_Weighted<int32_t,WT> adj_list;
        ConvertCOOtoCSR_weighted( (int*)graph->edgeList->dest_indices->data, (int*)graph->edgeList->src_indices->data, (WT*)graph->edgeList->edge_data->data, nnz, adj_list);
        gdf_column_view(graph->transposedAdjList->offsets, adj_list.rowOffsets,
                              nullptr, adj_list.size+1, graph->edgeList->src_indices->dtype);
        gdf_column_view(graph->transposedAdjList->indices, adj_list.colIndices,
                              nullptr, adj_list.nnz, graph->edgeList->src_indices->dtype);
        gdf_column_view(graph->transposedAdjList->edge_data, adj_list.edgeWeights,
                            nullptr, adj_list.nnz, graph->edgeList->edge_data->dtype);
      }
      else {

        CSR_Result<int> adj_list;
        ConvertCOOtoCSR((int*)graph->edgeList->dest_indices->data, (int*)graph->edgeList->src_indices->data, nnz, adj_list);
        gdf_column_view(graph->transposedAdjList->offsets, adj_list.rowOffsets,
                              nullptr, adj_list.size+1, graph->edgeList->src_indices->dtype);
        gdf_column_view(graph->transposedAdjList->indices, adj_list.colIndices,
                              nullptr, adj_list.nnz, graph->edgeList->src_indices->dtype);
      }
    }
}

template <typename VT, typename WT>
void add_adj_list(Graph *graph) {
  if (graph->adjList == nullptr) {
    CUGRAPH_EXPECTS( graph->edgeList != nullptr , "Invalid API parameter");
    CUGRAPH_EXPECTS( graph->edgeList->src_indices->dtype == GDF_INT32, "Unsupported data type" );

    if (graph->edgeList->edge_data != nullptr) {
      switch (graph->edgeList->edge_data->dtype) {
        case GDF_FLOAT32:   return cugraph::add_adj_list_impl<int32_t, float>(graph);
        case GDF_FLOAT64:   return cugraph::add_adj_list_impl<int32_t, double>(graph);
        default: CUGRAPH_FAIL("Unsupported data type");
      }
    }
    else {
      return cugraph::add_adj_list_impl<int32_t, float>(graph);
    }
  }
}

template <typename VT, typename WT>
void add_transposed_adj_list(Graph *graph) {
  if (graph->transposedAdjList == nullptr) {
    if (graph->edgeList == nullptr)
      cugraph::add_edge_list(graph);

    CUGRAPH_EXPECTS(graph->edgeList->src_indices->dtype == GDF_INT32, "Unsupported data type");
    CUGRAPH_EXPECTS(graph->edgeList->dest_indices->dtype == GDF_INT32, "Unsupported data type");

    if (graph->edgeList->edge_data != nullptr) {
      switch (graph->edgeList->edge_data->dtype) {
        case GDF_FLOAT32:   return cugraph::add_transposed_adj_list_impl<float>(graph);
        case GDF_FLOAT64:   return cugraph::add_transposed_adj_list_impl<double>(graph);
        default: CUGRAPH_FAIL("Unsupported data type");
      }
    }
    else {
      return cugraph::add_transposed_adj_list_impl<float>(graph);
    }
  }
}

void delete_adj_list(Graph *graph) {
  if (graph->adjList) {
    delete graph->adjList;
  }
  graph->adjList = nullptr;
  
}

void delete_edge_list(Graph *graph) {
  if (graph->edgeList) {
    delete graph->edgeList;
  }
  graph->edgeList = nullptr;
  
}

void delete_transposed_adj_list(Graph *graph) {
  if (graph->transposedAdjList) {
    delete graph->transposedAdjList;
  }
  graph->transposedAdjList = nullptr;
  
}

template <typename VT, typename WT>
void number_of_vertices(Graph *graph) {
  if (graph->v != 0)
    

  //
  //  int32_t implementation for now, since that's all that
  //  is supported elsewhere.
  //
  CUGRAPH_EXPECTS( (graph->edgeList != nullptr), "Invalid API parameter");
  CUGRAPH_EXPECTS( (graph->edgeList->src_indices->dtype == GDF_INT32), "Unsupported data type" );

  int32_t  h_max[2];
  int32_t *d_max;
  void    *d_temp_storage = nullptr;
  size_t   temp_storage_bytes = 0;
  
  ALLOC_TRY(&d_max, sizeof(int32_t), nullptr);
  
  //
  //  Compute size of temp storage
  //
  int32_t *tmp = static_cast<int32_t *>(graph->edgeList->src_indices->data);

  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tmp, d_max, graph->edgeList->src_indices->size);

  //
  //  Compute max of src indices and copy to host
  //
  ALLOC_TRY(&d_temp_storage, temp_storage_bytes, nullptr);
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tmp, d_max, graph->edgeList->src_indices->size);

  CUDA_TRY(cudaMemcpy(h_max, d_max, sizeof(int32_t), cudaMemcpyDeviceToHost));

  //
  //  Compute max of dest indices and copy to host
  //
  tmp = static_cast<int32_t *>(graph->edgeList->dest_indices->data);
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tmp, d_max, graph->edgeList->src_indices->size);
  CUDA_TRY(cudaMemcpy(h_max + 1, d_max, sizeof(int32_t), cudaMemcpyDeviceToHost));

  ALLOC_FREE_TRY(d_temp_storage, nullptr);
  ALLOC_FREE_TRY(d_max, nullptr);
  
  graph->v = 1 + std::max(h_max[0], h_max[1]);
  
}

} //namespace

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
/*
 * cudf has gdf_column_free and using this is, in general, better design than
 * creating our own, but we will keep this as cudf is planning to remove the
 * function. cudf plans to redesign cudf::column to fundamentally solve this
 * problem, so once they finished the redesign, we need to update this code to
 * use their new features. Until that time, we may rely on this as a temporary
 * solution.
 */

namespace cugraph {
int get_device(const void *ptr) {
    cudaPointerAttributes att;
    cudaPointerGetAttributes(&att, ptr);
    return att.device;
}

void gdf_col_delete(gdf_column* col) {
  if (col != nullptr) {
    cudaStream_t stream {nullptr};
    if (col->data != nullptr) {
      ALLOC_FREE_TRY(col->data, stream);
    }
    if (col->valid != nullptr) {
      ALLOC_FREE_TRY(col->valid, stream);
    }
#if 0
    /* Currently, gdf_column_view does not set col_name, and col_name can have
        an arbitrary value, so freeing col_name can lead to freeing a ranodom
        address. This problem should be cleaned up once cudf finishes
        redesigning cudf::column. */
    if (col->col_name != nullptr) {
      free(col->col_name);
    }
#endif
    delete col;
  }
}

void gdf_col_release(gdf_column* col) {
  delete col;
}

void cpy_column_view(const gdf_column *in, gdf_column *out) {
  if (in != nullptr && out !=nullptr) {
    gdf_column_view(out, in->data, in->valid, in->size, in->dtype);
  }
}

void transposed_adj_list_view(Graph *graph, const gdf_column *offsets,
                            const gdf_column *indices,
                            const gdf_column *edge_data) {
  //This function returns an error if this graph object has at least one graph
  //representation to prevent a single object storing two different graphs.
  CUGRAPH_EXPECTS( ((graph->edgeList == nullptr) && (graph->adjList == nullptr) &&
    (graph->transposedAdjList == nullptr)), "Invalid API parameter");
  CUGRAPH_EXPECTS( offsets->null_count == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( indices->null_count == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( (offsets->dtype == indices->dtype), "Unsupported data type" );
  CUGRAPH_EXPECTS( ((offsets->dtype == GDF_INT32)), "Unsupported data type" );
  CUGRAPH_EXPECTS( (offsets->size > 0), "Column is empty");

  graph->transposedAdjList = new gdf_adj_list;
  graph->transposedAdjList->offsets = new gdf_column;
  graph->transposedAdjList->indices = new gdf_column;
  graph->transposedAdjList->ownership = 0;

  cpy_column_view(offsets, graph->transposedAdjList->offsets);
  cpy_column_view(indices, graph->transposedAdjList->indices);
  
  if (!graph->prop)
      graph->prop = new Graph_properties();

  if (edge_data) {
    CUGRAPH_EXPECTS(indices->size == edge_data->size, "Column size mismatch");
    graph->transposedAdjList->edge_data = new gdf_column;
    cpy_column_view(edge_data, graph->transposedAdjList->edge_data);
    
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
        (has_neg_val) ? GDF_PROP_TRUE : GDF_PROP_FALSE;
  } else {
    graph->adjList->edge_data = nullptr;
    graph->prop->has_negative_edges = GDF_PROP_FALSE;
  }

  graph->numberOfVertices = graph->transposedAdjList->offsets->size - 1;
}

void adj_list_view(Graph *graph, const gdf_column *offsets,
                            const gdf_column *indices,
                            const gdf_column *edge_data) {
  //This function returns an error if this graph object has at least one graph
  //representation to prevent a single object storing two different graphs.
  CUGRAPH_EXPECTS( ((graph->edgeList == nullptr) && (graph->adjList == nullptr) &&
    (graph->transposedAdjList == nullptr)), "Invalid API parameter");
  CUGRAPH_EXPECTS( offsets->null_count == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( indices->null_count == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( (offsets->dtype == indices->dtype), "Unsupported data type" );
  CUGRAPH_EXPECTS( ((offsets->dtype == GDF_INT32)), "Unsupported data type" );
  CUGRAPH_EXPECTS( (offsets->size > 0), "Column is empty");

  graph->adjList = new gdf_adj_list;
  graph->adjList->offsets = new gdf_column;
  graph->adjList->indices = new gdf_column;
  graph->adjList->ownership = 0;

  cpy_column_view(offsets, graph->adjList->offsets);
  cpy_column_view(indices, graph->adjList->indices);
  
  if (!graph->prop)
      graph->prop = new Graph_properties();

  if (edge_data) {
    CUGRAPH_EXPECTS(indices->size == edge_data->size, "Column size mismatch");
    graph->adjList->edge_data = new gdf_column;
    cpy_column_view(edge_data, graph->adjList->edge_data);
    
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
        (has_neg_val) ? GDF_PROP_TRUE : GDF_PROP_FALSE;
  } else {
    graph->adjList->edge_data = nullptr;
    graph->prop->has_negative_edges = GDF_PROP_FALSE;
  }

  graph->numberOfVertices = graph->adjList->offsets->size - 1;
  
}

void gdf_adj_list::get_vertex_identifiers(gdf_column *identifiers) {
  CUGRAPH_EXPECTS( offsets != nullptr , "Invalid API parameter");
  CUGRAPH_EXPECTS( offsets->data != nullptr , "Invalid API parameter");
  cugraph::detail::sequence<int>((int)offsets->size-1, (int*)identifiers->data);

  
}

void gdf_adj_list::get_source_indices (gdf_column *src_indices) {
  CUGRAPH_EXPECTS( offsets != nullptr , "Invalid API parameter");
  CUGRAPH_EXPECTS( offsets->data != nullptr , "Invalid API parameter");
  CUGRAPH_EXPECTS( src_indices->size == indices->size, "Column size mismatch" );
  CUGRAPH_EXPECTS( src_indices->dtype == indices->dtype, "Unsupported data type" );
  CUGRAPH_EXPECTS( src_indices->size > 0, "Column is empty");
  
  cugraph::detail::offsets_to_indices<int>((int*)offsets->data, offsets->size-1, (int*)src_indices->data);

  
}

void edge_list_view(Graph *graph, const gdf_column *src_indices,
                             const gdf_column *dest_indices, 
                             const gdf_column *edge_data) {
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


  graph->edgeList = new gdf_edge_list;
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
        (has_neg_val) ? GDF_PROP_TRUE : GDF_PROP_FALSE;

  } else {
    graph->edgeList->edge_data = nullptr;
    graph->prop->has_negative_edges = GDF_PROP_FALSE;
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
      graph->adjList = new gdf_adj_list;
      graph->adjList->offsets = new gdf_column;
      graph->adjList->indices = new gdf_column;
      graph->adjList->ownership = 1;

    if (graph->edgeList->edge_data!= nullptr) {
      graph->adjList->edge_data = new gdf_column;

      CSR_Result_Weighted<int32_t,WT> adj_list;
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
    graph->numberOfVertices = graph->adjList->offsets->size - 1;
  }
}

void add_edge_list (Graph *graph) {
    if (graph->edgeList == nullptr) {
      CUGRAPH_EXPECTS( graph->adjList != nullptr , "Invalid API parameter");
      int *d_src;
      graph->edgeList = new gdf_edge_list;
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


template <typename WT>
void add_transposed_adj_list_impl (Graph *graph) {
    if (graph->transposedAdjList == nullptr ) {
      CUGRAPH_EXPECTS( graph->edgeList != nullptr , "Invalid API parameter");
      int nnz = graph->edgeList->src_indices->size;
      graph->transposedAdjList = new gdf_adj_list;
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
      graph->numberOfVertices = graph->transposedAdjList->offsets->size - 1;
    }
    
}

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

void number_of_vertices(Graph *graph) {
  if (graph->numberOfVertices != 0)
    

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
  
  graph->numberOfVertices = 1 + std::max(h_max[0], h_max[1]);
  
}

} //namespace

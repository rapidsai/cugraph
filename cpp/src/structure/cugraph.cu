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

gdf_error gdf_adj_list_view(gdf_graph *graph, const gdf_column *offsets,
                            const gdf_column *indices,
                            const gdf_column *edge_data) {
  //This function returns an error if this graph object has at least one graph
  //representation to prevent a single object storing two different graphs.
  GDF_REQUIRE( ((graph->edgeList == nullptr) && (graph->adjList == nullptr) &&
    (graph->transposedAdjList == nullptr)), GDF_INVALID_API_CALL);
  /*GDF_REQUIRE( offsets->null_count == 0 , GDF_VALIDITY_UNSUPPORTED );
  GDF_REQUIRE( indices->null_count == 0 , GDF_VALIDITY_UNSUPPORTED );
  GDF_REQUIRE( (offsets->dtype == indices->dtype), GDF_UNSUPPORTED_DTYPE );
  GDF_REQUIRE( ((offsets->dtype == GDF_INT32)), GDF_UNSUPPORTED_DTYPE );
  GDF_REQUIRE( (offsets->size > 0), GDF_DATASET_EMPTY );
  */

  graph->adjList = new gdf_adj_list;
  graph->adjList->offsets = new gdf_column;
  graph->adjList->indices = new gdf_column;

  cpy_column_view(offsets, graph->adjList->offsets);
  cpy_column_view(indices, graph->adjList->indices);
  
  if (!graph->prop)
      graph->prop = new gdf_graph_properties();

  if (edge_data) {
    GDF_REQUIRE(indices->size == edge_data->size, GDF_COLUMN_SIZE_MISMATCH);
    graph->adjList->edge_data = new gdf_column;
    cpy_column_view(edge_data, graph->adjList->edge_data);
    
    bool has_neg_val;
    
    switch (graph->adjList->edge_data->dtype) {
    case GDF_INT8:
      has_neg_val = cugraph::has_negative_val(
          static_cast<int8_t *>(graph->adjList->edge_data->data),
          graph->adjList->edge_data->size);
      break;
    case GDF_INT16:
      has_neg_val = cugraph::has_negative_val(
          static_cast<int16_t *>(graph->adjList->edge_data->data),
          graph->adjList->edge_data->size);
      break;
    case GDF_INT32:
      has_neg_val = cugraph::has_negative_val(
          static_cast<int32_t *>(graph->adjList->edge_data->data),
          graph->adjList->edge_data->size);
      break;
    case GDF_INT64:
      has_neg_val = cugraph::has_negative_val(
          static_cast<int64_t *>(graph->adjList->edge_data->data),
          graph->adjList->edge_data->size);
      break;
    case GDF_FLOAT32:
      has_neg_val = cugraph::has_negative_val(
          static_cast<float *>(graph->adjList->edge_data->data),
          graph->adjList->edge_data->size);
      break;
    case GDF_FLOAT64:
      has_neg_val = cugraph::has_negative_val(
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
  return GDF_SUCCESS;
}

gdf_error gdf_adj_list::get_vertex_identifiers(gdf_column *identifiers) {
  GDF_REQUIRE( offsets != nullptr , GDF_INVALID_API_CALL);
  GDF_REQUIRE( offsets->data != nullptr , GDF_INVALID_API_CALL);
  cugraph::sequence<int>((int)offsets->size-1, (int*)identifiers->data);
  return GDF_SUCCESS;
}

gdf_error gdf_adj_list::get_source_indices (gdf_column *src_indices) {
  GDF_REQUIRE( offsets != nullptr , GDF_INVALID_API_CALL);
  GDF_REQUIRE( offsets->data != nullptr , GDF_INVALID_API_CALL);
  GDF_REQUIRE( src_indices->size == indices->size, GDF_COLUMN_SIZE_MISMATCH );
  GDF_REQUIRE( src_indices->dtype == indices->dtype, GDF_UNSUPPORTED_DTYPE );
  GDF_REQUIRE( src_indices->size > 0, GDF_DATASET_EMPTY );
  cugraph::offsets_to_indices<int>((int*)offsets->data, offsets->size-1, (int*)src_indices->data);

  return GDF_SUCCESS;
}

gdf_error gdf_edge_list_view(gdf_graph *graph, const gdf_column *src_indices,
                             const gdf_column *dest_indices, 
                             const gdf_column *edge_data) {
  //This function returns an error if this graph object has at least one graph
  //representation to prevent a single object storing two different graphs.
  GDF_REQUIRE( ((graph->edgeList == nullptr) && (graph->adjList == nullptr) &&
    (graph->transposedAdjList == nullptr)), GDF_INVALID_API_CALL);
  /*GDF_REQUIRE( src_indices->size == dest_indices->size, GDF_COLUMN_SIZE_MISMATCH );
  GDF_REQUIRE( src_indices->dtype == dest_indices->dtype, GDF_UNSUPPORTED_DTYPE );
  GDF_REQUIRE( ((src_indices->dtype == GDF_INT32)), GDF_UNSUPPORTED_DTYPE );
  GDF_REQUIRE( src_indices->size > 0, GDF_DATASET_EMPTY );
  GDF_REQUIRE( src_indices->null_count == 0 , GDF_VALIDITY_UNSUPPORTED );
  GDF_REQUIRE( dest_indices->null_count == 0 , GDF_VALIDITY_UNSUPPORTED );*/

  graph->edgeList = new gdf_edge_list;
  graph->edgeList->src_indices = new gdf_column;
  graph->edgeList->dest_indices = new gdf_column;

  cpy_column_view(src_indices, graph->edgeList->src_indices);
  cpy_column_view(dest_indices, graph->edgeList->dest_indices);

  if (!graph->prop)
    graph->prop = new gdf_graph_properties();

  if (edge_data) {
    GDF_REQUIRE(src_indices->size == edge_data->size, GDF_COLUMN_SIZE_MISMATCH);
    graph->edgeList->edge_data = new gdf_column;
    cpy_column_view(edge_data, graph->edgeList->edge_data);

    bool has_neg_val;

    switch (graph->edgeList->edge_data->dtype) {
    case GDF_INT8:
      has_neg_val = cugraph::has_negative_val(
          static_cast<int8_t *>(graph->edgeList->edge_data->data),
          graph->edgeList->edge_data->size);
      break;
    case GDF_INT16:
      has_neg_val = cugraph::has_negative_val(
          static_cast<int16_t *>(graph->edgeList->edge_data->data),
          graph->edgeList->edge_data->size);
      break;
    case GDF_INT32:
      has_neg_val = cugraph::has_negative_val(
          static_cast<int32_t *>(graph->edgeList->edge_data->data),
          graph->edgeList->edge_data->size);
      break;
    case GDF_INT64:
      has_neg_val = cugraph::has_negative_val(
          static_cast<int64_t *>(graph->edgeList->edge_data->data),
          graph->edgeList->edge_data->size);
      break;
    case GDF_FLOAT32:
      has_neg_val = cugraph::has_negative_val(
          static_cast<float *>(graph->edgeList->edge_data->data),
          graph->edgeList->edge_data->size);
      break;
    case GDF_FLOAT64:
      has_neg_val = cugraph::has_negative_val(
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

  gdf_error status;
  status = cugraph::indexing_check<int> (
                                static_cast<int*>(graph->edgeList->src_indices->data), 
                                static_cast<int*>(graph->edgeList->dest_indices->data), 
                                graph->edgeList->dest_indices->size);

  return status;
}

template <typename T, typename WT>
gdf_error gdf_add_adj_list_impl (gdf_graph *graph) {
    if (graph->adjList == nullptr) {
      GDF_REQUIRE( graph->edgeList != nullptr , GDF_INVALID_API_CALL);
      int nnz = graph->edgeList->src_indices->size, status = 0;
      graph->adjList = new gdf_adj_list;
      graph->adjList->offsets = new gdf_column;
      graph->adjList->indices = new gdf_column;
      graph->adjList->ownership = 1;

    if (graph->edgeList->edge_data!= nullptr) {
      graph->adjList->edge_data = new gdf_column;

      CSR_Result_Weighted<int32_t,WT> adj_list;
      status = ConvertCOOtoCSR_weighted((int*)graph->edgeList->src_indices->data, (int*)graph->edgeList->dest_indices->data, (WT*)graph->edgeList->edge_data->data, nnz, adj_list);

      gdf_column_view(graph->adjList->offsets, adj_list.rowOffsets,
                            nullptr, adj_list.size+1, graph->edgeList->src_indices->dtype);
      gdf_column_view(graph->adjList->indices, adj_list.colIndices,
                            nullptr, adj_list.nnz, graph->edgeList->src_indices->dtype);
      gdf_column_view(graph->adjList->edge_data, adj_list.edgeWeights,
                          nullptr, adj_list.nnz, graph->edgeList->edge_data->dtype);
    }
    else {
      CSR_Result<int> adj_list;
      status = ConvertCOOtoCSR((int*)graph->edgeList->src_indices->data,(int*)graph->edgeList->dest_indices->data, nnz, adj_list);
      gdf_column_view(graph->adjList->offsets, adj_list.rowOffsets,
                            nullptr, adj_list.size+1, graph->edgeList->src_indices->dtype);
      gdf_column_view(graph->adjList->indices, adj_list.colIndices,
                            nullptr, adj_list.nnz, graph->edgeList->src_indices->dtype);
    }
    if (status !=0) {
      std::cerr << "Could not generate the adj_list" << std::endl;
      return GDF_CUDA_ERROR;
    }

    graph->numberOfVertices = graph->adjList->offsets->size - 1;
  }
  return GDF_SUCCESS;
}

gdf_error gdf_add_edge_list (gdf_graph *graph) {
    if (graph->edgeList == nullptr) {
      GDF_REQUIRE( graph->adjList != nullptr , GDF_INVALID_API_CALL);
      int *d_src;
      graph->edgeList = new gdf_edge_list;
      graph->edgeList->src_indices = new gdf_column;
      graph->edgeList->dest_indices = new gdf_column;
      graph->edgeList->ownership = 2;

      cudaStream_t stream{nullptr};
      ALLOC_TRY((void**)&d_src, sizeof(int) * graph->adjList->indices->size, stream);

      cugraph::offsets_to_indices<int>((int*)graph->adjList->offsets->data,
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
  return GDF_SUCCESS;
}


template <typename WT>
gdf_error gdf_add_transposed_adj_list_impl (gdf_graph *graph) {
    if (graph->transposedAdjList == nullptr ) {
      GDF_REQUIRE( graph->edgeList != nullptr , GDF_INVALID_API_CALL);
      int nnz = graph->edgeList->src_indices->size, status = 0;
      graph->transposedAdjList = new gdf_adj_list;
      graph->transposedAdjList->offsets = new gdf_column;
      graph->transposedAdjList->indices = new gdf_column;
      graph->transposedAdjList->ownership = 1;

      if (graph->edgeList->edge_data) {
        graph->transposedAdjList->edge_data = new gdf_column;
        CSR_Result_Weighted<int32_t,WT> adj_list;
        status = ConvertCOOtoCSR_weighted( (int*)graph->edgeList->dest_indices->data, (int*)graph->edgeList->src_indices->data, (WT*)graph->edgeList->edge_data->data, nnz, adj_list);
        gdf_column_view(graph->transposedAdjList->offsets, adj_list.rowOffsets,
                              nullptr, adj_list.size+1, graph->edgeList->src_indices->dtype);
        gdf_column_view(graph->transposedAdjList->indices, adj_list.colIndices,
                              nullptr, adj_list.nnz, graph->edgeList->src_indices->dtype);
        gdf_column_view(graph->transposedAdjList->edge_data, adj_list.edgeWeights,
                            nullptr, adj_list.nnz, graph->edgeList->edge_data->dtype);
      }
      else {

        CSR_Result<int> adj_list;
        status = ConvertCOOtoCSR((int*)graph->edgeList->dest_indices->data, (int*)graph->edgeList->src_indices->data, nnz, adj_list);
        gdf_column_view(graph->transposedAdjList->offsets, adj_list.rowOffsets,
                              nullptr, adj_list.size+1, graph->edgeList->src_indices->dtype);
        gdf_column_view(graph->transposedAdjList->indices, adj_list.colIndices,
                              nullptr, adj_list.nnz, graph->edgeList->src_indices->dtype);
      }
      if (status !=0) {
        std::cerr << "Could not generate the adj_list" << std::endl;
        return GDF_CUDA_ERROR;
      }

      graph->numberOfVertices = graph->transposedAdjList->offsets->size - 1;
    }
    return GDF_SUCCESS;
}

gdf_error gdf_add_adj_list(gdf_graph *graph) {
  if (graph->adjList != nullptr)
    return GDF_SUCCESS;

  GDF_REQUIRE( graph->edgeList != nullptr , GDF_INVALID_API_CALL);
  GDF_REQUIRE( graph->edgeList->src_indices->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE );

  if (graph->edgeList->edge_data != nullptr) {
    switch (graph->edgeList->edge_data->dtype) {
      case GDF_FLOAT32:   return gdf_add_adj_list_impl<int32_t, float>(graph);
      case GDF_FLOAT64:   return gdf_add_adj_list_impl<int32_t, double>(graph);
      default: return GDF_UNSUPPORTED_DTYPE;
    }
  }
  else {
    return gdf_add_adj_list_impl<int32_t, float>(graph);
  }
}

gdf_error gdf_add_transposed_adj_list(gdf_graph *graph) {
  if (graph->edgeList == nullptr)
    gdf_add_edge_list(graph);

  GDF_REQUIRE(graph->edgeList->src_indices->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
  GDF_REQUIRE(graph->edgeList->dest_indices->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);

  if (graph->edgeList->edge_data != nullptr) {
    switch (graph->edgeList->edge_data->dtype) {
      case GDF_FLOAT32:   return gdf_add_transposed_adj_list_impl<float>(graph);
      case GDF_FLOAT64:   return gdf_add_transposed_adj_list_impl<double>(graph);
      default: return GDF_UNSUPPORTED_DTYPE;
    }
  }
  else {
    return gdf_add_transposed_adj_list_impl<float>(graph);
  }
}

gdf_error gdf_delete_adj_list(gdf_graph *graph) {
  if (graph->adjList) {
    delete graph->adjList;
  }
  graph->adjList = nullptr;
  return GDF_SUCCESS;
}

gdf_error gdf_delete_edge_list(gdf_graph *graph) {
  if (graph->edgeList) {
    delete graph->edgeList;
  }
  graph->edgeList = nullptr;
  return GDF_SUCCESS;
}

gdf_error gdf_delete_transposed_adj_list(gdf_graph *graph) {
  if (graph->transposedAdjList) {
    delete graph->transposedAdjList;
  }
  graph->transposedAdjList = nullptr;
  return GDF_SUCCESS;
}

gdf_error gdf_number_of_vertices(gdf_graph *graph) {
  if (graph->numberOfVertices != 0)
    return GDF_SUCCESS;

  //
  //  int32_t implementation for now, since that's all that
  //  is supported elsewhere.
  //
  GDF_REQUIRE( (graph->edgeList != nullptr), GDF_INVALID_API_CALL);
  GDF_REQUIRE( (graph->edgeList->src_indices->dtype == GDF_INT32), GDF_UNSUPPORTED_DTYPE );

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
  return GDF_SUCCESS;
}

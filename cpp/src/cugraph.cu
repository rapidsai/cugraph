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
// Author: Alex Fender afender@nvidia.com

#include <cugraph.h>
#include "graph_utils.cuh"
#include "pagerank.cuh"
#include "COOtoCSR.cuh"
#include "utilities/error_utils.h"
#include "bfs.cuh"

#include <library_types.h>
#include <nvgraph/nvgraph.h>

#include <rmm_utils.h>

void gdf_col_delete(gdf_column* col) {
  if (col) {
    col->size = 0; 
    if(col->data) {
      ALLOC_FREE_TRY(col->data, nullptr);
    }
#if 1
// If delete col is executed, the memory pointed by col is no longer valid and
// can be used in another memory allocation, so executing col->data = nullptr
// after delete col is dangerous, also, col = nullptr has no effect here (the
// address is passed by value, for col = nullptr should work, the input
// parameter should be gdf_column*& col (or alternatively, gdf_column** col and
// *col = nullptr also work)
    col->data = nullptr;
    delete col;
#else
    delete col;
    col->data = nullptr;
    col = nullptr;
#endif
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
                                 const gdf_column *indices, const gdf_column *edge_data) {
  GDF_REQUIRE( offsets->null_count == 0 , GDF_VALIDITY_UNSUPPORTED );                    
  GDF_REQUIRE( indices->null_count == 0 , GDF_VALIDITY_UNSUPPORTED );
  GDF_REQUIRE( (offsets->dtype == indices->dtype), GDF_UNSUPPORTED_DTYPE );
  GDF_REQUIRE( ((offsets->dtype == GDF_INT32) || (offsets->dtype == GDF_INT64)), GDF_UNSUPPORTED_DTYPE );
  GDF_REQUIRE( (offsets->size > 0), GDF_DATASET_EMPTY ); 
  GDF_REQUIRE( (graph->adjList == nullptr) , GDF_INVALID_API_CALL);

  graph->adjList = new gdf_adj_list;
  graph->adjList->offsets = new gdf_column;
  graph->adjList->indices = new gdf_column;
  graph->adjList->ownership = 0;

  cpy_column_view(offsets, graph->adjList->offsets);
  cpy_column_view(indices, graph->adjList->indices);
  if (edge_data) {
      GDF_REQUIRE( indices->size == edge_data->size, GDF_COLUMN_SIZE_MISMATCH );
      graph->adjList->edge_data = new gdf_column;
      cpy_column_view(edge_data, graph->adjList->edge_data);
  }
  else {
    graph->adjList->edge_data = nullptr;
  }
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
                                 const gdf_column *dest_indices, const gdf_column *edge_data) {
  GDF_REQUIRE( src_indices->size == dest_indices->size, GDF_COLUMN_SIZE_MISMATCH );
  GDF_REQUIRE( src_indices->dtype == dest_indices->dtype, GDF_UNSUPPORTED_DTYPE );
  GDF_REQUIRE( ((src_indices->dtype == GDF_INT32) || (src_indices->dtype == GDF_INT64)), GDF_UNSUPPORTED_DTYPE );
  GDF_REQUIRE( src_indices->size > 0, GDF_DATASET_EMPTY ); 
  GDF_REQUIRE( src_indices->null_count == 0 , GDF_VALIDITY_UNSUPPORTED );                    
  GDF_REQUIRE( dest_indices->null_count == 0 , GDF_VALIDITY_UNSUPPORTED );
  GDF_REQUIRE( graph->edgeList == nullptr , GDF_INVALID_API_CALL);

  graph->edgeList = new gdf_edge_list;
  graph->edgeList->src_indices = new gdf_column;
  graph->edgeList->dest_indices = new gdf_column;
  graph->edgeList->ownership = 0;

  cpy_column_view(src_indices, graph->edgeList->src_indices);
  cpy_column_view(dest_indices, graph->edgeList->dest_indices);
  if (edge_data) {
      GDF_REQUIRE( src_indices->size == edge_data->size, GDF_COLUMN_SIZE_MISMATCH );
      graph->edgeList->edge_data = new gdf_column;
      cpy_column_view(edge_data, graph->edgeList->edge_data);
  }
  else {
    graph->edgeList->edge_data = nullptr;
  }
  return GDF_SUCCESS;
}

template <typename WT>
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

      CSR_Result_Weighted<int,WT> adj_list;
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

      CUDA_TRY(cudaMallocManaged ((void**)&d_src, sizeof(int) * graph->adjList->indices->size));

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
        CSR_Result_Weighted<int,WT> adj_list;
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
    }
    return GDF_SUCCESS;
}

template <typename WT>
gdf_error gdf_pagerank_impl (gdf_graph *graph,
                      gdf_column *pagerank, float alpha = 0.85,
                      float tolerance = 1e-4, int max_iter = 200,
                      bool has_guess = false) {
  GDF_REQUIRE( graph->edgeList != nullptr, GDF_VALIDITY_UNSUPPORTED );
  GDF_REQUIRE( graph->edgeList->src_indices->size == graph->edgeList->dest_indices->size, GDF_COLUMN_SIZE_MISMATCH ); 
  GDF_REQUIRE( graph->edgeList->src_indices->dtype == graph->edgeList->dest_indices->dtype, GDF_UNSUPPORTED_DTYPE );  
  GDF_REQUIRE( graph->edgeList->src_indices->null_count == 0 , GDF_VALIDITY_UNSUPPORTED );                 
  GDF_REQUIRE( graph->edgeList->dest_indices->null_count == 0 , GDF_VALIDITY_UNSUPPORTED );  
  GDF_REQUIRE( pagerank != nullptr , GDF_INVALID_API_CALL ); 
  GDF_REQUIRE( pagerank->data != nullptr , GDF_INVALID_API_CALL ); 
  GDF_REQUIRE( pagerank->null_count == 0 , GDF_VALIDITY_UNSUPPORTED );          
  GDF_REQUIRE( pagerank->size > 0 , GDF_INVALID_API_CALL );         

  int m=pagerank->size, nnz = graph->edgeList->src_indices->size, status = 0;
  WT *d_pr, *d_val = nullptr, *d_leaf_vector = nullptr; 
  WT res = 1.0;
  WT *residual = &res;

  if (graph->transposedAdjList == nullptr) {
    gdf_add_transposed_adj_list(graph);
  }
  cudaStream_t stream{nullptr};
  ALLOC_MANAGED_TRY((void**)&d_leaf_vector, sizeof(WT) * m, stream);
  ALLOC_MANAGED_TRY((void**)&d_val, sizeof(WT) * nnz , stream);
  ALLOC_MANAGED_TRY((void**)&d_pr,    sizeof(WT) * m, stream);

  cugraph::HT_matrix_csc_coo(m, nnz, (int*)graph->transposedAdjList->offsets->data, (int*)graph->transposedAdjList->indices->data, d_val, d_leaf_vector);

  if (has_guess)
  {
    GDF_REQUIRE( pagerank->data != nullptr, GDF_VALIDITY_UNSUPPORTED );
    cugraph::copy<WT>(m, (WT*)pagerank->data, d_pr);
  }

  status = cugraph::pagerank<int,WT>( m,nnz, (int*)graph->transposedAdjList->offsets->data, (int*)graph->transposedAdjList->indices->data, 
    d_val, alpha, d_leaf_vector, false, tolerance, max_iter, d_pr, residual);
 
  if (status !=0)
    switch ( status ) { 
      case -1: std::cerr<< "Error : bad parameters in Pagerank"<<std::endl; return GDF_CUDA_ERROR; 
      case 1: std::cerr<< "Warning : Pagerank did not reached the desired tolerance"<<std::endl;  return GDF_CUDA_ERROR; 
      default:  std::cerr<< "Pagerank failed"<<std::endl;  return GDF_CUDA_ERROR; 
    }   
 
  cugraph::copy<WT>(m, d_pr, (WT*)pagerank->data);

  ALLOC_FREE_TRY(d_val, stream);
  ALLOC_FREE_TRY(d_pr, stream);
  ALLOC_FREE_TRY(d_leaf_vector, stream);

  return GDF_SUCCESS;
}

gdf_error gdf_add_adj_list(gdf_graph *graph) {
  if (graph->adjList != nullptr)
    return GDF_SUCCESS;

  GDF_REQUIRE( graph->edgeList != nullptr , GDF_INVALID_API_CALL);
  GDF_REQUIRE( graph->adjList == nullptr , GDF_INVALID_API_CALL);

  if (graph->edgeList->edge_data != nullptr) {
    switch (graph->edgeList->edge_data->dtype) {
      case GDF_FLOAT32:   return gdf_add_adj_list_impl<float>(graph);
      case GDF_FLOAT64:   return gdf_add_adj_list_impl<double>(graph);
      default: return GDF_UNSUPPORTED_DTYPE;
    }
  }
  else {
    return gdf_add_adj_list_impl<float>(graph);
  }
}

gdf_error gdf_add_transposed_adj_list(gdf_graph *graph) {
  if (graph->edgeList == nullptr)
    gdf_add_edge_list(graph);
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

gdf_error gdf_pagerank(gdf_graph *graph, gdf_column *pagerank, float alpha, float tolerance, int max_iter, bool has_guess) {
  switch (pagerank->dtype) {
    case GDF_FLOAT32:   return gdf_pagerank_impl<float>(graph, pagerank, alpha, tolerance, max_iter, has_guess);
    case GDF_FLOAT64:   return gdf_pagerank_impl<double>(graph, pagerank, alpha, tolerance, max_iter, has_guess);
    default: return GDF_UNSUPPORTED_DTYPE;
  }
}

gdf_error gdf_bfs(gdf_graph *graph, gdf_column *distances, gdf_column *predecessors, int start_node, bool directed) {
  GDF_REQUIRE(graph->adjList != nullptr || graph->edgeList != nullptr, GDF_INVALID_API_CALL);
  gdf_error err = gdf_add_adj_list(graph);
  if (err != GDF_SUCCESS)
    return err;
  GDF_REQUIRE(graph->adjList->offsets->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
  GDF_REQUIRE(graph->adjList->indices->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
  GDF_REQUIRE(distances->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
  GDF_REQUIRE(predecessors->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);

  int n = graph->adjList->offsets->size - 1;
  int e = graph->adjList->indices->size;
  int* offsets_ptr = (int*)graph->adjList->offsets->data;
  int* indices_ptr = (int*)graph->adjList->indices->data;
  int* distances_ptr = (int*)distances->data;
  int* predecessors_ptr = (int*)predecessors->data;
  int alpha = 15;
  int beta = 18;

  cugraph::Bfs<int> bfs(n, e, offsets_ptr, indices_ptr, directed, alpha, beta);
  bfs.configure(distances_ptr, predecessors_ptr, nullptr);
  bfs.traverse(start_node);
  return GDF_SUCCESS;
}

gdf_error gdf_jaccard(gdf_graph *graph, void *c_gamma, gdf_column *weights, gdf_column *weight_j) {
  GDF_REQUIRE(graph->adjList != nullptr || graph->edgeList != nullptr, GDF_INVALID_API_CALL);
  gdf_error err = gdf_add_adj_list(graph);
  if (err != GDF_SUCCESS)
    return err;
  //GDF_REQUIRE(weight_j->dtype == GDF_FLOAT32, GDF_UNSUPPORTED_DTYPE);
  
  size_t n = graph->adjList->offsets->size - 1;
  size_t e = graph->adjList->indices->size;

  void* offsets_ptr = graph->adjList->offsets->data;
  void* indices_ptr = graph->adjList->indices->data;
  void* value_ptr = graph->adjList->edge_data? graph->adjList->edge_data->data: NULL;
  void* weight_j_ptr = weight_j->data;

  void* weights_ptr;
  if(weights == NULL)
      weights_ptr = NULL;
  else
      weights_ptr = weights->data;

  auto gdf_to_cudadtype= [](gdf_column *col){
    cudaDataType_t cuda_dtype; 
    switch(col->dtype){
      case GDF_INT8: cuda_dtype = CUDA_R_8I; break;
      case GDF_INT32: cuda_dtype = CUDA_R_32I; break;
      case GDF_FLOAT32: cuda_dtype = CUDA_R_32F; break;
      case GDF_FLOAT64: cuda_dtype = CUDA_R_64F; break;
      }return cuda_dtype;
  };

  cudaDataType_t index_type = gdf_to_cudadtype(graph->adjList->indices);
  cudaDataType_t val_type = graph->adjList->edge_data? gdf_to_cudadtype(graph->adjList->edge_data): CUDA_R_32F;

  nvgraphJaccard(index_type, val_type, n, e, offsets_ptr, indices_ptr, NULL,
                 0, weights_ptr, c_gamma, weight_j_ptr);
  return GDF_SUCCESS;
}

gdf_error gdf_louvain(gdf_graph *graph, void *final_modularity, void *num_level, gdf_column *louvain_parts) {
  GDF_REQUIRE(graph->adjList != nullptr || graph->edgeList != nullptr, GDF_INVALID_API_CALL);
  gdf_error err = gdf_add_adj_list(graph);
  if (err != GDF_SUCCESS)
    return err;

  size_t n = graph->adjList->offsets->size - 1;
  size_t e = graph->adjList->indices->size;

  void* offsets_ptr = graph->adjList->offsets->data;
  void* indices_ptr = graph->adjList->indices->data;
  void* value_ptr = graph->adjList->edge_data? graph->adjList->edge_data->data: NULL;
  void* louvain_parts_ptr = louvain_parts->data;

  auto gdf_to_cudadtype= [](gdf_column *col){
    cudaDataType_t cuda_dtype;
    switch(col->dtype){
      case GDF_INT8: cuda_dtype = CUDA_R_8I; break;
      case GDF_INT32: cuda_dtype = CUDA_R_32I; break;
      case GDF_FLOAT32: cuda_dtype = CUDA_R_32F; break;
      case GDF_FLOAT64: cuda_dtype = CUDA_R_64F; break;
      }return cuda_dtype;
  };

  cudaDataType_t index_type = gdf_to_cudadtype(graph->adjList->indices);
  cudaDataType_t val_type = graph->adjList->edge_data? gdf_to_cudadtype(graph->adjList->edge_data): CUDA_R_32F;

  nvgraphLouvain(index_type, val_type, n, e, offsets_ptr, indices_ptr, value_ptr, 1, 0, NULL, 
                 final_modularity, louvain_parts_ptr, num_level);
  return GDF_SUCCESS;
}

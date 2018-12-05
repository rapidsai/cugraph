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
//#include <functions.h>

void gdf_col_delete(gdf_column* col) {
  if (col)
  {
    col->size = 0; 
    if(col->data)
      if (cudaFree(col->data) != cudaSuccess) 
        std::cerr << "CUDA ERROR : " << cudaGetErrorString(cudaGetLastError()) <<std::endl;    
    delete col;
    col->data = nullptr;
    col = nullptr;  
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
    GDF_REQUIRE( graph->edgeList != nullptr , GDF_INVALID_API_CALL);
    GDF_REQUIRE( graph->adjList == nullptr , GDF_INVALID_API_CALL);
    
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
                          nullptr, adj_list.size, graph->edgeList->src_indices->dtype);
    gdf_column_view(graph->adjList->indices, adj_list.colIndices, 
                          nullptr, adj_list.nnz, graph->edgeList->src_indices->dtype);
    gdf_column_view(graph->adjList->edge_data, adj_list.edgeWeights, 
                        nullptr, adj_list.nnz, graph->edgeList->edge_data->dtype);
  }
  else {
    CSR_Result<int> adj_list;
    status = ConvertCOOtoCSR((int*)graph->edgeList->src_indices->data,(int*)graph->edgeList->dest_indices->data, nnz, adj_list);      
    gdf_column_view(graph->adjList->offsets, adj_list.rowOffsets, 
                          nullptr, adj_list.size, graph->edgeList->src_indices->dtype);
    gdf_column_view(graph->adjList->indices, adj_list.colIndices, 
                          nullptr, adj_list.nnz, graph->edgeList->src_indices->dtype);
  }
  if (status !=0) {
    std::cerr << "Could not generate the adj_list" << std::endl;
    return GDF_CUDA_ERROR;
  }
  return GDF_SUCCESS;
}


template <typename WT>
gdf_error gdf_add_transpose_impl (gdf_graph *graph) {
    GDF_REQUIRE( graph->edgeList != nullptr , GDF_INVALID_API_CALL);
    GDF_REQUIRE( graph->transposedAdjList == nullptr , GDF_INVALID_API_CALL);
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
                          nullptr, adj_list.size, graph->edgeList->src_indices->dtype);
    gdf_column_view(graph->transposedAdjList->indices, adj_list.colIndices, 
                          nullptr, adj_list.nnz, graph->edgeList->src_indices->dtype);
    gdf_column_view(graph->transposedAdjList->edge_data, adj_list.edgeWeights, 
                        nullptr, adj_list.nnz, graph->edgeList->edge_data->dtype);
  }
  else {

    CSR_Result<int> adj_list;
    status = ConvertCOOtoCSR((int*)graph->edgeList->dest_indices->data, (int*)graph->edgeList->src_indices->data, nnz, adj_list);      
    gdf_column_view(graph->transposedAdjList->offsets, adj_list.rowOffsets, 
                          nullptr, adj_list.size, graph->edgeList->src_indices->dtype);
    gdf_column_view(graph->transposedAdjList->indices, adj_list.colIndices, 
                          nullptr, adj_list.nnz, graph->edgeList->src_indices->dtype);
  }
    if (status !=0) {
      std::cerr << "Could not generate the adj_list" << std::endl;
      return GDF_CUDA_ERROR;
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
    gdf_add_transpose(graph);
  }

  CUDA_TRY(cudaMallocManaged ((void**)&d_leaf_vector,    sizeof(WT) * m));
  CUDA_TRY(cudaMallocManaged ((void**)&d_val, sizeof(WT) * nnz )); 
  CUDA_TRY(cudaMallocManaged ((void**)&d_pr,    sizeof(WT) * m));

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
  cudaFree(d_val);
  cudaFree(d_pr);    
  cudaFree(d_leaf_vector);  
  return GDF_SUCCESS;
}


gdf_error gdf_add_adj_list(gdf_graph *graph)
{
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

gdf_error gdf_add_edge_list(gdf_graph *graph)
{
  return GDF_UNSUPPORTED_METHOD;
}

gdf_error gdf_add_transpose(gdf_graph *graph)
{
  if (graph->edgeList->edge_data != nullptr) {
    switch (graph->edgeList->edge_data->dtype) {
      case GDF_FLOAT32:   return gdf_add_transpose_impl<float>(graph);
      case GDF_FLOAT64:   return gdf_add_transpose_impl<double>(graph);
      default: return GDF_UNSUPPORTED_DTYPE;
    }
  }
  else {
    return gdf_add_transpose_impl<float>(graph);
  }
}

gdf_error gdf_delete_adj_list(gdf_graph *graph) {
  if (graph->adjList) {
    graph->adjList->ownership = 1;
    delete graph->adjList;
  }
  graph->adjList = nullptr;
  return GDF_SUCCESS;
}
gdf_error gdf_delete_edge_list(gdf_graph *graph) {
  if (graph->edgeList) {
    graph->edgeList->ownership = 1;
    delete graph->edgeList;
  }
  graph->edgeList = nullptr;
  return GDF_SUCCESS;
}
gdf_error gdf_delete_transpose(gdf_graph *graph) {
  if (graph->transposedAdjList) {
    graph->transposedAdjList->ownership = 1;
    delete graph->transposedAdjList;
  }
  graph->transposedAdjList = nullptr;
  return GDF_SUCCESS;
}


gdf_error gdf_pagerank(gdf_graph *graph, gdf_column *pagerank, float alpha, float tolerance, int max_iter, bool has_guess)
{ 
  switch (pagerank->dtype) {
    case GDF_FLOAT32:   return gdf_pagerank_impl<float>(graph, pagerank, alpha, tolerance, max_iter, has_guess);
    case GDF_FLOAT64:   return gdf_pagerank_impl<double>(graph, pagerank, alpha, tolerance, max_iter, has_guess);
    default: return GDF_UNSUPPORTED_DTYPE;
  }
}

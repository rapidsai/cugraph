 /*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

#include <cugraph.h>
#include "utilities/graph_utils.cuh"
#include "utilities/error_utils.h"
#include "cub/cub.cuh"
#include <rmm_utils.h>

namespace cugraph
{
  
#ifdef DEBUG
  #define PR_VERBOSE
#endif
template <typename IndexType, typename ValueType>
bool  pagerankIteration( IndexType n, IndexType e, IndexType *cscPtr, IndexType *cscInd,ValueType *cscVal,
                                     ValueType alpha, ValueType *a, ValueType *b, float tolerance, int iter, int max_iter, 
                                     ValueType * &tmp,  void* cub_d_temp_storage, size_t  cub_temp_storage_bytes, 
                                     ValueType * &pr, ValueType *residual) {
    
    ValueType  dot_res;
    cub::DeviceSpmv::CsrMV(cub_d_temp_storage, cub_temp_storage_bytes, cscVal,
        cscPtr, cscInd, tmp, pr,
        n, n, e);
   
    scal(n, alpha, pr);
    dot_res = dot( n, a, tmp);
    axpy(n, dot_res,  b,  pr);
    scal(n, (ValueType)1.0/nrm2(n, pr) , pr);
    axpy(n, (ValueType)-1.0,  pr,  tmp);
    *residual = nrm2(n, tmp);
    if (*residual < tolerance)
    {
        scal(n, (ValueType)1.0/nrm1(n,pr), pr);
        return true;
    }
    else
    {
        if (iter< max_iter)
        {
            std::swap(pr, tmp);
        }
        else
        {
           scal(n, (ValueType)1.0/nrm1(n,pr), pr);
        }
        return false;
    }
}

template <typename IndexType, typename ValueType>
int pagerank (  IndexType n, IndexType e, IndexType *cscPtr, IndexType *cscInd, ValueType *cscVal,
                       ValueType alpha, ValueType *a, bool has_guess, float tolerance, int max_iter, 
                       ValueType * &pagerank_vector, ValueType * &residual) {
  int max_it, i = 0 ;
  float tol;
  bool converged = false;
  ValueType randomProbability =  static_cast<ValueType>( 1.0/n);
  ValueType *b=0, *tmp=0;
  void*    cub_d_temp_storage = NULL;
  size_t   cub_temp_storage_bytes = 0;

  if (max_iter > 0 )
      max_it = max_iter;
  else
      max_it =  500;

  if (tolerance == 0.0f)
      tol =  1.0E-6f;
  else if (tolerance < 1.0f && tolerance > 0.0f)
      tol = tolerance;
  else
      return -1;

  if (alpha <= 0.0f || alpha >= 1.0f)
          return -1;

  cudaStream_t stream{nullptr};
	
  ALLOC_MANAGED_TRY ((void**)&b,    sizeof(ValueType) * n, stream);
  ALLOC_MANAGED_TRY ((void**)&tmp,    sizeof(ValueType) * n, stream);
  cudaCheckError();

  if (!has_guess)  {
       fill(n, pagerank_vector, randomProbability);
       fill(n, tmp, randomProbability);
  }
  else {
    copy(n, pagerank_vector, tmp);
  }


  fill(n, b, randomProbability);
  update_dangling_nodes(n, a, alpha);

  cub::DeviceSpmv::CsrMV(cub_d_temp_storage, cub_temp_storage_bytes, cscVal,
                                             cscPtr, cscInd, tmp, pagerank_vector, n, n, e);
   // Allocate temporary storage
 ALLOC_MANAGED_TRY ((void**)&cub_d_temp_storage, cub_temp_storage_bytes, stream);
 cudaCheckError()
  #ifdef PR_VERBOSE
      std::stringstream ss;
      ss.str(std::string());
      ss <<" ------------------PageRank------------------"<< std::endl;
      ss <<" --------------------------------------------"<< std::endl;
      ss << std::setw(10) << "Iteration" << std::setw(15) << "Residual" << std::endl;
      ss <<" --------------------------------------------"<< std::endl;
      std::cout<<ss.str();
  #endif

  while (!converged && i < max_it)
  { 
      i++;
      converged = pagerankIteration(n, e, cscPtr, cscInd, cscVal,
                                           alpha, a, b, tol, i, max_it, tmp, 
                                           cub_d_temp_storage, cub_temp_storage_bytes, 
                                           pagerank_vector, residual);
       #ifdef PR_VERBOSE
          ss.str(std::string());
          ss << std::setw(10) << i ;
          ss.precision(3);
          ss << std::setw(15) << std::scientific << *residual  << std::endl;
          std::cout<<ss.str();
      #endif
  }
  #ifdef PR_VERBOSE
      std::cout <<" --------------------------------------------"<< std::endl;
  #endif
  //printv(n,pagerank_vector,0);

  ALLOC_FREE_TRY(b, stream);  
  ALLOC_FREE_TRY(tmp, stream);
  ALLOC_FREE_TRY(cub_d_temp_storage, stream);    
  
  return converged ? 0 : 1;
}

//template int pagerank<int, half> (  int n, int e, int *cscPtr, int *cscInd,half *cscVal, half alpha, half *a, bool has_guess, float tolerance, int max_iter, half * &pagerank_vector, half * &residual);
template int pagerank<int, float> (  int n, int e, int *cscPtr, int *cscInd,float *cscVal, float alpha, float *a, bool has_guess, float tolerance, int max_iter, float * &pagerank_vector, float * &residual);
template int pagerank<int, double> (  int n, int e, int *cscPtr, int *cscInd,double *cscVal, double alpha, double *a, bool has_guess, float tolerance, int max_iter, double * &pagerank_vector, double * &residual);

} //namespace cugraph

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

  //  The templating for HT_matrix_csc_coo assumes that m, nnz and data are all the same type
  cugraph::HT_matrix_csc_coo(m, nnz, (int *)graph->transposedAdjList->offsets->data, (int *)graph->transposedAdjList->indices->data, d_val, d_leaf_vector);

  if (has_guess)
  {
    GDF_REQUIRE( pagerank->data != nullptr, GDF_VALIDITY_UNSUPPORTED );
    cugraph::copy<WT>(m, (WT*)pagerank->data, d_pr);
  }

  status = cugraph::pagerank<int32_t,WT>( m,nnz, (int*)graph->transposedAdjList->offsets->data, (int*)graph->transposedAdjList->indices->data,
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

gdf_error gdf_pagerank(gdf_graph *graph, gdf_column *pagerank, float alpha, float tolerance, int max_iter, bool has_guess) {
  //
  //  page rank operates on CSR and can't currently support 64-bit integers.
  //
  //  If csr doesn't exist, create it.  Then check type to make sure it is 32-bit.
  //
  GDF_REQUIRE(graph->adjList != nullptr || graph->edgeList != nullptr, GDF_INVALID_API_CALL);
  gdf_error err = gdf_add_adj_list(graph);
  if (err != GDF_SUCCESS)
    return err;

  GDF_REQUIRE(graph->adjList->offsets->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
  GDF_REQUIRE(graph->adjList->indices->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);

  switch (pagerank->dtype) {
    case GDF_FLOAT32:   return gdf_pagerank_impl<float>(graph, pagerank, alpha, tolerance, max_iter, has_guess);
    case GDF_FLOAT64:   return gdf_pagerank_impl<double>(graph, pagerank, alpha, tolerance, max_iter, has_guess);
    default: return GDF_UNSUPPORTED_DTYPE;
  }
}

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

// Pagerank solver
// Author: Alex Fender afender@nvidia.com

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string>
 #include <sstream>
#include <iostream>
#include <iomanip>
#include "cub/cub.cuh"
#include <algorithm>
#include <iomanip>

#include <rmm_utils.h>

#include "utilities/graph_utils.cuh"
#include "utilities/error_utils.h"
#include <cugraph.h>

namespace cugraph { 
namespace detail {

#ifdef DEBUG
  #define PR_VERBOSE
#endif
template <typename IndexType, typename ValueType>
bool pagerankIteration(IndexType n, IndexType e, IndexType *cscPtr, IndexType *cscInd,ValueType *cscVal,
                       ValueType alpha, ValueType *a, ValueType *b, float tolerance, int iter, int max_iter,
                       ValueType * &tmp,  void* cub_d_temp_storage, size_t  cub_temp_storage_bytes,
                       ValueType * &pr, ValueType *residual) {
    ValueType  dot_res;
    CUDA_TRY(cub::DeviceSpmv::CsrMV(cub_d_temp_storage, cub_temp_storage_bytes, cscVal,
                                    cscPtr, cscInd, tmp, pr, n, n, e));

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
int pagerankSolver(IndexType n, IndexType e, IndexType *cscPtr, IndexType *cscInd, ValueType *cscVal,
             IndexType *prsVtx, ValueType *prsVal, IndexType prsLen, bool has_personalization,
             ValueType alpha, ValueType *a, bool has_guess, float tolerance, int max_iter,
             ValueType * &pagerank_vector, ValueType * &residual) {
  int max_it, i = 0 ;
  float tol;
  bool converged = false;
  ValueType randomProbability = static_cast<ValueType>( 1.0/n);
  ValueType *b=0, *tmp=0;
  void* cub_d_temp_storage = NULL;
  size_t cub_temp_storage_bytes = 0;

  if (max_iter > 0)
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

  ALLOC_TRY((void**)&b, sizeof(ValueType) * n, stream);
#if 1/* temporary solution till https://github.com/NVlabs/cub/issues/162 is resolved */
  CUDA_TRY(cudaMalloc((void**)&tmp, sizeof(ValueType) * n));
#else
  ALLOC_TRY((void**)&tmp, sizeof(ValueType) * n, stream);
#endif
  CUDA_CHECK_LAST();

  if (!has_guess) {
       fill(n, pagerank_vector, randomProbability);
       fill(n, tmp, randomProbability);
  }
  else {
    copy(n, pagerank_vector, tmp);
  }

  if (has_personalization) {
    ValueType sum = nrm1(prsLen, prsVal);
    if (static_cast<ValueType>(0) == sum) {
      fill(n, b, randomProbability);
    } else {
      scal(n, static_cast<ValueType>(1.0/sum), prsVal);
      fill(n, b, static_cast<ValueType>(0));
      scatter(prsLen, prsVal, b, prsVtx);
    }
  } else {
    fill(n, b, randomProbability);
  }
  update_dangling_nodes(n, a, alpha);

  CUDA_TRY(cub::DeviceSpmv::CsrMV(cub_d_temp_storage, cub_temp_storage_bytes, cscVal,
                                  cscPtr, cscInd, tmp, pagerank_vector, n, n, e));
   // Allocate temporary storage
  ALLOC_TRY ((void**)&cub_d_temp_storage, cub_temp_storage_bytes, stream);
  CUDA_CHECK_LAST()
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
#if 1/* temporary solution till https://github.com/NVlabs/cub/issues/162 is resolved */
  CUDA_TRY(cudaFree(tmp));
#else
  ALLOC_FREE_TRY(tmp, stream);
#endif
  ALLOC_FREE_TRY(cub_d_temp_storage, stream);

  return converged ? 0 : 1;
}

//template int pagerankSolver<int, half> (  int n, int e, int *cscPtr, int *cscInd,half *cscVal, half alpha, half *a, bool has_guess, float tolerance, int max_iter, half * &pagerank_vector, half * &residual);
template int pagerankSolver<int, float> (  int n, int e, int *cscPtr, int *cscInd,float *cscVal,
        int *prsVtx, float *prsVal, int prsLen, bool has_personalization,
        float alpha, float *a, bool has_guess, float tolerance, int max_iter, float * &pagerank_vector, float * &residual);
template int pagerankSolver<int, double> (  int n, int e, int *cscPtr, int *cscInd,double *cscVal,
        int *prsVtx,  double *prsVal, int prsLen, bool has_personalization,
        double alpha, double *a, bool has_guess, float tolerance, int max_iter, double * &pagerank_vector, double * &residual);

template <typename WT>
void pagerank_impl (Graph *graph,
                      gdf_column *pagerank,
                      gdf_column *personalization_subset, gdf_column *personalization_values,
                      float alpha = 0.85,
                      float tolerance = 1e-4, int max_iter = 200,
                      bool has_guess = false) {
  bool has_personalization = false;
  int *prsVtx = nullptr;
  WT  *prsVal = nullptr;
  int prsLen = 0;
  CUGRAPH_EXPECTS((personalization_subset == nullptr) == (personalization_values == nullptr), "Invalid API parameter");
  if (personalization_subset != nullptr) {
    has_personalization = true;
    prsVtx = reinterpret_cast<int*>(personalization_subset->data);
    prsVal = reinterpret_cast<WT* >(personalization_values->data);
    prsLen = reinterpret_cast<int >(personalization_subset->size);
    CUGRAPH_EXPECTS(pagerank->dtype == personalization_values->dtype, "Invalid API parameter");
    CUGRAPH_EXPECTS(personalization_subset->dtype == GDF_INT32, "Unsupported data type");
    CUGRAPH_EXPECTS(personalization_subset->size == personalization_values->size, "Column size mismatch");
    CUGRAPH_EXPECTS(personalization_subset->null_count == 0 , "Input column has non-zero null count");
    CUGRAPH_EXPECTS(personalization_values->null_count == 0 , "Input column has non-zero null count");
  }

  CUGRAPH_EXPECTS( pagerank != nullptr , "Invalid API parameter" );
  CUGRAPH_EXPECTS( pagerank->data != nullptr , "Invalid API parameter" );
  CUGRAPH_EXPECTS( pagerank->null_count == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( pagerank->size > 0 , "Invalid API parameter" );

  int m=pagerank->size, nnz = graph->transposedAdjList->indices->size, status = 0;
  WT *d_pr, *d_val = nullptr, *d_leaf_vector = nullptr;
  WT res = 1.0;
  WT *residual = &res;

  if (graph->transposedAdjList == nullptr) {
    add_transposed_adj_list(graph);
  }
  cudaStream_t stream{nullptr};
  ALLOC_TRY((void**)&d_leaf_vector, sizeof(WT) * m, stream);
  ALLOC_TRY((void**)&d_val, sizeof(WT) * nnz , stream);
#if 1/* temporary solution till https://github.com/NVlabs/cub/issues/162 is resolved */
  CUDA_TRY(cudaMalloc((void**)&d_pr, sizeof(WT) * m));
#else
  ALLOC_TRY((void**)&d_pr, sizeof(WT) * m, stream);
#endif

  //  The templating for HT_matrix_csc_coo assumes that m, nnz and data are all the same type
  HT_matrix_csc_coo(m, nnz, (int *)graph->transposedAdjList->offsets->data, (int *)graph->transposedAdjList->indices->data, d_val, d_leaf_vector);

  if (has_guess)
  {
    CUGRAPH_EXPECTS( pagerank->data != nullptr, "Column must be valid" );
    copy<WT>(m, (WT*)pagerank->data, d_pr);
  }

  status = pagerankSolver<int32_t,WT>( m,nnz, (int*)graph->transposedAdjList->offsets->data, (int*)graph->transposedAdjList->indices->data, d_val,
          prsVtx, prsVal, prsLen, has_personalization,
    alpha, d_leaf_vector, has_guess, tolerance, max_iter, d_pr, residual);

  if (status !=0)
    switch ( status ) {
      case -1: CUGRAPH_FAIL("Error : bad parameters in Pagerank");
      case 1: CUGRAPH_FAIL("Warning : Pagerank did not reached the desired tolerance");
      default:  CUGRAPH_FAIL("Pagerank exec failed");
    }

  copy<WT>(m, d_pr, (WT*)pagerank->data);

  ALLOC_FREE_TRY(d_val, stream);
#if 1/* temporary solution till https://github.com/NVlabs/cub/issues/162 is resolved */
  CUDA_TRY(cudaFree(d_pr));
#else
  ALLOC_FREE_TRY(d_pr, stream);
#endif
  ALLOC_FREE_TRY(d_leaf_vector, stream);

  
}

}
void pagerank(Graph *graph, gdf_column *pagerank,
        gdf_column *personalization_subset, gdf_column *personalization_values,
        float alpha, float tolerance, int max_iter, bool has_guess) {
  //
  //  page rank operates on CSR and can't currently support 64-bit integers.
  //
  //  If csr doesn't exist, create it.  Then check type to make sure it is 32-bit.
  //
  CUGRAPH_EXPECTS(graph->transposedAdjList != nullptr, "Invalid API parameter");

  switch (pagerank->dtype) {
    case GDF_FLOAT32:   return detail::pagerank_impl<float>(graph, pagerank,
                                personalization_subset, personalization_values,
                                alpha, tolerance, max_iter, has_guess);
    case GDF_FLOAT64:   return detail::pagerank_impl<double>(graph, pagerank,
                                personalization_subset, personalization_values,
                                alpha, tolerance, max_iter, has_guess);
    default: CUGRAPH_FAIL("Unsupported data type");
  }
}

} //namespace cugraph 
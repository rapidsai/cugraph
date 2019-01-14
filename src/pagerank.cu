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
#include "graph_utils.cuh"
#include "pagerank.cuh"
#include "cub/cub.cuh"
#include <algorithm>
#include <iomanip>

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

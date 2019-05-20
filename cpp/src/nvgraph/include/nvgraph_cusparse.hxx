/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#pragma once

#include <cusparse_v2.h>
#include <cusparse_internal.h>
#include "valued_csr_graph.hxx"
#include "nvgraph_vector.hxx"

#include <iostream>
#include "debug_macros.h"

namespace nvgraph
{
class Cusparse 
{
private:
  // global CUSPARSE handle for nvgraph
  static cusparseHandle_t m_handle; // Constructor.
  Cusparse();
  // Destructor.
  ~Cusparse();

public:

  // Get the handle.
  static cusparseHandle_t get_handle()
  {
      if (m_handle == 0)
          CHECK_CUSPARSE(cusparseCreate(&m_handle));
      return m_handle;
  }
  // Destroy handle
  static void destroy_handle()
  {
    if (m_handle != 0)
      CHECK_CUSPARSE( cusparseDestroy(m_handle) );
    m_handle = 0;
  }
  static void setStream(cudaStream_t stream) 
  {   
      cusparseHandle_t handle = Cusparse::get_handle();
      CHECK_CUSPARSE(cusparseSetStream(handle, stream));
  }
  // Set pointer mode
  static void set_pointer_mode_device();
  static void set_pointer_mode_host();

  // operate on all rows and columns y= alpha*A.x + beta*y
  template <typename IndexType_, typename ValueType_>
  static void csrmv( const bool transposed,
                     const bool sym,
                     const int m, const int n, const int nnz, 
                     const ValueType_* alpha, 
                     const ValueType_* csrVal,
                     const IndexType_ *csrRowPtr, 
                     const IndexType_ *csrColInd, 
                     const ValueType_* x,
                     const ValueType_* beta, 
                     ValueType_* y);
  
  template <typename IndexType_, typename ValueType_>
  static void csrmv( const bool transposed,
                     const bool sym,
                     const ValueType_* alpha, 
                     const ValuedCsrGraph<IndexType_, ValueType_>& G,
                     const Vector<ValueType_>& x,
                     const ValueType_* beta, 
                     Vector<ValueType_>& y
                     );
  
  // future possible features
  /*
  template <class TConfig>
  static void csrmv_with_mask( const typename TConfig::MatPrec alphaConst, 
                     Matrix<TConfig> &A, 
                     Vector<TConfig> &x,
                     const typename TConfig::MatPrec betaConst, 
                     Vector<TConfig> &y );

  template <class TConfig>
  static void csrmv_with_mask_restriction( const typename TConfig::MatPrec alphaConst, 
                     Matrix<TConfig> &A, 
                     Vector<TConfig> &x,
                     const typename TConfig::MatPrec betaConst, 
                     Vector<TConfig> &y, 
                     Matrix<TConfig> &P);

  // E is a vector that represents a diagonal matrix
  // operate on all rows and columns
  // y= alpha*E.x + beta*y
  template <class TConfig>
  static void csrmv( const typename TConfig::MatPrec alphaConst, 
                     Matrix<TConfig> &A, 
                     const typename Matrix<TConfig>::MVector &E,
                     Vector<TConfig> &x,
                     const typename TConfig::MatPrec betaConst, 
                     Vector<TConfig> &y, 
                     ViewType view = OWNED );

  // operate only on columns specified by columnColorSelector, see enum ColumnColorSelector above
  // operate only on rows of specified color, given by A.offsets_rows_per_color, A.sorted_rows_by_color
  // y= alpha*A.x + beta*y
  template <class TConfig>
  static void csrmv( ColumnColorSelector columnColorSelector, 
                     const int color,
                     const typename TConfig::MatPrec alphaConst, 
                     Matrix<TConfig> &A, 
                     Vector<TConfig> &x,
                     const typename TConfig::MatPrec betaConst, 
                     Vector<TConfig> &y, 
                     ViewType view = OWNED );

  // E is a vector that represents a diagonal matrix
  // operate only on rows of specified color, given by A.offsets_rows_per_color, A.sorted_rows_by_color
  // y= alpha*E.x + beta*y
  template <class TConfig>
  static void csrmv( const int color,
                     typename TConfig::MatPrec alphaConst, 
                     Matrix<TConfig> &A, 
                     const typename Matrix<TConfig>::MVector &E, 
                     Vector<TConfig> &x,
                     typename TConfig::MatPrec betaConst, 
                     Vector<TConfig> &y, 
                     ViewType view=OWNED );

  template <class TConfig>
  static void csrmm(typename TConfig::MatPrec alpha,
                    Matrix<TConfig> &A,
                    Vector<TConfig> &V,
                    typename TConfig::VecPrec beta,
                    Vector<TConfig> &Res);

*/

 template <typename IndexType_, typename ValueType_>
 static void csrmm(const bool transposed,
                   const bool sym,
                   const int m, 
                   const int n, 
                   const int k,
                   const int nnz, 
                   const ValueType_* alpha, 
                   const ValueType_* csrVal,
                   const IndexType_* csrRowPtr, 
                   const IndexType_* csrColInd, 
                   const ValueType_* x,
                   const int ldx,
                   const ValueType_* beta, 
                   ValueType_* y,
                   const int ldy);   

 //template <typename IndexType_, typename ValueType_>
 static void csr2coo( const int n, 
                                     const int nnz, 
                                     const int *csrRowPtr,
                                     int *cooRowInd);   
};

} // end namespace nvgraph


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
#include <cusparse.h>
#include "rmm_utils.h"
#include "cusparse_helper.h"

namespace cugraph
{
cusparseHandle_t Cusparse::m_handle = 0;

template <typename ValueType>
CusparseCsrMV<ValueType>::CusparseCsrMV() {
  if (sizeof(ValueType) == 4) 
    cuda_type = CUDA_R_32F;
  else
    cuda_type = CUDA_R_64F;
  CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
  CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO));
  CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL ));
  //alg = CUSPARSE_ALG_NAIVE;
  alg = CUSPARSE_ALG_MERGE_PATH;
  stream = nullptr;
}

template <typename ValueType>
CusparseCsrMV<ValueType>::~CusparseCsrMV() {
  ALLOC_FREE_TRY(spmv_d_temp_storage, stream);
}

template <typename ValueType>
void CusparseCsrMV<ValueType>::setup(int m,
               int n,
               int nnz,
               const ValueType* alpha,
               const ValueType* csrValA,
               const int* csrRowPtrA,
               const int* csrColIndA,
               const ValueType* x,
               const ValueType* beta,
               ValueType* y) {
  
  CHECK_CUSPARSE (cusparseCsrmvEx_bufferSize(Cusparse::get_handle(),
                             alg,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             m,
                             n,
                             nnz,
                             alpha,
                             cuda_type,
                             descrA,
                             csrValA,
                             cuda_type,
                             csrRowPtrA,
                             csrColIndA,
                             x,
                             cuda_type,
                             beta,
                             cuda_type,
                             y,
                             cuda_type,
                             cuda_type,
                             &spmv_temp_storage_bytes));
  ALLOC_TRY ((void**)&spmv_d_temp_storage, spmv_temp_storage_bytes, stream);
}
template <typename ValueType>
void CusparseCsrMV<ValueType>::run(int m,
             int n,
             int nnz,
             const ValueType* alpha,
             const ValueType* csrValA,
             const int* csrRowPtrA,
             const int* csrColIndA,
             const ValueType* x,
             const ValueType* beta,
             ValueType* y) {

  CHECK_CUSPARSE(cusparseCsrmvEx(Cusparse::get_handle(),
                  alg,
                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                  m,
                  n,
                  nnz,
                  alpha,
                  cuda_type,
                  descrA,
                  csrValA,
                  cuda_type,
                  csrRowPtrA,
                  csrColIndA,
                  x,
                  cuda_type,
                  beta,
                  cuda_type,
                  y,
                  cuda_type,
                  cuda_type,
                  spmv_d_temp_storage));

}

template class CusparseCsrMV<double>;
template class CusparseCsrMV<float>;

} //namespace
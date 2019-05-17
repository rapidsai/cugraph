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
#include <cusparse.h>

namespace cugraph
{

#define CHECK_CUSPARSE(call)                                              \
{                                                                         \
    cusparseStatus_t _e = (call);                                         \
    if (_e != CUSPARSE_STATUS_SUCCESS)                                    \
    {                                                                     \
        std::stringstream _error;                                         \
        _error << "CUSPARSE failure: '#" << _e << "'";                    \
        throw std::string(_error.str());                                        \
    }                                                                     \
}


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
};

template <typename ValueType>
class CusparseCsrMV
{
  private:
      cusparseHandle_t cusparseH;
      cusparseMatDescr_t descrA;
      cudaDataType cuda_type;
      cusparseAlgMode_t alg = CUSPARSE_ALG_MERGE_PATH;
      void* spmv_d_temp_storage;
      size_t spmv_temp_storage_bytes;
      cudaStream_t stream;
  
  public:
    CusparseCsrMV() {
      if (sizeof(ValueType) == 4)
        cuda_type = CUDA_R_32F;
      else
        cuda_type = CUDA_R_64F;
      cusparseH = Cusparse::get_handle();
      CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
      CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO));
      CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL ));
      alg = CUSPARSE_ALG_MERGE_PATH;
      stream = nullptr;
    }

    ~CusparseCsrMV() {
      ALLOC_FREE_TRY(spmv_d_temp_storage, stream);
    }

    void setup(int m,
               int n,
               int nnz,
               const ValueType* alpha,
               const ValueType* csrValA,
               const int* csrRowPtrA,
               const int* csrColIndA,
               const ValueType* x,
               const ValueType* beta,
               ValueType* y) {
      CHECK_CUSPARSE (cusparseCsrmvEx_bufferSize(cusparseH,
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
    void run(int m,
             int n,
             int nnz,
             const ValueType* alpha,
             const ValueType* csrValA,
             const int* csrRowPtrA,
             const int* csrColIndA,
             const ValueType* x,
             const ValueType* beta,
             ValueType* y) {
      CHECK_CUSPARSE(cusparseCsrmvEx(cusparseH,
                      alg,
                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                      m,
                      n,
                      nnz,
                      &alpha,
                      cuda_type,
                      descrA,
                      csrValA,
                      cuda_type,
                      csrRowPtrA,
                      csrColIndA,
                      x,
                      cuda_type,
                      &beta,
                      cuda_type,
                      y,
                      cuda_type,
                      cuda_type,
                      spmv_d_temp_storage));
    }
};
} //namespace
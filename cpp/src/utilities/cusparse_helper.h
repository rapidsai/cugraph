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
#include "rmm_utils.h"
#include "utilities/graph_utils.cuh"

namespace cugraph
{

#define CHECK_CUSPARSE(call)                                              \
{                                                                         \
    cusparseStatus_t _e = (call);                                         \
    if (_e != CUSPARSE_STATUS_SUCCESS)                                    \
    {                                                                     \
        std::stringstream _error;                                         \
        _error << "CUSPARSE failure: '#" << _e << "'";                    \
        throw std::string(_error.str());                                  \
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
      cusparseMatDescr_t descrA;
      cudaDataType cuda_type;
      cusparseAlgMode_t alg;
      void* spmv_d_temp_storage;
      size_t spmv_temp_storage_bytes;
      cudaStream_t stream;
  
  public:
    CusparseCsrMV();

    ~CusparseCsrMV();
    void setup(int m,
               int n,
               int nnz,
               const ValueType* alpha,
               const ValueType* csrValA,
               const int* csrRowPtrA,
               const int* csrColIndA,
               const ValueType* x,
               const ValueType* beta,
               ValueType* y);
    void run(int m,
             int n,
             int nnz,
             const ValueType* alpha,
             const ValueType* csrValA,
             const int* csrRowPtrA,
             const int* csrColIndA,
             const ValueType* x,
             const ValueType* beta,
             ValueType* y);
};

} //namespace
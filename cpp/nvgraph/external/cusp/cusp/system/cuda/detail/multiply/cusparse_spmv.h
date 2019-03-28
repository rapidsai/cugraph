/*
 *  Copyright 2008-2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#pragma once

#include <cusp/exception.h>
#include <cusp/system/cuda/cusparse_csr_matrix.h>

#include <cusparse_v2.h>
#include <algorithm>

namespace cusp
{
namespace system
{
namespace cuda
{

template <typename DerivedPolicy,
          typename ValueType,
          typename VectorType1,
          typename VectorType2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void multiply(cuda::execution_policy<DerivedPolicy>& exec,
              const cusp::cuda::cusparse_csr_matrix<ValueType>& A,
              const VectorType1& x,
              VectorType2& y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce,
              cusp::csr_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    cusparseHandle_t cusparse;

    if(CUSPARSE_STATUS_SUCCESS != cusparseCreate(&cusparse)) {
        throw cusp::runtime_exception("Could not initialize Cusparse library");
    }

    ValueType alpha = 1.0;
    ValueType beta  = 0.0;

    if(CUSPARSE_STATUS_SUCCESS != cusparseScsrmv(cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE, A.num_rows, A.num_cols, A.num_entries, &alpha,
            A.descr, thrust::raw_pointer_cast(&A.values[0]), thrust::raw_pointer_cast(&A.row_offsets[0]), thrust::raw_pointer_cast(&A.column_indices[0]),
            thrust::raw_pointer_cast(&x[0]), &beta, thrust::raw_pointer_cast(&y[0]))) {
        throw cusp::runtime_exception("Cusparse Spmv failed");
    }
}

} // end namespace cuda
} // end namespace system
} // end namespace cusp



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

#include <cusp/csr_matrix.h>

#include <cusparse_v2.h>

#include <thrust/system/cuda/vector.h>

namespace cusp
{
namespace system
{
namespace cuda
{

template <typename ValueType>
class cusparse_csr_matrix : public cusp::csr_matrix_view< cusp::array1d_view<typename thrust::cuda::vector<int>::iterator>,
                                                          cusp::array1d_view<typename thrust::cuda::vector<int>::iterator>,
                                                          cusp::array1d_view<typename thrust::cuda::vector<ValueType>::iterator> >
{
private:

    typedef typename thrust::cuda::vector<int>::iterator                    IndexIterator;
    typedef typename thrust::cuda::vector<ValueType>::iterator              ValueIterator;
    typedef cusp::array1d_view<IndexIterator>                               IndicesView;
    typedef cusp::array1d_view<ValueIterator>                               ValuesView;
    typedef cusp::csr_matrix_view<IndicesView,IndicesView,ValuesView>       Parent;

public:

    cusparseMatDescr_t descr;

    /*! Construct an empty \p cusparse_csr_matrix.
     */
    cusparse_csr_matrix(void) : descr(0) {}

    /*! Construct a \p csr_matrix with a specific shape and number of nonzero entries.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     */
    cusparse_csr_matrix(cusp::csr_matrix<int,ValueType,cusp::device_memory>& A)
        : Parent(A)
    {
        /* create and setup matrix descriptor */
        if (cusparseCreateMatDescr(&descr) != CUSPARSE_STATUS_SUCCESS) { throw cusp::runtime_exception("cusparseCreateMatDescr failed"); }
    }

}; // class csr_matrix

} // end namespace cuda
} // end namespace system

// alias system::cuda names at top-level
namespace cuda
{

using cusp::system::cuda::cusparse_csr_matrix;

} // end cuda

} // end namespace cusp


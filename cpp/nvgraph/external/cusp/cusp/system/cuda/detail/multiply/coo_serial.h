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

#include <thrust/device_ptr.h>

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{

// COO format SpMV kernel that uses only one thread
// This is incredibly slow, so it is only useful for testing purposes,
// *extremely* small matrices, or a few elements at the end of a
// larger matrix

template <typename RowIterator,     typename ColumnIterator,
          typename ValueIterator1,  typename ValueIterator2, typename ValueIterator3,
          typename BinaryFunction1, typename BinaryFunction2>
__global__ void
spmv_coo_serial_kernel(const int num_entries,
                       const RowIterator I,
                       const ColumnIterator J,
                       const ValueIterator1 V,
                       const ValueIterator2 x,
                       ValueIterator3 y,
                       BinaryFunction1 combine,
                       BinaryFunction2 reduce)
{
    typedef typename thrust::iterator_value<RowIterator>::type IndexType;

    for(IndexType n = 0; n < num_entries; n++)
    {
        y[I[n]] = reduce(y[I[n]], combine(V[n], x[J[n]]));
    }
}


template <typename Matrix,
          typename Array1,
          typename Array2,
          typename BinaryFunction1,
          typename BinaryFunction2>
void spmv_coo_serial_device(const Matrix& A,
                            const Array1& x,
                                  Array2& y,
                            BinaryFunction1 combine,
                            BinaryFunction2 reduce)
{
    typedef typename Matrix::row_indices_array_type::const_iterator    RowIterator;
    typedef typename Matrix::column_indices_array_type::const_iterator ColumnIterator;
    typedef typename Matrix::values_array_type::const_iterator         ValueIterator1;
    typedef typename Array1::const_iterator                            ValueIterator2;
    typedef typename Array2::iterator                                  ValueIterator3;

    spmv_coo_serial_kernel<RowIterator,ColumnIterator,ValueIterator1,ValueIterator2,ValueIterator3,BinaryFunction1,BinaryFunction2> <<<1,1>>>
    (A.num_entries, A.row_indices.begin(), A.column_indices.begin(), A.values.begin(), x.begin(), y.begin(), combine, reduce);
}

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp


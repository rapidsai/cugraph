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

#include <cusp/coo_matrix.h>
#include <cusp/sort.h>

#include <cusp/format_utils.h>

#include <thrust/gather.h>
#include <thrust/sort.h>

namespace cusp
{

template <typename IndexType, typename MemorySpace>
void
permutation_matrix<IndexType,MemorySpace>
::resize(const size_t num_rows)
{
    Parent::resize(num_rows, num_rows, num_rows);
    permutation.resize(num_rows);
}

template <typename IndexType, typename MemorySpace>
void
permutation_matrix<IndexType,MemorySpace>
::swap(permutation_matrix& matrix)
{
    Parent::swap(matrix);
    permutation.swap(matrix.permutation);
}

template <typename IndexType, typename MemorySpace>
template <typename MatrixType>
void permutation_matrix<IndexType,MemorySpace>
::symmetric_permute(MatrixType& A)
{
    typedef typename MatrixType::index_type IndexType2;
    typedef typename MatrixType::value_type ValueType2;

    cusp::coo_matrix<IndexType2,ValueType2,MemorySpace> B(A);

    // reorder rows and column according to permutation
    thrust::gather(B.row_indices.begin(), B.row_indices.end(), permutation.begin(), B.row_indices.begin());
    thrust::gather(B.column_indices.begin(), B.column_indices.end(), permutation.begin(), B.column_indices.begin());

    // order COO matrix
    cusp::sort_by_row_and_column(B.row_indices, B.column_indices, B.values);

    // store permuted matrix
    A = B;
}

///////////////////////////
// View Member Functions //
///////////////////////////
//
template <typename Array, typename IndexType, typename MemorySpace>
void
permutation_matrix_view<Array,IndexType,MemorySpace>
::resize(const size_t num_rows)
{
    Parent::resize(num_rows, num_rows, num_rows);
    permutation.resize(num_rows);
}

template <typename Array, typename IndexType, typename MemorySpace>
template <typename MatrixType>
void
permutation_matrix_view<Array,IndexType,MemorySpace>
::symmetric_permute(MatrixType& A)
{
    typedef typename MatrixType::index_type IndexType2;
    typedef typename MatrixType::value_type ValueType2;

    cusp::coo_matrix<IndexType2,ValueType2,MemorySpace> B(A);

    // reorder rows and column according to permutation
    thrust::gather(B.row_indices.begin(), B.row_indices.end(), permutation.begin(), B.row_indices.begin());
    thrust::gather(B.column_indices.begin(), B.column_indices.end(), permutation.begin(), B.column_indices.begin());

    // order COO matrix
    cusp::sort_by_row_and_column(B.row_indices, B.column_indices, B.values);

    // store permuted matrix
    A = B;
}

} // end namespace cusp

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

#include <cusp/format_utils.h>

namespace cusp
{

// Forward definitions
template <typename T1, typename T2> void convert(const T1&, T2&);

//////////////////
// Constructors //
//////////////////

// construct from a different matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
csr_matrix<IndexType,ValueType,MemorySpace>
::csr_matrix(const MatrixType& matrix)
{
    cusp::convert(matrix, *this);
}

//////////////////////
// Member Functions //
//////////////////////

template <typename IndexType, typename ValueType, class MemorySpace>
void
csr_matrix<IndexType,ValueType,MemorySpace>
::resize(const size_t num_rows, const size_t num_cols, const size_t num_entries)
{
    Parent::resize(num_rows, num_cols, num_entries);
    row_offsets.resize(num_rows + 1);
    column_indices.resize(num_entries);
    values.resize(num_entries);
}

template <typename IndexType, typename ValueType, class MemorySpace>
void
csr_matrix<IndexType,ValueType,MemorySpace>
::swap(csr_matrix& matrix)
{
    Parent::swap(matrix);
    row_offsets.swap(matrix.row_offsets);
    column_indices.swap(matrix.column_indices);
    values.swap(matrix.values);
}

// assignment from another matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
csr_matrix<IndexType,ValueType,MemorySpace>&
csr_matrix<IndexType,ValueType,MemorySpace>
::operator=(const MatrixType& matrix)
{
    cusp::convert(matrix, *this);

    return *this;
}

///////////////////////
// View Constructors //
///////////////////////

///////////////////////////
// View Member Functions //
///////////////////////////

// construct from a different matrix
template <typename ArrayType1,typename ArrayType2,typename ArrayType3,
          typename IndexType, typename ValueType, typename MemorySpace>
void
csr_matrix_view<ArrayType1,ArrayType2,ArrayType3,IndexType,ValueType,MemorySpace>
::resize(const size_t num_rows, const size_t num_cols, const size_t num_entries)
{
    Parent::resize(num_rows, num_cols, num_entries);
    row_offsets.resize(num_rows + 1);
    column_indices.resize(num_entries);
    values.resize(num_entries);
}

} // end namespace cusp

#include <cusp/convert.h>


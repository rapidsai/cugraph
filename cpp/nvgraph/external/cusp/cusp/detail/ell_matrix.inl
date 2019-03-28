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

#include <cusp/array2d.h>
#include <cusp/convert.h>
#include <cusp/detail/utils.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////

template <typename IndexType, typename ValueType, class MemorySpace>
ell_matrix<IndexType,ValueType,MemorySpace>
::ell_matrix(const size_t num_rows, const size_t num_cols, const size_t num_entries,
             const size_t num_entries_per_row, const size_t alignment)
    : Parent(num_rows, num_cols, num_entries)
{
    // TODO use array2d constructor when it can accept pitch
    column_indices.resize(num_rows, num_entries_per_row, cusp::detail::round_up(num_rows, alignment));
    values.resize        (num_rows, num_entries_per_row, cusp::detail::round_up(num_rows, alignment));
}

// construct from a different matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
ell_matrix<IndexType,ValueType,MemorySpace>
::ell_matrix(const MatrixType& matrix)
{
    cusp::convert(matrix, *this);
}

//////////////////////
// Member Functions //
//////////////////////

template <typename IndexType, typename ValueType, class MemorySpace>
void
ell_matrix<IndexType,ValueType,MemorySpace>
::swap(ell_matrix& matrix)
{
    Parent::swap(matrix);
    column_indices.swap(matrix.column_indices);
    values.swap(matrix.values);
}

template <typename IndexType, typename ValueType, class MemorySpace>
void
ell_matrix<IndexType,ValueType,MemorySpace>
::resize(const size_t num_rows, const size_t num_cols, const size_t num_entries,
         const size_t num_entries_per_row)
{
    Parent::resize(num_rows, num_cols, num_entries);
    column_indices.resize(num_rows, num_entries_per_row);
    values.resize(num_rows, num_entries_per_row);
}

template <typename IndexType, typename ValueType, class MemorySpace>
void
ell_matrix<IndexType,ValueType,MemorySpace>
::resize(const size_t num_rows, const size_t num_cols, const size_t num_entries,
         const size_t num_entries_per_row, const size_t alignment)
{
    Parent::resize(num_rows, num_cols, num_entries);
    column_indices.resize(num_rows, num_entries_per_row, cusp::detail::round_up(num_rows, alignment));
    values.resize        (num_rows, num_entries_per_row, cusp::detail::round_up(num_rows, alignment));
}

// assignment from another matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
ell_matrix<IndexType,ValueType,MemorySpace>&
ell_matrix<IndexType,ValueType,MemorySpace>
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

template <typename Array1, typename Array2, typename IndexType, typename ValueType, class MemorySpace>
void
ell_matrix_view<Array1,Array2,IndexType,ValueType,MemorySpace>
::resize(const size_t num_rows, const size_t num_cols, const size_t num_entries,
         const size_t num_entries_per_row)
{
    Parent::resize(num_rows, num_cols, num_entries);
    column_indices.resize(num_rows, num_entries_per_row);
    values.resize(num_rows, num_entries_per_row);
}

template <typename Array1, typename Array2, typename IndexType, typename ValueType, class MemorySpace>
void
ell_matrix_view<Array1,Array2,IndexType,ValueType,MemorySpace>
::resize(const size_t num_rows, const size_t num_cols, const size_t num_entries,
         const size_t num_entries_per_row, const size_t alignment)
{
    Parent::resize(num_rows, num_cols, num_entries);
    column_indices.resize(num_rows, num_entries_per_row, cusp::detail::round_up(num_rows, alignment));
    values.resize        (num_rows, num_entries_per_row, cusp::detail::round_up(num_rows, alignment));
}

} // end namespace cusp


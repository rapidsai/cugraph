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

#include <cusp/convert.h>
#include <cusp/ell_matrix.h>
#include <cusp/coo_matrix.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////

// construct from another matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
hyb_matrix<IndexType,ValueType,MemorySpace>
::hyb_matrix(const MatrixType& matrix)
{
    cusp::convert(matrix, *this);
}

//////////////////////
// Member Functions //
//////////////////////

template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
hyb_matrix<IndexType,ValueType,MemorySpace>&
hyb_matrix<IndexType,ValueType,MemorySpace>
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

template <typename Matrix1, typename Matrix2, typename IndexType, typename ValueType, typename MemorySpace>
void
hyb_matrix_view<Matrix1,Matrix2,IndexType,ValueType,MemorySpace>
::resize(size_t num_rows, size_t num_cols,
         size_t num_ell_entries, size_t num_coo_entries,
         size_t num_entries_per_row, size_t alignment)
{
    Parent::resize(num_rows, num_cols, num_ell_entries + num_coo_entries);
    ell.resize(num_rows, num_cols, num_ell_entries, num_entries_per_row, alignment);
    coo.resize(num_rows, num_cols, num_coo_entries);
}

} // end namespace cusp


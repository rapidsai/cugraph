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

#include <algorithm>
#include <numeric>
#include <vector>

#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>

#include <thrust/count.h>

// TODO remove std::

namespace cusp
{
namespace detail
{
namespace host
{
namespace detail
{

template <typename Matrix>
size_t count_diagonals(const Matrix& csr, cusp::csr_format)
{
    typedef typename Matrix::index_type IndexType;

    cusp::array1d<bool,cusp::host_memory> occupied_diagonals(csr.num_rows + csr.num_cols, false);

    for(size_t i = 0; i < csr.num_rows; i++)
    {
        for(IndexType jj = csr.row_offsets[i]; jj < csr.row_offsets[i+1]; jj++) {
            IndexType j = csr.column_indices[jj];
            IndexType diagonal_offset = (csr.num_rows - i) + j; //offset shifted by + num_rows
            occupied_diagonals[diagonal_offset] = true;
        }
    }

    return thrust::count(occupied_diagonals.begin(), occupied_diagonals.end(), true);
}

template <typename Matrix>
size_t compute_max_entries_per_row(const Matrix& csr, cusp::csr_format)
{
    size_t max_entries_per_row = 0;
    for(size_t i = 0; i < csr.num_rows; i++)
        max_entries_per_row = std::max<size_t>(max_entries_per_row, csr.row_offsets[i+1] - csr.row_offsets[i]);
    return max_entries_per_row;
}

template <typename Matrix>
size_t compute_optimal_entries_per_row(const Matrix& csr,
                                       float relative_speed,
                                       size_t breakeven_threshold,
                                       cusp::csr_format)
{
    typedef typename Matrix::index_type IndexType;

    // compute maximum row length
    size_t max_cols_per_row = 0;
    for(size_t i = 0; i < csr.num_rows; i++)
        max_cols_per_row = std::max<size_t>(max_cols_per_row, csr.row_offsets[i+1] - csr.row_offsets[i]);

    // compute distribution of nnz per row
    std::vector<IndexType> histogram(max_cols_per_row + 1, 0);
    for(size_t i = 0; i < csr.num_rows; i++)
        histogram[csr.row_offsets[i+1] - csr.row_offsets[i]]++;

    // compute optimal ELL column size
    size_t num_cols_per_row = max_cols_per_row;
    for(size_t i = 0, rows = csr.num_rows; i < max_cols_per_row; i++)
    {
        rows -= histogram[i];  //number of rows of length > i
        if(relative_speed * rows < csr.num_rows || (size_t) rows < breakeven_threshold)
        {
            num_cols_per_row = i;
            break;
        }
    }

    return num_cols_per_row;
}

} // end namespace detail

template <typename Matrix>
size_t count_diagonals(const Matrix& m)
{
    return cusp::detail::host::detail::count_diagonals(m, typename Matrix::format());
}

template <typename Matrix>
size_t compute_max_entries_per_row(const Matrix& m)
{
    return cusp::detail::host::detail::compute_max_entries_per_row(m, typename Matrix::format());
}


////////////////////////////////////////////////////////////////////////////////
//! Compute Optimal Number of Columns per Row in the ELL part of the HYB format
//! Examines the distribution of nonzeros per row of the input CSR matrix to find
//! the optimal tradeoff between the ELL and COO portions of the hybrid (HYB)
//! sparse matrix format under the assumption that ELL performance is a fixed
//! multiple of COO performance.  Furthermore, since ELL performance is also
//! sensitive to the absolute number of rows (and COO is not), a threshold is
//! used to ensure that the ELL portion contains enough rows to be worthwhile.
//! The default values were chosen empirically for a GTX280.
//!
//! @param csr                  CSR matrix
//! @param relative_speed       Speed of ELL relative to COO (e.g. 2.0 -> ELL is twice as fast)
//! @param breakeven_threshold  Minimum threshold at which ELL is faster than COO
////////////////////////////////////////////////////////////////////////////////
template <typename Matrix>
size_t compute_optimal_entries_per_row(const Matrix& m,
                                       float relative_speed = 3.0f,
                                       size_t breakeven_threshold = 4096)
{
    return cusp::detail::host::detail::compute_optimal_entries_per_row
           (m, relative_speed, breakeven_threshold, typename Matrix::format());
}

} // end namespace host
} // end namespace detail
} // end namespace cusp


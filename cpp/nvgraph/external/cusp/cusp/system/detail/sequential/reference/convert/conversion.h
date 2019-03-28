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

#include <cusp/ell_matrix.h>
#include <cusp/exception.h>

#include <cusp/detail/host/conversion_utils.h>

#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <thrust/count.h>

namespace cusp
{
namespace detail
{
namespace host
{

/////////////////////
// COO Conversions //
/////////////////////

template <typename Matrix1, typename Matrix2>
void coo_to_csr(const Matrix1& src, Matrix2& dst)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    // compute number of non-zero entries per row of A
    thrust::fill(dst.row_offsets.begin(), dst.row_offsets.end(), IndexType(0));

    for (size_t n = 0; n < src.num_entries; n++)
        dst.row_offsets[src.row_indices[n]]++;

    // cumsum the num_entries per row to get dst.row_offsets[]
    IndexType cumsum = 0;
    for(size_t i = 0; i < src.num_rows; i++)
    {
        IndexType temp = dst.row_offsets[i];
        dst.row_offsets[i] = cumsum;
        cumsum += temp;
    }
    dst.row_offsets[src.num_rows] = cumsum;

    // write Aj,Ax into dst.column_indices,dst.values
    for(size_t n = 0; n < src.num_entries; n++)
    {
        IndexType row  = src.row_indices[n];
        IndexType dest = dst.row_offsets[row];

        dst.column_indices[dest] = src.column_indices[n];
        dst.values[dest]         = src.values[n];

        dst.row_offsets[row]++;
    }

    IndexType last = 0;
    for(size_t i = 0; i <= src.num_rows; i++)
    {
        IndexType temp = dst.row_offsets[i];
        dst.row_offsets[i]  = last;
        last   = temp;
    }

    //csr may contain duplicates
}

template <typename Matrix1, typename Matrix2>
void coo_to_array2d(const Matrix1& src, Matrix2& dst)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    dst.resize(src.num_rows, src.num_cols);

    thrust::fill(dst.values.begin(), dst.values.end(), ValueType(0));

    for(size_t n = 0; n < src.num_entries; n++)
        dst(src.row_indices[n], src.column_indices[n]) += src.values[n]; //sum duplicates
}

/////////////////////
// CSR Conversions //
/////////////////////

template <typename Matrix1, typename Matrix2>
void csr_to_coo(const Matrix1& src, Matrix2& dst)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    // TODO replace with offsets_to_indices
    for(size_t i = 0; i < src.num_rows; i++)
        for(IndexType jj = src.row_offsets[i]; jj < src.row_offsets[i + 1]; jj++)
            dst.row_indices[jj] = i;

    cusp::copy(src.column_indices, dst.column_indices);
    cusp::copy(src.values,         dst.values);
}

template <typename Matrix1, typename Matrix2>
void csr_to_dia(const Matrix1& src, Matrix2& dst,
                const size_t alignment = 32)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    // compute number of occupied diagonals and enumerate them
    size_t num_diagonals = 0;

    cusp::array1d<IndexType,cusp::host_memory> diag_map(src.num_rows + src.num_cols, 0);

    for(size_t i = 0; i < src.num_rows; i++)
    {
        for(IndexType jj = src.row_offsets[i]; jj < src.row_offsets[i+1]; jj++)
        {
            size_t j         = src.column_indices[jj];
            size_t map_index = (src.num_rows - i) + j; //offset shifted by + num_rows

            if(diag_map[map_index] == 0)
            {
                diag_map[map_index] = 1;
                num_diagonals++;
            }
        }
    }


    // allocate DIA structure
    dst.resize(src.num_rows, src.num_cols, src.num_entries, num_diagonals, alignment);

    // fill in diagonal_offsets array
    for(size_t n = 0, diag = 0; n < src.num_rows + src.num_cols; n++)
    {
        if(diag_map[n] == 1)
        {
            diag_map[n] = diag;
            dst.diagonal_offsets[diag] = (IndexType) n - (IndexType) src.num_rows;
            diag++;
        }
    }

    // fill in values array
    thrust::fill(dst.values.values.begin(), dst.values.values.end(), ValueType(0));

    for(size_t i = 0; i < src.num_rows; i++)
    {
        for(IndexType jj = src.row_offsets[i]; jj < src.row_offsets[i+1]; jj++)
        {
            size_t j = src.column_indices[jj];
            size_t map_index = (src.num_rows - i) + j; //offset shifted by + num_rows
            size_t diag = diag_map[map_index];

            dst.values(i, diag) = src.values[jj];
        }
    }
}


template <typename Matrix1, typename Matrix2>
void csr_to_hyb(const Matrix1& src, Matrix2& dst,
                const size_t num_entries_per_row,
                const size_t alignment = 32)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    // The ELL portion of the HYB matrix will have 'num_entries_per_row' columns.
    // Nonzero values that do not fit within the ELL structure are placed in the
    // COO format portion of the HYB matrix.

    // compute number of nonzeros in the ELL and COO portions
    size_t num_ell_entries = 0;
    for(size_t i = 0; i < src.num_rows; i++)
        num_ell_entries += thrust::min<size_t>(num_entries_per_row, src.row_offsets[i+1] - src.row_offsets[i]);

    IndexType num_coo_entries = src.num_entries - num_ell_entries;

    dst.resize(src.num_rows, src.num_cols,
               num_ell_entries, num_coo_entries,
               num_entries_per_row, alignment);

    const IndexType invalid_index = cusp::ell_matrix<IndexType, ValueType, cusp::host_memory>::invalid_index;

    // pad out ELL format with zeros
    thrust::fill(dst.ell.column_indices.values.begin(), dst.ell.column_indices.values.end(), invalid_index);
    thrust::fill(dst.ell.values.values.begin(),         dst.ell.values.values.end(),         ValueType(0));

    for(size_t i = 0, coo_nnz = 0; i < src.num_rows; i++)
    {
        size_t n = 0;
        IndexType jj = src.row_offsets[i];

        // copy up to num_cols_per_row values of row i into the ELL
        while(jj < src.row_offsets[i+1] && n < num_entries_per_row)
        {
            dst.ell.column_indices(i,n) = src.column_indices[jj];
            dst.ell.values(i,n)         = src.values[jj];
            jj++, n++;
        }

        // copy any remaining values in row i into the COO
        while(jj < src.row_offsets[i+1])
        {
            dst.coo.row_indices[coo_nnz]    = i;
            dst.coo.column_indices[coo_nnz] = src.column_indices[jj];
            dst.coo.values[coo_nnz]         = src.values[jj];
            jj++;
            coo_nnz++;
        }
    }
}


template <typename Matrix1, typename Matrix2>
void csr_to_ell(const Matrix1& src, Matrix2& dst,
                const size_t num_entries_per_row, const size_t alignment = 32)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    // compute number of nonzeros
    size_t num_entries = 0;
    for(size_t i = 0; i < src.num_rows; i++)
        num_entries += thrust::min<size_t>(num_entries_per_row, src.row_offsets[i+1] - src.row_offsets[i]);

    dst.resize(src.num_rows, src.num_cols, num_entries, num_entries_per_row, alignment);

    const IndexType invalid_index = cusp::ell_matrix<IndexType, ValueType, cusp::host_memory>::invalid_index;

    // pad out ELL format with zeros
    thrust::fill(dst.column_indices.values.begin(), dst.column_indices.values.end(), invalid_index);
    thrust::fill(dst.values.values.begin(),         dst.values.values.end(),         ValueType(0));

    for(size_t i = 0; i < src.num_rows; i++)
    {
        size_t n = 0;
        IndexType jj = src.row_offsets[i];

        // copy up to num_cols_per_row values of row i into the ELL
        while(jj < src.row_offsets[i+1] && n < num_entries_per_row)
        {
            dst.column_indices(i,n) = src.column_indices[jj];
            dst.values(i,n)         = src.values[jj];
            jj++, n++;
        }
    }
}


template <typename Matrix1, typename Matrix2>
void csr_to_array2d(const Matrix1& src, Matrix2& dst)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    dst.resize(src.num_rows, src.num_cols);

    thrust::fill(dst.values.begin(), dst.values.end(), ValueType(0));

    for(size_t i = 0; i < src.num_rows; i++)
        for(IndexType jj = src.row_offsets[i]; jj < src.row_offsets[i+1]; jj++)
            dst(i, src.column_indices[jj]) += src.values[jj]; //sum duplicates
}


/////////////////////
// DIA Conversions //
/////////////////////

template <typename Matrix1, typename Matrix2>
void dia_to_csr(const Matrix1& src, Matrix2& dst)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    size_t num_entries = 0;
    size_t num_diagonals = src.diagonal_offsets.size();

    // count nonzero entries
    for(size_t i = 0; i < src.num_rows; i++)
    {
        for(size_t n = 0; n < num_diagonals; n++)
        {
            const IndexType j = i + src.diagonal_offsets[n];

            if(j >= 0 && static_cast<size_t>(j) < src.num_cols && src.values(i,n) != ValueType(0))
                num_entries++;
        }
    }

    dst.resize(src.num_rows, src.num_cols, num_entries);

    num_entries = 0;
    dst.row_offsets[0] = 0;

    // copy nonzero entries to CSR structure
    for(size_t i = 0; i < src.num_rows; i++)
    {
        for(size_t n = 0; n < num_diagonals; n++)
        {
            const IndexType j = i + src.diagonal_offsets[n];

            if(j >= 0 && static_cast<size_t>(j) < src.num_cols)
            {
                const ValueType value = src.values(i, n);

                if (value != ValueType(0))
                {
                    dst.column_indices[num_entries] = j;
                    dst.values[num_entries] = value;
                    num_entries++;
                }
            }
        }

        dst.row_offsets[i + 1] = num_entries;
    }
}

/////////////////////
// ELL Conversions //
/////////////////////

template <typename Matrix1, typename Matrix2>
void ell_to_coo(const Matrix1& src, Matrix2& dst)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    const IndexType invalid_index = cusp::ell_matrix<IndexType, ValueType, cusp::host_memory>::invalid_index;

    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    size_t num_entries = 0;

    const size_t num_entries_per_row = src.column_indices.num_cols;

    for(size_t i = 0; i < src.num_rows; i++)
    {
        for(size_t n = 0; n < num_entries_per_row; n++)
        {
            const IndexType j = src.column_indices(i,n);
            const ValueType v = src.values(i,n);

            if(j != invalid_index)
            {
                dst.row_indices[num_entries]    = i;
                dst.column_indices[num_entries] = j;
                dst.values[num_entries]         = v;
                num_entries++;
            }
        }
    }
}

template <typename Matrix1, typename Matrix2>
void ell_to_csr(const Matrix1& src, Matrix2& dst)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    const IndexType invalid_index = cusp::ell_matrix<IndexType, ValueType, cusp::host_memory>::invalid_index;

    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    size_t num_entries = 0;
    dst.row_offsets[0] = 0;

    const size_t num_entries_per_row = src.column_indices.num_cols;

    for(size_t i = 0; i < src.num_rows; i++)
    {
        for(size_t n = 0; n < num_entries_per_row; n++)
        {
            const IndexType j = src.column_indices(i,n);
            const ValueType v = src.values(i,n);

            if(j != invalid_index)
            {
                dst.column_indices[num_entries] = j;
                dst.values[num_entries]         = v;
                num_entries++;
            }
        }

        dst.row_offsets[i + 1] = num_entries;
    }
}

/////////////////////
// HYB Conversions //
/////////////////////

template <typename Matrix1, typename Matrix2>
void hyb_to_coo(const Matrix1& src, Matrix2& dst)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    const IndexType invalid_index = cusp::ell_matrix<IndexType, ValueType, cusp::host_memory>::invalid_index;

    const size_t num_entries_per_row = src.ell.column_indices.num_cols;

    size_t num_entries  = 0;
    size_t coo_progress = 0;

    // merge each row of the ELL and COO parts into a single COO row
    for(size_t i = 0; i < src.num_rows; i++)
    {
        // append the i-th row from the ELL part
        for(size_t n = 0; n < num_entries_per_row; n++)
        {
            const IndexType j = src.ell.column_indices(i,n);
            const ValueType v = src.ell.values(i,n);

            if(j != invalid_index)
            {
                dst.row_indices[num_entries]    = i;
                dst.column_indices[num_entries] = j;
                dst.values[num_entries]         = v;
                num_entries++;
            }
        }

        // append the i-th row from the COO part
        while (coo_progress < src.coo.num_entries && static_cast<size_t>(src.coo.row_indices[coo_progress]) == i)
        {
            dst.row_indices[num_entries]    = i;
            dst.column_indices[num_entries] = src.coo.column_indices[coo_progress];
            dst.values[num_entries]         = src.coo.values[coo_progress];
            num_entries++;
            coo_progress++;
        }
    }
}

template <typename Matrix1, typename Matrix2>
void hyb_to_csr(const Matrix1& src, Matrix2& dst)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    const IndexType invalid_index = cusp::ell_matrix<IndexType, ValueType, cusp::host_memory>::invalid_index;

    const size_t num_entries_per_row = src.ell.column_indices.num_cols;

    size_t num_entries = 0;
    dst.row_offsets[0] = 0;

    size_t coo_progress = 0;

    // merge each row of the ELL and COO parts into a single CSR row
    for(size_t i = 0; i < src.num_rows; i++)
    {
        // append the i-th row from the ELL part
        for(size_t n = 0; n < num_entries_per_row; n++)
        {
            const IndexType j = src.ell.column_indices(i,n);
            const ValueType v = src.ell.values(i,n);

            if(j != invalid_index)
            {
                dst.column_indices[num_entries] = j;
                dst.values[num_entries]         = v;
                num_entries++;
            }
        }

        // append the i-th row from the COO part
        while (coo_progress < src.coo.num_entries && static_cast<size_t>(src.coo.row_indices[coo_progress]) == i)
        {
            dst.column_indices[num_entries] = src.coo.column_indices[coo_progress];
            dst.values[num_entries]         = src.coo.values[coo_progress];
            num_entries++;
            coo_progress++;
        }

        dst.row_offsets[i + 1] = num_entries;
    }
}

/////////////////
// Permutation //
/////////////////

template <typename Matrix1, typename Matrix2>
void permutation_to_csr(const Matrix1& src, Matrix2& dst)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    dst.row_offsets = cusp::counting_array<IndexType>(src.num_rows + 1);
    dst.column_indices = src.permutation;
    thrust::fill(dst.values.begin(), dst.values.end(), ValueType(1));
}

template <typename Matrix1, typename Matrix2>
void permutation_to_coo(const Matrix1& src, Matrix2& dst)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    dst.row_indices = cusp::counting_array<IndexType>(src.num_rows);
    dst.column_indices = src.permutation;
    thrust::fill(dst.values.begin(), dst.values.end(), ValueType(1));
}

/////////////////////////
// Array1d Conversions //
/////////////////////////
template <typename Matrix1, typename Matrix2>
void array2d_to_array1d(const Matrix1& src, Matrix2& dst)
{
    if (src.num_rows == 0 && src.num_cols == 0)
    {
        dst.resize(0);
    }
    else if (src.num_cols == 1)
    {
        dst.resize(src.num_rows);

        for (size_t i = 0; i < src.num_rows; i++)
            dst[i] = src(i,0);
    }
    else if (src.num_rows == 1)
    {
        dst.resize(src.num_cols);

        for (size_t j = 0; j < src.num_cols; j++)
            dst[j] = src(0,j);
    }
    else
    {
        throw cusp::format_conversion_exception("array2d to array1d conversion is only defined for row or column vectors");
    }
}

/////////////////////////
// Array2d Conversions //
/////////////////////////
template <typename Matrix1, typename Matrix2>
void array1d_to_array2d(const Matrix1& src, Matrix2& dst)
{
    dst.resize(src.size(),1);

    for (size_t i = 0; i < src.size(); i++)
        dst(i,0) = src[i];
}

template <typename Matrix1, typename Matrix2>
void array2d_to_coo(const Matrix1& src, Matrix2& dst)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    // count number of nonzero entries in array
    size_t nnz = 0;

    for(size_t i = 0; i < src.num_rows; i++)
    {
        for(size_t j = 0; j < src.num_cols; j++)
        {
            if (src(i,j) != ValueType(0))
                nnz++;
        }
    }

    dst.resize(src.num_rows, src.num_cols, nnz);

    nnz = 0;

    for(size_t i = 0; i < src.num_rows; i++)
    {
        for(size_t j = 0; j < src.num_cols; j++)
        {
            if (src(i,j) != ValueType(0))
            {
                dst.row_indices[nnz]    = i;
                dst.column_indices[nnz] = j;
                dst.values[nnz]         = src(i,j);
                nnz++;
            }
        }
    }
}

template <typename Matrix1, typename Matrix2>
void array2d_to_csr(const Matrix1& src, Matrix2& dst)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    IndexType nnz = src.num_entries - thrust::count(src.values.begin(), src.values.end(), ValueType(0));

    dst.resize(src.num_rows, src.num_cols, nnz);

    IndexType num_entries = 0;

    for(size_t i = 0; i < src.num_rows; i++)
    {
        dst.row_offsets[i] = num_entries;

        for(size_t j = 0; j < src.num_cols; j++)
        {
            if (src(i,j) != ValueType(0))
            {
                dst.column_indices[num_entries] = j;
                dst.values[num_entries]         = src(i,j);
                num_entries++;
            }
        }
    }

    dst.row_offsets[src.num_rows] = num_entries;
}

} // end namespace host
} // end namespace detail
} // end namespace cusp



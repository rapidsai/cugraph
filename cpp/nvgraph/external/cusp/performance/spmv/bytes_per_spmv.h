#pragma once

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

template <typename IndexType, typename ValueType>
size_t bytes_per_spmv(const cusp::dia_matrix<IndexType,ValueType,cusp::host_memory>& mtx)
{
    // note: this neglects diag_offsets, which is < 1% of other parts
    size_t bytes = 0;
    bytes += 2*sizeof(ValueType) * mtx.num_entries;  // A[i,j] and x[j]
    bytes += 2*sizeof(ValueType) * mtx.num_rows;     // y[i] = y[i] + ...
    return bytes;
}

template <typename IndexType, typename ValueType>
size_t bytes_per_spmv(const cusp::ell_matrix<IndexType,ValueType,cusp::host_memory>& mtx)
{
    size_t bytes = 0;
    bytes += 1*sizeof(ValueType) * mtx.num_rows * mtx.values.num_cols; // A[i,j] and padding
    bytes += 1*sizeof(IndexType) * mtx.num_entries;  // column index
    bytes += 1*sizeof(ValueType) * mtx.num_entries;  // x[j]
    bytes += 2*sizeof(ValueType) * mtx.num_rows;     // y[i] = y[i] + ...
    return bytes;
}

template <typename IndexType, typename ValueType>
size_t bytes_per_spmv(const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& mtx)
{
    size_t bytes = 0;
    bytes += 2*sizeof(IndexType) * mtx.num_rows;     // row pointer
    bytes += 1*sizeof(IndexType) * mtx.num_entries;  // column index
    bytes += 2*sizeof(ValueType) * mtx.num_entries;  // A[i,j] and x[j]
    bytes += 2*sizeof(ValueType) * mtx.num_rows;     // y[i] = y[i] + ...
    return bytes;
}

template <typename IndexType, typename ValueType>
size_t bytes_per_spmv_block(const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& mtx, size_t num_cols)
{
    size_t bytes = 0;
    bytes += 2*sizeof(IndexType) * mtx.num_rows;     // row pointer
    bytes += 1*sizeof(IndexType) * mtx.num_entries;  // column index
    bytes += (num_cols + 1)*sizeof(ValueType) * mtx.num_entries;  // A[i,j] and x[j]
    bytes += (num_cols + 1)*sizeof(ValueType) * mtx.num_rows;     // y[i] = y[i] + ...
    return bytes;
}

template <typename IndexType, typename ValueType>
size_t bytes_per_spmv(const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& mtx)
{
    size_t bytes = 0;
    bytes += 2*sizeof(IndexType) * mtx.num_entries; // row and column indices
    bytes += 2*sizeof(ValueType) * mtx.num_entries; // A[i,j] and x[j]

    std::vector<size_t> occupied_rows(mtx.num_rows, 0);
    for(size_t n = 0; n < mtx.num_entries; n++)
        occupied_rows[mtx.row_indices[n]] = 1;
    for(size_t n = 0; n < mtx.num_rows; n++)
        if(occupied_rows[n] == 1)
            bytes += 2*sizeof(ValueType);            // y[i] = y[i] + ...
    return bytes;
}

template <typename IndexType, typename ValueType>
size_t bytes_per_spmv(const cusp::hyb_matrix<IndexType,ValueType,cusp::host_memory>& mtx)
{
    return bytes_per_spmv(mtx.ell) + bytes_per_spmv(mtx.coo);
}


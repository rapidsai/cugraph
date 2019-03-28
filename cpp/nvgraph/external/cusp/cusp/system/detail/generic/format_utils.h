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

#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/functional.h>

#include <cusp/detail/format.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename Matrix, typename Array>
void extract_diagonal(thrust::execution_policy<DerivedPolicy> &exec,
                      const Matrix& A,
                      Array& output,
                      cusp::coo_format);

template <typename DerivedPolicy, typename Matrix, typename Array>
void extract_diagonal(thrust::execution_policy<DerivedPolicy> &exec,
                      const Matrix& A,
                      Array& output,
                      cusp::csr_format);

template <typename DerivedPolicy, typename Matrix, typename Array>
void extract_diagonal(thrust::execution_policy<DerivedPolicy> &exec,
                      const Matrix& A,
                      Array& output,
                      cusp::dia_format);

template <typename DerivedPolicy, typename Matrix, typename Array>
void extract_diagonal(thrust::execution_policy<DerivedPolicy> &exec,
                      const Matrix& A,
                      Array& output,
                      cusp::ell_format);

template <typename DerivedPolicy, typename Matrix, typename Array>
void extract_diagonal(thrust::execution_policy<DerivedPolicy> &exec,
                      const Matrix& A,
                      Array& output,
                      cusp::hyb_format);

template <typename DerivedPolicy, typename OffsetArray, typename IndexArray>
void offsets_to_indices(thrust::execution_policy<DerivedPolicy> &exec,
                        const OffsetArray& offsets, IndexArray& indices);

template <typename DerivedPolicy, typename IndexArray, typename OffsetArray>
void indices_to_offsets(thrust::execution_policy<DerivedPolicy> &exec,
                        const IndexArray& indices, OffsetArray& offsets);

template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2>
size_t count_diagonals(thrust::execution_policy<DerivedPolicy> &exec,
                       const size_t num_rows,
                       const size_t num_cols,
                       const ArrayType1& row_indices,
                       const ArrayType2& column_indices );

template <typename DerivedPolicy, typename ArrayType>
size_t compute_max_entries_per_row(thrust::execution_policy<DerivedPolicy> &exec,
                                   const ArrayType& row_offsets);

template <typename DerivedPolicy, typename ArrayType>
size_t compute_optimal_entries_per_row(thrust::execution_policy<DerivedPolicy> &exec,
                                       const ArrayType& row_offsets,
                                       float relative_speed,
                                       size_t breakeven_threshold);

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

#include <cusp/system/detail/generic/format_utils.inl>


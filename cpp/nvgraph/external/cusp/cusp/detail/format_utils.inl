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

#include <cusp/system/detail/adl/format_utils.h>
#include <cusp/system/detail/generic/format_utils.h>

namespace cusp
{

template <typename DerivedPolicy, typename Matrix, typename Array>
void extract_diagonal(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                      const Matrix& A, Array& output)
{
    using cusp::system::detail::generic::extract_diagonal;

    return extract_diagonal(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, output);
}

template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Matrix::memory_space System1;
    typedef typename Array::memory_space  System2;

    System1 system1;
    System2 system2;

    return cusp::extract_diagonal(select_system(system1,system2), A, output);
}

template <typename DerivedPolicy, typename OffsetArray, typename IndexArray>
void offsets_to_indices(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        const OffsetArray& offsets, IndexArray& indices)
{
    using cusp::system::detail::generic::offsets_to_indices;

    return offsets_to_indices(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), offsets, indices);
}

template <typename OffsetArray, typename IndexArray>
void offsets_to_indices(const OffsetArray& offsets, IndexArray& indices)
{
    using thrust::system::detail::generic::select_system;

    typedef typename IndexArray::memory_space  System1;
    typedef typename OffsetArray::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::offsets_to_indices(select_system(system1,system2), offsets, indices);
}

template <typename DerivedPolicy, typename IndexArray, typename OffsetArray>
void indices_to_offsets(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        const IndexArray& indices, OffsetArray& offsets)
{
    using cusp::system::detail::generic::indices_to_offsets;

    return indices_to_offsets(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), indices, offsets);
}

template <typename IndexArray, typename OffsetArray>
void indices_to_offsets(const IndexArray& indices, OffsetArray& offsets)
{
    using thrust::system::detail::generic::select_system;

    typedef typename IndexArray::memory_space  System1;
    typedef typename OffsetArray::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::indices_to_offsets(select_system(system1,system2), indices, offsets);
}

template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2>
size_t count_diagonals(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                       const size_t num_rows,
                       const size_t num_cols,
                       const ArrayType1& row_indices,
                       const ArrayType2& column_indices)
{
    using cusp::system::detail::generic::count_diagonals;

    return count_diagonals(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), num_rows, num_cols, row_indices, column_indices);
}

template <typename ArrayType1, typename ArrayType2>
size_t count_diagonals(const size_t num_rows,
                       const size_t num_cols,
                       const ArrayType1& row_indices,
                       const ArrayType2& column_indices)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType1::memory_space System1;
    typedef typename ArrayType2::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::count_diagonals(select_system(system1,system2), num_rows, num_cols, row_indices, column_indices);
}

template <typename DerivedPolicy, typename ArrayType>
size_t compute_max_entries_per_row(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                   const ArrayType& row_offsets)
{
    using cusp::system::detail::generic::compute_max_entries_per_row;

    return compute_max_entries_per_row(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), row_offsets);
}

template <typename ArrayType>
size_t compute_max_entries_per_row(const ArrayType& row_offsets)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType::memory_space System;

    System system;

    return cusp::compute_max_entries_per_row(select_system(system), row_offsets);
}

template <typename DerivedPolicy, typename ArrayType>
size_t compute_optimal_entries_per_row(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       const ArrayType& row_offsets,
                                       float relative_speed,
                                       size_t breakeven_threshold)
{
    using cusp::system::detail::generic::compute_optimal_entries_per_row;

    return compute_optimal_entries_per_row(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), row_offsets, relative_speed, breakeven_threshold);
}

template <typename ArrayType>
size_t compute_optimal_entries_per_row(const ArrayType& row_offsets,
                                       float relative_speed,
                                       size_t breakeven_threshold)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType::memory_space System;

    System system;

    return cusp::compute_optimal_entries_per_row(select_system(system), row_offsets, relative_speed, breakeven_threshold);
}

} // end namespace cusp


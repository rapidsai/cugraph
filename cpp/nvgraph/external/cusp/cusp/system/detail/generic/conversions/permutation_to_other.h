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

#include <cusp/array1d.h>

#include <cusp/detail/execution_policy.h>
#include <thrust/fill.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::permutation_format&,
        cusp::csr_format&)
{
    typedef typename DestinationType::index_type IndexType;
    typedef typename DestinationType::value_type ValueType;

    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    dst.row_offsets = cusp::counting_array<IndexType>(src.num_rows + 1);
    dst.column_indices = src.permutation;
    thrust::fill(exec, dst.values.begin(), dst.values.end(), ValueType(1));
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::permutation_format&,
        cusp::coo_format&)
{
    typedef typename DestinationType::index_type IndexType;
    typedef typename DestinationType::value_type ValueType;

    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    dst.row_indices = cusp::counting_array<IndexType>(src.num_rows);
    dst.column_indices = src.permutation;
    thrust::fill(exec, dst.values.begin(), dst.values.end(), ValueType(1));
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

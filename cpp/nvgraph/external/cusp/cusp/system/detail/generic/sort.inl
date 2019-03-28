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

#include <cusp/detail/format.h>
#include <cusp/detail/temporary_array.h>

#include <cusp/array1d.h>
#include <cusp/system/detail/adl/sort.h>
#include <cusp/system/detail/generic/sort.h>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row(thrust::execution_policy<DerivedPolicy> &exec,
                 ArrayType1& row_indices, ArrayType2& column_indices, ArrayType3& values,
                 typename ArrayType1::value_type min_row,
                 typename ArrayType1::value_type max_row)
{
    typedef typename ArrayType1::value_type IndexType;
    typedef typename ArrayType3::value_type ValueType;

    size_t N = row_indices.size();

    IndexType minr = min_row;
    IndexType maxr = max_row;

    if(max_row == 0)
        maxr = *thrust::max_element(exec, row_indices.begin(), row_indices.end());

    cusp::detail::temporary_array<IndexType, DerivedPolicy> permutation(exec, N);
    thrust::sequence(exec, permutation.begin(), permutation.end());

    // compute permutation that sorts the row_indices
    cusp::counting_sort_by_key(exec, row_indices, permutation, minr, maxr);

    // copy column_indices and values to temporary buffers
    cusp::detail::temporary_array<IndexType, DerivedPolicy> temp1(exec, column_indices);
    cusp::detail::temporary_array<ValueType, DerivedPolicy> temp2(exec, values);

    // use permutation to reorder the values
    thrust::gather(exec,
                   permutation.begin(), permutation.end(),
                   thrust::make_zip_iterator(thrust::make_tuple(temp1.begin(),   temp2.begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(column_indices.begin(), values.begin())));
}

template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row_and_column(thrust::execution_policy<DerivedPolicy> &exec,
                            ArrayType1& row_indices, ArrayType2& column_indices, ArrayType3& values,
                            typename ArrayType1::value_type min_row,
                            typename ArrayType1::value_type max_row,
                            typename ArrayType2::value_type min_col,
                            typename ArrayType2::value_type max_col)
{
    typedef typename ArrayType1::value_type IndexType1;
    typedef typename ArrayType2::value_type IndexType2;
    typedef typename ArrayType3::value_type ValueType;

    size_t N = row_indices.size();

    cusp::detail::temporary_array<IndexType1, DerivedPolicy> permutation(exec, N);
    thrust::sequence(exec, permutation.begin(), permutation.end());

    IndexType1 minr = min_row;
    IndexType1 maxr = max_row;
    IndexType2 minc = min_col;
    IndexType2 maxc = max_col;

    if(maxr == 0)
        maxr = *thrust::max_element(exec, row_indices.begin(), row_indices.end());
    if(maxc == 0)
        maxc = *thrust::max_element(exec, column_indices.begin(), column_indices.end());

    // compute permutation and sort by (I,J)
    {
        cusp::detail::temporary_array<IndexType1, DerivedPolicy> temp(exec, column_indices);
        cusp::counting_sort_by_key(exec, temp, permutation, minc, maxc);

        thrust::copy(exec, row_indices.begin(), row_indices.end(), temp.begin());
        thrust::gather(exec, permutation.begin(), permutation.end(), temp.begin(), row_indices.begin());
        cusp::counting_sort_by_key(exec, row_indices, permutation, minr, maxr);

        thrust::copy(exec, column_indices.begin(), column_indices.end(), temp.begin());
        thrust::gather(exec, permutation.begin(), permutation.end(), temp.begin(), column_indices.begin());
    }

    // use permutation to reorder the values
    {
        cusp::detail::temporary_array<ValueType, DerivedPolicy> temp(exec, values);
        thrust::gather(exec, permutation.begin(), permutation.end(), temp.begin(), values.begin());
    }
}

template <typename DerivedPolicy, typename ArrayType>
void counting_sort(thrust::execution_policy<DerivedPolicy>& exec,
                   ArrayType& keys, typename ArrayType::value_type min, typename ArrayType::value_type max)
{
    thrust::stable_sort(exec, keys.begin(), keys.end());
}

template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2>
void counting_sort_by_key(thrust::execution_policy<DerivedPolicy>& exec,
                          ArrayType1& keys, ArrayType2& vals,
                          typename ArrayType1::value_type min, typename ArrayType1::value_type max)
{
    thrust::stable_sort_by_key(exec, keys.begin(), keys.end(), vals.begin());
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp


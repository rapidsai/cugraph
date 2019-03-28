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
#include <cusp/convert.h>
#include <cusp/copy.h>
#include <cusp/csr_matrix.h>
#include <cusp/format_utils.h>
#include <cusp/functional.h>
#include <cusp/transpose.h>

#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{
namespace detail
{

using namespace thrust::placeholders;

template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2,
          typename MatrixType,
          typename ArrayType3>
void fit_candidates(thrust::execution_policy<DerivedPolicy> &exec,
                    const ArrayType1& aggregates,
                    const ArrayType2& B,
                    MatrixType& Q_,
                    ArrayType3& R)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    IndexType num_unaggregated = thrust::count(exec, aggregates.begin(), aggregates.end(), -1);
    IndexType num_aggregates   = *thrust::max_element(exec, aggregates.begin(), aggregates.end()) + 1;

    cusp::coo_matrix<IndexType,ValueType,MemorySpace> Q;
    Q.resize(aggregates.size(), num_aggregates, aggregates.size() - num_unaggregated);
    R.resize(num_aggregates);

    // gather values into Q
    thrust::copy_if(exec,
                    thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), aggregates.begin(), B.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(aggregates.size()), aggregates.end(), B.end())),
                    aggregates.begin(),
                    thrust::make_zip_iterator(thrust::make_tuple(Q.row_indices.begin(), Q.column_indices.begin(), Q.values.begin())),
                    _1 != -1);

    // compute norm over each aggregate
    {
        // compute Qt
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> Qt;
        cusp::transpose(exec, Q, Qt);

        // compute sum of squares for each column of Q (rows of Qt)
        cusp::detail::temporary_array<IndexType, DerivedPolicy> temp(exec, num_aggregates);

        thrust::transform(exec, Qt.values.begin(), Qt.values.end(), Qt.values.begin(), cusp::square_functor<ValueType>());

        thrust::reduce_by_key(exec,
                              Qt.row_indices.begin(), Qt.row_indices.end(),
                              Qt.values.begin(),
                              temp.begin(),
                              R.begin());

        // compute square root of each column sum
        thrust::transform(exec, R.begin(), R.end(), R.begin(), cusp::sqrt_functor<ValueType>());
    }

    // rescale columns of Q
    thrust::transform(exec,
                      Q.values.begin(), Q.values.end(),
                      thrust::make_permutation_iterator(R.begin(), Q.column_indices.begin()),
                      Q.values.begin(),
                      thrust::divides<ValueType>());

    // copy/convert Q to output matrix Q_
    cusp::convert(exec, Q, Q_);
}

} // end namepace detail
} // end namespace aggregation
} // end namespace precond
} // end namespace cusp


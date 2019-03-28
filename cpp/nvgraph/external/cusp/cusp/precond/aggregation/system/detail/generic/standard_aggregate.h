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
#include <cusp/copy.h>
#include <cusp/csr_matrix.h>

#include <cusp/precond/aggregation/system/detail/sequential/standard_aggregate.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{
namespace detail
{

template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType1,
          typename ArrayType2>
void standard_aggregate(thrust::execution_policy<DerivedPolicy> &exec,
                        const MatrixType& A,
                              ArrayType1& aggregates,
                              ArrayType2& roots)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> A_csr(A);
    cusp::array1d<IndexType, cusp::host_memory> aggregates_host(aggregates);
    cusp::array1d<IndexType, cusp::host_memory> roots_host(roots);

    cusp::precond::aggregation::standard_aggregate(A_csr, aggregates_host, roots_host);

    cusp::copy(aggregates_host, aggregates);
    cusp::copy(roots_host, roots);
}

} // end namespace detail
} // end namespace aggregation
} // end namespace precond
} // end namespace cusp


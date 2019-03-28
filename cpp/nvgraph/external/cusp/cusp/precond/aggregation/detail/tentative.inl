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


#include <cusp/detail/config.h>

#include <cusp/precond/aggregation/system/detail/generic/tentative.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{

template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2,
          typename MatrixType,
          typename ArrayType3>
void fit_candidates(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                    const ArrayType1& aggregates,
                    const ArrayType2& B,
                          MatrixType& Q,
                          ArrayType3& R)
{
    using cusp::precond::aggregation::detail::fit_candidates;

    return fit_candidates(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), aggregates, B, Q, R);
}

template <typename ArrayType1,
          typename ArrayType2,
          typename MatrixType,
          typename ArrayType3>
void fit_candidates(const ArrayType1& aggregates,
                    const ArrayType2& B,
                          MatrixType& Q,
                          ArrayType3& R)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType1::memory_space System1;
    typedef typename ArrayType2::memory_space System2;
    typedef typename ArrayType3::memory_space System3;
    typedef typename MatrixType::memory_space System4;

    System1 system1;
    System2 system2;
    System3 system3;
    System4 system4;

    return cusp::precond::aggregation::fit_candidates(select_system(system1,system2,system3,system4), aggregates, B, Q, R);
}

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp



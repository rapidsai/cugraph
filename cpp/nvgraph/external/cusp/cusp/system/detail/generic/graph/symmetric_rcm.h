/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#include <cusp/graph/pseudo_peripheral.h>

#include <cusp/detail/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template<typename DerivedPolicy,
         typename MatrixType,
         typename PermutationType>
void symmetric_rcm(thrust::execution_policy<DerivedPolicy>& exec,
                   const MatrixType& G,
                         PermutationType& P,
                   cusp::csr_format)
{
    typedef typename MatrixType::index_type IndexType;

    if(G.num_rows != G.num_cols)
        throw cusp::invalid_input_exception("matrix must be square");

    assert(P.num_rows == G.num_rows);

    // find peripheral vertex and return BFS levels from vertex
    cusp::graph::pseudo_peripheral_vertex(exec, G, P.permutation);

    // sort vertices by level in BFS traversal
    cusp::detail::temporary_array<IndexType,DerivedPolicy> levels(exec, G.num_rows);
    thrust::sequence(exec, levels.begin(), levels.end());
    thrust::sort_by_key(exec, P.permutation.begin(), P.permutation.end(), levels.begin());

    // form RCM permutation matrix
    thrust::scatter(exec,
                    thrust::counting_iterator<IndexType>(0),
                    thrust::counting_iterator<IndexType>(G.num_rows),
                    levels.begin(), P.permutation.begin());
}

template <typename DerivedPolicy,
          typename MatrixType,
          typename PermutationType>
void symmetric_rcm(thrust::execution_policy<DerivedPolicy>& exec,
                   const MatrixType& G,
                         PermutationType& P,
                   cusp::known_format)
{
    typedef typename cusp::detail::as_csr_type<MatrixType>::type CsrMatrix;

    CsrMatrix G_csr(G);

    cusp::graph::symmetric_rcm(exec, G_csr, P);
}

template <typename DerivedPolicy,
          typename MatrixType,
          typename PermutationType>
void symmetric_rcm(thrust::execution_policy<DerivedPolicy>& exec,
                   const MatrixType& G,
                         PermutationType& P)
{
    typedef typename MatrixType::format Format;

    Format format;

    symmetric_rcm(thrust::detail::derived_cast(exec), G, P, format);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp


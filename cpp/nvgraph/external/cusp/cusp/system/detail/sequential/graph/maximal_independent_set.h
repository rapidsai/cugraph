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

#include <cusp/detail/config.h>
#include <cusp/detail/format.h>
#include <cusp/detail/temporary_array.h>

#include <cusp/system/detail/sequential/execution_policy.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace sequential
{
namespace detail
{

template <typename MatrixType, typename IndexType, typename ArrayType>
void propagate_distances(const MatrixType& A,
                         const IndexType i,
                         const size_t d,
                         const size_t k,
                         ArrayType& distance)
{
    distance[i] = d;

    if (d < k)
    {
        for(IndexType jj = A.row_offsets[i]; jj < A.row_offsets[i + 1]; jj++)
        {
            IndexType j = A.column_indices[jj];

            // update only if necessary
            if (d + 1 < distance[j])
                propagate_distances(A, j, d + 1, k, distance);
        }
    }
}

} // end namespace detail

template <typename DerivedPolicy, typename MatrixType, typename ArrayType>
size_t maximal_independent_set(thrust::cpp::execution_policy<DerivedPolicy>& exec,
                               const MatrixType& G,
                               ArrayType& stencil,
                               const size_t k,
                               cusp::csr_format)
{
    typedef typename MatrixType::index_type IndexType;

    const IndexType N = G.num_rows;

    // distance to nearest MIS node
    cusp::detail::temporary_array<size_t, DerivedPolicy> distance(exec, N, k + 1);

    // count number of MIS nodes
    size_t set_nodes = 0;

    // pick MIS-k nodes greedily and deactivate all their k-neighbors
    for(IndexType i = 0; i < N; i++)
    {
        if (distance[i] > k)
        {
            set_nodes++;

            // reset distances on all k-ring neighbors
            detail::propagate_distances(G, i, 0, k, distance);
        }
    }

    // write output
    stencil.resize(N);

    for (IndexType i = 0; i < N; i++)
        stencil[i] = distance[i] == 0;

    return set_nodes;
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp


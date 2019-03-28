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

#include <stack>

#include <cusp/detail/config.h>
#include <cusp/detail/format.h>

#include <cusp/array1d.h>
#include <cusp/exception.h>

#include <cusp/system/detail/sequential/execution_policy.h>

#include <thrust/fill.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace sequential
{

template<typename DerivedPolicy, typename MatrixType, typename ArrayType>
size_t connected_components(thrust::cpp::execution_policy<DerivedPolicy>& exec,
                            const MatrixType& G,
                            ArrayType& components,
                            csr_format)
{
    typedef typename MatrixType::index_type VertexId;

    VertexId num_nodes = G.num_rows;

    thrust::fill(components.begin(), components.begin() + num_nodes, -1);
    std::stack<VertexId> DFS;
    VertexId component = 0;

    for(VertexId i = 0; i < num_nodes; i++)
    {
        if(components[i] == -1)
        {
            DFS.push(i);
            components[i] = component;

            while (!DFS.empty())
            {
                VertexId top = DFS.top();
                DFS.pop();

                for(VertexId jj = G.row_offsets[top]; jj < G.row_offsets[top + 1]; jj++) {
                    const VertexId j = G.column_indices[jj];
                    if(components[j] == -1) {
                        DFS.push(j);
                        components[j] = component;
                    }
                }
            }

            component++;
        }
    }

    return component;
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp

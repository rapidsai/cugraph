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

#include <cusp/array1d.h>
#include <cusp/exception.h>

#include <cusp/system/detail/sequential/execution_policy.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace sequential
{

template<typename DerivedPolicy, typename MatrixType, typename ArrayType>
size_t vertex_coloring(thrust::cpp::execution_policy<DerivedPolicy>& exec,
                       const MatrixType& G,
                       ArrayType& colors,
                       cusp::csr_format)
{
    typedef typename MatrixType::index_type IndexType;

    if(G.num_rows != G.num_cols)
        throw cusp::invalid_input_exception("matrix must be square");

    size_t max_color = 0;
    size_t N = G.num_rows;

    thrust::fill(exec, colors.begin(), colors.end(), N-1);

    cusp::detail::temporary_array<size_t, DerivedPolicy> mark(exec, N, std::numeric_limits<IndexType>::max());

    for(size_t vertex = 0; vertex < N; vertex++)
    {
        IndexType row_begin = G.row_offsets[vertex];
        IndexType row_end   = G.row_offsets[vertex + 1];

        for(IndexType offset = row_begin; offset < row_end; offset++)
        {
            IndexType neighbor = G.column_indices[offset];
            mark[colors[neighbor]] = vertex;
        }

        size_t vertex_color = 0;
        while(vertex_color < max_color && mark[vertex_color] == vertex)
            vertex_color++;

        if(vertex_color == max_color)
            max_color++;

        colors[vertex] = vertex_color;
    }

    return max_color;
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp


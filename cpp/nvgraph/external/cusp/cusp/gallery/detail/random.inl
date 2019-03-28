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

#include <cusp/coo_matrix.h>
#include <thrust/unique.h>
#include <thrust/sort.h>

#include <stdlib.h> // XXX remove when we switch RNGs

namespace cusp
{
namespace gallery
{

template <typename MatrixType>
void random(MatrixType& matrix,
            const size_t m,
            const size_t n,
            const size_t num_samples)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> coo(m, n, num_samples);

    srand(m ^ n ^ num_samples);

    for(size_t k = 0; k < num_samples; k++)
    {
        coo.row_indices[k]    = rand() % m;
        coo.column_indices[k] = rand() % n;
        coo.values[k]         = ValueType(1);
    }

    // sort indices by (row,column)
    coo.sort_by_row_and_column();

    size_t num_entries = thrust::unique(thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin())),
                                        thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.end(),   coo.column_indices.end())))
                         - thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin()));

    coo.resize(m, n, num_entries);

    matrix = coo;
}

} // end namespace gallery
} // end namespace cusp


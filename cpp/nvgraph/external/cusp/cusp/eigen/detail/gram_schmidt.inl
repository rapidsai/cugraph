/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

#include <cusp/array1d.h>
#include <cusp/blas/blas.h>

#include <limits>

namespace cusp
{
namespace eigen
{
namespace detail
{

template<typename Array2d, typename Array1d>
void modifiedGramSchmidt(const Array2d& Q, Array1d& v)
{
    typedef typename Array2d::value_type ValueType;

    for(size_t i = 0; i < Q.num_cols; i++)
    {
        ValueType dot_product = cusp::blas::dot(Q.column(i), v);
        cusp::blas::axpy(Q.column(i), v, -dot_product);
    }
}

template<typename Array2d, typename Array1d>
void modifiedGramSchmidt(const Array2d& Q, Array1d& v,
                         const cusp::array1d<bool,cusp::host_memory>& flag,
                         size_t num_cols = 0,
                         size_t num_iter = 1)
{
    typedef typename Array2d::value_type ValueType;

    if( num_cols == 0 ) num_cols = Q.num_cols;
    //size_t start = 0;
    size_t start = std::max(int(num_cols)-10, 0);

    for(size_t n = 0; n < num_iter; n++)
    {
        for(size_t i = start; i < num_cols; i++)
        {
            if(flag.empty() || flag[i])
            {
                ValueType dot_product = cusp::blas::dot(Q.column(i), v);
                cusp::blas::axpy(Q.column(i), v, -dot_product);
            }
        }
    }
}

template<typename ValueType, typename MemorySpace1, typename MemorySpace2>
void modifiedGramSchmidt(cusp::array2d<ValueType,MemorySpace1,cusp::column_major>& Q,
                         cusp::array2d<ValueType,MemorySpace2>& R)
{
    for(size_t i = 0; i < Q.num_cols; i++)
    {
        R(i,i) = cusp::blas::nrm2(Q.column(i));

        if(R(i,i) < std::numeric_limits<ValueType>::epsilon())
            break;

        cusp::blas::scal(Q.column(i), 1.0/R(i,i));

        for(size_t j = i+1; j < Q.num_cols; j++)
        {
            R(i,j) = cusp::blas::dot(Q.column(i), Q.column(j));
            cusp::blas::axpy(Q.column(i), Q.column(j), -R(i,j));
        }
    }
}

} // end namespace detail
} // end namespace eigen
} // end namespace cusp


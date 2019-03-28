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

/*! \file sor.inl
 *  \brief Inline file for sor.h
 */

#include <cusp/blas/blas.h>

namespace cusp
{
namespace relaxation
{

// linear_operator
template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
void sor<ValueType,MemorySpace>
::operator()(const MatrixType& A, const VectorType1& b, VectorType2& x)
{
    sor<ValueType,MemorySpace>::operator()(A,b,x,default_omega,gs.default_direction);
}

// override default sweep direction
template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
void sor<ValueType,MemorySpace>
::operator()(const MatrixType& A, const VectorType1& b, VectorType2& x, const ValueType omega, sweep direction)
{
    temp = x;
    gs(A, b, x, direction);
    cusp::blas::axpby(temp, x, x, ValueType(1)-omega, omega);
}

} // end namespace relaxation
} // end namespace cusp


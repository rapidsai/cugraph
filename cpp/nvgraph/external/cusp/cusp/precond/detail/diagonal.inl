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

/*! \file diagonal.inl
 *  \brief Inline file for diagonal.h
 */

#include <cusp/format_utils.h>
#include <cusp/functional.h>

#include <cusp/blas/blas.h>

#include <thrust/transform.h>

namespace cusp
{
namespace precond
{

// constructor
template <typename ValueType, typename MemorySpace>
template<typename MatrixType>
diagonal<ValueType,MemorySpace>
::diagonal(const MatrixType& A)
    : linear_operator<ValueType,MemorySpace>(A.num_rows, A.num_cols, A.num_rows),
      diagonal_reciprocals(A.num_rows)
{
    // extract the main diagonal
    cusp::extract_diagonal(A, diagonal_reciprocals);

    // invert the entries
    thrust::transform(diagonal_reciprocals.begin(), diagonal_reciprocals.end(),
                      diagonal_reciprocals.begin(), cusp::reciprocal_functor<ValueType>());
}

// linear operator
template <typename ValueType, typename MemorySpace>
template <typename VectorType1, typename VectorType2>
void diagonal<ValueType, MemorySpace>
::operator()(const VectorType1& x, VectorType2& y) const
{
    cusp::blas::xmy(diagonal_reciprocals, x, y);
}

} // end namespace precond
} // end namespace cusp


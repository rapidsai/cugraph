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

/*! \file smoothed_aggregation.h
 *  \brief Algebraic multigrid preconditoner based on smoothed aggregation.
 *
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>
#include <cusp/format_utils.h>
#include <cusp/graph/vertex_coloring.h>
#include <cusp/relaxation/gauss_seidel.h>

#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

namespace cusp
{
namespace precond
{

/*! \addtogroup preconditioners Preconditioners
 *  \ingroup preconditioners
 *  \{
 */

template <typename ValueType, typename MemorySpace>
class gauss_seidel_smoother
{
private:

    typedef cusp::relaxation::gauss_seidel<ValueType,MemorySpace> BaseSmoother;

public:
    size_t num_iters;
    BaseSmoother M;

    gauss_seidel_smoother(void) {}

    template <typename ValueType2, typename MemorySpace2>
    gauss_seidel_smoother(const gauss_seidel_smoother<ValueType2,MemorySpace2>& A)
      : num_iters(A.num_iters), M(A.M) {}

    template <typename MatrixType, typename Level>
    gauss_seidel_smoother(const MatrixType& A, const Level& L)
    {
        initialize(A, L);
    }

    template <typename MatrixType, typename Level>
    void initialize(const MatrixType& A, const Level& L)
    {
        num_iters = L.num_iters;
        M = BaseSmoother(A);
    }

    // smooths initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void presmooth(const MatrixType& A, const VectorType1& b, VectorType2& x)
    {
        for(size_t i = 0; i < num_iters; i++)
            M(A, b, x);
    }

    // smooths initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void postsmooth(const MatrixType& A, const VectorType1& b, VectorType2& x)
    {
        for(size_t i = 0; i < num_iters; i++)
            M(A, b, x);
    }
};
/*! \}
 */

} // end namespace precond
} // end namespace cusp


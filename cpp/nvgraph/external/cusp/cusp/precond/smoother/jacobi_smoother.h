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

#include <cusp/format_utils.h>
#include <cusp/eigen/spectral_radius.h>
#include <cusp/relaxation/jacobi.h>

#include <thrust/transform.h>

namespace cusp
{
namespace precond
{

template <typename ValueType>
struct jacobi_presmooth_functor
{
    ValueType omega;

    jacobi_presmooth_functor(ValueType omega) : omega(omega) {}

    __host__ __device__
    ValueType operator()(const ValueType& b, const ValueType& d) const
    {
        return omega * b / d;
    }
};

/*! \addtogroup preconditioners Preconditioners
 *  \ingroup preconditioners
 *  \{
 */

template <typename ValueType, typename MemorySpace>
class jacobi_smoother
{
private:

    typedef cusp::relaxation::jacobi<ValueType,MemorySpace> BaseSmoother;

public:
    size_t num_iters;
    BaseSmoother M;

    jacobi_smoother(void) {}

    template <typename ValueType2, typename MemorySpace2>
    jacobi_smoother(const jacobi_smoother<ValueType2,MemorySpace2>& A) : num_iters(A.num_iters), M(A.M) {}

    template <typename MatrixType, typename Level>
    jacobi_smoother(const MatrixType& A, const Level& L, double weight=4.0/3.0)
    {
        initialize(A, L, weight);
    }

    template <typename MatrixType, typename Level>
    void initialize(const MatrixType& A, const Level& L, double weight=4.0/3.0)
    {
        num_iters = L.num_iters;

        if(L.rho_DinvA == ValueType(0))
            M = BaseSmoother(A, weight / cusp::eigen::estimate_rho_Dinv_A(A));
        else
            M = BaseSmoother(A, weight / L.rho_DinvA);
    }

    // ignores initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void presmooth(const MatrixType& A, const VectorType1& b, VectorType2& x)
    {
        for(size_t i = 0; i < num_iters; i++)
          // x <- omega * D^-1 * b
          thrust::transform(b.begin(), b.end(), M.diagonal.begin(), x.begin(),
                            jacobi_presmooth_functor<ValueType>(M.default_omega));
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


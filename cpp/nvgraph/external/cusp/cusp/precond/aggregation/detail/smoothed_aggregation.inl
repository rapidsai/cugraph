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

#include <cusp/array1d.h>
#include <cusp/precond/aggregation/strength.h>
#include <cusp/precond/aggregation/aggregate.h>
#include <cusp/precond/aggregation/tentative.h>
#include <cusp/precond/aggregation/smooth_prolongator.h>
#include <cusp/precond/aggregation/restrict.h>
#include <cusp/precond/aggregation/galerkin_product.h>

#include <cusp/detail/temporary_array.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType, typename Format>
template <typename MatrixType>
smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType,Format>
::smoothed_aggregation(const MatrixType& A)
    : ML()
{
    initialize(A);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType, typename Format>
template <typename MatrixType, typename ArrayType>
smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType,Format>
::smoothed_aggregation(const MatrixType& A, const ArrayType& B)
    : ML()
{
    initialize(A, B);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType, typename Format>
template <typename MemorySpace2, typename SmootherType2, typename SolverType2, typename Format2>
smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType,Format>
::smoothed_aggregation(const smoothed_aggregation<IndexType,ValueType,MemorySpace2,SmootherType2,SolverType2,Format2>& M)
    : ML(M)
{
    for( size_t lvl = 0; lvl < M.sa_levels.size(); lvl++ )
        sa_levels.push_back(M.sa_levels[lvl]);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType, typename Format>
template <typename MatrixType>
void smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType,Format>
::initialize(const MatrixType& A)
{
    cusp::constant_array<ValueType> B(A.num_rows, 1);

    initialize(A, B);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType, typename Format>
template <typename MatrixType, typename ArrayType>
void smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType,Format>
::initialize(const MatrixType& A, const ArrayType& B)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType::memory_space System1;
    typedef typename ArrayType::memory_space  System2;

    System1 system1;
    System2 system2;

    initialize(select_system(system1,system2), A, B);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType, typename Format>
template <typename DerivedPolicy, typename MatrixType, typename ArrayType>
void smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType,Format>
::initialize(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
             const MatrixType& A, const ArrayType& B)
{
    typedef typename detail::select_sa_matrix_view<MatrixType>::type View;
    typedef typename ML::level Level;

    if(sa_levels.size() > 0)
    {
        sa_levels.resize(0);
        ML::levels.resize(0);
    }

    ML::resize(A.num_rows, A.num_cols, A.num_entries);
    ML::levels.reserve(ML::max_levels); // avoid reallocations which force matrix copies
    ML::levels.push_back(Level());

    sa_levels.push_back(sa_level<SetupMatrixType>());
    sa_levels.back().B = B;

    // Setup the first level using a COO view
    if(A.num_rows > ML::min_level_size)
    {
        View A_(A);
        extend_hierarchy(exec, A_);
        ML::setup_level(0, A, sa_levels[0]);
    }

    // Iteratively setup lower levels until stopping criteria are reached
    while ((sa_levels.back().A_.num_rows > ML::min_level_size) &&
            (sa_levels.size() < ML::max_levels))
        extend_hierarchy(exec, sa_levels.back().A_);

    // Setup multilevel arrays and matrices on each level
    for( size_t lvl = 1; lvl < sa_levels.size(); lvl++ )
        ML::setup_level(lvl, sa_levels[lvl].A_, sa_levels[lvl]);

    // Initialize coarse solver
    ML::initialize_coarse_solver();
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType, typename Format>
template <typename DerivedPolicy, typename MatrixType>
void smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType,Format>
::extend_hierarchy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   const MatrixType& A)
{
    typedef typename ML::level Level;

    {
        // compute stength of connection matrix
        SetupMatrixType C;
        strength_of_connection(exec, A, C, sa_levels.back());

        // compute aggregates
        sa_levels.back().aggregates.resize(A.num_rows, IndexType(0));
        sa_levels.back().roots.resize(A.num_rows);
        aggregate(exec, C, sa_levels.back().aggregates, sa_levels.back().roots);
    }

    SetupMatrixType P;
    cusp::array1d<ValueType, MemorySpace> B_coarse;

    // compute tenative prolongator and coarse nullspace vector
    fit_candidates(exec, sa_levels.back().aggregates, sa_levels.back().B, sa_levels.back().T, B_coarse);

    // compute prolongation operator
    smooth_prolongator(exec, A, sa_levels.back().T, P, sa_levels.back().rho_DinvA);  // TODO if C != A then compute rho_Dinv_C

    // compute restriction operator (transpose of prolongator)
    SetupMatrixType R;
    form_restriction(exec, P, R);

    // construct Galerkin product R*A*P
    SetupMatrixType RAP;
    galerkin_product(exec, R, A, P, RAP);

    // Setup components for next level in hierarchy
    sa_levels.push_back(sa_level<SetupMatrixType>());
    sa_levels.back().A_.swap(RAP);
    // sa_levels.back().B.swap(B_coarse);
    cusp::copy(exec, B_coarse, sa_levels.back().B);

    ML::copy_or_swap_matrix(ML::levels.back().R, R);
    ML::copy_or_swap_matrix(ML::levels.back().P, P);
    ML::levels.push_back(Level());
}

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp


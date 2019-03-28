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

/*! \file multilevel.h
 *  \brief Multilevel hierarchy
 *
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/lu.h>
#include <cusp/detail/type_traits.h>

#include <cusp/array1d.h>
#include <cusp/linear_operator.h>

#include <cusp/precond/smoother/jacobi_smoother.h>

#include <thrust/detail/use_default.h>

namespace cusp
{
namespace detail
{
  template <typename FormatType, typename MemorySpace>
  struct select_format_type
  {
    typedef typename thrust::detail::eval_if<
        thrust::detail::is_same<MemorySpace, cusp::host_memory>::value
      , thrust::detail::identity_<cusp::csr_format>
      , thrust::detail::identity_<cusp::hyb_format>
      >::type DefaultFormat;

    typedef typename thrust::detail::eval_if<
          thrust::detail::is_same<FormatType, thrust::use_default>::value
        , thrust::detail::identity_<DefaultFormat>
        , thrust::detail::identity_<FormatType>
      >::type type;
  };

  template <typename SmootherType, typename ValueType, typename MemorySpace>
  struct select_smoother_type
  {
    typedef cusp::precond::jacobi_smoother<ValueType,MemorySpace> JacobiSmoother;

    typedef typename thrust::detail::eval_if<
          thrust::detail::is_same<SmootherType, thrust::use_default>::value
        , thrust::detail::identity_<JacobiSmoother>
        , thrust::detail::identity_<SmootherType>
      >::type type;
  };

  template <typename SolverType, typename ValueType, typename MemorySpace>
  struct select_solver_type
  {
    typedef cusp::detail::lu_solver<ValueType,cusp::host_memory> LUSolver;

    typedef typename thrust::detail::eval_if<
          thrust::detail::is_same<SolverType, thrust::use_default>::value
        , thrust::detail::identity_<LUSolver>
        , thrust::detail::identity_<SolverType>
      >::type type;
  };
} // end detail namespace

/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup preconditioners Preconditioners
 *  \ingroup iterative_solvers
 *  \{
 */

/*! \p multilevel : multilevel hierarchy
 *
 *
 *  TODO
 */
template <typename IndexType,
          typename ValueType,
          typename MemorySpace,
          typename FormatType,
          typename SmootherType,
          typename SolverType>
class multilevel
: public cusp::linear_operator<ValueType,MemorySpace>
{
private:

    typedef typename detail::select_format_type<FormatType,MemorySpace>::type					        MatrixFormat;
    typedef typename detail::matrix_type<IndexType,ValueType,MemorySpace,MatrixFormat>::type	SolveMatrixType;
    typedef typename detail::select_smoother_type<SmootherType,ValueType,MemorySpace>::type		Smoother;
    typedef typename detail::select_solver_type<SolverType,ValueType,MemorySpace>::type			  Solver;

public:

	typedef cusp::multilevel<IndexType, ValueType, MemorySpace, MatrixFormat, Smoother, Solver>	container;

    /* \cond */
    struct level
    {
        SolveMatrixType R;  // restriction operator
        SolveMatrixType A;  // matrix
        SolveMatrixType P;  // prolongation operator
        cusp::array1d<ValueType,MemorySpace> x;               // per-level solution
        cusp::array1d<ValueType,MemorySpace> b;               // per-level rhs
        cusp::array1d<ValueType,MemorySpace> residual;        // per-level residual

        Smoother smoother;

        level(void) {}

        template<typename LevelType>
        level(const LevelType& level)
          : R(level.R), A(level.A), P(level.P),
            x(level.x), b(level.b), residual(level.residual),
            smoother(level.smoother) {}
    };
    /* \endcond */

    SolveMatrixType* A_ptr;

    size_t min_level_size;
    size_t max_levels;

    Solver solver;

    std::vector<level> levels;

    multilevel(size_t min_level_size=500, size_t max_levels=10) : A_ptr(NULL), min_level_size(min_level_size), max_levels(max_levels) {};

    template <typename MemorySpace2, typename Format2, typename SmootherType2, typename SolverType2>
    multilevel(const multilevel<IndexType,ValueType,MemorySpace2,Format2,SmootherType2,SolverType2>& M);

    template <typename Array1, typename Array2>
    void operator()(const Array1& x, Array2& y);

    template <typename Array1, typename Array2>
    void solve(const Array1& b, Array2& x);

    template <typename Array1, typename Array2, typename Monitor>
    void solve(const Array1& b, Array2& x, Monitor& monitor);

    void print( void );

    void set_min_level_size(size_t min_size);

    void set_max_levels(size_t max_depth);

    double operator_complexity( void );

    double grid_complexity( void );

protected:

    SolveMatrixType A;

    cusp::array1d<ValueType, MemorySpace> update;
    cusp::array1d<ValueType, MemorySpace> residual;
    cusp::array1d<ValueType, cusp::host_memory> temp_b;
    cusp::array1d<ValueType, cusp::host_memory> temp_x;

    template <typename Array1, typename Array2>
    void _solve(const Array1& b, Array2& x, const size_t i);

    template <typename MatrixType2, typename Level>
    void setup_level(const size_t lvl, const MatrixType2& A, const Level& L);

    template <typename Level>
    void set_multilevel_matrix(const SolveMatrixType& A, const Level& L);

    template <typename SolveMatrixType2, typename Level>
    void set_multilevel_matrix(const SolveMatrixType2& A, const Level& L);

    void copy_or_swap_matrix(SolveMatrixType& dst, SolveMatrixType& src);

    template <typename SolveMatrixType2>
    void copy_or_swap_matrix(SolveMatrixType& dst, SolveMatrixType2& src);

    void initialize_coarse_solver(void);
};
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/multilevel.inl>


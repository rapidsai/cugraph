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

#include <cusp/multiply.h>
#include <cusp/monitor.h>
#include <cusp/blas/blas.h>

namespace cusp
{

template <typename IndexType, typename ValueType, typename MemorySpace, typename Format, typename SmootherType, typename SolverType>
template <typename MemorySpace2, typename Format2, typename SmootherType2, typename SolverType2>
multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>
::multilevel(const multilevel<IndexType,ValueType,MemorySpace2,Format2,SmootherType2,SolverType2>& M)
    : min_level_size(M.min_level_size), max_levels(M.max_levels), solver(M.solver)
{
    for( size_t lvl = 0; lvl < M.levels.size(); lvl++ )
        levels.push_back(M.levels[lvl]);

    levels[0].A = *M.A_ptr;
    A_ptr = &levels[0].A;

    residual.resize(A_ptr->num_rows);
    update.resize(A_ptr->num_rows);
    temp_b.resize(levels.back().A.num_rows);
    temp_x.resize(levels.back().A.num_rows);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename Format, typename SmootherType, typename SolverType>
template <typename MatrixType2, typename Level>
void multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>
::setup_level(const size_t lvl, const MatrixType2& A, const Level& L)
{
    size_t N = A.num_rows;

    // Allocate arrays used during cycling
    levels[lvl].x.resize(N);
    levels[lvl].b.resize(N);
    levels[lvl].residual.resize(N);

    // Setup solve matrix for each level
    if(lvl == 0)
    {
        set_multilevel_matrix(A, L);
    }
    else
    {
        copy_or_swap_matrix(levels[lvl].A, const_cast<MatrixType2&>(A));

        // Initialize smoother for each level
        levels[lvl].smoother.initialize(levels[lvl].A, L);
    }
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename Format, typename SmootherType, typename SolverType>
void multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>
::copy_or_swap_matrix(SolveMatrixType& dst, SolveMatrixType& src)
{
    dst.swap(src);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename Format, typename SmootherType, typename SolverType>
template <typename SolveMatrixType2>
void multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>
::copy_or_swap_matrix(SolveMatrixType& dst, SolveMatrixType2& src)
{
    dst = src;
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename Format, typename SmootherType, typename SolverType>
template <typename SolveMatrixType2, typename Level>
void multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>
::set_multilevel_matrix(const SolveMatrixType2& A, const Level& L)
{
    this->A = A;
    A_ptr = &this->A;

    levels[0].smoother.initialize(this->A, L);

    residual.resize(A.num_rows);
    update.resize(A.num_rows);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename Format, typename SmootherType, typename SolverType>
template <typename Level>
void multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>
::set_multilevel_matrix(const SolveMatrixType& A, const Level& L)
{
    A_ptr = const_cast<SolveMatrixType*>(&A);

    levels[0].smoother.initialize(A, L);

    residual.resize(A.num_rows);
    update.resize(A.num_rows);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename Format, typename SmootherType, typename SolverType>
void multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>
::set_min_level_size(size_t min_size)
{
    min_level_size = min_size;
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename Format, typename SmootherType, typename SolverType>
void multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>
::set_max_levels(size_t max_depth)
{
    max_levels = max_depth;
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename Format, typename SmootherType, typename SolverType>
void multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>
::initialize_coarse_solver(void)
{
    temp_b.resize(levels.back().A.num_rows);
    temp_x.resize(levels.back().A.num_rows);

    solver = Solver(levels.back().A);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename Format, typename SmootherType, typename SolverType>
template <typename Array1, typename Array2>
void multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>
::operator()(const Array1& b, Array2& x)
{
    // perform 1 V-cycle
    _solve(b, x, 0);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename Format, typename SmootherType, typename SolverType>
template <typename Array1, typename Array2>
void multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>
::solve(const Array1& b, Array2& x)
{
    cusp::monitor<ValueType> monitor(b);

    solve(b, x, monitor);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename Format, typename SmootherType, typename SolverType>
template <typename Array1, typename Array2, typename Monitor>
void multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>
::solve(const Array1& b, Array2& x, Monitor& monitor)
{
    // use simple iteration
    // compute initial residual
    cusp::multiply(*A_ptr, x, residual);
    cusp::blas::axpby(b, residual, residual, ValueType(1.0), ValueType(-1.0));

    while(!monitor.finished(residual))
    {
        _solve(residual, update, 0);

        // x += M * r
        cusp::blas::axpy(update, x, ValueType(1.0));

        // update residual
        cusp::multiply(*A_ptr, x, residual);
        cusp::blas::axpby(b, residual, residual, ValueType(1.0), ValueType(-1.0));
        ++monitor;
    }
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename Format, typename SmootherType, typename SolverType>
template <typename Array1, typename Array2>
void multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>
::_solve(const Array1& b, Array2& x, const size_t i)
{
    if (i + 1 == levels.size())
    {
        // coarse grid solve
        // TODO streamline
        cusp::copy(b, temp_b);
        solver(temp_b, temp_x);
        cusp::copy(temp_x, x);
    }
    else
    {
        // initialize solution
        cusp::blas::fill(x, ValueType(0));

        // presmooth
        if(i == 0)
            levels[i].smoother.presmooth(*A_ptr, b, x);
        else
            levels[i].smoother.presmooth(levels[i].A, b, x);

        // compute residual <- b - A*x
        if(i == 0)
            cusp::multiply(*A_ptr, x, levels[i].residual);
        else
            cusp::multiply(levels[i].A, x, levels[i].residual);

        cusp::blas::axpby(b, levels[i].residual, levels[i].residual, ValueType(1.0), ValueType(-1.0));

        // restrict to coarse grid
        cusp::multiply(levels[i].R, levels[i].residual, levels[i + 1].b);

        // compute coarse grid solution
        _solve(levels[i + 1].b, levels[i + 1].x, i + 1);

        // apply coarse grid correction
        cusp::multiply(levels[i].P, levels[i + 1].x, levels[i].residual);
        cusp::blas::axpy(levels[i].residual, x, ValueType(1.0));

        // postsmooth
        if(i == 0)
            levels[i].smoother.postsmooth(*A_ptr, b, x);
        else
            levels[i].smoother.postsmooth(levels[i].A, b, x);
    }
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename Format, typename SmootherType, typename SolverType>
void multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>
::print( void )
{
    size_t num_levels = levels.size();
    double nnz = this->num_entries;

    std::cout << "\tNumber of Levels    :\t" << num_levels << std::endl;
    std::cout << "\tOperator Complexity :\t" << operator_complexity() << std::endl;
    std::cout << "\tGrid Complexity     :\t" << grid_complexity() << std::endl;
    std::cout << "\tlevel\tunknowns\tnonzeros" << std::endl;

    for(size_t index = 1; index < num_levels; index++)
        nnz += levels[index].A.num_entries;

    double percent = this->num_entries / nnz;

    std::cout << "\t" << 0 << "\t" << std::setw(8) << std::right << this->num_cols << "\t" \
              << std::setw(8) << std::right << this->num_entries << "  [" << 100*percent << "%]" \
              << std::endl;

    for(size_t index = 1; index < num_levels; index++)
    {
        percent = levels[index].A.num_entries / nnz;
        std::cout << "\t" << index << "\t" << std::setw(8) << std::right << levels[index].A.num_cols << "\t" \
                  << std::setw(8) << std::right << levels[index].A.num_entries << "  [" << 100*percent << "%]" \
                  << std::endl;
    }
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename Format, typename SmootherType, typename SolverType>
double multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>
::operator_complexity( void )
{
    size_t nnz = this->num_entries;

    for(size_t index = 1; index < levels.size(); index++)
        nnz += levels[index].A.num_entries;

    return (double) nnz / (double) this->num_entries;
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename Format, typename SmootherType, typename SolverType>
double multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>
::grid_complexity( void )
{
    size_t unknowns = this->num_rows;

    for(size_t index = 1; index < levels.size(); index++)
        unknowns += levels[index].A.num_rows;

    return (double) unknowns / (double) this->num_rows;
}

} // end namespace cusp



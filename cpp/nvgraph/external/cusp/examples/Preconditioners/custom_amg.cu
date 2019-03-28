#include <cusp/csr_matrix.h>

#include <cusp/gallery/diffusion.h>
#include <cusp/krylov/cg.h>
#include <cusp/precond/aggregation/smoothed_aggregation.h>

#include <thrust/execution_policy.h>

#include <iostream>

struct custom_amg_policy : cusp::cuda::execution_policy<custom_amg_policy> {};

// Use evolution strength measure
template <typename MatrixType1, typename MatrixType2, typename SALevelType>
void strength_of_connection(custom_amg_policy, const MatrixType1& A, MatrixType2& S, SALevelType& level)
{
    std::cout << "Calling my strength" << std::endl;
    cusp::precond::aggregation::evolution_strength_of_connection(A, S, level.B);
}

// Always use standard aggregation
template <typename MatrixType, typename ArrayType1, typename ArrayType2>
void aggregate(custom_amg_policy, const MatrixType& C, ArrayType1& aggregates, ArrayType2& roots)
{
    std::cout << "Calling my aggregation" << std::endl;
    cusp::precond::aggregation::standard_aggregate(C, aggregates, roots);
}

// Use default fit_candidates
template <typename ArrayType1, typename ArrayType2, typename MatrixType, typename ArrayType3>
void fit_candidates(custom_amg_policy, const ArrayType1& aggregates, const ArrayType2& B, MatrixType& Q, ArrayType3& R)
{
    std::cout << "Calling my tentative constructor" << std::endl;
    cusp::precond::aggregation::fit_candidates(aggregates, B, Q, R);
}

// Use default prolongator smoother
template <typename MatrixType1, typename MatrixType2, typename MatrixType3, typename ValueType>
void smooth_prolongator(custom_amg_policy, const MatrixType1& S, const MatrixType2& T, MatrixType3& P,
                        const ValueType rho_Dinv_S, const ValueType omega)
{
    std::cout << "Calling my tentative smoother" << std::endl;
    cusp::precond::aggregation::smooth_prolongator(S, T, P, rho_Dinv_S, omega);
}

// Use default Galerkin product to form coarse grid
template <typename MatrixType1, typename MatrixType2, typename MatrixType3>
void galerkin_product(custom_amg_policy, const MatrixType1& R, const MatrixType2& A, const MatrixType1& P, MatrixType3& RAP)
{
    std::cout << "Calling my Galerkin product\n" << std::endl;
    cusp::precond::aggregation::galerkin_product(R, A, P, RAP);
}

int main(void)
{
    typedef int                 IndexType;
    typedef float               ValueType;
    typedef cusp::device_memory MemorySpace;

    // create an empty sparse matrix structure
    cusp::coo_matrix<IndexType, ValueType, MemorySpace> A;

    // create 2D Poisson problem
    cusp::gallery::diffusion<cusp::gallery::FE>(A, 256, 256);

    cusp::array1d<ValueType, MemorySpace> x0(A.num_rows, 0);
    cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);
    cusp::monitor<ValueType> monitor(b, 1000, 1e-6);

    // solve with customized smoothed aggregation algebraic multigrid preconditioner
    std::cout << "\nSolving with customized smoothed aggregation preconditioner..." << std::endl;

    // allocate storage for solution (x)
    cusp::array1d<ValueType, MemorySpace> x(x0);

    // setup preconditioner
    cusp::precond::aggregation::smoothed_aggregation<IndexType, ValueType, MemorySpace> M;

    // instantiate instance of my custom AMG policy
    custom_amg_policy exec;

    // initialize construction using custom policy to control inner routines
    cusp::constant_array<ValueType> B(A.num_rows, ValueType(1));
    M.initialize(exec, A, B);

    // solve
    cusp::krylov::cg(A, x, b, monitor, M);

    // report status
    monitor.print();

    // print hierarchy information
    std::cout << "\nPreconditioner statistics" << std::endl;
    M.print();

    return 0;
}


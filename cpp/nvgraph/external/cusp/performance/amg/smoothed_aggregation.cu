#include <cusp/csr_matrix.h>
#include <cusp/gallery/diffusion.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>
#include <cusp/precond/aggregation/smoothed_aggregation.h>
#include <cusp/precond/smoother/gauss_seidel_smoother.h>
#include <cusp/precond/smoother/polynomial_smoother.h>

#include <iostream>

#include "../timer.h"

using namespace cusp::precond::aggregation;

template<typename MatrixType, typename Prec>
void run_amg(const MatrixType& A, Prec& M)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
    cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);

    // set stopping criteria (iteration_limit = 1000, relative_tolerance = 1e-10)
    cusp::monitor<ValueType> monitor(b, 1000, 1e-10);

    // solve
    timer t1;
    cusp::krylov::cg(A, x, b, monitor, M);
    std::cout << "solved system  in " << t1.milliseconds_elapsed() << " ms " << std::endl;

    // report status
    monitor.print();
}

int main(int argc, char ** argv)
{
    typedef int                 IndexType;
    typedef double              ValueType;
    typedef cusp::device_memory MemorySpace;

    // create an empty sparse matrix structure
    cusp::hyb_matrix<IndexType, ValueType, MemorySpace> A;

    size_t N = 1024;

    // create 2D Poisson problem
    cusp::gallery::poisson5pt(A, N, N);

    std::cout << "Constructed test matrix with shape ("  << A.num_rows << "," << A.num_cols << ") and "
              << A.num_entries << " entries" << std::endl;

    // solve without preconditioning
    {
        std::cout << "\nSolving with no preconditioner" << std::endl;

        // allocate storage for solution (x) and right hand side (b)
        cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
        cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);

        // set stopping criteria (iteration_limit = 10000, relative_tolerance = 1e-10)
        cusp::monitor<ValueType> monitor(b, 10000, 1e-10);

        // solve
        timer t0;
        cusp::krylov::cg(A, x, b, monitor);
        std::cout << "solved system  in " << t0.milliseconds_elapsed() << " ms " << std::endl;

        // report status
        monitor.print();
    }

    // solve with smoothed aggregation algebraic multigrid preconditioner and jacobi smoother
    {
        std::cout << "\nSolving with smoothed aggregation preconditioner and jacobi smoother" << std::endl;

        // setup preconditioner
        timer t0;
        cusp::precond::aggregation::smoothed_aggregation<IndexType, ValueType, MemorySpace> M(A);
        std::cout << "constructed hierarchy in " << t0.milliseconds_elapsed() << " ms " << std::endl;

        run_amg(A,M);
    }

    // solve with smoothed aggregation algebraic multigrid preconditioner and polynomial smoother
    {
        typedef cusp::precond::polynomial_smoother<ValueType,MemorySpace> Smoother;
        std::cout << "\nSolving with smoothed aggregation preconditioner and polynomial smoother" << std::endl;

        timer t0;
        cusp::precond::aggregation::smoothed_aggregation<IndexType, ValueType, MemorySpace, Smoother> M(A);
        std::cout << "constructed hierarchy in " << t0.milliseconds_elapsed() << " ms " << std::endl;

        run_amg(A,M);
    }

    // solve with smoothed aggregation algebraic multigrid preconditioner and gauss-seidel smoother
    {
        typedef cusp::precond::gauss_seidel_smoother<ValueType,MemorySpace> Smoother;
        std::cout << "\nSolving with smoothed aggregation preconditioner and gauss-seidel smoother" << std::endl;

        // Input matrix must be in CSR format for GS smoother
        cusp::csr_matrix<IndexType, ValueType, MemorySpace> A_csr(A);

        timer t0;
        cusp::precond::aggregation::smoothed_aggregation<IndexType, ValueType, MemorySpace, Smoother, thrust::use_default, cusp::csr_format> M(A_csr);
        std::cout << "constructed hierarchy in " << t0.milliseconds_elapsed() << " ms " << std::endl;

        run_amg(A_csr,M);
    }

    return 0;
}


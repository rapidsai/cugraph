#include <cusp/precond/diagonal.h>
#include <cusp/krylov/cg.h>
#include <cusp/csr_matrix.h>
#include <cusp/io/matrix_market.h>

#include <iostream>

int main(void)
{
    // where to perform the computation
    typedef cusp::device_memory MemorySpace;

    // which floating point type to use
    typedef float ValueType;

    // create an empty sparse matrix structure (HYB format)
    cusp::csr_matrix<int, ValueType, MemorySpace> A;

    // load a matrix stored in MatrixMarket format
    cusp::io::read_matrix_market_file(A, "A.mtx");

    // Note: A has poorly scaled rows & columns
    // set stopping criteria (iteration_limit = 100, relative_tolerance = 1e-6, absolute_tolerance = 0, verbose = true)
    cusp::array1d<ValueType, MemorySpace> x0(A.num_rows, 0);
    cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);
    cusp::monitor<ValueType> monitor(b, 100, 1e-6, 0, true);

    // solve without preconditioning
    {
        std::cout << "\nSolving with no preconditioner" << std::endl;

        // allocate storage for solution (x)
        cusp::array1d<ValueType, MemorySpace> x(x0);

        // solve
        cusp::krylov::cg(A, x, b, monitor);
    }

    // solve with diagonal preconditioner
    {
        std::cout << "\nSolving with diagonal preconditioner (M = D^-1)" << std::endl;

        // allocate storage for solution (x)
        cusp::array1d<ValueType, MemorySpace> x(x0);

        // reset the monitor
        monitor.reset(b);

        // setup preconditioner
        cusp::precond::diagonal<ValueType, MemorySpace> M(A);

        // solve
        cusp::krylov::cg(A, x, b, monitor, M);
    }

    return 0;
}


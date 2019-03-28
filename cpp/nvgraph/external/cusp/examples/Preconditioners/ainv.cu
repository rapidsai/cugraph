#include <cusp/csr_matrix.h>
#include <cusp/monitor.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>
#include <cusp/precond/ainv.h>

#include <iostream>

int main(void)
{
    typedef int                 IndexType;
    typedef float               ValueType;
    typedef cusp::device_memory MemorySpace;

    // create an empty sparse matrix structure
    cusp::coo_matrix<IndexType, ValueType, MemorySpace> A;

    // create 2D Poisson problem
    cusp::gallery::poisson5pt(A, 256, 256);

    // set stopping criteria (iteration_limit = 1000, relative_tolerance = 1e-6)
    cusp::array1d<ValueType, MemorySpace> x0(A.num_rows, 0);
    cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);
    cusp::monitor<ValueType> monitor(b);

    // solve without preconditioning
    {
        std::cout << "\nSolving with no preconditioner" << std::endl;

        // allocate storage for solution (x)
        cusp::array1d<ValueType, MemorySpace> x(x0);

        // solve
        cusp::krylov::cg(A, x, b, monitor);

        // report status
        monitor.print();
    }

    // solve AINV preconditioner, using standard drop tolerance strategy
    {
        std::cout << "\nSolving with scaled bridson preconditioner (drop tolerance .1)" << std::endl;

        // allocate storage for solution (x)
        cusp::array1d<ValueType, MemorySpace> x(x0);

        // reset the monitor
        monitor.reset(b);

        // setup preconditioner
        cusp::precond::scaled_bridson_ainv<ValueType, MemorySpace> M(A, .1);

        // solve
        cusp::krylov::cg(A, x, b, monitor, M);

        // report status
        monitor.print();
    }


    // solve AINV preconditioner, using static dropping strategy
    {
        std::cout << "\nSolving with scaled bridson preconditioner (10 nonzeroes per row)" << std::endl;

        // allocate storage for solution (x)
        cusp::array1d<ValueType, MemorySpace> x(x0);

        // reset the monitor
        monitor.reset(b);

        // setup preconditioner
        cusp::precond::scaled_bridson_ainv<ValueType, MemorySpace> M(A, 0, 10);

        // solve
        cusp::krylov::cg(A, x, b, monitor, M);

        // report status
        monitor.print();
    }


    // solve AINV preconditioner, using novel dropping strategy
    {
        std::cout << "\nSolving with AINV preconditioner (Lin strategy, p=2)" << std::endl;

        // allocate storage for solution (x)
        cusp::array1d<ValueType, MemorySpace> x(x0);

        // reset the monitor
        monitor.reset(b);

        // setup preconditioner
        cusp::precond::bridson_ainv<ValueType, MemorySpace> M(A, 0, -1, true, 2);

        // solve
        cusp::krylov::cg(A, x, b, monitor, M);

        // report status
        monitor.print();
    }

    return 0;
}


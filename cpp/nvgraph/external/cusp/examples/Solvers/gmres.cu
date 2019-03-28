#include <cusp/hyb_matrix.h>
#include <cusp/monitor.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/gmres.h>

// where to perform the computation
typedef cusp::device_memory MemorySpace;

// which floating point type to use
typedef float ValueType;

int main(void)
{
    // create an empty sparse matrix structure (HYB format)
    cusp::hyb_matrix<int, ValueType, MemorySpace> A;

    // create a 2d Poisson problem on a 10x10 mesh
    cusp::gallery::poisson5pt(A, 10, 10);

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<ValueType, MemorySpace> x(A.num_rows, ValueType(1));
    cusp::array1d<ValueType, MemorySpace> b(A.num_rows);

    cusp::multiply(A,x,b);

    // set initial guess
    thrust::fill( x.begin(), x.end(), ValueType(0) );

    // set stopping criteria:
    //  iteration_limit    = 100
    //  relative_tolerance = 1e-6
    cusp::monitor<ValueType> monitor(b, 100, 1e-6, 0, true);
    int restart = 50;
    // solve the linear system A * x = b with the GMRES
    cusp::krylov::gmres(A, x, b,restart, monitor);

    return 0;
}


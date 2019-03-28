#include <cusp/hyb_matrix.h>
#include <cusp/monitor.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>

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
    cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
    cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);

    // set stopping criteria:
    //  iteration_limit    = 100
    //  relative_tolerance = 1e-3
    //  absolute_tolerance = 0
    //  verbose            = true
    cusp::monitor<ValueType> monitor(b, 100, 1e-3, 0, true);

    // set preconditioner (identity)
    cusp::identity_operator<ValueType, MemorySpace> M(A.num_rows, A.num_rows);

    // solve the linear system A * x = b with the Conjugate Gradient method
    cusp::krylov::cg(A, x, b, monitor, M);

    return 0;
}


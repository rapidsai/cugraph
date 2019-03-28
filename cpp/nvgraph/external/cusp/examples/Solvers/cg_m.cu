#include <cusp/hyb_matrix.h>
#include <cusp/monitor.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg_m.h>

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
    size_t N_s = 4;
    cusp::array1d<ValueType, MemorySpace> x(A.num_rows*N_s, ValueType(0));  // TODO replace with array2d when cg_m supports it
    cusp::array1d<ValueType, MemorySpace> b(A.num_rows, ValueType(1));

    // set sigma values
    cusp::array1d<ValueType, MemorySpace> sigma(N_s);
    sigma[0] = ValueType(0.1);
    sigma[1] = ValueType(0.5);
    sigma[2] = ValueType(1.0);
    sigma[3] = ValueType(5.0);

    // set stopping criteria:
    //  iteration_limit    = 100
    //  relative_tolerance = 1e-6
    cusp::monitor<ValueType> monitor(b, 100, 1e-6, 0, true);

    // solve the linear systems (A + \sigma_i * I) * x = b for each 
    // sigma_i with the Conjugate Gradient method
    cusp::krylov::cg_m(A, x, b, sigma, monitor);

    return 0;
}


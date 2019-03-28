# CUSP : A Parallel Solver Library
# What is CUSP?
<blockquote>
Cusp is a library for sparse linear algebra and graph computations based on Thrust. Cusp provides a flexible, high-level interface for manipulating sparse matrices and solving sparse linear systems.
</blockquote>

# News
- Cusp v0.5.1 has been released with bug fixes. See [CHANGELOG](https://github.com/cusplibrary/cusplibrary/blob/master/CHANGELOG#L1) for release information.

- Cusp v0.5.0 has been released with support for CUDA 7.0. See [CHANGELOG](https://github.com/cusplibrary/cusplibrary/blob/master/CHANGELOG#L16) for release information.

- Cusp v0.4.0 has been released with support for CUDA 5.5. See [CHANGELOG](https://github.com/cusplibrary/cusplibrary/blob/master/CHANGELOG#L51) for release information.

- Cusp v0.3.0 has been released with support for CUDA 4.1. See [CHANGELOG](https://github.com/cusplibrary/cusplibrary/blob/master/CHANGELOG#L84) for release information.

- Cusp v0.2.0 has been released! See [CHANGELOG](https://github.com/cusplibrary/cusplibrary/blob/master/CHANGELOG#L112) for release information.

- Cusp v0.1.2 has been released! v0.1.2 contains compatibility fixes for Thrust v1.3.0.

- Cusp v0.1.1 has been released! v0.1.1 contains compatibility fixes for CUDA 3.1.

- Cusp v0.1.0 has been released!.

# Example
~~~{.cpp}
#include <cusp/hyb_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/krylov/cg.h>

int main(void)
{
    // create an empty sparse matrix structure (HYB format)
    cusp::hyb_matrix<int, float, cusp::device_memory> A;

    // load a matrix stored in MatrixMarket format
    cusp::io::read_matrix_market_file(A, "5pt_10x10.mtx");

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
    cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);

    // solve the linear system A * x = b with the Conjugate Gradient method
    cusp::krylov::cg(A, x, b);

    return 0;
}
~~~

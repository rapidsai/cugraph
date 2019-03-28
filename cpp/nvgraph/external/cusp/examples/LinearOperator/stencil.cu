#include <cusp/linear_operator.h>
#include <cusp/krylov/cg.h>

// This example shows how to use cusp::linear_operator to solve
// a linear system with a user-defined linear operator A.  The
// linear_operator is a way to interface custom sparse matrix
// formats or so-called "matrix-free" methods with the iterative
// solvers in Cusp.  In this example, we illustrate a matrix-free
// implementation of a simple 5-point finite-difference stencil,
//
//                [  0 -1  0 ]
//                [ -1  4 -1 ]
//                [  0 -1  0 ]
//
// using a CUDA kernel.  We combine the linear_operator with the
// Conjugate Gradient method to solve a 2D Poisson problem.


__global__
void stencil_kernel(int N, const float * x, float * y)
{
    // compute y = A*x, where A is the 5-point stencil
    // note: pre-caching a window of x into __shared__ memory
    // would make this a lot faster.

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < N && j < N)
    {
        // linear index into 2D grid
        int index = N * i + j;

        float result = 4.0f * x[index];         // center point

        if (i > 0    ) result -= x[index - N];  // lower neighbor
        if (i < N - 1) result -= x[index + N];  // upper neighbor
        if (j > 0    ) result -= x[index - 1];  // left neighbor
        if (j < N - 1) result -= x[index + 1];  // right neighbor

        // write result
        y[N * i + j] = result;
    }
}

class stencil : public cusp::linear_operator<float,cusp::device_memory>
{
public:
    typedef cusp::linear_operator<float,cusp::device_memory> super;

    int N;

    // constructor
    stencil(int N)
        : super(N*N,N*N), N(N) {}

    // linear operator y = A*x
    template <typename VectorType1,
             typename VectorType2>
    void operator()(const VectorType1& x, VectorType2& y) const
    {
        // obtain a raw pointer to device memory
        const float * x_ptr = thrust::raw_pointer_cast(&x[0]);
        float * y_ptr = thrust::raw_pointer_cast(&y[0]);

        dim3 dimBlock(16,16);
        dim3 dimGrid((N + 15) / 16, (N + 15) / 16);

        stencil_kernel<<<dimGrid,dimBlock>>>(N, x_ptr, y_ptr);
    }
};


int main(void)
{
    // number of grid points in each dimension
    const int N = 10;

    // create a matrix-free linear operator
    stencil A(N);

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
    cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);

    // set stopping criteria:
    //  iteration_limit    = 100
    //  relative_tolerance = 1e-6
    cusp::monitor<float> monitor(b, 100, 1e-5, 0, false);

    // solve the linear system A * x = b with the Conjugate Gradient method
    cusp::krylov::cg(A, x, b, monitor);

    return 0;
}


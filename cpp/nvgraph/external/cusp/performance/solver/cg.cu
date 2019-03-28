#define CUSP_USE_TEXTURE_MEMORY

#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>
#include <cusp/krylov/cg.h>

#include <iostream>

#include "../timer.h"

template <typename Matrix>
void benchmark_matrix(const Matrix& A)
{
    typedef typename Matrix::memory_space MemorySpace;
    typedef typename Matrix::value_type   ValueType;

    const size_t N = A.num_rows;

    cusp::array1d<ValueType, MemorySpace> x(N,0);
    cusp::array1d<ValueType, MemorySpace> b(N,1);

    cusp::monitor<ValueType> monitor(b, 2000, 1e-5);

    // time solver
    timer t;

    cusp::krylov::cg(A, x, b, monitor);

    float time = t.seconds_elapsed();

    cudaThreadSynchronize();

    if (monitor.converged())
        std::cout << "  Successfully converged";
    else
        std::cout << "  Failed to converge";
    std::cout << " after " << monitor.iteration_count() << " iterations." << std::endl;

    std::cout << "  Solver time " << time << " seconds (" << (1e3 * time / monitor.iteration_count()) << "ms per iteration)" << std::endl;
}


int main(int argc, char** argv)
{
    typedef int    IndexType;
    typedef double ValueType;

    typedef cusp::hyb_matrix<IndexType,ValueType,cusp::host_memory>   HostMatrix;
    typedef cusp::hyb_matrix<IndexType,ValueType,cusp::device_memory> DeviceMatrix;

    HostMatrix A;

    if (argc == 1)
    {
        std::cout << "Using default matrix (5-pt Laplacian stencil)" << std::endl;
        cusp::gallery::poisson5pt(A, 1000, 1000);
    }
    else
    {
        std::cout << "Reading matrix from file: " << argv[1] << std::endl;
        cusp::io::read_matrix_market_file(A, std::string(argv[1]));
    }

    std::cout << "Running solver on host..." << std::endl;
    benchmark_matrix(A);

    std::cout << "Running solver on device..." << std::endl;
    benchmark_matrix(DeviceMatrix(A));

    return 0;
}


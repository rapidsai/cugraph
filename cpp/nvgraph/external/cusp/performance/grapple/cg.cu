#include <cusp/csr_matrix.h>
#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>
#include <cusp/krylov/cg.h>

#include <iostream>

#include "my_execution_policy.h"

int main(int argc, char** argv)
{
    typedef int                 IndexType;
    typedef double              ValueType;
    typedef cusp::device_memory MemorySpace;

    cusp::csr_matrix<IndexType,ValueType,MemorySpace> A;

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

    size_t N = A.num_rows;

    cusp::array1d<ValueType, MemorySpace> x(N,0);
    cusp::array1d<ValueType, MemorySpace> b(N,1);
    cusp::monitor<ValueType> monitor(b, 2, 1e-5);
    cusp::identity_operator<ValueType, MemorySpace> M(N, N);

    my_policy exec;
    cusp::krylov::cg(exec, A, x, b, monitor, M);

    return 0;
}


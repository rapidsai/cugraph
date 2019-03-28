#include <cusp/csr_matrix.h>
#include <cusp/monitor.h>
#include <cusp/eigen/lanczos.h>
#include <cusp/gallery/poisson.h>

int main(void)
{
    // create an empty sparse matrix structure (CSR format)
    cusp::csr_matrix<int, float, cusp::device_memory> A;
    // initialize matrix
    cusp::gallery::poisson5pt(A, 1024, 1024);
    // allocate storage and initialize eigenpairs
    cusp::array1d<float, cusp::device_memory> S(5,0);
    cusp::array2d<float, cusp::device_memory, cusp::column_major> V;

    // Compute the largest eigenpair of A
    cusp::eigen::lanczos_options<float> options;

    options.tol = 1e-6;
    options.maxIter = 100;
    options.verbose = true;
    options.computeEigVecs = false;
    options.reorth = cusp::eigen::Full;

    cusp::eigen::lanczos(A, S, V, options);
    std::cout << "Largest eigenvalue : " << S[4] << std::endl;
    return 0;
}

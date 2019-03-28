#include <cusp/csr_matrix.h>
#include <cusp/monitor.h>

#include <cusp/eigen/lobpcg.h>
#include <cusp/gallery/poisson.h>
#include <cusp/precond/aggregation/smoothed_aggregation.h>

int main(void)
{
    // create an empty sparse matrix structure (CSR format)
    cusp::csr_matrix<int, double, cusp::device_memory> A;

    // initialize matrix
    cusp::gallery::poisson5pt(A, 1024, 1024);

    // allocate storage and initialize eigenpairs
    cusp::random_array<double> randx(A.num_rows);
    cusp::array1d<double, cusp::device_memory> X(randx);
    cusp::array1d<double, cusp::device_memory> S(1,0);

    // set stopping criteria:
    //  iteration_limit    = 100
    //  relative_tolerance = 1e-6
    //  absolute_tolerance = 0
    //  verbose            = true
    cusp::monitor<double> monitor(X, 10, 1e-6, 0, true);

    // set preconditioner (identity)
    cusp::precond::aggregation::smoothed_aggregation<int, double, cusp::device_memory> M(A);

    // Compute the largest eigenpair of A
    cusp::eigen::lobpcg(A, S, X, monitor, M, true);
    std::cout << "Largest eigenvalue : " << S[0] << std::endl;

    return 0;
}


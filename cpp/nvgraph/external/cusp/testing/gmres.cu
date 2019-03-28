#include <unittest/unittest.h>

#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>

#include <cusp/gallery/poisson.h>
#include <cusp/krylov/gmres.h>

template <class LinearOperator, class VectorType1, class VectorType2, class Monitor, class Preconditioner>
void gmres(my_system& system, const LinearOperator& A, VectorType1& x, const VectorType2& b, const size_t restart, Monitor& monitor, Preconditioner& M)
{
    system.validate_dispatch();
    return;
}

void TestGeneralizedMinResDispatch()
{
    // initialize testing variables
    size_t restart = 20;
    cusp::csr_matrix<int, float, cusp::device_memory> A;
    cusp::gallery::poisson5pt(A, 10, 10);
    cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0.0f);
    cusp::monitor<float> monitor(x, 20, 1e-4);
    cusp::identity_operator<float,cusp::device_memory> M(A.num_rows, A.num_cols);

    {
        my_system sys(0);

        // call gmres with explicit dispatching
        cusp::krylov::gmres(sys, A, x, x, restart, monitor, M);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }
}
DECLARE_UNITTEST(TestGeneralizedMinResDispatch);

template <class MemorySpace>
void TestGeneralizedMinRes(void)
{
    size_t restart = 20;

    cusp::csr_matrix<int, float, MemorySpace> A;

    cusp::gallery::poisson5pt(A, 10, 10);

    cusp::array1d<float, MemorySpace> x(A.num_rows, 0.0f);
    cusp::array1d<float, MemorySpace> b(A.num_rows, 1.0f);

    cusp::monitor<float> monitor(b, 20, 1e-4);

    cusp::krylov::gmres(A, x, b, restart, monitor);

    // check residual norm
    cusp::array1d<float, MemorySpace> residual(A.num_rows, 0.0f);
    cusp::multiply(A, x, residual);
    cusp::blas::axpby(residual, b, residual, -1.0f, 1.0f);

    ASSERT_EQUAL(cusp::blas::nrm2(residual) < 1e-4 * cusp::blas::nrm2(b), true);
}
DECLARE_HOST_DEVICE_UNITTEST(TestGeneralizedMinRes);


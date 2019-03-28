#include <unittest/unittest.h>

#include <cusp/csr_matrix.h>

#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg_m.h>
#include <cusp/krylov/bicgstab_m.h>

template <class LinearOperator,
          class VectorType1,
          class VectorType2,
          class VectorType3,
          class Monitor>
void cg_m(my_system& system,
          const LinearOperator& A,
                VectorType1& x,
          const VectorType2& b,
          const VectorType3& sigma,
                Monitor& monitor)
{
    system.validate_dispatch();
    return;
}

void TestConjugateGradientMDispatch()
{
    // initialize testing variables
    cusp::csr_matrix<int, float, cusp::device_memory> A;
    cusp::gallery::poisson5pt(A, 10, 10);
    cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0.0f);
    cusp::monitor<float> monitor(x, 20, 1e-4);

    my_system sys(0);

    // call with explicit dispatching
    cusp::krylov::cg_m(sys, A, x, x, x, monitor);

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestConjugateGradientMDispatch);

template <class LinearOperator, class VectorType1, class VectorType2, class VectorType3>
void check_residuals(LinearOperator& A, VectorType1& xs, VectorType2& b, VectorType3& sigma)
{
    typedef typename LinearOperator::value_type   ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    size_t N = A.num_rows;

    for (size_t i = 0; i < sigma.size(); i++)
    {
        // compute residual = b - (A + \sigma * I) x
        ValueType s = sigma[i];

        cusp::array1d<ValueType, MemorySpace> residual(A.num_rows, 0.0f);

        // TODO replace this with a array1d view of a array2d
        cusp::array1d<ValueType, MemorySpace> x(xs.begin() + i * N, xs.begin() + (i + 1) * N);
        cusp::multiply(A, x, residual);
        cusp::blas::axpby(residual, x, residual,  1.0f,     s);
        cusp::blas::axpby(residual, b, residual, -1.0f,  1.0f);

        ASSERT_EQUAL(cusp::blas::nrm2(residual) < 1e-4 * cusp::blas::nrm2(b), true);

        //std::cout << "Residual for sigma = " << s << " is " << cusp::blas::nrm2(residual) << std::endl;
    }
} // end check_residuals

template <class MemorySpace>
void TestConjugateGradientM(void)
{
    // which floating point type to use
    typedef float ValueType;

    // create an empty sparse matrix structure (HYB format)
    cusp::csr_matrix<int, ValueType, MemorySpace> A;

    // create a 2d Poisson problem on a 10x10 mesh
    cusp::gallery::poisson5pt(A, 10, 10);

    // allocate storage for solution (x) and right hand side (b)
    size_t N_s = 4;
    cusp::array1d<ValueType, MemorySpace> x(A.num_rows*N_s, ValueType(0));
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
    cusp::monitor<ValueType> monitor(b, 100, 1e-6);

    // solve the linear systems (A + \sigma_i * I) * x = b for each
    // sigma_i with the Conjugate Gradient method
    cusp::krylov::cg_m(A, x, b, sigma, monitor);

    check_residuals(A, x, b, sigma);
}
DECLARE_HOST_DEVICE_UNITTEST(TestConjugateGradientM);

template <class LinearOperator, class VectorType1, class VectorType2, class VectorType3>
void bicgstab_m(my_system& system, LinearOperator& A, VectorType1& x, VectorType2& b, VectorType3& sigma)
{
    system.validate_dispatch();
    return;
}

template <class LinearOperator, class VectorType1, class VectorType2, class VectorType3, class Monitor>
void bicgstab_m(my_system& system, LinearOperator& A, VectorType1& x, VectorType2& b, VectorType3& sigma, Monitor& monitor)
{
    system.validate_dispatch();
    return;
}

void TestBiConjugateGradientStabilizedMDispatch()
{
    // initialize testing variables
    cusp::csr_matrix<int, float, cusp::device_memory> A;
    cusp::gallery::poisson5pt(A, 10, 10);
    cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0.0f);
    cusp::monitor<float> monitor(x, 20, 1e-4);

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::krylov::bicgstab_m(sys, A, x, x, x);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::krylov::bicgstab_m(sys, A, x, x, x, monitor);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }
}
DECLARE_UNITTEST(TestBiConjugateGradientStabilizedMDispatch);


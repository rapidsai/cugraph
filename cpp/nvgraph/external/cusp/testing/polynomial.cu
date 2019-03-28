#include <unittest/unittest.h>

#include <cusp/relaxation/polynomial.h>
#include <cusp/blas/blas.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/multiply.h>

#include <thrust/sequence.h>

template <typename Matrix>
void TestPolynomialRelaxation(void)
{
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space Space;

    cusp::array2d<ValueType, Space> M(5,5);
    M(0,0) =  2.0;
    M(0,1) = -1.0;
    M(0,2) =  0.0;
    M(0,3) =  0.0;
    M(0,4) =  0.0;
    M(1,0) = -1.0;
    M(1,1) =  2.0;
    M(1,2) = -1.0;
    M(1,3) =  0.0;
    M(1,4) =  0.0;
    M(2,0) =  0.0;
    M(2,1) = -1.0;
    M(2,2) =  2.0;
    M(2,3) = -1.0;
    M(2,4) =  0.0;
    M(3,0) =  0.0;
    M(3,1) =  0.0;
    M(3,2) = -1.0;
    M(3,3) =  2.0;
    M(3,4) = -1.0;
    M(4,0) =  0.0;
    M(4,1) =  0.0;
    M(4,2) =  0.0;
    M(4,3) = -1.0;
    M(4,4) =  2.0;

    cusp::array1d<ValueType, Space> b(5,  0.0);
    cusp::array1d<ValueType, Space> x0(5);
    x0[0] = 0.0;
    x0[1] = 1.0;
    x0[2] = 2.0;
    x0[3] = 3.0;
    x0[4] = 4.0;

    Matrix A(M);
    cusp::array1d<ValueType, Space> residual(A.num_rows);

    // compute residual <- b - A*x
    cusp::multiply(A, x0, residual);
    cusp::blas::axpby(b, residual, residual, ValueType(1), ValueType(-1));

    {
        cusp::array1d<ValueType, Space> x(x0);
        cusp::array1d<ValueType, Space> coef(1,-1.0/3.0);
        cusp::relaxation::polynomial<ValueType, Space> relax(A, coef);
        cusp::array1d<ValueType, Space> expected(5);
        cusp::blas::axpby(x0, residual, expected, ValueType(1), ValueType(-1.0/3.0));

        relax(A, b, x, coef);

        ASSERT_ALMOST_EQUAL(x, expected);
    }

    {
        cusp::array1d<ValueType, Space> coef(3);
        coef[0] = -0.14285714;
        coef[1] = 1.0;
        coef[2] = -2.0;
        cusp::relaxation::polynomial<ValueType, Space> relax(A, coef);

        cusp::array1d<ValueType, Space> Ar(5);
        cusp::multiply(A, residual, Ar);

        cusp::array1d<ValueType, Space> A2r(5);
        cusp::multiply(A, Ar, A2r);

        cusp::array1d<ValueType, Space> expected(5);
        cusp::blas::axpby(x0, A2r, expected, ValueType(1), ValueType(-0.14285714));
        cusp::blas::axpby(expected, Ar, expected, ValueType(1), ValueType(1));
        cusp::blas::axpby(expected, residual, expected, ValueType(1), ValueType(-2));

        cusp::array1d<ValueType, Space> x(x0);
        relax(A, b, x, coef);

        ASSERT_ALMOST_EQUAL(x, expected);
    }

}
DECLARE_SPARSE_MATRIX_UNITTEST(TestPolynomialRelaxation);


void TestChebyshevCoefficients(void)
{
    cusp::array1d<double,cusp::host_memory> coef;
    cusp::relaxation::detail::chebyshev_polynomial_coefficients(1.0,coef,1.0,2.0);

    cusp::array1d<double,cusp::host_memory> expected(4);
    expected[0] = -0.32323232;
    expected[1] = 1.45454545;
    expected[2] = -2.12121212;
    expected[3] = 1.0;

    ASSERT_ALMOST_EQUAL(coef, expected);
}
DECLARE_UNITTEST(TestChebyshevCoefficients);


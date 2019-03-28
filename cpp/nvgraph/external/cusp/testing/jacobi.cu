#include <unittest/unittest.h>

#include <cusp/relaxation/jacobi.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

template <typename Matrix>
void TestJacobiRelaxation(void)
{
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space Space;

    cusp::array2d<ValueType, Space> M(5,5);
    M(0,0) = 1.0;
    M(0,1) = 1.0;
    M(0,2) = 2.0;
    M(0,3) = 0.0;
    M(0,4) = 0.0;
    M(1,0) = 3.0;
    M(1,1) = 2.0;
    M(1,2) = 0.0;
    M(1,3) = 0.0;
    M(1,4) = 5.0;
    M(2,0) = 0.0;
    M(2,1) = 0.0;
    M(2,2) = 0.5;
    M(2,3) = 0.0;
    M(2,4) = 0.0;
    M(3,0) = 0.0;
    M(3,1) = 6.0;
    M(3,2) = 7.0;
    M(3,3) = 4.0;
    M(3,4) = 0.0;
    M(4,0) = 0.0;
    M(4,1) = 8.0;
    M(4,2) = 0.0;
    M(4,3) = 0.0;
    M(4,4) = 8.0;

    cusp::array1d<ValueType, Space> b(5,  5.0);
    cusp::array1d<ValueType, Space> x(5, -1.0);
    cusp::array1d<ValueType, Space> expected(5);
    expected[0] =  8.000;  // (5 + 1 + 2) / 1   = 8
    expected[1] =  6.500;  // (5 + 3 + 5) / 2   = 6.5
    expected[2] = 10.000;  // (5 + 0    ) / 0.5 = 10
    expected[3] =  4.500;  // (5 + 6 + 7) / 4   = 4.5
    expected[4] =  1.625;  // (5 + 8    ) / 8   = 1.625

    Matrix A(M);
    cusp::relaxation::jacobi<ValueType, Space> relax(A);

    relax(A, b, x);

    ASSERT_ALMOST_EQUAL(x, expected);
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestJacobiRelaxation);


template <typename Matrix>
void TestJacobiRelaxationWithWeighting(void)
{
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space Space;

    cusp::array2d<ValueType, Space> M(2,2);
    M(0,0) = 2.0;
    M(0,1) = 1.0;
    M(1,0) = 1.0;
    M(1,1) = 3.0;

    Matrix A(M);

    // use default omega
    {
        cusp::array1d<ValueType, Space> b(2,  5.0);
        cusp::array1d<ValueType, Space> x(2, -1.0);
        cusp::relaxation::jacobi<ValueType, Space> relax(A, 0.5);
        relax(A, b, x);
        ASSERT_ALMOST_EQUAL(ValueType(x[0]), ValueType(1.0));
        ASSERT_ALMOST_EQUAL(ValueType(x[1]), ValueType(0.5));
    }

    // override default omega
    {
        cusp::array1d<ValueType, Space> b(2,  5.0);
        cusp::array1d<ValueType, Space> x(2, -1.0);
        cusp::relaxation::jacobi<ValueType, Space> relax(A, 1.0);
        relax(A, b, x, 0.5);
        ASSERT_ALMOST_EQUAL(ValueType(x[0]), ValueType(1.0));
        ASSERT_ALMOST_EQUAL(ValueType(x[1]), ValueType(0.5));
    }
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestJacobiRelaxationWithWeighting);


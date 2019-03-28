#include <unittest/unittest.h>

#include <cusp/relaxation/gauss_seidel.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

template <typename Space>
void TestGaussSeidelRelaxation(void)
{
    typedef cusp::csr_matrix<int,float,Space> Matrix;

    cusp::array2d<float, Space> M(5,5);
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

    cusp::array1d<float, Space> b(5,  5.0);
    cusp::array1d<float, Space> x(5, -1.0);
    cusp::array1d<float, Space> expected(5);
    expected[0] = -1.4375;
    expected[1] = -13.5625;
    expected[2] =  10.000;
    expected[3] =  4.09375;
    expected[4] =  14.1875;


    Matrix A(M);
    cusp::relaxation::gauss_seidel<float, Space> relax(A);

    relax(A, b, x);

    ASSERT_ALMOST_EQUAL(x, expected);
}
DECLARE_HOST_DEVICE_UNITTEST(TestGaussSeidelRelaxation);


template <typename Space>
void TestGaussSeidelRelaxationSweeps(void)
{
    typedef cusp::csr_matrix<int,float,Space> Matrix;

    cusp::array2d<float, Space> M(2,2);
    M(0,0) = 2.0;
    M(0,1) = 1.0;
    M(1,0) = 1.0;
    M(1,1) = 3.0;

    Matrix A(M);

    // use default omega
    {
        cusp::array1d<float, Space> b(2,  5.0);
        cusp::array1d<float, Space> x(2, -1.0);
        cusp::relaxation::gauss_seidel<float, Space> relax(A);
        relax(A, b, x, cusp::relaxation::FORWARD);
        ASSERT_ALMOST_EQUAL(x[0], 3.0);
        ASSERT_ALMOST_EQUAL(x[1], 0.666667);
    }

    // override default omega
    {
        cusp::array1d<float, Space> b(2,  5.0);
        cusp::array1d<float, Space> x(2, -1.0);
        cusp::relaxation::gauss_seidel<float, Space> relax(A);
        relax(A, b, x, cusp::relaxation::BACKWARD);
        ASSERT_ALMOST_EQUAL(x[0], 1.5);
        ASSERT_ALMOST_EQUAL(x[1], 2.0);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestGaussSeidelRelaxationSweeps);


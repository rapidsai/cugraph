#include <unittest/unittest.h>

#include <cusp/blas/blas.h>
#include <cusp/gallery/diffusion.h>

void TestDiffusionFE(void)
{
    cusp::dia_matrix<int, double, cusp::host_memory> matrix;
    cusp::gallery::diffusion<cusp::gallery::FE>(matrix, 2, 2, 0.0001, M_PI/6.0);

    // convert result to array2d
    cusp::array2d<double, cusp::host_memory> R(matrix);
    cusp::array2d<double, cusp::host_memory> E(4,4);

    E(0,0) =  1.33347;
    E(0,1) =  0.0832917;
    E(0,2) = -0.416658;
    E(0,3) = -0.383168;
    E(1,0) =  0.0832917;
    E(1,1) =  1.33347;
    E(1,2) = 0.0498014;
    E(1,3) = -0.416658;
    E(2,0) = -0.416658;
    E(2,1) =  0.0498014;
    E(2,2) = 1.33347;
    E(2,3) = 0.0832917;
    E(3,0) = -0.383168;
    E(3,1) = -0.416658;
    E(3,2) = 0.0832917;
    E(3,3) = 1.33347;

    // TODO Replace with a matrix norm
    cusp::blas::axpy(R.values,E.values,-1.0);
    ASSERT_EQUAL(cusp::blas::nrmmax(E.values) < 1e-5, true);
}
DECLARE_UNITTEST(TestDiffusionFE);

void TestDiffusionFD(void)
{
    cusp::dia_matrix<int, double, cusp::host_memory> matrix;
    cusp::gallery::diffusion<cusp::gallery::FD>(matrix, 2, 2, 0.0001, M_PI/6.0);

    // convert result to array2d
    cusp::array2d<double, cusp::host_memory> R(matrix);
    cusp::array2d<double, cusp::host_memory> E(4,4);

    E(0,0) =  2.0002;
    E(0,1) = -0.250075;
    E(0,2) = -0.750025;
    E(0,3) = -0.2164847;
    E(1,0) = -0.250075;
    E(1,1) =  2.0002;
    E(1,2) =  0.2164847;
    E(1,3) = -0.750025;
    E(2,0) = -0.750025;
    E(2,1) =  0.2164847;
    E(2,2) =  2.0002;
    E(2,3) = -0.250075;
    E(3,0) = -0.2164847;
    E(3,1) = -0.750025;
    E(3,2) = -0.250075;
    E(3,3) =  2.0002;

    // TODO Replace with a matrix norm
    cusp::blas::axpy(R.values,E.values,-1.0);
    ASSERT_EQUAL(cusp::blas::nrmmax(E.values) < 1e-5, true);
}
DECLARE_UNITTEST(TestDiffusionFD);

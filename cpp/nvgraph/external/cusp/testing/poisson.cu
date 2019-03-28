#include <unittest/unittest.h>

#include <cusp/print.h>
#include <cusp/gallery/poisson.h>

void TestPoisson5pt(void)
{
    // grid is 2x3
    cusp::dia_matrix<int, float, cusp::host_memory> matrix;
    cusp::gallery::poisson5pt(matrix, 2, 3);

    // convert result to array2d
    cusp::array2d<float, cusp::host_memory> R(matrix);
    cusp::array2d<float, cusp::host_memory> E(6,6);

    E(0,0) =  4; E(0,1) = -1; E(0,2) = -1; E(0,3) =  0; E(0,4) =  0; E(0,5) =  0;
    E(1,0) = -1; E(1,1) =  4; E(1,2) =  0; E(1,3) = -1; E(1,4) =  0; E(1,5) =  0;
    E(2,0) = -1; E(2,1) =  0; E(2,2) =  4; E(2,3) = -1; E(2,4) = -1; E(2,5) =  0;
    E(3,0) =  0; E(3,1) = -1; E(3,2) = -1; E(3,3) =  4; E(3,4) =  0; E(3,5) = -1;
    E(4,0) =  0; E(4,1) =  0; E(4,2) = -1; E(4,3) =  0; E(4,4) =  4; E(4,5) = -1;
    E(5,0) =  0; E(5,1) =  0; E(5,2) =  0; E(5,3) = -1; E(5,4) = -1; E(5,5) =  4;

    ASSERT_EQUAL_QUIET(R, E);
}
DECLARE_UNITTEST(TestPoisson5pt);

void TestPoisson9pt(void)
{
    // grid is 2x3
    cusp::dia_matrix<int, float, cusp::host_memory> matrix;
    cusp::gallery::poisson9pt(matrix, 2, 3);

    // convert result to array2d
    cusp::array2d<float, cusp::host_memory> R(matrix);
    cusp::array2d<float, cusp::host_memory> E(6,6);

    E(0,0) =  8; E(0,1) = -1; E(0,2) = -1; E(0,3) = -1; E(0,4) =  0; E(0,5) =  0;
    E(1,0) = -1; E(1,1) =  8; E(1,2) = -1; E(1,3) = -1; E(1,4) =  0; E(1,5) =  0;
    E(2,0) = -1; E(2,1) = -1; E(2,2) =  8; E(2,3) = -1; E(2,4) = -1; E(2,5) = -1;
    E(3,0) = -1; E(3,1) = -1; E(3,2) = -1; E(3,3) =  8; E(3,4) = -1; E(3,5) = -1;
    E(4,0) =  0; E(4,1) =  0; E(4,2) = -1; E(4,3) = -1; E(4,4) =  8; E(4,5) = -1;
    E(5,0) =  0; E(5,1) =  0; E(5,2) = -1; E(5,3) = -1; E(5,4) = -1; E(5,5) =  8;

    ASSERT_EQUAL_QUIET(R, E);
}
DECLARE_UNITTEST(TestPoisson9pt);

void TestPoisson7pt(void)
{
    // grid is 3x3x3
    cusp::dia_matrix<int, float, cusp::host_memory> matrix;
    cusp::gallery::poisson7pt(matrix, 2, 2, 2);

    // convert result to array2d
    cusp::array2d<float, cusp::host_memory> R(matrix);
    cusp::array2d<float, cusp::host_memory> E(8,8);

    E(0,0) =  6; E(0,1) = -1; E(0,2) = -1; E(0,3) =  0; E(0,4) = -1; E(0,5) =  0; E(0,6) =  0; E(0,7) =  0;
    E(1,0) = -1; E(1,1) =  6; E(1,2) =  0; E(1,3) = -1; E(1,4) =  0; E(1,5) = -1; E(1,6) =  0; E(1,7) =  0;
    E(2,0) = -1; E(2,1) =  0; E(2,2) =  6; E(2,3) = -1; E(2,4) =  0; E(2,5) =  0; E(2,6) = -1; E(2,7) =  0;
    E(3,0) =  0; E(3,1) = -1; E(3,2) = -1; E(3,3) =  6; E(3,4) =  0; E(3,5) =  0; E(3,6) =  0; E(3,7) = -1;
    E(4,0) = -1; E(4,1) =  0; E(4,2) =  0; E(4,3) =  0; E(4,4) =  6; E(4,5) = -1; E(4,6) = -1; E(4,7) =  0;
    E(5,0) =  0; E(5,1) = -1; E(5,2) =  0; E(5,3) =  0; E(5,4) = -1; E(5,5) =  6; E(5,6) =  0; E(5,7) = -1;
    E(6,0) =  0; E(6,1) =  0; E(6,2) = -1; E(6,3) =  0; E(6,4) = -1; E(6,5) =  0; E(6,6) =  6; E(6,7) = -1;
    E(7,0) =  0; E(7,1) =  0; E(7,2) =  0; E(7,3) = -1; E(7,4) =  0; E(7,5) = -1; E(7,6) = -1; E(7,7) =  6;

    ASSERT_EQUAL_QUIET(R, E);
}
DECLARE_UNITTEST(TestPoisson7pt);

void TestPoisson27pt(void)
{
    // grid is 3x3x3
    cusp::dia_matrix<int, float, cusp::host_memory> matrix;
    cusp::gallery::poisson27pt(matrix, 2, 2, 2);

    // convert result to array2d
    cusp::array2d<float, cusp::host_memory> R(matrix);
    cusp::array2d<float, cusp::host_memory> E(8,8,-1);

    E(0,0) = 26;
    E(1,1) = 26;
    E(2,2) = 26;
    E(3,3) = 26;
    E(4,4) = 26;
    E(5,5) = 26;
    E(6,6) = 26;
    E(7,7) = 26;

    ASSERT_EQUAL_QUIET(R, E);
}
DECLARE_UNITTEST(TestPoisson27pt);


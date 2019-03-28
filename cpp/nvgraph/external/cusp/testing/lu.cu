#include <unittest/unittest.h>

#include <cusp/detail/lu.h>

void TestLUFactorAndSolve(void)
{
    cusp::array2d<float, cusp::host_memory> A(4,4);
    A(0,0) = 0.83228434;
    A(0,1) = 0.41106598;
    A(0,2) = 0.72609841;
    A(0,3) = 0.80428486;
    A(1,0) = 0.00890590;
    A(1,1) = 0.29940800;
    A(1,2) = 0.60630740;
    A(1,3) = 0.33654542;
    A(2,0) = 0.22525064;
    A(2,1) = 0.93054253;
    A(2,2) = 0.37939225;
    A(2,3) = 0.16235888;
    A(3,0) = 0.83911960;
    A(3,1) = 0.21176293;
    A(3,2) = 0.21010691;
    A(3,3) = 0.52911885;

    cusp::array1d<float, cusp::host_memory> b(4);
    b[0] = 1.31699541;
    b[1] = 0.87768331;
    b[2] = 1.18994714;
    b[3] = 0.61914723;

//    std::cout << "\nA" << std::endl;
//    cusp::print_matrix(A);
//    std::cout << "b" << std::endl;
//    cusp::print_matrix(b);

    cusp::array1d<int, cusp::host_memory>   pivot(4);
    cusp::array1d<float, cusp::host_memory> x(4);
    cusp::detail::lu_factor(A, pivot);
    cusp::detail::lu_solve(A, pivot, b, x);

//    std::cout << "LU" << std::endl;
//    cusp::print_matrix(A);
//    std::cout << "pivot" << std::endl;
//    cusp::print_matrix(pivot);
//    std::cout << "x" << std::endl;
//    cusp::print_matrix(x);

    cusp::array1d<float, cusp::host_memory> expected(4);
    expected[0] = 0.21713221;
    expected[1] = 0.80528582;
    expected[2] = 0.98416811;
    expected[3] = 0.11271028;

    ASSERT_EQUAL(std::fabs(expected[0] - x[0]) < 1e-4, true);
    ASSERT_EQUAL(std::fabs(expected[1] - x[1]) < 1e-4, true);
    ASSERT_EQUAL(std::fabs(expected[2] - x[2]) < 1e-4, true);
    ASSERT_EQUAL(std::fabs(expected[3] - x[3]) < 1e-4, true);
}
DECLARE_UNITTEST(TestLUFactorAndSolve);

void TestLUSolver(void)
{
    cusp::array2d<float, cusp::host_memory> A(3,3);
    A(0,0) = 2.0;
    A(0,1) = 0.0;
    A(0,2) = 0.0;
    A(1,0) = 0.0;
    A(1,1) = 4.0;
    A(1,2) = 0.0;
    A(2,0) = 0.0;
    A(2,1) = 0.0;
    A(2,2) = 8.0;

    cusp::array1d<float, cusp::host_memory> b(3, 1.0);
    cusp::array1d<float, cusp::host_memory> x(3, 0.0);

    cusp::detail::lu_solver<float, cusp::host_memory> lu(A);
    lu(b, x);

    ASSERT_EQUAL(x[0], 0.500);
    ASSERT_EQUAL(x[1], 0.250);
    ASSERT_EQUAL(x[2], 0.125);
}
DECLARE_UNITTEST(TestLUSolver);


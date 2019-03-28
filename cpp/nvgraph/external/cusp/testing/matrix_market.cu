#include <unittest/unittest.h>

#include <cusp/io/matrix_market.h>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/array2d.h>

#include <stdio.h>

const char random_file_name[] = "test_93298409283221.mtx";

void TestReadWriteMarketFileRealArray1d(void)
{
    // create a column vector
    cusp::array1d<float, cusp::host_memory> a(5);
    a[0] = 10;
    a[1] =  0;
    a[2] = 20;
    a[3] =  0;
    a[4] = 30;

    // save a to disk in MatrixMarket format
    cusp::io::write_matrix_market_file(a, random_file_name);

    // load A from disk into an array1d
    cusp::array1d<float, cusp::device_memory> b;
    cusp::io::read_matrix_market_file(b, random_file_name);

    remove(random_file_name);

    ASSERT_EQUAL(a == b, true);
}
DECLARE_UNITTEST(TestReadWriteMarketFileRealArray1d);

void TestReadWriteMarketFileComplexArray1d(void)
{
    // create a column vector
    cusp::array1d<cusp::complex<float>, cusp::host_memory> a(5);
    a[0] = cusp::complex<float>(10, 1);
    a[1] = cusp::complex<float>( 0, 2);
    a[2] = cusp::complex<float>(20, 3);
    a[3] = cusp::complex<float>( 0, 4);
    a[4] = cusp::complex<float>(30, 5);

    // save a to disk in MatrixMarket format
    cusp::io::write_matrix_market_file(a, random_file_name);

    // load A from disk into an array1d
    cusp::array1d<cusp::complex<float>, cusp::device_memory> b;
    cusp::io::read_matrix_market_file(b, random_file_name);

    remove(random_file_name);

    ASSERT_EQUAL(a == b, true);
}
DECLARE_UNITTEST(TestReadWriteMarketFileComplexArray1d);


void TestReadMatrixMarketFileCoordinateRealGeneral(void)
{
    // load matrix
    cusp::coo_matrix<int, float, cusp::host_memory> coo;
    cusp::io::read_matrix_market_file(coo, "../data/test/coordinate_real_general.mtx");

    // convert to array2d
    cusp::array2d<float, cusp::host_memory> D(coo);

    // expected result
    cusp::array2d<float, cusp::host_memory> E(5, 5);
    E(0,0) =  1.000e+00;
    E(0,1) =  0.000e+00;
    E(0,2) =  0.000e+00;
    E(0,3) =  6.000e+00;
    E(0,4) =  0.000e+00;
    E(1,0) =  0.000e+00;
    E(1,1) =  1.050e+01;
    E(1,2) =  0.000e+00;
    E(1,3) =  0.000e+00;
    E(1,4) =  0.000e+00;
    E(2,0) =  0.000e+00;
    E(2,1) =  0.000e+00;
    E(2,2) =  2.500e-01;
    E(2,3) =  0.000e+00;
    E(2,4) =  0.000e+00;
    E(3,0) =  0.000e+00;
    E(3,1) =  2.505e+02;
    E(3,2) =  0.000e+00;
    E(3,3) = -2.500e+02;
    E(3,4) =  3.875e+01;
    E(4,0) =  0.000e+00;
    E(4,1) =  0.000e+00;
    E(4,2) =  0.000e+00;
    E(4,3) =  0.000e+00;
    E(4,4) =  1.200e+01;

    ASSERT_EQUAL(D == E, true);
}
DECLARE_UNITTEST(TestReadMatrixMarketFileCoordinateRealGeneral);

void TestReadMatrixMarketFileCoordinateComplexGeneral(void)
{
    // load matrix
    cusp::coo_matrix<int, cusp::complex<float>, cusp::host_memory> coo;
    cusp::io::read_matrix_market_file(coo, "../data/test/coordinate_complex_general.mtx");

    // convert to array2d
    cusp::array2d<cusp::complex<float>, cusp::host_memory> D(coo);

    // expected result
    cusp::array2d<cusp::complex<float>, cusp::host_memory> E(5, 5);
    E(0,0) = cusp::complex<float>(1.000e+00,1.040e+00);
    E(1,0) = cusp::complex<float>(0.000e+00,0.000e+00);
    E(2,0) = cusp::complex<float>(0.000e+00,0.000e+00);
    E(3,0) = cusp::complex<float>(0.000e+00,0.000e+00);
    E(4,0) = cusp::complex<float>(0.000e+00,0.000e+00);

    E(0,1) = cusp::complex<float>(0.000e+00,0.000e+00);
    E(1,1) = cusp::complex<float>(1.050e+01,3.000e+01);
    E(2,1) = cusp::complex<float>(0.000e+00,0.000e+00);
    E(3,1) = cusp::complex<float>(2.505e+02,-3.000e+00);
    E(4,1) = cusp::complex<float>(0.000e+00,0.000e+00);

    E(0,2) = cusp::complex<float>(0.000e+00,0.000e+00);
    E(1,2) = cusp::complex<float>(0.000e+00,0.000e+00);
    E(2,2) = cusp::complex<float>(2.500e-01,-5.300e+00);
    E(3,2) = cusp::complex<float>(0.000e+00,0.000e+00);
    E(4,2) = cusp::complex<float>(0.000e+00,0.000e+00);

    E(0,3) = cusp::complex<float>(6.000e+00,3.000e-01);
    E(1,3) = cusp::complex<float>(0.000e+00,0.000e+00);
    E(2,3) = cusp::complex<float>(0.000e+00,0.000e+00);
    E(3,3) = cusp::complex<float>(-2.500e+02,9.500e+02);
    E(4,3) = cusp::complex<float>(0.000e+00,0.000e+00);

    E(0,4) = cusp::complex<float>(0.000e+00,0.000e+00);
    E(1,4) = cusp::complex<float>(0.000e+00,0.000e+00);
    E(2,4) = cusp::complex<float>(0.000e+00,0.000e+00);
    E(3,4) = cusp::complex<float>(3.875e+01,-8.000e+00);
    E(4,4) = cusp::complex<float>(1.200e+01,6.200e+02);

    ASSERT_EQUAL(D == E, true);
}
DECLARE_UNITTEST(TestReadMatrixMarketFileCoordinateComplexGeneral);

void TestReadMatrixMarketFileCoordinatePatternSymmetric(void)
{
    // load matrix
    cusp::coo_matrix<int, float, cusp::host_memory> coo;
    cusp::io::read_matrix_market_file(coo, "../data/test/coordinate_pattern_symmetric.mtx");

    // convert to array2d
    cusp::array2d<float, cusp::host_memory> D(coo);

    // expected result
    cusp::array2d<float, cusp::host_memory> E(5, 5);
    E(0,0) =  1.000e+00;
    E(0,1) =  0.000e+00;
    E(0,2) =  0.000e+00;
    E(0,3) =  0.000e+00;
    E(0,4) =  0.000e+00;
    E(1,0) =  0.000e+00;
    E(1,1) =  1.000e+00;
    E(1,2) =  0.000e+00;
    E(1,3) =  1.000e+00;
    E(1,4) =  0.000e+00;
    E(2,0) =  0.000e+00;
    E(2,1) =  0.000e+00;
    E(2,2) =  1.000e+00;
    E(2,3) =  0.000e+00;
    E(2,4) =  0.000e+00;
    E(3,0) =  0.000e+00;
    E(3,1) =  1.000e+00;
    E(3,2) =  0.000e+00;
    E(3,3) =  1.000e+00;
    E(3,4) =  1.000e+00;
    E(4,0) =  0.000e+00;
    E(4,1) =  0.000e+00;
    E(4,2) =  0.000e+00;
    E(4,3) =  1.000e+00;
    E(4,4) =  1.000e+00;

    ASSERT_EQUAL(D == E, true);
}
DECLARE_UNITTEST(TestReadMatrixMarketFileCoordinatePatternSymmetric);

void TestReadMatrixMarketFileArrayRealGeneral(void)
{
    // load matrix
    cusp::coo_matrix<int, float, cusp::host_memory> coo;
    cusp::io::read_matrix_market_file(coo, "../data/test/array_real_general.mtx");

    // convert to array2d
    cusp::array2d<float, cusp::host_memory> D(coo);

    // expected result
    cusp::array2d<float, cusp::host_memory> E(4, 3);
    E(0,0) =  1.0;
    E(0,1) =  5.0;
    E(0,2) =  9.0;
    E(1,0) =  2.0;
    E(1,1) =  6.0;
    E(1,2) = 10.0;
    E(2,0) =  3.0;
    E(2,1) =  7.0;
    E(2,2) = 11.0;
    E(3,0) =  4.0;
    E(3,1) =  8.0;
    E(3,2) = 12.0;

    ASSERT_EQUAL(D == E, true);
}
DECLARE_UNITTEST(TestReadMatrixMarketFileArrayRealGeneral);

template <typename MemorySpace>
void TestReadMatrixMarketFileToCsrMatrix(void)
{
    // load matrix
    cusp::csr_matrix<int, float, MemorySpace> csr;
    cusp::io::read_matrix_market_file(csr, "../data/test/coordinate_real_general.mtx");

    // convert to array2d
    cusp::array2d<float, cusp::host_memory> D(csr);

    // expected result
    cusp::array2d<float, cusp::host_memory> E(5, 5);
    E(0,0) =  1.000e+00;
    E(0,1) =  0.000e+00;
    E(0,2) =  0.000e+00;
    E(0,3) =  6.000e+00;
    E(0,4) =  0.000e+00;
    E(1,0) =  0.000e+00;
    E(1,1) =  1.050e+01;
    E(1,2) =  0.000e+00;
    E(1,3) =  0.000e+00;
    E(1,4) =  0.000e+00;
    E(2,0) =  0.000e+00;
    E(2,1) =  0.000e+00;
    E(2,2) =  2.500e-01;
    E(2,3) =  0.000e+00;
    E(2,4) =  0.000e+00;
    E(3,0) =  0.000e+00;
    E(3,1) =  2.505e+02;
    E(3,2) =  0.000e+00;
    E(3,3) = -2.500e+02;
    E(3,4) =  3.875e+01;
    E(4,0) =  0.000e+00;
    E(4,1) =  0.000e+00;
    E(4,2) =  0.000e+00;
    E(4,3) =  0.000e+00;
    E(4,4) =  1.200e+01;

    ASSERT_EQUAL(D == E, true);
}
DECLARE_HOST_DEVICE_UNITTEST(TestReadMatrixMarketFileToCsrMatrix);

template <typename MemorySpace>
void TestWriteMatrixMarketFileCoordinateRealGeneral(void)
{
    // initial matrix
    cusp::array2d<float, cusp::host_memory> E(4, 3);
    E(0,0) =  1.000e+00;
    E(0,1) =  0.000e+00;
    E(0,2) =  0.000e+00;
    E(1,0) =  0.000e+00;
    E(1,1) =  1.050e+01;
    E(1,2) =  0.000e+00;
    E(2,0) =  0.000e+00;
    E(2,1) =  0.000e+00;
    E(2,2) =  2.500e-01;
    E(3,0) =  0.000e+00;
    E(3,1) =  2.505e+02;
    E(3,2) =  0.000e+00;

    // convert to coo
    cusp::coo_matrix<int, float, MemorySpace> coo(E);

    // write coo to file
    cusp::io::write_matrix_market_file(coo, random_file_name);

    // read file back
    cusp::io::read_matrix_market_file(coo, random_file_name);

    remove(random_file_name);

    // compare to initial matrix
    cusp::array2d<float, cusp::host_memory> D(coo);
    ASSERT_EQUAL(D == E, true);
}
DECLARE_HOST_DEVICE_UNITTEST(TestWriteMatrixMarketFileCoordinateRealGeneral);

template <typename MemorySpace>
void TestWriteMatrixMarketFileCoordinateComplexGeneral(void)
{
    // initial matrix
    cusp::array2d<cusp::complex<float>, cusp::host_memory> E(4, 3);
    E(0,0) = cusp::complex<float>(1.000e+00, 1);
    E(0,1) = cusp::complex<float>(0.000e+00, 5);
    E(0,2) = cusp::complex<float>(0.000e+00,  9);
    E(1,0) = cusp::complex<float>(0.000e+00, 2);
    E(1,1) = cusp::complex<float>(1.050e+01, 6);
    E(1,2) = cusp::complex<float>(0.000e+00, 10);
    E(2,0) = cusp::complex<float>(0.000e+00, 3);
    E(2,1) = cusp::complex<float>(0.000e+00, 7);
    E(2,2) = cusp::complex<float>(2.500e-01, 11);
    E(3,0) = cusp::complex<float>(0.000e+00, 4);
    E(3,1) = cusp::complex<float>(2.505e+02, 8);
    E(3,2) = cusp::complex<float>(0.000e+00, 12);

    // convert to coo
    cusp::coo_matrix<int, cusp::complex<float>, MemorySpace> coo(E);

    // write coo to file
    cusp::io::write_matrix_market_file(coo, random_file_name);

    // read file back
    cusp::io::read_matrix_market_file(coo, random_file_name);

    remove(random_file_name);

    // compare to initial matrix
    cusp::array2d<cusp::complex<float>, cusp::host_memory> D(coo);
    ASSERT_EQUAL(D == E, true);
}
DECLARE_HOST_DEVICE_UNITTEST(TestWriteMatrixMarketFileCoordinateComplexGeneral);


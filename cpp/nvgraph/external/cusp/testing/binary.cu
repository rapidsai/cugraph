#include <unittest/unittest.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <cusp/io/binary.h>

#include <stdio.h>

const char random_file_name[] = "test_93298409283221.bin";

void TestReadBinaryFileCoordinateRealGeneral(void)
{
    // load matrix
    cusp::coo_matrix<int, float, cusp::host_memory> coo;
    cusp::io::read_binary_file(coo, "../data/test/coordinate_real_general.bin");

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
DECLARE_UNITTEST(TestReadBinaryFileCoordinateRealGeneral);

template <typename MemorySpace>
void TestReadBinaryFileToCsrMatrix(void)
{
    // load matrix
    cusp::csr_matrix<int, float, MemorySpace> csr;
    cusp::io::read_binary_file(csr, "../data/test/coordinate_real_general.bin");

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
DECLARE_HOST_DEVICE_UNITTEST(TestReadBinaryFileToCsrMatrix)

template <typename MemorySpace>
void TestWriteBinaryFileCoordinateRealGeneral(void)
{
    // initial matrix
    cusp::array2d<float, cusp::host_memory> E(4, 4);
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
    cusp::io::write_binary_file(coo, random_file_name);

    // read file back
    cusp::io::read_binary_file(coo, random_file_name);

    remove(random_file_name);

    // compare to initial matrix
    cusp::array2d<float, cusp::host_memory> D(coo);
    ASSERT_EQUAL(D == E, true);
}
DECLARE_HOST_DEVICE_UNITTEST(TestWriteBinaryFileCoordinateRealGeneral)

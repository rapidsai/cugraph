#include <unittest/unittest.h>

#include <cusp/io/dimacs.h>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/array2d.h>

#include <stdio.h>

const char random_file_name[] = "test_93298409283221.dimacs";

void TestReadDimacsFileCoordinateRealGeneral(void)
{
    // load matrix
    thrust::tuple<int,int> nodes(-1,-1);
    cusp::coo_matrix<int, float, cusp::host_memory> coo;
    nodes = cusp::io::read_dimacs_file(coo, "../data/test/coordinate_real_general.dimacs");

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
    ASSERT_EQUAL(thrust::get<0>(nodes), 0);
    ASSERT_EQUAL(thrust::get<1>(nodes), 3);
}
DECLARE_UNITTEST(TestReadDimacsFileCoordinateRealGeneral);

template <typename MemorySpace>
void TestReadDimacsFileToCsrMatrix(void)
{
    // load matrix
    thrust::tuple<int,int> nodes(-1,-1);
    cusp::csr_matrix<int, float, MemorySpace> csr;
    nodes = cusp::io::read_dimacs_file(csr, "../data/test/coordinate_real_general.dimacs");

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
    ASSERT_EQUAL(thrust::get<0>(nodes), 0);
    ASSERT_EQUAL(thrust::get<1>(nodes), 3);
}
DECLARE_HOST_DEVICE_UNITTEST(TestReadDimacsFileToCsrMatrix);

template <typename MemorySpace>
void TestWriteDimacsFileCoordinateRealGeneral(void)
{
    // initial matrix
    cusp::array2d<int, cusp::host_memory> E(4, 4);
    E(0,0) =  1;
    E(0,1) =  0;
    E(0,2) =  0;
    E(1,0) =  0;
    E(1,1) =  10;
    E(1,2) =  0;
    E(2,0) =  0;
    E(2,1) =  0;
    E(2,2) =  0;
    E(3,0) =  0;
    E(3,1) =  250;
    E(3,2) =  0;

    // convert to coo
    cusp::coo_matrix<int, int, MemorySpace> coo(E);

    thrust::tuple<int,int> nodes(0,3);

    // write coo to file
    cusp::io::write_dimacs_file(coo, nodes, random_file_name);

    thrust::get<0>(nodes) = -1;
    thrust::get<1>(nodes) = -1;

    // read file back
    nodes = cusp::io::read_dimacs_file(coo, random_file_name);

    remove(random_file_name);

    // compare to initial matrix
    cusp::array2d<int, cusp::host_memory> D(coo);
    ASSERT_EQUAL(D == E, true);
    ASSERT_EQUAL(thrust::get<0>(nodes), 0);
    ASSERT_EQUAL(thrust::get<1>(nodes), 3);
}
DECLARE_HOST_DEVICE_UNITTEST(TestWriteDimacsFileCoordinateRealGeneral);


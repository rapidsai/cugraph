#include <unittest/unittest.h>
#include <cusp/dia_matrix.h>

template <class Space>
void TestDiaMatrixBasicConstructor(void)
{
    cusp::dia_matrix<int, float, Space> matrix(4, 5, 7, 3, 8);

    ASSERT_EQUAL(matrix.num_rows,                4);
    ASSERT_EQUAL(matrix.num_cols,                5);
    ASSERT_EQUAL(matrix.num_entries,             7);
    ASSERT_EQUAL(matrix.diagonal_offsets.size(), 3);
    ASSERT_EQUAL(matrix.values.num_rows,         4);
    ASSERT_EQUAL(matrix.values.num_cols,         3);
    ASSERT_EQUAL(matrix.values.pitch,            8);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDiaMatrixBasicConstructor);

template <class Space>
void TestDiaMatrixCopyConstructor(void)
{
    cusp::dia_matrix<int, float, Space> matrix(4, 5, 7, 3, 1);

    matrix.diagonal_offsets[0] = -2;
    matrix.diagonal_offsets[1] =  0;
    matrix.diagonal_offsets[2] =  1;

    matrix.values.values[ 0] =  0;
    matrix.values.values[ 1] =  0;
    matrix.values.values[ 2] = 13;
    matrix.values.values[ 3] = 16;
    matrix.values.values[ 4] = 10;
    matrix.values.values[ 5] =  0;
    matrix.values.values[ 6] = 14;
    matrix.values.values[ 7] =  0;
    matrix.values.values[ 8] = 11;
    matrix.values.values[ 9] = 12;
    matrix.values.values[10] = 15;
    matrix.values.values[11] =  0;

    cusp::dia_matrix<int, float, Space> copy_of_matrix(matrix);

    ASSERT_EQUAL(copy_of_matrix.num_rows,                 4);
    ASSERT_EQUAL(copy_of_matrix.num_cols,                 5);
    ASSERT_EQUAL(copy_of_matrix.num_entries,              7);
    ASSERT_EQUAL(copy_of_matrix.diagonal_offsets.size(),  3);
    ASSERT_EQUAL(copy_of_matrix.values.num_rows,          4);
    ASSERT_EQUAL(copy_of_matrix.values.num_cols,          3);
    ASSERT_EQUAL(copy_of_matrix.values.pitch,             4);

    ASSERT_EQUAL(copy_of_matrix.diagonal_offsets, matrix.diagonal_offsets);
    ASSERT_EQUAL_QUIET(copy_of_matrix.values,           matrix.values);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDiaMatrixCopyConstructor);

template <class Space>
void TestDiaMatrixResize(void)
{
    cusp::dia_matrix<int, float, Space> matrix;

    matrix.resize(4, 5, 7, 3, 8);

    ASSERT_EQUAL(matrix.num_rows,                4);
    ASSERT_EQUAL(matrix.num_cols,                5);
    ASSERT_EQUAL(matrix.num_entries,             7);
    ASSERT_EQUAL(matrix.diagonal_offsets.size(), 3);
    ASSERT_EQUAL(matrix.values.num_rows,         4);
    ASSERT_EQUAL(matrix.values.num_cols,         3);
    ASSERT_EQUAL(matrix.values.pitch,            8);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDiaMatrixResize);

template <class Space>
void TestDiaMatrixSwap(void)
{
    cusp::dia_matrix<int, float, Space> A(2, 2, 4, 3, 1);
    cusp::dia_matrix<int, float, Space> B(1, 3, 2, 2, 1);

    A.diagonal_offsets[0] = -1;
    A.diagonal_offsets[1] =  0;
    A.diagonal_offsets[2] =  1;

    A.values(0,0) = 10;
    A.values(0,1) = 30;
    A.values(0,2) = 50;
    A.values(1,0) = 20;
    A.values(1,1) = 40;
    A.values(1,2) = 60;

    B.diagonal_offsets[0] = 1;
    B.diagonal_offsets[1] = 2;

    B.values(0,0) = 10;
    B.values(0,1) = 20;

    cusp::dia_matrix<int, float, Space> A_copy(A);
    cusp::dia_matrix<int, float, Space> B_copy(B);

    A.swap(B);

    ASSERT_EQUAL(A.num_rows,         B_copy.num_rows);
    ASSERT_EQUAL(A.num_cols,         B_copy.num_cols);
    ASSERT_EQUAL(A.num_entries,      B_copy.num_entries);
    ASSERT_EQUAL(A.diagonal_offsets, B_copy.diagonal_offsets);
    ASSERT_EQUAL_QUIET(A.values,           B_copy.values);

    ASSERT_EQUAL(B.num_rows,         A_copy.num_rows);
    ASSERT_EQUAL(B.num_cols,         A_copy.num_cols);
    ASSERT_EQUAL(B.num_entries,      A_copy.num_entries);
    ASSERT_EQUAL(B.diagonal_offsets, A_copy.diagonal_offsets);
    ASSERT_EQUAL_QUIET(B.values,           A_copy.values);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDiaMatrixSwap);

void TestDiaMatrixRebind(void)
{
    typedef cusp::dia_matrix<int, float, cusp::host_memory> HostMatrix;
    typedef HostMatrix::rebind<cusp::device_memory>::type   DeviceMatrix;

    HostMatrix   h_matrix(10,10,100,19);
    DeviceMatrix d_matrix(h_matrix);

    ASSERT_EQUAL(h_matrix.num_entries, d_matrix.num_entries);
}
DECLARE_UNITTEST(TestDiaMatrixRebind);


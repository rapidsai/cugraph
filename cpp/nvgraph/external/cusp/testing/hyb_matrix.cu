#include <unittest/unittest.h>
#include <cusp/hyb_matrix.h>

template <class Space>
void TestHybMatrixBasicConstructor(void)
{
    cusp::hyb_matrix<int, float, Space> matrix(10, 10, 42, 13, 5, 16);

    ASSERT_EQUAL(matrix.num_rows,                 10);
    ASSERT_EQUAL(matrix.num_cols,                 10);
    ASSERT_EQUAL(matrix.num_entries,              55);

    ASSERT_EQUAL(matrix.ell.num_rows,                10);
    ASSERT_EQUAL(matrix.ell.num_cols,                10);
    ASSERT_EQUAL(matrix.ell.num_entries,             42);
    ASSERT_EQUAL(matrix.ell.column_indices.num_rows, 10);
    ASSERT_EQUAL(matrix.ell.column_indices.num_cols,  5);
    ASSERT_EQUAL(matrix.ell.column_indices.pitch,    16);
    ASSERT_EQUAL(matrix.ell.values.num_rows,         10);
    ASSERT_EQUAL(matrix.ell.values.num_cols,          5);
    ASSERT_EQUAL(matrix.ell.values.pitch,            16);

    ASSERT_EQUAL(matrix.coo.num_rows,              10);
    ASSERT_EQUAL(matrix.coo.num_cols,              10);
    ASSERT_EQUAL(matrix.coo.num_entries,           13);
    ASSERT_EQUAL(matrix.coo.row_indices.size(),    13);
    ASSERT_EQUAL(matrix.coo.column_indices.size(), 13);
    ASSERT_EQUAL(matrix.coo.values.size(),         13);
}
DECLARE_HOST_DEVICE_UNITTEST(TestHybMatrixBasicConstructor);

template <class Space>
void TestHybMatrixCopyConstructor(void)
{
    // [0356]
    // [1xx7]
    // [24xx]

    cusp::hyb_matrix<int, float, Space> matrix(3, 4, 5,  3, 2, 1);

    matrix.ell.column_indices.values[0] = 0;
    matrix.ell.values.values[0] = 0;
    matrix.ell.column_indices.values[1] = 0;
    matrix.ell.values.values[1] = 1;
    matrix.ell.column_indices.values[2] = 0;
    matrix.ell.values.values[2] = 2;
    matrix.ell.column_indices.values[3] = 1;
    matrix.ell.values.values[3] = 3;
    matrix.ell.column_indices.values[4] = 1;
    matrix.ell.values.values[4] = 0;
    matrix.ell.column_indices.values[5] = 1;
    matrix.ell.values.values[5] = 4;

    matrix.coo.row_indices[0] = 0;
    matrix.coo.column_indices[0] = 2;
    matrix.coo.values[0] = 5;
    matrix.coo.row_indices[1] = 0;
    matrix.coo.column_indices[1] = 3;
    matrix.coo.values[1] = 6;
    matrix.coo.row_indices[2] = 1;
    matrix.coo.column_indices[2] = 3;
    matrix.coo.values[2] = 7;

    cusp::hyb_matrix<int, float, Space> copy_of_matrix(matrix);

    ASSERT_EQUAL(copy_of_matrix.num_rows,    3);
    ASSERT_EQUAL(copy_of_matrix.num_cols,    4);
    ASSERT_EQUAL(copy_of_matrix.num_entries, 8);

    ASSERT_EQUAL(copy_of_matrix.ell.num_rows,                 3);
    ASSERT_EQUAL(copy_of_matrix.ell.num_cols,                 4);
    ASSERT_EQUAL(copy_of_matrix.ell.num_entries,              5);
    ASSERT_EQUAL(copy_of_matrix.ell.column_indices.num_rows,  3);
    ASSERT_EQUAL(copy_of_matrix.ell.column_indices.num_cols,  2);
    ASSERT_EQUAL(copy_of_matrix.ell.column_indices.pitch,     3);
    ASSERT_EQUAL(copy_of_matrix.ell.values.num_rows,          3);
    ASSERT_EQUAL(copy_of_matrix.ell.values.num_cols,          2);
    ASSERT_EQUAL(copy_of_matrix.ell.values.pitch,             3);
    ASSERT_EQUAL_QUIET(copy_of_matrix.ell.column_indices, matrix.ell.column_indices);
    ASSERT_EQUAL_QUIET(copy_of_matrix.ell.values,         matrix.ell.values);

    ASSERT_EQUAL(copy_of_matrix.coo.num_rows,              3);
    ASSERT_EQUAL(copy_of_matrix.coo.num_cols,              4);
    ASSERT_EQUAL(copy_of_matrix.coo.num_entries,           3);
    ASSERT_EQUAL(copy_of_matrix.coo.row_indices,    matrix.coo.row_indices);
    ASSERT_EQUAL(copy_of_matrix.coo.column_indices, matrix.coo.column_indices);
    ASSERT_EQUAL(copy_of_matrix.coo.values,         matrix.coo.values);
}
DECLARE_HOST_DEVICE_UNITTEST(TestHybMatrixCopyConstructor);

template <class Space>
void TestHybMatrixResize(void)
{
    cusp::hyb_matrix<int, float, Space> matrix;

    matrix.resize(10, 10, 42, 13, 5, 16);

    ASSERT_EQUAL(matrix.num_rows,                 10);
    ASSERT_EQUAL(matrix.num_cols,                 10);
    ASSERT_EQUAL(matrix.num_entries,              55);

    ASSERT_EQUAL(matrix.ell.num_rows,                10);
    ASSERT_EQUAL(matrix.ell.num_cols,                10);
    ASSERT_EQUAL(matrix.ell.num_entries,             42);
    ASSERT_EQUAL(matrix.ell.column_indices.num_rows, 10);
    ASSERT_EQUAL(matrix.ell.column_indices.num_cols,  5);
    ASSERT_EQUAL(matrix.ell.column_indices.pitch,    16);
    ASSERT_EQUAL(matrix.ell.values.num_rows,         10);
    ASSERT_EQUAL(matrix.ell.values.num_cols,          5);
    ASSERT_EQUAL(matrix.ell.values.pitch,            16);

    ASSERT_EQUAL(matrix.coo.num_rows,              10);
    ASSERT_EQUAL(matrix.coo.num_cols,              10);
    ASSERT_EQUAL(matrix.coo.num_entries,           13);
    ASSERT_EQUAL(matrix.coo.row_indices.size(),    13);
    ASSERT_EQUAL(matrix.coo.column_indices.size(), 13);
    ASSERT_EQUAL(matrix.coo.values.size(),         13);
}
DECLARE_HOST_DEVICE_UNITTEST(TestHybMatrixResize);

template <class Space>
void TestHybMatrixSwap(void)
{
    cusp::hyb_matrix<int, float, Space> A(1, 2, 1, 1, 1, 1);
    cusp::hyb_matrix<int, float, Space> B(1, 3, 0, 3, 0, 1);

    A.ell.column_indices(0,0) = 0;
    A.ell.values(0,0) = 0;
    A.coo.row_indices[0] = 0;
    A.coo.column_indices[0] = 1;
    A.coo.values[0] = 1;

    B.coo.row_indices[0] = 0;
    B.coo.column_indices[0] = 0;
    B.coo.values[0] = 0;
    B.coo.row_indices[1] = 0;
    B.coo.column_indices[1] = 1;
    B.coo.values[1] = 1;
    B.coo.row_indices[2] = 0;
    B.coo.column_indices[2] = 2;
    B.coo.values[2] = 2;

    cusp::hyb_matrix<int, float, Space> A_copy(A);
    cusp::hyb_matrix<int, float, Space> B_copy(B);

    A.swap(B);

    ASSERT_EQUAL(A.num_rows,                  1);
    ASSERT_EQUAL(A.num_cols,                  3);
    ASSERT_EQUAL(A.num_entries,               3);
    ASSERT_EQUAL(A.ell.num_rows,              1);
    ASSERT_EQUAL(A.ell.num_cols,              3);
    ASSERT_EQUAL(A.ell.num_entries,           0);
    ASSERT_EQUAL(A.coo.num_rows,              1);
    ASSERT_EQUAL(A.coo.num_cols,              3);
    ASSERT_EQUAL(A.coo.num_entries,           3);
    ASSERT_EQUAL_QUIET(A.ell.column_indices, B_copy.ell.column_indices);
    ASSERT_EQUAL_QUIET(A.ell.values,         B_copy.ell.values);
    ASSERT_EQUAL(A.coo.row_indices,    B_copy.coo.row_indices);
    ASSERT_EQUAL(A.coo.column_indices, B_copy.coo.column_indices);
    ASSERT_EQUAL(A.coo.values,         B_copy.coo.values);

    ASSERT_EQUAL(B.num_rows,                  1);
    ASSERT_EQUAL(B.num_cols,                  2);
    ASSERT_EQUAL(B.num_entries,               2);
    ASSERT_EQUAL(B.ell.num_rows,              1);
    ASSERT_EQUAL(B.ell.num_cols,              2);
    ASSERT_EQUAL(B.ell.num_entries,           1);
    ASSERT_EQUAL(B.coo.num_rows,              1);
    ASSERT_EQUAL(B.coo.num_cols,              2);
    ASSERT_EQUAL(B.coo.num_entries,           1);
    ASSERT_EQUAL_QUIET(B.ell.column_indices, A_copy.ell.column_indices);
    ASSERT_EQUAL_QUIET(B.ell.values,         A_copy.ell.values);
    ASSERT_EQUAL(B.coo.row_indices,    A_copy.coo.row_indices);
    ASSERT_EQUAL(B.coo.column_indices, A_copy.coo.column_indices);
    ASSERT_EQUAL(B.coo.values,         A_copy.coo.values);
}
DECLARE_HOST_DEVICE_UNITTEST(TestHybMatrixSwap);

void TestHybMatrixRebind(void)
{
    typedef cusp::hyb_matrix<int, float, cusp::host_memory> HostMatrix;
    typedef HostMatrix::rebind<cusp::device_memory>::type   DeviceMatrix;

    HostMatrix   h_matrix(10,10,50,50,5,10);
    DeviceMatrix d_matrix(h_matrix);

    ASSERT_EQUAL(h_matrix.num_entries, d_matrix.num_entries);
}
DECLARE_UNITTEST(TestHybMatrixRebind);


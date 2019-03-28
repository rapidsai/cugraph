#include <unittest/unittest.h>
#include <cusp/coo_matrix.h>

template <class Space>
void TestCooMatrixBasicConstructor(void)
{
    cusp::coo_matrix<int, float, Space> matrix(3, 2, 6);

    ASSERT_EQUAL(matrix.num_rows,              3);
    ASSERT_EQUAL(matrix.num_cols,              2);
    ASSERT_EQUAL(matrix.num_entries,           6);
    ASSERT_EQUAL(matrix.row_indices.size(),    6);
    ASSERT_EQUAL(matrix.column_indices.size(), 6);
    ASSERT_EQUAL(matrix.values.size(),         6);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCooMatrixBasicConstructor);

template <class Space>
void TestCooMatrixCopyConstructor(void)
{
    cusp::coo_matrix<int, float, Space> matrix(3, 2, 6);

    matrix.row_indices[0] = 0;
    matrix.column_indices[0] = 0;
    matrix.values[0] = 0;
    matrix.row_indices[1] = 0;
    matrix.column_indices[1] = 1;
    matrix.values[1] = 1;
    matrix.row_indices[2] = 1;
    matrix.column_indices[2] = 0;
    matrix.values[2] = 2;
    matrix.row_indices[3] = 1;
    matrix.column_indices[3] = 1;
    matrix.values[3] = 3;
    matrix.row_indices[4] = 2;
    matrix.column_indices[4] = 0;
    matrix.values[4] = 4;
    matrix.row_indices[5] = 2;
    matrix.column_indices[5] = 1;
    matrix.values[5] = 5;

    cusp::coo_matrix<int, float, Space> copy_of_matrix(matrix);

    ASSERT_EQUAL(copy_of_matrix.num_rows,              3);
    ASSERT_EQUAL(copy_of_matrix.num_cols,              2);
    ASSERT_EQUAL(copy_of_matrix.num_entries,           6);
    ASSERT_EQUAL(copy_of_matrix.row_indices.size(),    6);
    ASSERT_EQUAL(copy_of_matrix.column_indices.size(), 6);
    ASSERT_EQUAL(copy_of_matrix.values.size(),         6);

    ASSERT_EQUAL(copy_of_matrix.row_indices,    matrix.row_indices);
    ASSERT_EQUAL(copy_of_matrix.column_indices, matrix.column_indices);
    ASSERT_EQUAL(copy_of_matrix.values,         matrix.values);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCooMatrixCopyConstructor);

template <class Space>
void TestCooMatrixResize(void)
{
    cusp::coo_matrix<int, float, Space> matrix;

    matrix.resize(3, 2, 6);

    ASSERT_EQUAL(matrix.num_rows,              3);
    ASSERT_EQUAL(matrix.num_cols,              2);
    ASSERT_EQUAL(matrix.num_entries,           6);
    ASSERT_EQUAL(matrix.row_indices.size(),    6);
    ASSERT_EQUAL(matrix.column_indices.size(), 6);
    ASSERT_EQUAL(matrix.values.size(),         6);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCooMatrixResize);

template <class Space>
void TestCooMatrixSwap(void)
{
    cusp::coo_matrix<int, float, Space> A(1, 2, 2);
    cusp::coo_matrix<int, float, Space> B(3, 1, 3);

    A.row_indices[0] = 0;
    A.column_indices[0] = 0;
    A.values[0] = 0;
    A.row_indices[1] = 0;
    A.column_indices[1] = 1;
    A.values[1] = 1;

    B.row_indices[0] = 0;
    B.column_indices[0] = 0;
    B.values[0] = 0;
    B.row_indices[1] = 1;
    B.column_indices[1] = 0;
    B.values[1] = 1;
    B.row_indices[2] = 2;
    B.column_indices[2] = 0;
    B.values[2] = 2;

    cusp::coo_matrix<int, float, Space> A_copy(A);
    cusp::coo_matrix<int, float, Space> B_copy(B);

    A.swap(B);

    ASSERT_EQUAL(A.num_rows,              3);
    ASSERT_EQUAL(A.num_cols,              1);
    ASSERT_EQUAL(A.num_entries,           3);
    ASSERT_EQUAL(A.row_indices,    B_copy.row_indices);
    ASSERT_EQUAL(A.column_indices, B_copy.column_indices);
    ASSERT_EQUAL(A.values,         B_copy.values);

    ASSERT_EQUAL(B.num_rows,              1);
    ASSERT_EQUAL(B.num_cols,              2);
    ASSERT_EQUAL(B.num_entries,           2);
    ASSERT_EQUAL(B.row_indices,    A_copy.row_indices);
    ASSERT_EQUAL(B.column_indices, A_copy.column_indices);
    ASSERT_EQUAL(B.values,         A_copy.values);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCooMatrixSwap);

void TestCooMatrixRebind(void)
{
    typedef cusp::coo_matrix<int, float, cusp::host_memory> HostMatrix;
    typedef HostMatrix::rebind<cusp::device_memory>::type   DeviceMatrix;

    HostMatrix   h_matrix(10,10,100);
    DeviceMatrix d_matrix(h_matrix);

    ASSERT_EQUAL(h_matrix.num_entries, d_matrix.num_entries);
}
DECLARE_UNITTEST(TestCooMatrixRebind);

template <class Space>
void TestCooMatrixSortByRow(void)
{
    cusp::coo_matrix<int, float, Space> A(5, 5, 4);

    A.row_indices[0] = 3;
    A.column_indices[0] = 1;
    A.values[0] = 1;
    A.row_indices[1] = 4;
    A.column_indices[1] = 2;
    A.values[1] = 2;
    A.row_indices[2] = 1;
    A.column_indices[2] = 3;
    A.values[2] = 3;
    A.row_indices[3] = 2;
    A.column_indices[3] = 4;
    A.values[3] = 4;

    A.sort_by_row();

    ASSERT_EQUAL(A.row_indices[0], 1);
    ASSERT_EQUAL(A.column_indices[0], 3);
    ASSERT_EQUAL(A.values[0], 3);
    ASSERT_EQUAL(A.row_indices[1], 2);
    ASSERT_EQUAL(A.column_indices[1], 4);
    ASSERT_EQUAL(A.values[1], 4);
    ASSERT_EQUAL(A.row_indices[2], 3);
    ASSERT_EQUAL(A.column_indices[2], 1);
    ASSERT_EQUAL(A.values[2], 1);
    ASSERT_EQUAL(A.row_indices[3], 4);
    ASSERT_EQUAL(A.column_indices[3], 2);
    ASSERT_EQUAL(A.values[3], 2);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCooMatrixSortByRow);

template <class Space>
void TestCooMatrixSortByRowAndColumn(void)
{
    cusp::coo_matrix<int, float, Space> A(5, 5, 7);

    A.row_indices[0] = 3;
    A.column_indices[0] = 1;
    A.values[0] = 1;
    A.row_indices[1] = 4;
    A.column_indices[1] = 2;
    A.values[1] = 2;
    A.row_indices[2] = 1;
    A.column_indices[2] = 3;
    A.values[2] = 3;
    A.row_indices[3] = 2;
    A.column_indices[3] = 2;
    A.values[3] = 4;
    A.row_indices[4] = 1;
    A.column_indices[4] = 2;
    A.values[4] = 5;
    A.row_indices[5] = 0;
    A.column_indices[5] = 3;
    A.values[5] = 6;
    A.row_indices[6] = 2;
    A.column_indices[6] = 1;
    A.values[6] = 7;

    A.sort_by_row_and_column();

    ASSERT_EQUAL(A.row_indices[0], 0);
    ASSERT_EQUAL(A.column_indices[0], 3);
    ASSERT_EQUAL(A.values[0], 6);
    ASSERT_EQUAL(A.row_indices[1], 1);
    ASSERT_EQUAL(A.column_indices[1], 2);
    ASSERT_EQUAL(A.values[1], 5);
    ASSERT_EQUAL(A.row_indices[2], 1);
    ASSERT_EQUAL(A.column_indices[2], 3);
    ASSERT_EQUAL(A.values[2], 3);
    ASSERT_EQUAL(A.row_indices[3], 2);
    ASSERT_EQUAL(A.column_indices[3], 1);
    ASSERT_EQUAL(A.values[3], 7);
    ASSERT_EQUAL(A.row_indices[4], 2);
    ASSERT_EQUAL(A.column_indices[4], 2);
    ASSERT_EQUAL(A.values[4], 4);
    ASSERT_EQUAL(A.row_indices[5], 3);
    ASSERT_EQUAL(A.column_indices[5], 1);
    ASSERT_EQUAL(A.values[5], 1);
    ASSERT_EQUAL(A.row_indices[6], 4);
    ASSERT_EQUAL(A.column_indices[6], 2);
    ASSERT_EQUAL(A.values[6], 2);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCooMatrixSortByRowAndColumn);

template <class Space>
void TestCooMatrixIsSortedByRow(void)
{
    cusp::coo_matrix<int, float, Space> A(5, 5, 4);

    A.row_indices[0] = 3;
    A.column_indices[0] = 1;
    A.values[0] = 1;
    A.row_indices[1] = 4;
    A.column_indices[1] = 2;
    A.values[1] = 2;
    A.row_indices[2] = 1;
    A.column_indices[2] = 3;
    A.values[2] = 3;
    A.row_indices[3] = 2;
    A.column_indices[3] = 4;
    A.values[3] = 4;

    ASSERT_EQUAL(A.is_sorted_by_row(), false);

    A.sort_by_row();

    ASSERT_EQUAL(A.is_sorted_by_row(), true);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCooMatrixIsSortedByRow);

template <class Space>
void TestCooMatrixIsSortedByRowAndColumn(void)
{
    cusp::coo_matrix<int, float, Space> A(5, 5, 4);

    A.row_indices[0] = 3;
    A.column_indices[0] = 1;
    A.values[0] = 1;
    A.row_indices[1] = 4;
    A.column_indices[1] = 2;
    A.values[1] = 2;
    A.row_indices[2] = 1;
    A.column_indices[2] = 3;
    A.values[2] = 3;
    A.row_indices[3] = 2;
    A.column_indices[3] = 4;
    A.values[3] = 4;

    ASSERT_EQUAL(A.is_sorted_by_row_and_column(), false);

    A.sort_by_row_and_column();

    ASSERT_EQUAL(A.is_sorted_by_row_and_column(), true);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCooMatrixIsSortedByRowAndColumn);


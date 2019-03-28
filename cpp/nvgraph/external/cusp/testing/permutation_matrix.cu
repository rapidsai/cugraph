#include <unittest/unittest.h>
#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/multiply.h>
#include <cusp/permutation_matrix.h>

template <class Space>
void TestPermutationMatrixBasicConstructor(void)
{
    cusp::permutation_matrix<int, Space> matrix(3);

    ASSERT_EQUAL(matrix.num_rows,              3);
    ASSERT_EQUAL(matrix.num_cols,              3);
    ASSERT_EQUAL(matrix.num_entries,           3);
    ASSERT_EQUAL(matrix.permutation.size(),    3);
}
DECLARE_HOST_DEVICE_UNITTEST(TestPermutationMatrixBasicConstructor);

template <class Space>
void TestPermutationMatrixCopyConstructor(void)
{
    cusp::permutation_matrix<int, Space> matrix(3);

    matrix.permutation[0] = 1;
    matrix.permutation[1] = 0;
    matrix.permutation[2] = 2;

    cusp::permutation_matrix<int, Space> copy_of_matrix(matrix);

    ASSERT_EQUAL(copy_of_matrix.num_rows,              3);
    ASSERT_EQUAL(copy_of_matrix.num_cols,              3);
    ASSERT_EQUAL(copy_of_matrix.num_entries,           3);
    ASSERT_EQUAL(copy_of_matrix.permutation.size(),    3);

    ASSERT_EQUAL(copy_of_matrix.permutation,    matrix.permutation);
}
DECLARE_HOST_DEVICE_UNITTEST(TestPermutationMatrixCopyConstructor);

template <class Space>
void TestPermutationMatrixResize(void)
{
    cusp::permutation_matrix<int, Space> matrix;

    matrix.resize(3);

    ASSERT_EQUAL(matrix.num_rows,              3);
    ASSERT_EQUAL(matrix.num_cols,              3);
    ASSERT_EQUAL(matrix.num_entries,           3);
    ASSERT_EQUAL(matrix.permutation.size(),    3);
}
DECLARE_HOST_DEVICE_UNITTEST(TestPermutationMatrixResize);

template <class Space>
void TestPermutationMatrixSwap(void)
{
    cusp::permutation_matrix<int, Space> A(2);
    cusp::permutation_matrix<int, Space> B(3);

    A.permutation[0] = 1;
    A.permutation[1] = 0;

    B.permutation[0] = 2;
    B.permutation[1] = 1;
    B.permutation[2] = 0;

    cusp::permutation_matrix<int, Space> A_copy(A);
    cusp::permutation_matrix<int, Space> B_copy(B);

    A.swap(B);

    ASSERT_EQUAL(A.num_rows,              3);
    ASSERT_EQUAL(A.num_cols,              3);
    ASSERT_EQUAL(A.num_entries,           3);
    ASSERT_EQUAL(A.permutation,    B_copy.permutation);

    ASSERT_EQUAL(B.num_rows,              2);
    ASSERT_EQUAL(B.num_cols,              2);
    ASSERT_EQUAL(B.num_entries,           2);
    ASSERT_EQUAL(B.permutation,    A_copy.permutation);
}
DECLARE_HOST_DEVICE_UNITTEST(TestPermutationMatrixSwap);

void TestPermutationMatrixRebind(void)
{
    typedef cusp::permutation_matrix<int, cusp::host_memory> HostMatrix;
    typedef HostMatrix::rebind<cusp::device_memory>::type   DeviceMatrix;

    HostMatrix   h_matrix(10);
    DeviceMatrix d_matrix(h_matrix);

    ASSERT_EQUAL(h_matrix.num_entries, d_matrix.num_entries);
}
DECLARE_UNITTEST(TestPermutationMatrixRebind);


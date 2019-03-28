#include <unittest/unittest.h>

#include <cusp/verify.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

template <typename MemorySpace>
void TestIsValidMatrixCoo(void)
{
    cusp::array2d<float, MemorySpace> A(3,3);
    A(0,0) = 0;
    A(0,1) = 1;
    A(0,2) = 0;
    A(1,0) = 1;
    A(1,1) = 0;
    A(1,2) = 1;
    A(2,0) = 0;
    A(2,1) = 1;
    A(2,2) = 0;

    // basic tests
    {
        cusp::coo_matrix<int, float, MemorySpace> M(A);
        ASSERT_EQUAL(cusp::is_valid_matrix(M), true);
    }

    // invalid row_indices
    {
        cusp::coo_matrix<int, float, MemorySpace> M(A);
        M.row_indices[0] = 1;
        M.row_indices[1] = 0;
        ASSERT_EQUAL(cusp::is_valid_matrix(M), false);
    }

    // invalid column_indices
    {
        cusp::coo_matrix<int, float, MemorySpace> M(A);
        M.column_indices[2] = -1;
        ASSERT_EQUAL(cusp::is_valid_matrix(M), false);
    }
    {
        cusp::coo_matrix<int, float, MemorySpace> M(A);
        M.column_indices[2] = 4;
        ASSERT_EQUAL(cusp::is_valid_matrix(M), false);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestIsValidMatrixCoo);


template <typename MemorySpace>
void TestIsValidMatrixCsr(void)
{
    cusp::array2d<float, MemorySpace> A(3,3);
    A(0,0) = 0;
    A(0,1) = 1;
    A(0,2) = 0;
    A(1,0) = 1;
    A(1,1) = 0;
    A(1,2) = 1;
    A(2,0) = 0;
    A(2,1) = 1;
    A(2,2) = 0;

    // basic tests
    {
        cusp::csr_matrix<int, float, MemorySpace> M(A);
        ASSERT_EQUAL(cusp::is_valid_matrix(M), true);
    }

    // invalid row_offsets
    {
        cusp::csr_matrix<int, float, MemorySpace> M(A);
        M.row_offsets[0] = 0;
        M.row_offsets[1] = 1;
        M.row_offsets[2] = 5;
        M.row_offsets[3] = 4;
        ASSERT_EQUAL(cusp::is_valid_matrix(M), false);
    }

    // invalid column_indices
    {
        cusp::csr_matrix<int, float, MemorySpace> M(A);
        M.column_indices[2] = -1;
        ASSERT_EQUAL(cusp::is_valid_matrix(M), false);
    }
    {
        cusp::csr_matrix<int, float, MemorySpace> M(A);
        M.column_indices[2] = 4;
        ASSERT_EQUAL(cusp::is_valid_matrix(M), false);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestIsValidMatrixCsr);


template <typename MemorySpace>
void TestIsValidMatrixDia(void)
{
    cusp::array2d<float, MemorySpace> A(3,3);
    A(0,0) = 0;
    A(0,1) = 1;
    A(0,2) = 0;
    A(1,0) = 1;
    A(1,1) = 0;
    A(1,2) = 1;
    A(2,0) = 0;
    A(2,1) = 1;
    A(2,2) = 0;

    // basic tests
    {
        cusp::dia_matrix<int, float, MemorySpace> M(A);
        ASSERT_EQUAL(cusp::is_valid_matrix(M), true);
    }

    // invalid shapes
    {
        cusp::dia_matrix<int, float, MemorySpace> M(A);
        M.values.num_rows = 2;
        ASSERT_EQUAL(cusp::is_valid_matrix(M), false);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestIsValidMatrixDia);


template <typename MemorySpace>
void TestIsValidMatrixEll(void)
{
    cusp::array2d<float, MemorySpace> A(3,3);
    A(0,0) = 0;
    A(0,1) = 1;
    A(0,2) = 0;
    A(1,0) = 1;
    A(1,1) = 0;
    A(1,2) = 1;
    A(2,0) = 0;
    A(2,1) = 1;
    A(2,2) = 0;

    // basic tests
    {
        cusp::ell_matrix<int, float, MemorySpace> M(A);
        ASSERT_EQUAL(cusp::is_valid_matrix(M), true);
    }

    // invalid shapes
    {
        cusp::ell_matrix<int, float, MemorySpace> M(A);
        M.column_indices.num_rows = 3;
        M.column_indices.num_cols = 3;
        M.values.num_rows = 3;
        M.values.num_cols = 2;
        M.column_indices.values.resize(3*3);
        M.values.values.resize(3*2);
        ASSERT_EQUAL(cusp::is_valid_matrix(M), false);
    }
    {
        cusp::ell_matrix<int, float, MemorySpace> M(A);
        M.column_indices.num_rows = 2;
        M.column_indices.num_cols = 3;
        M.values.num_rows = 2;
        M.values.num_cols = 3;
        M.column_indices.values.resize(2*3);
        M.values.values.resize(2*3);
        ASSERT_EQUAL(cusp::is_valid_matrix(M), false);
    }

    // invalid column_indices
    {
        cusp::ell_matrix<int, float, MemorySpace> M(A);
        int invalid_index = cusp::ell_matrix<int, float, MemorySpace>::invalid_index;
        M.column_indices(0,0) = invalid_index;
        ASSERT_EQUAL(cusp::is_valid_matrix(M), false);
    }
    {
        cusp::ell_matrix<int, float, MemorySpace> M(A);
        M.column_indices(0,0) = -2;
        ASSERT_EQUAL(cusp::is_valid_matrix(M), false);
    }
    {
        cusp::ell_matrix<int, float, MemorySpace> M(A);
        M.column_indices(0,0) = 3;
        ASSERT_EQUAL(cusp::is_valid_matrix(M), false);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestIsValidMatrixEll);


template <typename MemorySpace>
void TestIsValidMatrixHyb(void)
{
    cusp::array2d<float, MemorySpace> A(3,3);
    A(0,0) = 0;
    A(0,1) = 1;
    A(0,2) = 0;
    A(1,0) = 1;
    A(1,1) = 0;
    A(1,2) = 1;
    A(2,0) = 0;
    A(2,1) = 1;
    A(2,2) = 0;

    // basic tests
    {
        cusp::hyb_matrix<int, float, MemorySpace> M(A);
        ASSERT_EQUAL(cusp::is_valid_matrix(M), true);
    }

    // invalid shapes
    {
        cusp::hyb_matrix<int, float, MemorySpace> M(A);
        M.ell.num_rows = 4;
        ASSERT_EQUAL(cusp::is_valid_matrix(M), false);
    }
    {
        cusp::hyb_matrix<int, float, MemorySpace> M(A);
        M.coo.num_rows = 4;
        ASSERT_EQUAL(cusp::is_valid_matrix(M), false);
    }

    // invalid num_entries
    {
        cusp::hyb_matrix<int, float, MemorySpace> M(A);
        M.num_entries = 5;
        ASSERT_EQUAL(cusp::is_valid_matrix(M), false);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestIsValidMatrixHyb);


template <typename MemorySpace>
void TestIsValidMatrixArray2d(void)
{
    cusp::array2d<float, MemorySpace> A(3,3);
    A(0,0) = 0;
    A(0,1) = 1;
    A(0,2) = 0;
    A(1,0) = 1;
    A(1,1) = 0;
    A(1,2) = 1;
    A(2,0) = 0;
    A(2,1) = 1;
    A(2,2) = 0;

    // basic tests
    {
        cusp::array2d<float, MemorySpace> M(A);
        ASSERT_EQUAL(cusp::is_valid_matrix(M), true);
    }

    // invalid shapes
    {
        cusp::array2d<float, MemorySpace> M(A);
        M.num_rows = 10;
        M.num_cols = 10;
        ASSERT_EQUAL(cusp::is_valid_matrix(M), false);
    }
    {
        cusp::array2d<float, MemorySpace> M(A);
        M.num_rows = 10;
        M.num_cols = 10;
        M.num_cols = 100;
        ASSERT_EQUAL(cusp::is_valid_matrix(M), false);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestIsValidMatrixArray2d);


template <typename MatrixType>
void TestAssertIsValidMatrix(void)
{
    typedef typename MatrixType::value_type ValueType;

    cusp::array2d<ValueType, cusp::host_memory> A(3,3);
    A(0,0) = 0;
    A(0,1) = 1;
    A(0,2) = 0;
    A(1,0) = 1;
    A(1,1) = 0;
    A(1,2) = 1;
    A(2,0) = 0;
    A(2,1) = 1;
    A(2,2) = 0;

    // should not throw
    {
        MatrixType M(A);
        cusp::assert_is_valid_matrix(M);
    }

    // should throw
    {
        MatrixType M(A);
        M.num_rows = 100;
        M.num_cols = 100;
        M.num_entries = 100;
        ASSERT_THROWS(cusp::assert_is_valid_matrix(M), cusp::format_exception);
    }
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestAssertIsValidMatrix);


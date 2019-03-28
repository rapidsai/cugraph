#include <unittest/unittest.h>

#include <cusp/transpose.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

template <typename MatrixType>
void initialize_matrix(MatrixType& matrix)
{
    typedef typename MatrixType::value_type ValueType;

    cusp::array2d<ValueType, cusp::host_memory> D(4,3);

    D(0,0) = 10.25;
    D(0,1) = 11.00;
    D(0,2) =  0.00;
    D(1,0) =  0.00;
    D(1,1) =  0.00;
    D(1,2) = 12.50;
    D(2,0) = 13.75;
    D(2,1) =  0.00;
    D(2,2) = 14.00;
    D(3,0) =  0.00;
    D(3,1) = 16.50;
    D(3,2) =  0.00;

    matrix = D;
}

template <typename MatrixType>
void verify_result(const MatrixType& matrix)
{
    typedef typename MatrixType::value_type ValueType;

    ASSERT_EQUAL(matrix.num_rows,    3);
    ASSERT_EQUAL(matrix.num_cols,    4);

    cusp::array2d<ValueType, cusp::host_memory> dense(matrix);

    ASSERT_EQUAL(dense(0,0), 10.25);
    ASSERT_EQUAL(dense(0,1),  0.00);
    ASSERT_EQUAL(dense(0,2), 13.75);
    ASSERT_EQUAL(dense(0,3),  0.00);
    ASSERT_EQUAL(dense(1,0), 11.00);
    ASSERT_EQUAL(dense(1,1),  0.00);
    ASSERT_EQUAL(dense(1,2),  0.00);
    ASSERT_EQUAL(dense(1,3), 16.50);
    ASSERT_EQUAL(dense(2,0),  0.00);
    ASSERT_EQUAL(dense(2,1), 12.50);
    ASSERT_EQUAL(dense(2,2), 14.00);
    ASSERT_EQUAL(dense(2,3),  0.00);
}

template <typename Matrix1, typename Matrix2>
void TestTranspose(void)
{
    typedef typename Matrix1::view View1;
    typedef typename Matrix2::view View2;

    Matrix1 A;

    initialize_matrix(A);

    {
        Matrix2 At;
        cusp::transpose(A, At);
        verify_result(At);
    }
    {
        View1 V(A);
        Matrix2 At;
        cusp::transpose(V, At);
        verify_result(At);
    }

    Matrix2 At;
    cusp::transpose(A, At);

    {
        View2 Vt(At);
        cusp::transpose(A, Vt);
        verify_result(Vt);
    }
    {
        View1 V(A);
        View2 Vt(At);
        cusp::transpose(V, Vt);
        verify_result(Vt);
    }
}

///////////////////////
// Instantiate Tests //
///////////////////////
template <class Space>
void TestTransposeArray2dVariablePitch(void)
{
    typedef typename cusp::array2d<float, Space, cusp::row_major>    RowMajor;
    typedef typename cusp::array2d<float, Space, cusp::column_major> ColumnMajor;

    TestTranspose<RowMajor,    RowMajor>();
    TestTranspose<ColumnMajor, ColumnMajor>();
    TestTranspose<RowMajor,    ColumnMajor>();
    TestTranspose<ColumnMajor, RowMajor>();

    // test with non-trivial pitch
    {
        RowMajor A(4,3);
        A.resize(4,3,5);
        A(0,0) = 10.25;
        A(0,1) = 11.00;
        A(0,2) =  0.00;
        A(1,0) =  0.00;
        A(1,1) =  0.00;
        A(1,2) = 12.50;
        A(2,0) = 13.75;
        A(2,1) =  0.00;
        A(2,2) = 14.00;
        A(3,0) =  0.00;
        A(3,1) = 16.50;
        A(3,2) =  0.00;

        {
            RowMajor    At;
            cusp::transpose(A, At);
            verify_result(At);
        }
        {
            ColumnMajor At;
            cusp::transpose(A, At);
            verify_result(At);
        }
        {
            RowMajor    At;
            At.resize(3,4,5);
            cusp::transpose(A, At);
            verify_result(At);
            ASSERT_EQUAL(At.pitch, 5);
        }
        {
            ColumnMajor At;
            At.resize(3,4,5);
            cusp::transpose(A, At);
            verify_result(At);
            ASSERT_EQUAL(At.pitch, 5);
        }
    }
    {
        ColumnMajor A(4,3);
        A.resize(4,3,5);
        A(0,0) = 10.25;
        A(0,1) = 11.00;
        A(0,2) =  0.00;
        A(1,0) =  0.00;
        A(1,1) =  0.00;
        A(1,2) = 12.50;
        A(2,0) = 13.75;
        A(2,1) =  0.00;
        A(2,2) = 14.00;
        A(3,0) =  0.00;
        A(3,1) = 16.50;
        A(3,2) =  0.00;

        {
            RowMajor    At;
            cusp::transpose(A, At);
            verify_result(At);
        }
        {
            ColumnMajor At;
            cusp::transpose(A, At);
            verify_result(At);
        }
        {
            RowMajor    At;
            At.resize(3,4,5);
            cusp::transpose(A, At);
            verify_result(At);
            ASSERT_EQUAL(At.pitch, 5);
        }
        {
            ColumnMajor At;
            At.resize(3,4,5);
            cusp::transpose(A, At);
            verify_result(At);
            ASSERT_EQUAL(At.pitch, 5);
        }
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestTransposeArray2dVariablePitch);

template <class Matrix>
void TestTranspose(void)
{
    TestTranspose<Matrix, Matrix>();
}
DECLARE_MATRIX_UNITTEST(TestTranspose);

template <typename MatrixType1, typename MatrixType2>
void transpose(my_system& system, const MatrixType1& A, MatrixType2& At)
{
    system.validate_dispatch();
    return;
}

void TestTransposeDispatch()
{
    // initialize testing variables
    cusp::csr_matrix<int, float, cusp::device_memory> A, At;

    my_system sys(0);

    // call with explicit dispatching
    cusp::transpose(sys, A, At);

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestTransposeDispatch);


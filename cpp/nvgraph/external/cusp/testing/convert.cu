#include <unittest/unittest.h>

#include <cusp/convert.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

#include <cusp/verify.h>


template <typename Matrix>
void reset_view(Matrix& view, cusp::coo_format)
{
    thrust::fill(view.row_indices.begin(),    view.row_indices.end(),    -1);
    thrust::fill(view.column_indices.begin(), view.column_indices.end(), -1);
    thrust::fill(view.values.begin(),         view.values.end(),         -1);
    view.resize(0,0,0);
}

template <typename Matrix>
void reset_view(Matrix& view, cusp::csr_format)
{
    thrust::fill(view.row_offsets.begin(),    view.row_offsets.end(),    -1);
    thrust::fill(view.column_indices.begin(), view.column_indices.end(), -1);
    thrust::fill(view.values.begin(),         view.values.end(),         -1);
    view.resize(0,0,0);
}

template <typename Matrix>
void reset_view(Matrix& view, cusp::dia_format)
{
    thrust::fill(view.diagonal_offsets.begin(), view.diagonal_offsets.end(), -1);
    thrust::fill(view.values.values.begin(),    view.values.values.end(),    -1);
    view.resize(0,0,0,0);
}

template <typename Matrix>
void reset_view(Matrix& view, cusp::ell_format)
{
    thrust::fill(view.column_indices.values.begin(), view.column_indices.values.end(), -1);
    thrust::fill(view.values.values.begin(),         view.values.values.end(),         -1);
    view.resize(0,0,0,0);
}

template <typename Matrix>
void reset_view(Matrix& view, cusp::hyb_format)
{
    reset_view(view.ell, cusp::ell_format());
    reset_view(view.coo, cusp::coo_format());
    view.resize(0,0,0,0,0);
}

template <typename Matrix>
void reset_view(Matrix& view, cusp::array2d_format)
{
    thrust::fill(view.values.begin(), view.values.end(), -1);
    view.resize(0,0);
}

template <typename IndexType, typename ValueType, typename Space>
void initialize_conversion_example(cusp::csr_matrix<IndexType, ValueType, Space> & csr)
{
    csr.resize(4, 4, 7);

    csr.row_offsets[0] = 0;
    csr.row_offsets[1] = 2;
    csr.row_offsets[2] = 3;
    csr.row_offsets[3] = 6;
    csr.row_offsets[4] = 7;

    csr.column_indices[0] = 0;
    csr.values[0] = 10.25;
    csr.column_indices[1] = 1;
    csr.values[1] = 11.00;
    csr.column_indices[2] = 2;
    csr.values[2] = 12.50;
    csr.column_indices[3] = 0;
    csr.values[3] = 13.75;
    csr.column_indices[4] = 2;
    csr.values[4] = 14.00;
    csr.column_indices[5] = 3;
    csr.values[5] = 15.25;
    csr.column_indices[6] = 1;
    csr.values[6] = 16.50;
}

template <typename IndexType, typename ValueType, typename Space>
void initialize_conversion_example(cusp::coo_matrix<IndexType, ValueType, Space> & coo)
{
    coo.resize(4, 4, 7);

    coo.row_indices[0] = 0;
    coo.column_indices[0] = 0;
    coo.values[0] = 10.25;
    coo.row_indices[1] = 0;
    coo.column_indices[1] = 1;
    coo.values[1] = 11.00;
    coo.row_indices[2] = 1;
    coo.column_indices[2] = 2;
    coo.values[2] = 12.50;
    coo.row_indices[3] = 2;
    coo.column_indices[3] = 0;
    coo.values[3] = 13.75;
    coo.row_indices[4] = 2;
    coo.column_indices[4] = 2;
    coo.values[4] = 14.00;
    coo.row_indices[5] = 2;
    coo.column_indices[5] = 3;
    coo.values[5] = 15.25;
    coo.row_indices[6] = 3;
    coo.column_indices[6] = 1;
    coo.values[6] = 16.50;
}

template <typename IndexType, typename ValueType, typename Space>
void initialize_conversion_example(cusp::dia_matrix<IndexType, ValueType, Space> & dia)
{
    dia.resize(4, 4, 7, 3, 1);

    dia.diagonal_offsets[0] = -2;
    dia.diagonal_offsets[1] =  0;
    dia.diagonal_offsets[2] =  1;

    dia.values.values[ 0] =  0.00;
    dia.values.values[ 1] =  0.00;
    dia.values.values[ 2] = 13.75;
    dia.values.values[ 3] = 16.50;
    dia.values.values[ 4] = 10.25;
    dia.values.values[ 5] =  0.00;
    dia.values.values[ 6] = 14.00;
    dia.values.values[ 7] =  0.00;
    dia.values.values[ 8] = 11.00;
    dia.values.values[ 9] = 12.50;
    dia.values.values[10] = 15.25;
    dia.values.values[11] =  0.00;
}

template <typename IndexType, typename ValueType, typename Space>
void initialize_conversion_example(cusp::ell_matrix<IndexType, ValueType, Space> & ell)
{
    ell.resize(4, 4, 7, 3, 1);

    const int X = cusp::ell_matrix<IndexType, ValueType, Space>::invalid_index;

    ell.column_indices.values[ 0] =  0;
    ell.values.values[ 0] = 10.25;
    ell.column_indices.values[ 1] =  2;
    ell.values.values[ 1] = 12.50;
    ell.column_indices.values[ 2] =  0;
    ell.values.values[ 2] = 13.75;
    ell.column_indices.values[ 3] =  1;
    ell.values.values[ 3] = 16.50;

    ell.column_indices.values[ 4] =  1;
    ell.values.values[ 4] = 11.00;
    ell.column_indices.values[ 5] =  X;
    ell.values.values[ 5] =  0.00;
    ell.column_indices.values[ 6] =  2;
    ell.values.values[ 6] = 14.00;
    ell.column_indices.values[ 7] =  X;
    ell.values.values[ 7] =  0.00;

    ell.column_indices.values[ 8] =  X;
    ell.values.values[ 8] =  0.00;
    ell.column_indices.values[ 9] =  X;
    ell.values.values[ 9] =  0.00;
    ell.column_indices.values[10] =  3;
    ell.values.values[10] = 15.25;
    ell.column_indices.values[11] =  X;
    ell.values.values[11] =  0.00;
}

template <typename IndexType, typename ValueType, typename Space>
void initialize_conversion_example(cusp::hyb_matrix<IndexType, ValueType, Space> & hyb)
{
    hyb.resize(4, 4, 4, 3, 1, 1);

    hyb.ell.column_indices.values[0] = 0;
    hyb.ell.values.values[0] = 10.25;
    hyb.ell.column_indices.values[1] = 2;
    hyb.ell.values.values[1] = 12.50;
    hyb.ell.column_indices.values[2] = 0;
    hyb.ell.values.values[2] = 13.75;
    hyb.ell.column_indices.values[3] = 1;
    hyb.ell.values.values[3] = 16.50;

    hyb.coo.row_indices[0] = 0;
    hyb.coo.column_indices[0] = 1;
    hyb.coo.values[0] = 11.00;
    hyb.coo.row_indices[1] = 2;
    hyb.coo.column_indices[1] = 2;
    hyb.coo.values[1] = 14.00;
    hyb.coo.row_indices[2] = 2;
    hyb.coo.column_indices[2] = 3;
    hyb.coo.values[2] = 15.25;
}

template <typename ValueType, typename Space, class Orientation>
void initialize_conversion_example(cusp::array2d<ValueType, Space, Orientation> & dense)
{
    dense.resize(4, 4);

    dense(0,0) = 10.25;
    dense(0,1) = 11.00;
    dense(0,2) =  0.00;
    dense(0,3) =  0.00;
    dense(1,0) =  0.00;
    dense(1,1) =  0.00;
    dense(1,2) = 12.50;
    dense(1,3) =  0.00;
    dense(2,0) = 13.75;
    dense(2,1) =  0.00;
    dense(2,2) = 14.00;
    dense(2,3) = 15.25;
    dense(3,0) =  0.00;
    dense(3,1) = 16.50;
    dense(3,2) =  0.00;
    dense(3,3) =  0.00;
}

template <typename MatrixType>
void verify_conversion_example(const MatrixType& matrix)
{
    typedef typename MatrixType::value_type ValueType;

    cusp::array2d<ValueType, cusp::host_memory> dense(matrix);

    ASSERT_EQUAL(dense.num_rows,    4);
    ASSERT_EQUAL(dense.num_cols,    4);
    ASSERT_EQUAL(dense.num_entries, 16);

    ASSERT_EQUAL(dense(0,0), 10.25);
    ASSERT_EQUAL(dense(0,1), 11.00);
    ASSERT_EQUAL(dense(0,2),  0.00);
    ASSERT_EQUAL(dense(0,3),  0.00);
    ASSERT_EQUAL(dense(1,0),  0.00);
    ASSERT_EQUAL(dense(1,1),  0.00);
    ASSERT_EQUAL(dense(1,2), 12.50);
    ASSERT_EQUAL(dense(1,3),  0.00);
    ASSERT_EQUAL(dense(2,0), 13.75);
    ASSERT_EQUAL(dense(2,1),  0.00);
    ASSERT_EQUAL(dense(2,2), 14.00);
    ASSERT_EQUAL(dense(2,3), 15.25);
    ASSERT_EQUAL(dense(3,0),  0.00);
    ASSERT_EQUAL(dense(3,1), 16.50);
    ASSERT_EQUAL(dense(3,2),  0.00);
    ASSERT_EQUAL(dense(3,3),  0.00);
}


template <class DestinationType, class HostSourceType>
void TestConversionToMatrixFormat(void)
{
    typedef typename HostSourceType::template rebind<cusp::device_memory>::type DeviceSourceType;
    typedef typename HostSourceType::view   HostSourceViewType;
    typedef typename DeviceSourceType::view DeviceSourceViewType;
    typedef typename DestinationType::view  DestinationViewType;

    {
        HostSourceType src;

        initialize_conversion_example(src);

        HostSourceViewType view(src);

        {
            DestinationType dst(src);
            verify_conversion_example(dst);
            cusp::assert_is_valid_matrix(dst);
        }
        {
            DestinationType dst;
            dst = src;
            verify_conversion_example(dst);
            cusp::assert_is_valid_matrix(dst);
        }
        {
            DestinationType dst;
            cusp::convert(src,dst);
            verify_conversion_example(dst);
            cusp::assert_is_valid_matrix(dst);
        }
        {
            DestinationType dst;
            cusp::convert(src,dst);
            verify_conversion_example(dst);
            cusp::assert_is_valid_matrix(dst);
        }
        {
            DestinationType dst;
            cusp::convert(view,dst);
            verify_conversion_example(dst);
            cusp::assert_is_valid_matrix(dst);
        }
    }

    {
        DeviceSourceType src;

        initialize_conversion_example(src);

        DeviceSourceViewType view(src);

        {
            DestinationType dst(src);
            verify_conversion_example(dst);
            cusp::assert_is_valid_matrix(dst);
        }
        {
            DestinationType dst;
            dst = src;
            verify_conversion_example(dst);
            cusp::assert_is_valid_matrix(dst);
        }
        {
            DestinationType dst;
            cusp::convert(src,dst);
            verify_conversion_example(dst);
            cusp::assert_is_valid_matrix(dst);
        }
        {
            DestinationType dst;
            cusp::convert(view,dst);
            verify_conversion_example(dst);
            cusp::assert_is_valid_matrix(dst);
        }
    }

    {
        HostSourceType src;
        initialize_conversion_example(src);

        DestinationType dst(src);
        DestinationViewType view(dst);

        reset_view(view, typename DestinationViewType::format());

        cusp::convert(src, view);

        // check that view is correct
        verify_conversion_example(view);
        cusp::assert_is_valid_matrix(view);

        // check that dst is correct (detects lightweight copies)
        verify_conversion_example(dst);
        cusp::assert_is_valid_matrix(dst);
    }

    {
        DeviceSourceType src;
        initialize_conversion_example(src);

        DestinationType dst(src);
        DestinationViewType view(dst);

        reset_view(view, typename DestinationViewType::format());

        cusp::convert(src, view);

        // check that view is correct
        verify_conversion_example(view);
        cusp::assert_is_valid_matrix(view);

        // check that dst is correct (detects lightweight copies)
        verify_conversion_example(dst);
        cusp::assert_is_valid_matrix(dst);
    }
}


template <typename Matrix>
void TestConversionToMatrix(void)
{
    typedef typename Matrix::value_type ValueType;

    TestConversionToMatrixFormat<Matrix, cusp::coo_matrix<int, ValueType, cusp::host_memory> >();
    TestConversionToMatrixFormat<Matrix, cusp::csr_matrix<int, ValueType, cusp::host_memory> >();
    TestConversionToMatrixFormat<Matrix, cusp::dia_matrix<int, ValueType, cusp::host_memory> >();
    TestConversionToMatrixFormat<Matrix, cusp::ell_matrix<int, ValueType, cusp::host_memory> >();
    TestConversionToMatrixFormat<Matrix, cusp::hyb_matrix<int, ValueType, cusp::host_memory> >();
    TestConversionToMatrixFormat<Matrix, cusp::array2d<ValueType, cusp::host_memory, cusp::row_major>    >();
    TestConversionToMatrixFormat<Matrix, cusp::array2d<ValueType, cusp::host_memory, cusp::column_major> >();
}

///////////////////////////
// Main Conversion Tests //
///////////////////////////
template <class Matrix>
void TestConversionTo(void)
{
    TestConversionToMatrix<Matrix>();
}
DECLARE_MATRIX_UNITTEST(TestConversionTo);

//////////////////////////////
// Special Conversion Tests //
//////////////////////////////
void TestConvertCsrToDiaMatrixHost(void)
{
    cusp::csr_matrix<int, float, cusp::host_memory> csr;
    cusp::dia_matrix<int, float, cusp::host_memory> dia;

    // initialize host matrix
    initialize_conversion_example(csr);

    // make dia with an alignment of 1
    thrust::execution_policy<thrust::system::cpp::detail::tag> policy;
    cusp::csr_format format1;
    cusp::dia_format format2;
    cusp::system::detail::generic::convert(policy, csr, dia, format1, format2, 1);

    // compare csr and dia
    ASSERT_EQUAL(dia.num_rows,    csr.num_rows);
    ASSERT_EQUAL(dia.num_cols,    csr.num_cols);
    ASSERT_EQUAL(dia.num_entries, csr.num_entries);

    ASSERT_EQUAL(dia.diagonal_offsets[ 0],  -2);
    ASSERT_EQUAL(dia.diagonal_offsets[ 1],   0);
    ASSERT_EQUAL(dia.diagonal_offsets[ 2],   1);

    ASSERT_EQUAL(dia.values(0,0),  0.00);
    ASSERT_EQUAL(dia.values(1,0),  0.00);
    ASSERT_EQUAL(dia.values(2,0), 13.75);
    ASSERT_EQUAL(dia.values(3,0), 16.50);

    ASSERT_EQUAL(dia.values(0,1), 10.25);
    ASSERT_EQUAL(dia.values(1,1),  0.00);
    ASSERT_EQUAL(dia.values(2,1), 14.00);
    ASSERT_EQUAL(dia.values(3,1),  0.00);

    ASSERT_EQUAL(dia.values(0,2), 11.00);
    ASSERT_EQUAL(dia.values(1,2), 12.50);
    ASSERT_EQUAL(dia.values(2,2), 15.25);
    ASSERT_EQUAL(dia.values(3,2),  0.00);

    cusp::assert_is_valid_matrix(dia);
}
DECLARE_UNITTEST(TestConvertCsrToDiaMatrixHost);

void TestConvertCsrToEllMatrixHost(void)
{
    cusp::csr_matrix<int, float, cusp::host_memory> csr;
    cusp::ell_matrix<int, float, cusp::host_memory> ell;

    // initialize host matrix
    initialize_conversion_example(csr);

    // make ell with an alignment of 1
    thrust::execution_policy<thrust::system::cpp::detail::tag> policy;
    cusp::csr_format format1;
    cusp::ell_format format2;
    cusp::system::detail::generic::convert(policy, csr, ell, format1, format2, 3.0, 1);

    const int X = cusp::ell_matrix<int, float, cusp::host_memory>::invalid_index;

    // compare csr and dia
    ASSERT_EQUAL(ell.num_rows,    csr.num_rows);
    ASSERT_EQUAL(ell.num_cols,    csr.num_cols);
    ASSERT_EQUAL(ell.num_entries, csr.num_entries);
    ASSERT_EQUAL(ell.column_indices.num_rows, 4);
    ASSERT_EQUAL(ell.column_indices.num_cols, 3);
    ASSERT_EQUAL(ell.column_indices.values[ 0],  0);
    ASSERT_EQUAL(ell.values.values[ 0], 10.25);
    ASSERT_EQUAL(ell.column_indices.values[ 1],  2);
    ASSERT_EQUAL(ell.values.values[ 1], 12.50);
    ASSERT_EQUAL(ell.column_indices.values[ 2],  0);
    ASSERT_EQUAL(ell.values.values[ 2], 13.75);
    ASSERT_EQUAL(ell.column_indices.values[ 3],  1);
    ASSERT_EQUAL(ell.values.values[ 3], 16.50);

    ASSERT_EQUAL(ell.column_indices.values[ 4],  1);
    ASSERT_EQUAL(ell.values.values[ 4], 11.00);
    ASSERT_EQUAL(ell.column_indices.values[ 5],  X);
    ASSERT_EQUAL(ell.values.values[ 5],  0.00);
    ASSERT_EQUAL(ell.column_indices.values[ 6],  2);
    ASSERT_EQUAL(ell.values.values[ 6], 14.00);
    ASSERT_EQUAL(ell.column_indices.values[ 7],  X);
    ASSERT_EQUAL(ell.values.values[ 7],  0.00);

    ASSERT_EQUAL(ell.column_indices.values[ 8],  X);
    ASSERT_EQUAL(ell.values.values[ 8],  0.00);
    ASSERT_EQUAL(ell.column_indices.values[ 9],  X);
    ASSERT_EQUAL(ell.values.values[ 9],  0.00);
    ASSERT_EQUAL(ell.column_indices.values[10],  3);
    ASSERT_EQUAL(ell.values.values[10], 15.25);
    ASSERT_EQUAL(ell.column_indices.values[11],  X);
    ASSERT_EQUAL(ell.values.values[11],  0.00);

    cusp::assert_is_valid_matrix(ell);
}
DECLARE_UNITTEST(TestConvertCsrToEllMatrixHost);

template <class Matrix>
void TestConversionFromArray1dTo(void)
{
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space MemorySpace;
    typedef cusp::array1d<ValueType, MemorySpace> Array1d;
    typedef cusp::array2d<ValueType, MemorySpace> Array2d;

    Array1d a(4);
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;
    a[3] = 4;

    Matrix m;
    cusp::convert(a, m);

    Array2d b;
    cusp::convert(m, b);

    ASSERT_EQUAL(b.num_rows, 4);
    ASSERT_EQUAL(b.num_cols, 1);
    ASSERT_EQUAL(b(0,0), 1);
    ASSERT_EQUAL(b(1,0), 2);
    ASSERT_EQUAL(b(2,0), 3);
    ASSERT_EQUAL(b(3,0), 4);
}
DECLARE_MATRIX_UNITTEST(TestConversionFromArray1dTo);

template <class Matrix>
void TestConversionToArray1dFrom(void)
{
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space MemorySpace;
    typedef cusp::array1d<ValueType, MemorySpace> Array1d;
    typedef cusp::array2d<ValueType, MemorySpace> Array2d;

    // column vector
    {
        Array2d a(4,1);
        a(0,0) = 1;
        a(1,0) = 2;
        a(2,0) = 3;
        a(3,0) = 4;

        Matrix m(a);

        Array1d b;
        cusp::convert(m, b);

        ASSERT_EQUAL(b.size(), 4);
        ASSERT_EQUAL(b[0], 1);
        ASSERT_EQUAL(b[1], 2);
        ASSERT_EQUAL(b[2], 3);
        ASSERT_EQUAL(b[3], 4);
    }

    // row vector
    {
        Array2d a(1,4);
        a(0,0) = 1;
        a(0,1) = 2;
        a(0,2) = 3;
        a(0,3) = 4;

        Matrix m(a);

        Array1d b;
        cusp::convert(m, b);

        ASSERT_EQUAL(b.size(), 4);
        ASSERT_EQUAL(b[0], 1);
        ASSERT_EQUAL(b[1], 2);
        ASSERT_EQUAL(b[2], 3);
        ASSERT_EQUAL(b[3], 4);
    }

    // invalid case
    {
        Array2d a(2,2);
        a(0,0) = 1;
        a(0,1) = 2;
        a(1,0) = 3;
        a(1,1) = 4;

        Matrix m(a);

        Array1d b;
        ASSERT_THROWS(cusp::convert(m, b), cusp::format_conversion_exception);
    }
}
DECLARE_MATRIX_UNITTEST(TestConversionToArray1dFrom);

template <typename MemorySpace>
void TestConversionFromArray1dToPitchedArray2d(void)
{
    cusp::array1d<int, MemorySpace> A(3);
    A[0] = 1;
    A[1] = 2;
    A[2] = 3;

    // row-major
    {
        cusp::array2d<int, MemorySpace, cusp::row_major> B;
        B.resize(3,1,2);

        cusp::convert(A, B);

        ASSERT_EQUAL(B.values.size(), 6);
        ASSERT_EQUAL(B.pitch, 2);

        ASSERT_EQUAL(B(0,0), 1);
        ASSERT_EQUAL(B(1,0), 2);
        ASSERT_EQUAL(B(2,0), 3);
    }

    // column-major
    {
        cusp::array2d<int, MemorySpace, cusp::column_major> B;
        B.resize(3,1,4);

        cusp::convert(A, B);

        ASSERT_EQUAL(B.values.size(), 4);
        ASSERT_EQUAL(B.pitch, 4);

        ASSERT_EQUAL(B(0,0), 1);
        ASSERT_EQUAL(B(1,0), 2);
        ASSERT_EQUAL(B(2,0), 3);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestConversionFromArray1dToPitchedArray2d);

template <typename MemorySpace>
void TestConversionFromPitchedArray2dToArray1d(void)
{
    // row-major
    {
        cusp::array2d<int, MemorySpace, cusp::row_major> A;
        A.resize(3,1,2);
        A(0,0) = 1;
        A(1,0) = 2;
        A(2,0) = 3;

        cusp::array1d<int, MemorySpace> B(3);
        cusp::convert(A, B);

        ASSERT_EQUAL(B.size(), 3);
        ASSERT_EQUAL(B[0], 1);
        ASSERT_EQUAL(B[1], 2);
        ASSERT_EQUAL(B[2], 3);
    }

    // column-major
    {
        cusp::array2d<int, MemorySpace, cusp::column_major> A;
        A.resize(3,1,4);
        A(0,0) = 1;
        A(1,0) = 2;
        A(2,0) = 3;

        cusp::array1d<int, MemorySpace> B(3);
        cusp::convert(A, B);

        ASSERT_EQUAL(B.size(), 3);
        ASSERT_EQUAL(B[0], 1);
        ASSERT_EQUAL(B[1], 2);
        ASSERT_EQUAL(B[2], 3);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestConversionFromPitchedArray2dToArray1d);

template <typename MatrixType1, typename MatrixType2>
void convert(my_system& system, const MatrixType1& A, MatrixType2& At)
{
    system.validate_dispatch();
    return;
}

void TestConvertDispatch()
{
    // initialize testing variables
    cusp::csr_matrix<int, float, cusp::device_memory> A;
    cusp::hyb_matrix<int, float, cusp::device_memory> B;

    my_system sys(0);

    // call with explicit dispatching
    cusp::convert(sys, A, B);

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestConvertDispatch);


#include <unittest/unittest.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

#include <cusp/format_utils.h>

template <class Space>
void TestOffsetsToIndices(void)
{
    cusp::array1d<int, Space> offsets(8);
    offsets[0] =  0;
    offsets[1] =  0;
    offsets[2] =  0;
    offsets[3] =  1;
    offsets[4] =  1;
    offsets[5] =  2;
    offsets[6] =  5;
    offsets[7] = 10;

    cusp::array1d<int, Space> expected(10);
    expected[0] = 2;
    expected[1] = 4;
    expected[2] = 5;
    expected[3] = 5;
    expected[4] = 5;
    expected[5] = 6;
    expected[6] = 6;
    expected[7] = 6;
    expected[8] = 6;
    expected[9] = 6;

    cusp::array1d<int, Space> indices(10);
    cusp::offsets_to_indices(offsets, indices);

    ASSERT_EQUAL(indices, expected);
}
DECLARE_HOST_DEVICE_UNITTEST(TestOffsetsToIndices);


template <class Space>
void TestIndicesToOffsets(void)
{
    cusp::array1d<int, Space> indices(10);
    indices[0] = 2;
    indices[1] = 4;
    indices[2] = 5;
    indices[3] = 5;
    indices[4] = 5;
    indices[5] = 6;
    indices[6] = 6;
    indices[7] = 6;
    indices[8] = 6;
    indices[9] = 6;

    cusp::array1d<int, Space> expected(8);
    expected[0] =  0;
    expected[1] =  0;
    expected[2] =  0;
    expected[3] =  1;
    expected[4] =  1;
    expected[5] =  2;
    expected[6] =  5;
    expected[7] = 10;

    cusp::array1d<int, Space> offsets(8);
    cusp::indices_to_offsets(indices, offsets);

    ASSERT_EQUAL(offsets, expected);
}
DECLARE_HOST_DEVICE_UNITTEST(TestIndicesToOffsets);

template <class Matrix>
void TestExtractDiagonal(void)
{
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space Space;

    {
        cusp::array2d<ValueType, Space> A(2,2);
        A(0,0) = 1.0;
        A(0,1) = 2.0;
        A(1,0) = 3.0;
        A(1,1) = 4.0;

        cusp::array1d<ValueType, Space> expected(2);
        expected[0] = 1.0;
        expected[1] = 4.0;

        cusp::array1d<ValueType, Space> output;

        cusp::extract_diagonal(Matrix(A), output);

        ASSERT_EQUAL(output, expected);
    }

    {
        cusp::array2d<ValueType, Space> A(3,4);
        A(0,0) = 0.0;
        A(0,1) = 0.0;
        A(0,2) = 4.0;
        A(0,3) = 0.0;
        A(1,0) = 1.0;
        A(1,1) = 2.0;
        A(1,2) = 0.0;
        A(1,3) = 6.0;
        A(2,0) = 0.0;
        A(2,1) = 3.0;
        A(2,2) = 5.0;
        A(2,3) = 0.0;

        cusp::array1d<ValueType, Space> expected(3);
        expected[0] = 0.0;
        expected[1] = 2.0;
        expected[2] = 5.0;

        cusp::array1d<ValueType, Space> output;

        cusp::extract_diagonal(Matrix(A), output);

        ASSERT_EQUAL(output, expected);
    }

    {
        cusp::array2d<ValueType, Space> A(5,5);
        A(0,0) = 1.0;
        A(0,1) = 1.0;
        A(0,2) = 2.0;
        A(0,3) = 0.0;
        A(0,4) = 0.0;
        A(1,0) = 3.0;
        A(1,1) = 4.0;
        A(1,2) = 0.0;
        A(1,3) = 0.0;
        A(1,4) = 0.0;
        A(2,0) = 0.0;
        A(2,1) = 6.0;
        A(2,2) = 0.0;
        A(2,3) = 0.0;
        A(2,4) = 0.0;
        A(3,0) = 0.0;
        A(3,1) = 0.0;
        A(3,2) = 7.0;
        A(3,3) = 8.0;
        A(3,4) = 0.0;
        A(4,0) = 0.0;
        A(4,1) = 0.0;
        A(4,2) = 0.0;
        A(4,3) = 0.0;
        A(4,4) = 9.0;

        cusp::array1d<ValueType, Space> expected(5);
        expected[0] = 1.0;
        expected[1] = 4.0;
        expected[2] = 0.0;
        expected[3] = 8.0;
        expected[4] = 9.0;

        cusp::array1d<ValueType, Space> output;

        cusp::extract_diagonal(Matrix(A), output);

        ASSERT_EQUAL(output, expected);
    }
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestExtractDiagonal);


#include <unittest/unittest.h>

#include <cusp/precond/diagonal.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

#include <cusp/multiply.h>

template <class MatrixType>
void _TestDiagonalPreconditioner(void)
{
    typedef typename MatrixType::value_type   ValueType;
    typedef typename MatrixType::memory_space Space;

    cusp::array2d<ValueType, Space> A(5,5);
    A(0,0) = 1.0;
    A(0,1) = 1.0;
    A(0,2) = 2.0;
    A(0,3) = 0.0;
    A(0,4) = 0.0;
    A(1,0) = 3.0;
    A(1,1) = 2.0;
    A(1,2) = 0.0;
    A(1,3) = 0.0;
    A(1,4) = 5.0;
    A(2,0) = 0.0;
    A(2,1) = 0.0;
    A(2,2) = 0.5;
    A(2,3) = 0.0;
    A(2,4) = 0.0;
    A(3,0) = 0.0;
    A(3,1) = 6.0;
    A(3,2) = 7.0;
    A(3,3) = 4.0;
    A(3,4) = 0.0;
    A(4,0) = 0.0;
    A(4,1) = 8.0;
    A(4,2) = 0.0;
    A(4,3) = 0.0;
    A(4,4) = 0.25;

    cusp::array1d<ValueType, Space> input(5, 1.0);
    cusp::array1d<ValueType, Space> expected(5);
    expected[0] = 1.00;
    expected[1] = 0.50;
    expected[2] = 2.00;
    expected[3] = 0.25;
    expected[4] = 4.00;

    cusp::array1d<ValueType, Space> output(5, 0.0f);

    MatrixType M(A);
    cusp::precond::diagonal<ValueType, Space> D(M);

    ASSERT_EQUAL(D.num_rows,    5);
    ASSERT_EQUAL(D.num_cols,    5);
    ASSERT_EQUAL(D.num_entries, 5);

    D(input, output);

    ASSERT_EQUAL(output, expected);

    cusp::multiply(D, input, output);

    ASSERT_EQUAL(output, expected);
}

template <class SparseMatrix>
void TestDiagonalPreconditioner(void)
{
    _TestDiagonalPreconditioner<SparseMatrix>();
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestDiagonalPreconditioner);


#include <unittest/unittest.h>

#include <cusp/precond/aggregation/strength.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

template <typename SparseMatrix>
void TestSymmetricStrengthOfConnection(void)
{
    typedef typename SparseMatrix::value_type ValueType;
    typedef cusp::array2d<ValueType,cusp::host_memory> Matrix;

    // input
    Matrix M(4,4);
    M(0,0) =  3.0;
    M(0,1) =  0.0;
    M(0,2) =  1.0;
    M(0,3) =  2.0;
    M(1,0) =  0.0;
    M(1,1) =  4.0;
    M(1,2) =  3.0;
    M(1,3) =  4.0;
    M(2,0) = -1.0;
    M(2,1) = -3.0;
    M(2,2) =  5.0;
    M(2,3) =  5.0;
    M(3,0) = -2.0;
    M(3,1) = -4.0;
    M(3,2) = -5.0;
    M(3,3) =  6.0;

    // default: all connections are strong
    {
        SparseMatrix A = M;
        SparseMatrix S;
        cusp::precond::aggregation::symmetric_strength_of_connection(A, S);
        Matrix result = S;
        ASSERT_EQUAL(result == M, true);
    }

    // theta = 0.0: all connections are strong
    {
        SparseMatrix A = M;
        SparseMatrix S;
        cusp::precond::aggregation::symmetric_strength_of_connection(A, S, 0.0);
        Matrix result = S;
        ASSERT_EQUAL(result == M, true);
    }

    // theta = 0.5
    {
        SparseMatrix A = M;
        SparseMatrix S;
        cusp::precond::aggregation::symmetric_strength_of_connection(A, S, 0.5);
        Matrix result = S;

        // expected output
        Matrix N(4,4);
        N(0,0) =  3.0;
        N(0,1) =  0.0;
        N(0,2) =  0.0;
        N(0,3) =  0.0;
        N(1,0) =  0.0;
        N(1,1) =  4.0;
        N(1,2) =  3.0;
        N(1,3) =  4.0;
        N(2,0) =  0.0;
        N(2,1) = -3.0;
        N(2,2) =  5.0;
        N(2,3) =  5.0;
        N(3,0) =  0.0;
        N(3,1) = -4.0;
        N(3,2) = -5.0;
        N(3,3) =  6.0;
        ASSERT_EQUAL(result == N, true);
    }

    // theta = 0.75
    {
        SparseMatrix A = M;
        SparseMatrix S;
        cusp::precond::aggregation::symmetric_strength_of_connection(A, S, 0.75);
        Matrix result = S;

        // expected output
        Matrix N(4,4);
        N(0,0) =  3.0;
        N(0,1) =  0.0;
        N(0,2) =  0.0;
        N(0,3) =  0.0;
        N(1,0) =  0.0;
        N(1,1) =  4.0;
        N(1,2) =  0.0;
        N(1,3) =  4.0;
        N(2,0) =  0.0;
        N(2,1) =  0.0;
        N(2,2) =  5.0;
        N(2,3) =  5.0;
        N(3,0) =  0.0;
        N(3,1) = -4.0;
        N(3,2) = -5.0;
        N(3,3) =  6.0;
        ASSERT_EQUAL(result == N, true);
    }

    // theta = 0.9
    {
        SparseMatrix A = M;
        SparseMatrix S;
        cusp::precond::aggregation::symmetric_strength_of_connection(A, S, 0.9);
        Matrix result = S;

        // expected output
        Matrix N(4,4);
        N(0,0) =  3.0;
        N(0,1) =  0.0;
        N(0,2) =  0.0;
        N(0,3) =  0.0;
        N(1,0) =  0.0;
        N(1,1) =  4.0;
        N(1,2) =  0.0;
        N(1,3) =  0.0;
        N(2,0) =  0.0;
        N(2,1) =  0.0;
        N(2,2) =  5.0;
        N(2,3) =  5.0;
        N(3,0) =  0.0;
        N(3,1) =  0.0;
        N(3,2) = -5.0;
        N(3,3) =  6.0;
        ASSERT_EQUAL(result == N, true);
    }
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestSymmetricStrengthOfConnection);


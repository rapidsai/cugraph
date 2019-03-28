#include <unittest/unittest.h>

#include <cusp/elementwise.h>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

#include <cusp/gallery/poisson.h>
#include <cusp/gallery/random.h>

template <typename Vector>
void example_matrices(Vector& matrices)
{
    cusp::array2d<float,cusp::host_memory> A(3,2);
    A(0,0) = 1.0;
    A(0,1) = 2.0;
    A(1,0) = 3.0;
    A(1,1) = 0.0;
    A(2,0) = 5.0;
    A(2,1) = 6.0;

    cusp::array2d<float,cusp::host_memory> B(2,4);
    B(0,0) = 0.0;
    B(0,1) = 2.0;
    B(0,2) = 3.0;
    B(0,3) = 4.0;
    B(1,0) = 5.0;
    B(1,1) = 0.0;
    B(1,2) = 0.0;
    B(1,3) = 8.0;

    cusp::array2d<float,cusp::host_memory> C(2,2);
    C(0,0) = 0.0;
    C(0,1) = 0.0;
    C(1,0) = 3.0;
    C(1,1) = 5.0;

    cusp::array2d<float,cusp::host_memory> D(2,1);
    D(0,0) = 2.0;
    D(1,0) = 3.0;

    cusp::array2d<float,cusp::host_memory> E(2,2);
    E(0,0) = 0.0;
    E(0,1) = 0.0;
    E(1,0) = 0.0;
    E(1,1) = 0.0;

    cusp::array2d<float,cusp::host_memory> F(3,2);
    F(0,0) = 0.0;
    F(0,1) = 1.5;
    F(1,0) = 0.5;
    F(1,1) = 0.0;
    F(2,0) = 0.0;
    F(2,1) = 0.0;

    cusp::array2d<float,cusp::host_memory> G;
    cusp::gallery::poisson5pt(G, 4, 6);

    cusp::array2d<float,cusp::host_memory> H;
    cusp::gallery::poisson5pt(H, 8, 3);

    cusp::array2d<float,cusp::host_memory> I;
    cusp::gallery::random(I, 24, 24, 150);

    cusp::array2d<float,cusp::host_memory> J;
    cusp::gallery::random(J, 24, 24, 50);

    cusp::array2d<float,cusp::host_memory> K;
    cusp::gallery::random(K, 24, 12, 20);

    matrices.push_back(A);
    matrices.push_back(B);
    matrices.push_back(C);
    matrices.push_back(D);
    matrices.push_back(E);
    matrices.push_back(F);
    matrices.push_back(G);
    matrices.push_back(H);
    matrices.push_back(I);
    matrices.push_back(J);
    matrices.push_back(K);
}


template <typename SparseMatrix>
void TestAdd(void)
{
    typedef typename SparseMatrix::value_type ValueType;
    typedef cusp::array2d<ValueType,cusp::host_memory> DenseMatrix;

    //thrust::host_vector< DenseMatrix > matrices;
    std::vector< DenseMatrix > matrices;

    example_matrices(matrices);

    // test add for every pair of compatible matrices
    for(size_t i = 0; i < matrices.size(); i++)
    {
        for(size_t j = 0; j < matrices.size(); j++)
        {
            const DenseMatrix& A = matrices[i];
            const DenseMatrix& B = matrices[j];

            if (A.num_rows == B.num_rows && A.num_cols == B.num_cols)
            {
                DenseMatrix C;
                cusp::add(A, B, C);

                // test containers
                SparseMatrix _A(A), _B(B), _C;
                cusp::add(_A, _B, _C);

                ASSERT_EQUAL(C == DenseMatrix(_C), true);

                // test views
                typename SparseMatrix::view _Aview(_A), _Bview(_B), _Cview(_C);

                cusp::add(_Aview, _Bview, _C);
                ASSERT_EQUAL(C == DenseMatrix(_Cview), true);

                cusp::add(_Aview, _Bview, _Cview);
                ASSERT_EQUAL(C == DenseMatrix(_Cview), true);
            }
        }
    }

    SparseMatrix A = DenseMatrix(2,2,1);
    SparseMatrix B = DenseMatrix(2,3,1);
    SparseMatrix C = DenseMatrix(3,2,1);
    SparseMatrix D;

    ASSERT_THROWS(cusp::add(A,B,D), cusp::invalid_input_exception);
    ASSERT_THROWS(cusp::add(A,C,D), cusp::invalid_input_exception);
    ASSERT_THROWS(cusp::add(B,C,D), cusp::invalid_input_exception);
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestAdd);


template <typename SparseMatrix>
void TestSubtract(void)
{
    typedef typename SparseMatrix::value_type ValueType;
    typedef cusp::array2d<ValueType,cusp::host_memory> DenseMatrix;

    std::vector< DenseMatrix > matrices;

    example_matrices(matrices);

    // test add for every pair of compatible matrices
    for(size_t i = 0; i < matrices.size(); i++)
    {
        for(size_t j = 0; j < matrices.size(); j++)
        {
            const DenseMatrix& A = matrices[i];
            const DenseMatrix& B = matrices[j];

            if (A.num_rows == B.num_rows && A.num_cols == B.num_cols)
            {
                DenseMatrix C;
                cusp::subtract(A, B, C);

                SparseMatrix _A(A), _B(B), _C;
                cusp::subtract(_A, _B, _C);

                ASSERT_EQUAL(C == DenseMatrix(_C), true);
            }
        }
    }

    SparseMatrix A = DenseMatrix(2,2,1);
    SparseMatrix B = DenseMatrix(2,3,1);
    SparseMatrix C = DenseMatrix(3,2,1);
    SparseMatrix D;

    ASSERT_THROWS(cusp::subtract(A,B,D), cusp::invalid_input_exception);
    ASSERT_THROWS(cusp::subtract(A,C,D), cusp::invalid_input_exception);
    ASSERT_THROWS(cusp::subtract(B,C,D), cusp::invalid_input_exception);
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestSubtract);

template <typename MatrixType1, typename MatrixType2, typename MatrixType3, typename BinaryFunction>
void elementwise(my_system& system, const MatrixType1& A, const MatrixType2& B, MatrixType3& C, BinaryFunction op)
{
    system.validate_dispatch();
    return;
}

void TestElementwiseDispatch()
{
    // initialize testing variables
    cusp::csr_matrix<int, float, cusp::device_memory> A, B, C;

    my_system sys(0);

    // call with explicit dispatching
    cusp::elementwise(sys, A, B, C, thrust::plus<float>());

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestElementwiseDispatch);


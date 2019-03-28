#include <unittest/unittest.h>

#include <cusp/linear_operator.h>
#include <cusp/gallery/poisson.h>
#include <cusp/gallery/random.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/permutation_matrix.h>

#include <cusp/multiply.h>

/////////////////////////////////////////
// Sparse Matrix-Matrix Multiplication //
/////////////////////////////////////////

template <typename SparseMatrixType, typename DenseMatrixType>
void CompareSparseMatrixMatrixMultiply(DenseMatrixType A, DenseMatrixType B)
{
    DenseMatrixType C;
    cusp::multiply(A, B, C);

    SparseMatrixType _A(A), _B(B), _C;
    cusp::multiply(_A, _B, _C);

    ASSERT_EQUAL(C == DenseMatrixType(_C), true);

    typename SparseMatrixType::view _Aview(_A), _Bview(_B), _Cview(_C);
    cusp::multiply(_Aview, _Bview, _Cview);

    ASSERT_EQUAL(C == DenseMatrixType(_Cview), true);
}

template <typename TestMatrix>
void TestSparseMatrixMatrixMultiply(void)
{
    typedef typename TestMatrix::value_type ValueType;

    cusp::array2d<ValueType,cusp::host_memory> A(3,2);
    A(0,0) = 1.0;
    A(0,1) = 2.0;
    A(1,0) = 3.0;
    A(1,1) = 0.0;
    A(2,0) = 5.0;
    A(2,1) = 6.0;

    cusp::array2d<ValueType,cusp::host_memory> B(2,4);
    B(0,0) = 0.0;
    B(0,1) = 2.0;
    B(0,2) = 3.0;
    B(0,3) = 4.0;
    B(1,0) = 5.0;
    B(1,1) = 0.0;
    B(1,2) = 0.0;
    B(1,3) = 8.0;

    cusp::array2d<ValueType,cusp::host_memory> C(2,2);
    C(0,0) = 0.0;
    C(0,1) = 0.0;
    C(1,0) = 3.0;
    C(1,1) = 5.0;

    cusp::array2d<ValueType,cusp::host_memory> D(2,1);
    D(0,0) = 2.0;
    D(1,0) = 3.0;

    cusp::array2d<ValueType,cusp::host_memory> E(2,2);
    E(0,0) = 0.0;
    E(0,1) = 0.0;
    E(1,0) = 0.0;
    E(1,1) = 0.0;

    cusp::array2d<ValueType,cusp::host_memory> F(2,3);
    F(0,0) = 0.0;
    F(0,1) = 1.5;
    F(0,2) = 3.0;
    F(1,0) = 0.5;
    F(1,1) = 0.0;
    F(1,2) = 0.0;

    cusp::array2d<ValueType,cusp::host_memory> G;
    cusp::gallery::poisson5pt(G, 4, 6);

    cusp::array2d<ValueType,cusp::host_memory> H;
    cusp::gallery::poisson5pt(H, 8, 3);

    cusp::array2d<ValueType,cusp::host_memory> I;
    cusp::gallery::random(I, 24, 24, 150);

    cusp::array2d<ValueType,cusp::host_memory> J;
    cusp::gallery::random(J, 24, 24, 50);

    cusp::array2d<ValueType,cusp::host_memory> K;
    cusp::gallery::random(K, 24, 12, 20);

    //thrust::host_vector< cusp::array2d<float,cusp::host_memory> > matrices;
    std::vector< cusp::array2d<ValueType,cusp::host_memory> > matrices;
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

    // test matrix multiply for every pair of compatible matrices
    for(size_t i = 0; i < matrices.size(); i++)
    {
        const cusp::array2d<ValueType,cusp::host_memory>& left = matrices[i];
        for(size_t j = 0; j < matrices.size(); j++)
        {
            const cusp::array2d<ValueType,cusp::host_memory>& right = matrices[j];

            if (left.num_cols == right.num_rows)
                CompareSparseMatrixMatrixMultiply<TestMatrix>(left, right);
        }
    }

}
DECLARE_SPARSE_MATRIX_UNITTEST(TestSparseMatrixMatrixMultiply);

template <typename SparseMatrixType, typename DenseMatrixType>
void CompareScaledSparseMatrixMatrixMultiply(DenseMatrixType A, DenseMatrixType B)
{
    typedef typename SparseMatrixType::value_type ValueType;

    thrust::identity<ValueType>   initialize;
    thrust::multiplies<ValueType> combine;
    thrust::plus<ValueType>       reduce;

    DenseMatrixType C(A);
    cusp::multiply(A, B, C, initialize, combine, reduce);

    SparseMatrixType _A(A), _B(B), _C(A);
    cusp::multiply(_A, _B, _C, initialize, combine, reduce);

    ASSERT_EQUAL(C == DenseMatrixType(_C), true);

    typename SparseMatrixType::view _Aview(_A), _Bview(_B), _Cview(_C);
    cusp::multiply(_Aview, _Bview, _Cview, initialize, combine, reduce);

    ASSERT_EQUAL(C == DenseMatrixType(_Cview), true);
}

template <typename TestMatrix>
void TestScaledSparseMatrixMatrixMultiply(void)
{
    typedef typename TestMatrix::value_type ValueType;

    cusp::array2d<ValueType,cusp::host_memory> A(3,2);
    A(0,0) = 1.0;
    A(0,1) = 2.0;
    A(1,0) = 3.0;
    A(1,1) = 0.0;
    A(2,0) = 5.0;
    A(2,1) = 6.0;

    cusp::array2d<ValueType,cusp::host_memory> B(2,4);
    B(0,0) = 0.0;
    B(0,1) = 2.0;
    B(0,2) = 3.0;
    B(0,3) = 4.0;
    B(1,0) = 5.0;
    B(1,1) = 0.0;
    B(1,2) = 0.0;
    B(1,3) = 8.0;

    cusp::array2d<ValueType,cusp::host_memory> C(2,2);
    C(0,0) = 0.0;
    C(0,1) = 0.0;
    C(1,0) = 3.0;
    C(1,1) = 5.0;

    cusp::array2d<ValueType,cusp::host_memory> D(2,1);
    D(0,0) = 2.0;
    D(1,0) = 3.0;

    cusp::array2d<ValueType,cusp::host_memory> E(2,2);
    E(0,0) = 0.0;
    E(0,1) = 0.0;
    E(1,0) = 0.0;
    E(1,1) = 0.0;

    cusp::array2d<ValueType,cusp::host_memory> F(2,3);
    F(0,0) = 0.0;
    F(0,1) = 1.5;
    F(0,2) = 3.0;
    F(1,0) = 0.5;
    F(1,1) = 0.0;
    F(1,2) = 0.0;

    cusp::array2d<ValueType,cusp::host_memory> G;
    cusp::gallery::poisson5pt(G, 4, 6);

    cusp::array2d<ValueType,cusp::host_memory> H;
    cusp::gallery::poisson5pt(H, 8, 3);

    cusp::array2d<ValueType,cusp::host_memory> I;
    cusp::gallery::random(I, 24, 24, 150);

    cusp::array2d<ValueType,cusp::host_memory> J;
    cusp::gallery::random(J, 24, 24, 50);

    cusp::array2d<ValueType,cusp::host_memory> K;
    cusp::gallery::random(K, 24, 12, 20);

    //thrust::host_vector< cusp::array2d<float,cusp::host_memory> > matrices;
    std::vector< cusp::array2d<ValueType,cusp::host_memory> > matrices;
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

    // test matrix multiply for every pair of compatible matrices
    for(size_t i = 0; i < matrices.size(); i++)
    {
        const cusp::array2d<ValueType,cusp::host_memory>& left = matrices[i];
        for(size_t j = 0; j < matrices.size(); j++)
        {
            const cusp::array2d<ValueType,cusp::host_memory>& right = matrices[j];

            if (left.num_cols == right.num_rows)
                CompareScaledSparseMatrixMatrixMultiply<TestMatrix>(left, right);
        }
    }

}
/* DECLARE_SPARSE_MATRIX_UNITTEST(TestScaledSparseMatrixMatrixMultiply); */

///////////////////////////////////////////////
// Sparse Matrix-Dense Matrix Multiplication //
///////////////////////////////////////////////

template <typename SparseMatrixType, typename DenseMatrixType>
void CompareSparseMatrixDenseMatrixMultiply(DenseMatrixType A, DenseMatrixType B)
{
    typedef typename SparseMatrixType::value_type ValueType;
    typedef typename SparseMatrixType::memory_space MemorySpace;
    typedef cusp::array2d<ValueType,MemorySpace,cusp::column_major> DenseSpaceMatrixType;

    DenseMatrixType C(A.num_rows, B.num_cols);
    cusp::multiply(A, B, C);

    SparseMatrixType _A(A);

    // Copy B into the memory space
    DenseSpaceMatrixType B_space(B);
    // Allocate _B and ensure each column is properly aligned
    DenseSpaceMatrixType _B(B.num_rows, B.num_cols, ValueType(0), cusp::detail::round_up(B.num_rows, size_t(128)));
    // Copy columns of B into _B
    for(size_t i = 0; i < B.num_cols; i++ )
        cusp::blas::copy(B_space.column(i), _B.column(i));

    // test container
    {
        DenseSpaceMatrixType _C(C.num_rows, C.num_cols);
        cusp::multiply(_A, _B, _C);

        ASSERT_EQUAL(C == DenseMatrixType(_C), true);
    }

    {
        // test view
        DenseSpaceMatrixType _C(C.num_rows, C.num_cols);
        typename SparseMatrixType::view _Aview(_A);
        typename DenseSpaceMatrixType::view _Bview(_B), _Cview(_C);
        cusp::multiply(_Aview, _Bview, _Cview);

        ASSERT_EQUAL(C == DenseMatrixType(_C), true);
    }
}

template <typename TestMatrix>
void TestSparseMatrixDenseMatrixMultiply(void)
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

    cusp::array2d<float,cusp::host_memory> F(2,3);
    F(0,0) = 0.0;
    F(0,1) = 1.5;
    F(0,2) = 3.0;
    F(1,0) = 0.5;
    F(1,1) = 0.0;
    F(1,2) = 0.0;

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

    //thrust::host_vector< cusp::array2d<float,cusp::host_memory,cusp::column_major> > matrices;
    std::vector< cusp::array2d<float,cusp::host_memory,cusp::column_major> > matrices;
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

    // test matrix multiply for every pair of compatible matrices
    for(size_t i = 0; i < matrices.size(); i++)
    {
        const cusp::array2d<float,cusp::host_memory,cusp::column_major>& left = matrices[i];
        for(size_t j = 0; j < matrices.size(); j++)
        {
            const cusp::array2d<float,cusp::host_memory,cusp::column_major>& right = matrices[j];

            if (left.num_cols == right.num_rows)
                CompareSparseMatrixDenseMatrixMultiply<TestMatrix>(left, right);
        }
    }

}
/* DECLARE_SPARSE_MATRIX_UNITTEST(TestSparseMatrixDenseMatrixMultiply); */


/////////////////////////////////////////
// Sparse Matrix-Vector Multiplication //
/////////////////////////////////////////

template <typename SparseMatrixType, typename DenseMatrixType>
void CompareSparseMatrixVectorMultiply(DenseMatrixType A)
{
    typedef typename SparseMatrixType::value_type   ValueType;
    typedef typename SparseMatrixType::memory_space MemorySpace;

    // setup reference input
    cusp::array1d<ValueType, cusp::host_memory> x(A.num_cols);
    cusp::array1d<ValueType, cusp::host_memory> y(A.num_rows, 10);
    for(size_t i = 0; i < x.size(); i++)
        x[i] = i % 10;

    // compute reference output
    cusp::multiply(A, x, y);

    // test container
    {
        SparseMatrixType _A(A);
        cusp::array1d<ValueType, MemorySpace> _x(x);
        cusp::array1d<ValueType, MemorySpace> _y(A.num_rows, 10);

        cusp::multiply(_A, _x, _y);

        ASSERT_EQUAL(_y, y);
    }

    // test matrix view
    {
        SparseMatrixType _A(A);
        cusp::array1d<ValueType, MemorySpace> _x(x);
        cusp::array1d<ValueType, MemorySpace> _y(A.num_rows, 10);

        typename SparseMatrixType::view _V(_A);
        cusp::multiply(_V, _x, _y);

        ASSERT_EQUAL(_y, y);
    }

    // test array view
    {
        SparseMatrixType _A(A);
        cusp::array1d<ValueType, MemorySpace> _x(x);
        cusp::array1d<ValueType, MemorySpace> _y(A.num_rows, 10);

        typename cusp::array1d<ValueType, MemorySpace> _Vx(_x), _Vy(_y);
        cusp::multiply(_A, _Vx, _Vy);

        ASSERT_EQUAL(_Vy, y);
    }
}


// TODO use COO reference format and test larger problem sizes
template <class TestMatrix>
void TestSparseMatrixVectorMultiply()
{
    typedef typename TestMatrix::value_type   ValueType;

    cusp::array2d<ValueType, cusp::host_memory> A(5,4);
    A(0,0) = 13;
    A(0,1) = 80;
    A(0,2) =  0;
    A(0,3) =  0;
    A(1,0) =  0;
    A(1,1) = 27;
    A(1,2) =  0;
    A(1,3) =  0;
    A(2,0) = 55;
    A(2,1) =  0;
    A(2,2) = 24;
    A(2,3) = 42;
    A(3,0) =  0;
    A(3,1) = 69;
    A(3,2) =  0;
    A(3,3) = 83;
    A(4,0) =  0;
    A(4,1) =  0;
    A(4,2) = 27;
    A(4,3) =  0;

    cusp::array2d<ValueType,cusp::host_memory> B(2,4);
    B(0,0) = 0.0;
    B(0,1) = 2.0;
    B(0,2) = 3.0;
    B(0,3) = 4.0;
    B(1,0) = 5.0;
    B(1,1) = 0.0;
    B(1,2) = 0.0;
    B(1,3) = 8.0;

    cusp::array2d<ValueType,cusp::host_memory> C(2,2);
    C(0,0) = 0.0;
    C(0,1) = 0.0;
    C(1,0) = 3.0;
    C(1,1) = 5.0;

    cusp::array2d<ValueType,cusp::host_memory> D(2,1);
    D(0,0) = 2.0;
    D(1,0) = 3.0;

    cusp::array2d<ValueType,cusp::host_memory> E(2,2);
    E(0,0) = 0.0;
    E(0,1) = 0.0;
    E(1,0) = 0.0;
    E(1,1) = 0.0;

    cusp::array2d<ValueType,cusp::host_memory> F(2,3);
    F(0,0) = 0.0;
    F(0,1) = 1.5;
    F(0,2) = 3.0;
    F(1,0) = 0.5;
    F(1,1) = 0.0;
    F(1,2) = 0.0;

    cusp::array2d<ValueType,cusp::host_memory> G;
    cusp::gallery::poisson5pt(G, 4, 6);

    cusp::array2d<ValueType,cusp::host_memory> H;
    cusp::gallery::poisson5pt(H, 8, 3);

    CompareSparseMatrixVectorMultiply<TestMatrix>(A);
    CompareSparseMatrixVectorMultiply<TestMatrix>(B);
    CompareSparseMatrixVectorMultiply<TestMatrix>(C);
    CompareSparseMatrixVectorMultiply<TestMatrix>(D);
    CompareSparseMatrixVectorMultiply<TestMatrix>(E);
    CompareSparseMatrixVectorMultiply<TestMatrix>(F);
    CompareSparseMatrixVectorMultiply<TestMatrix>(G);
    CompareSparseMatrixVectorMultiply<TestMatrix>(H);
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestSparseMatrixVectorMultiply);

template <typename SparseMatrixType, typename DenseMatrixType>
void CompareScaledSparseMatrixVectorMultiply(DenseMatrixType A)
{
    typedef typename SparseMatrixType::value_type   ValueType;
    typedef typename SparseMatrixType::memory_space MemorySpace;

    // setup reference input
    cusp::array1d<ValueType, cusp::host_memory> x(A.num_cols);
    cusp::array1d<ValueType, cusp::host_memory> y(A.num_rows, 10);
    for(size_t i = 0; i < x.size(); i++)
        x[i] = i % 10;

    thrust::identity<ValueType>   initialize;
    thrust::multiplies<ValueType> combine;
    thrust::plus<ValueType>       reduce;

    // compute reference output
    cusp::multiply(A, x, y, initialize, combine, reduce);

    // test container
    {
        SparseMatrixType _A(A);
        cusp::array1d<ValueType, MemorySpace> _x(x);
        cusp::array1d<ValueType, MemorySpace> _y(A.num_rows, 10);

        cusp::multiply(_A, _x, _y, initialize, combine, reduce);

        ASSERT_EQUAL(_y, y);
    }

    // test matrix view
    {
        SparseMatrixType _A(A);
        cusp::array1d<ValueType, MemorySpace> _x(x);
        cusp::array1d<ValueType, MemorySpace> _y(A.num_rows, 10);

        typename SparseMatrixType::view _V(_A);
        cusp::multiply(_V, _x, _y, initialize, combine, reduce);

        ASSERT_EQUAL(_y, y);
    }

    // test array view
    {
        SparseMatrixType _A(A);
        cusp::array1d<ValueType, MemorySpace> _x(x);
        cusp::array1d<ValueType, MemorySpace> _y(A.num_rows, 10);

        typename cusp::array1d<ValueType, MemorySpace> _Vx(_x), _Vy(_y);
        cusp::multiply(_A, _Vx, _Vy, initialize, combine, reduce);

        ASSERT_EQUAL(_Vy, y);
    }
}

template <class TestMatrix>
void TestScaledSparseMatrixVectorMultiply()
{
    typedef typename TestMatrix::value_type   ValueType;

    cusp::array2d<ValueType, cusp::host_memory> A(5,4);
    A(0,0) = 13;
    A(0,1) = 80;
    A(0,2) =  0;
    A(0,3) =  0;
    A(1,0) =  0;
    A(1,1) = 27;
    A(1,2) =  0;
    A(1,3) =  0;
    A(2,0) = 55;
    A(2,1) =  0;
    A(2,2) = 24;
    A(2,3) = 42;
    A(3,0) =  0;
    A(3,1) = 69;
    A(3,2) =  0;
    A(3,3) = 83;
    A(4,0) =  0;
    A(4,1) =  0;
    A(4,2) = 27;
    A(4,3) =  0;

    cusp::array2d<ValueType,cusp::host_memory> B(2,4);
    B(0,0) = 0.0;
    B(0,1) = 2.0;
    B(0,2) = 3.0;
    B(0,3) = 4.0;
    B(1,0) = 5.0;
    B(1,1) = 0.0;
    B(1,2) = 0.0;
    B(1,3) = 8.0;

    cusp::array2d<ValueType,cusp::host_memory> C(2,2);
    C(0,0) = 0.0;
    C(0,1) = 0.0;
    C(1,0) = 3.0;
    C(1,1) = 5.0;

    cusp::array2d<ValueType,cusp::host_memory> D(2,1);
    D(0,0) = 2.0;
    D(1,0) = 3.0;

    cusp::array2d<ValueType,cusp::host_memory> E(2,2);
    E(0,0) = 0.0;
    E(0,1) = 0.0;
    E(1,0) = 0.0;
    E(1,1) = 0.0;

    cusp::array2d<ValueType,cusp::host_memory> F(2,3);
    F(0,0) = 0.0;
    F(0,1) = 1.5;
    F(0,2) = 3.0;
    F(1,0) = 0.5;
    F(1,1) = 0.0;
    F(1,2) = 0.0;

    cusp::array2d<ValueType,cusp::host_memory> G;
    cusp::gallery::poisson5pt(G, 4, 6);

    cusp::array2d<ValueType,cusp::host_memory> H;
    cusp::gallery::poisson5pt(H, 8, 3);

    CompareScaledSparseMatrixVectorMultiply<TestMatrix>(A);
    CompareScaledSparseMatrixVectorMultiply<TestMatrix>(B);
    CompareScaledSparseMatrixVectorMultiply<TestMatrix>(C);
    CompareScaledSparseMatrixVectorMultiply<TestMatrix>(D);
    CompareScaledSparseMatrixVectorMultiply<TestMatrix>(E);
    CompareScaledSparseMatrixVectorMultiply<TestMatrix>(F);
    CompareScaledSparseMatrixVectorMultiply<TestMatrix>(G);
    CompareScaledSparseMatrixVectorMultiply<TestMatrix>(H);
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestScaledSparseMatrixVectorMultiply);

//////////////////////////////
// General Linear Operators //
//////////////////////////////

template <class MemorySpace>
void TestMultiplyIdentityOperator(void)
{
    cusp::array1d<float, MemorySpace> x(4);
    cusp::array1d<float, MemorySpace> y(4);

    x[0] =  7.0f;
    y[0] =  0.0f;
    x[1] =  5.0f;
    y[1] = -2.0f;
    x[2] =  4.0f;
    y[2] =  0.0f;
    x[3] = -3.0f;
    y[3] =  5.0f;

    cusp::identity_operator<float, MemorySpace> A(4,4);

    cusp::multiply(A, x, y);

    ASSERT_EQUAL(y[0],  7.0f);
    ASSERT_EQUAL(y[1],  5.0f);
    ASSERT_EQUAL(y[2],  4.0f);
    ASSERT_EQUAL(y[3], -3.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestMultiplyIdentityOperator);


///////////////////////////
// Permutation Operators //
///////////////////////////

template <class MemorySpace>
void TestMultiplyPermutationOperator(void)
{
    cusp::array1d<float, MemorySpace> x(4);
    cusp::array1d<float, MemorySpace> y(4);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;

    cusp::permutation_matrix<int, MemorySpace> P(4);
    P.permutation[0] = 3;
    P.permutation[1] = 2;
    P.permutation[2] = 1;
    P.permutation[3] = 0;

    cusp::multiply(P, x, y);

    ASSERT_EQUAL(y[0], -3.0f);
    ASSERT_EQUAL(y[1],  4.0f);
    ASSERT_EQUAL(y[2],  5.0f);
    ASSERT_EQUAL(y[3],  7.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestMultiplyPermutationOperator);

template<typename TestMatrix>
void TestPermutationMatrixMultiply(void)
{
    typedef typename TestMatrix::index_type   IndexType;
    typedef typename TestMatrix::value_type   ValueType;
    typedef typename TestMatrix::memory_space MemorySpace;

    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> A(3,3,7);

    A.row_indices[0] = 0;
    A.column_indices[0] = 0;
    A.values[0] = 10;
    A.row_indices[1] = 0;
    A.column_indices[1] = 1;
    A.values[1] = 20;
    A.row_indices[2] = 0;
    A.column_indices[2] = 2;
    A.values[2] = 30;
    A.row_indices[3] = 1;
    A.column_indices[3] = 0;
    A.values[3] = 40;
    A.row_indices[4] = 1;
    A.column_indices[4] = 1;
    A.values[4] = 50;
    A.row_indices[5] = 2;
    A.column_indices[5] = 0;
    A.values[5] = 60;
    A.row_indices[6] = 2;
    A.column_indices[6] = 2;
    A.values[6] = 70;

    cusp::array1d<IndexType,MemorySpace> permutation(3);
    permutation[0] = 2;
    permutation[1] = 1;
    permutation[2] = 0;

    cusp::permutation_matrix<IndexType,MemorySpace> P(3, permutation);

    // Test row permutations
    {
        TestMatrix PA;
        TestMatrix A_(A);
        cusp::multiply(P, A_, PA);

        cusp::array2d<ValueType,cusp::host_memory> host_matrix(PA);

        ASSERT_EQUAL(PA.num_rows,    A.num_rows);
        ASSERT_EQUAL(PA.num_cols,    A.num_cols);
        ASSERT_EQUAL(PA.num_entries, A.num_entries);
        ASSERT_EQUAL(host_matrix(0,0), ValueType(60));
        ASSERT_EQUAL(host_matrix(0,1), ValueType( 0));
        ASSERT_EQUAL(host_matrix(0,2), ValueType(70));
        ASSERT_EQUAL(host_matrix(1,0), ValueType(40));
        ASSERT_EQUAL(host_matrix(1,1), ValueType(50));
        ASSERT_EQUAL(host_matrix(1,2), ValueType( 0));
        ASSERT_EQUAL(host_matrix(2,0), ValueType(10));
        ASSERT_EQUAL(host_matrix(2,1), ValueType(20));
        ASSERT_EQUAL(host_matrix(2,2), ValueType(30));
    }

    // Test column permutations
    {
        TestMatrix AP;
        TestMatrix A_(A);
        cusp::multiply(A_, P, AP);

        cusp::array2d<ValueType,cusp::host_memory> host_matrix(AP);

        ASSERT_EQUAL(AP.num_rows,    A.num_rows);
        ASSERT_EQUAL(AP.num_cols,    A.num_cols);
        ASSERT_EQUAL(AP.num_entries, A.num_entries);
        ASSERT_EQUAL(host_matrix(0,0), ValueType(30));
        ASSERT_EQUAL(host_matrix(0,1), ValueType(20));
        ASSERT_EQUAL(host_matrix(0,2), ValueType(10));
        ASSERT_EQUAL(host_matrix(1,0), ValueType( 0));
        ASSERT_EQUAL(host_matrix(1,1), ValueType(50));
        ASSERT_EQUAL(host_matrix(1,2), ValueType(40));
        ASSERT_EQUAL(host_matrix(2,0), ValueType(70));
        ASSERT_EQUAL(host_matrix(2,1), ValueType( 0));
        ASSERT_EQUAL(host_matrix(2,2), ValueType(60));
    }
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestPermutationMatrixMultiply);

template <typename MatrixType1, typename MatrixType2, typename MatrixType3>
void multiply(my_system& system, const MatrixType1& A, const MatrixType2& B, MatrixType3& C)
{
    system.validate_dispatch();
    return;
}

void TestMatrixMatrixMultiplyDispatch()
{
    // initialize testing variables
    cusp::csr_matrix<int, float, cusp::device_memory> A, B, C;

    my_system sys(0);

    // call with explicit dispatching
    cusp::multiply(sys, A, B, C);

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestMatrixMatrixMultiplyDispatch);

template <typename LinearOperator,
         typename MatrixOrVector1,
         typename MatrixOrVector2,
         typename UnaryFunction,
         typename BinaryFunction1,
         typename BinaryFunction2>
void multiply(my_system& system,
              const LinearOperator&  A,
              const MatrixOrVector1& B,
              MatrixOrVector2& C,
              UnaryFunction  initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce)
{
    system.validate_dispatch();
    return;
}

void TestMatrixVectorMultiplyDispatch()
{
    // initialize testing variables
    cusp::csr_matrix<int, float, cusp::device_memory> A, B, C;
    cusp::array1d<float, cusp::device_memory> x;

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::multiply(sys, A, x, x);

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }

    {
        my_system sys(0);

        // call with explicit dispatching
        cusp::multiply(sys, A, x, x, cusp::constant_functor<float>(), thrust::multiplies<float>(), thrust::plus<float>());

        // check if dispatch policy was used
        ASSERT_EQUAL(true, sys.is_valid());
    }
}
DECLARE_UNITTEST(TestMatrixVectorMultiplyDispatch);


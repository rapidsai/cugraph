#include <unittest/unittest.h>
#include <cusp/array2d.h>

template <class Space>
void TestArray2dBasicConstructor(void)
{
    cusp::array2d<float, Space> A(3, 2);

    ASSERT_EQUAL(A.num_rows,       3);
    ASSERT_EQUAL(A.num_cols,       2);
    ASSERT_EQUAL(A.num_entries,    6);
    ASSERT_EQUAL(A.values.size(),  6);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dBasicConstructor)

template <class Space>
void TestArray2dFillConstructor(void)
{
    cusp::array2d<float, Space> A(3, 2, 13.0f);

    ASSERT_EQUAL(A.num_rows,       3);
    ASSERT_EQUAL(A.num_cols,       2);
    ASSERT_EQUAL(A.num_entries,    6);
    ASSERT_EQUAL(A.values.size(),  6);
    ASSERT_EQUAL(A.values[0], 13.0f);
    ASSERT_EQUAL(A.values[1], 13.0f);
    ASSERT_EQUAL(A.values[2], 13.0f);
    ASSERT_EQUAL(A.values[3], 13.0f);
    ASSERT_EQUAL(A.values[4], 13.0f);
    ASSERT_EQUAL(A.values[5], 13.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dFillConstructor)

template <class Space>
void TestArray2dCopyConstructor(void)
{
    cusp::array2d<float, Space> A(3, 2);
    A(0,0) = 1;
    A(0,1) = 2;
    A(1,0) = 3;
    A(1,1) = 4;
    A(2,0) = 5;
    A(2,1) = 6;

    cusp::array2d<float, Space> B(A);
    ASSERT_EQUAL(A.num_rows,    B.num_rows);
    ASSERT_EQUAL(A.num_cols,    B.num_cols);
    ASSERT_EQUAL(A.num_entries, B.num_entries);
    ASSERT_EQUAL(A.pitch,       B.pitch);
    ASSERT_EQUAL(A.values,      B.values);

    cusp::array2d<float, Space, cusp::column_major> C(A);
    ASSERT_EQUAL(A.num_rows,    C.num_rows);
    ASSERT_EQUAL(A.num_cols,    C.num_cols);
    ASSERT_EQUAL(A.num_entries, C.num_entries);
    ASSERT_EQUAL(3,             C.pitch);
    ASSERT_EQUAL(A(0,0),        C(0,0));
    ASSERT_EQUAL(A(0,1),        C(0,1));
    ASSERT_EQUAL(A(1,0),        C(1,0));
    ASSERT_EQUAL(A(1,1),        C(1,1));
    ASSERT_EQUAL(A(2,0),        C(2,0));
    ASSERT_EQUAL(A(2,1),        C(2,1));

    // set pitch to 4
    A.resize(3,2,4);
    thrust::fill(A.values.begin(), A.values.end(), -1);
    A(0,0) = 1;
    A(0,1) = 2;
    A(1,0) = 3;
    A(1,1) = 4;
    A(2,0) = 5;
    A(2,1) = 6;

    cusp::array2d<float, Space> D(A);
    ASSERT_EQUAL(A.num_rows,    D.num_rows);
    ASSERT_EQUAL(A.num_cols,    D.num_cols);
    ASSERT_EQUAL(A.num_entries, D.num_entries);
    ASSERT_EQUAL(A.pitch,       D.pitch);
    ASSERT_EQUAL(A(0,0),        D(0,0));
    ASSERT_EQUAL(A(0,1),        D(0,1));
    ASSERT_EQUAL(A(1,0),        D(1,0));
    ASSERT_EQUAL(A(1,1),        D(1,1));
    ASSERT_EQUAL(A(2,0),        D(2,0));
    ASSERT_EQUAL(A(2,1),        D(2,1));

    cusp::array2d<float, Space, cusp::column_major> E(A);
    ASSERT_EQUAL(A.num_rows,    E.num_rows);
    ASSERT_EQUAL(A.num_cols,    E.num_cols);
    ASSERT_EQUAL(A.num_entries, E.num_entries);
    ASSERT_EQUAL(3,             E.pitch);
    ASSERT_EQUAL(A(0,0),        E(0,0));
    ASSERT_EQUAL(A(0,1),        E(0,1));
    ASSERT_EQUAL(A(1,0),        E(1,0));
    ASSERT_EQUAL(A(1,1),        E(1,1));
    ASSERT_EQUAL(A(2,0),        E(2,0));
    ASSERT_EQUAL(A(2,1),        E(2,1));
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dCopyConstructor)

template <class Space>
void TestArray2dRowMajor(void)
{
    cusp::array2d<float, Space, cusp::row_major> A(2,3);

    A(0,0) = 10;
    A(0,1) = 20;
    A(0,2) = 30;
    A(1,0) = 40;
    A(1,1) = 50;
    A(1,2) = 60;

    ASSERT_EQUAL(A(0,0), 10);
    ASSERT_EQUAL(A(0,1), 20);
    ASSERT_EQUAL(A(0,2), 30);
    ASSERT_EQUAL(A(1,0), 40);
    ASSERT_EQUAL(A(1,1), 50);
    ASSERT_EQUAL(A(1,2), 60);

    ASSERT_EQUAL(A.values[0], 10);
    ASSERT_EQUAL(A.values[1], 20);
    ASSERT_EQUAL(A.values[2], 30);
    ASSERT_EQUAL(A.values[3], 40);
    ASSERT_EQUAL(A.values[4], 50);
    ASSERT_EQUAL(A.values[5], 60);

    // test non-trivial pitch
    A.resize(2,3,4);
    thrust::fill(A.values.begin(), A.values.end(), 0);

    A(0,0) = 10;
    A(0,1) = 20;
    A(0,2) = 30;
    A(1,0) = 40;
    A(1,1) = 50;
    A(1,2) = 60;

    ASSERT_EQUAL(A.values[0], 10);
    ASSERT_EQUAL(A.values[1], 20);
    ASSERT_EQUAL(A.values[2], 30);
    ASSERT_EQUAL(A.values[3],  0);
    ASSERT_EQUAL(A.values[4], 40);
    ASSERT_EQUAL(A.values[5], 50);
    ASSERT_EQUAL(A.values[6], 60);
    ASSERT_EQUAL(A.values[7],  0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dRowMajor)

template <class Space>
void TestArray2dColumnMajor(void)
{
    cusp::array2d<float, Space, cusp::column_major> A(2,3);

    A(0,0) = 10;
    A(0,1) = 20;
    A(0,2) = 30;
    A(1,0) = 40;
    A(1,1) = 50;
    A(1,2) = 60;

    ASSERT_EQUAL(A(0,0), 10);
    ASSERT_EQUAL(A(0,1), 20);
    ASSERT_EQUAL(A(0,2), 30);
    ASSERT_EQUAL(A(1,0), 40);
    ASSERT_EQUAL(A(1,1), 50);
    ASSERT_EQUAL(A(1,2), 60);

    ASSERT_EQUAL(A.values[0], 10);
    ASSERT_EQUAL(A.values[1], 40);
    ASSERT_EQUAL(A.values[2], 20);
    ASSERT_EQUAL(A.values[3], 50);
    ASSERT_EQUAL(A.values[4], 30);
    ASSERT_EQUAL(A.values[5], 60);

    // test non-trivial pitch
    A.resize(2,3,4);
    thrust::fill(A.values.begin(), A.values.end(), 0);

    A(0,0) = 10;
    A(0,1) = 20;
    A(0,2) = 30;
    A(1,0) = 40;
    A(1,1) = 50;
    A(1,2) = 60;

    ASSERT_EQUAL(A.values[ 0], 10);
    ASSERT_EQUAL(A.values[ 1], 40);
    ASSERT_EQUAL(A.values[ 2],  0);
    ASSERT_EQUAL(A.values[ 3],  0);
    ASSERT_EQUAL(A.values[ 4], 20);
    ASSERT_EQUAL(A.values[ 5], 50);
    ASSERT_EQUAL(A.values[ 6],  0);
    ASSERT_EQUAL(A.values[ 7],  0);
    ASSERT_EQUAL(A.values[ 8], 30);
    ASSERT_EQUAL(A.values[ 9], 60);
    ASSERT_EQUAL(A.values[10],  0);
    ASSERT_EQUAL(A.values[11],  0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dColumnMajor)

template <class Space>
void TestArray2dMixedOrientations(void)
{
    cusp::array2d<float, Space, cusp::row_major>    R(2,3);
    cusp::array2d<float, Space, cusp::column_major> C(2,3);

    R(0,0) = 10;
    R(0,1) = 20;
    R(0,2) = 30;
    R(1,0) = 40;
    R(1,1) = 50;
    R(1,2) = 60;

    C = R;
    ASSERT_EQUAL(C(0,0), 10);
    ASSERT_EQUAL(C(0,1), 20);
    ASSERT_EQUAL(C(0,2), 30);
    ASSERT_EQUAL(C(1,0), 40);
    ASSERT_EQUAL(C(1,1), 50);
    ASSERT_EQUAL(C(1,2), 60);

    R = C;
    ASSERT_EQUAL(R(0,0), 10);
    ASSERT_EQUAL(R(0,1), 20);
    ASSERT_EQUAL(R(0,2), 30);
    ASSERT_EQUAL(R(1,0), 40);
    ASSERT_EQUAL(R(1,1), 50);
    ASSERT_EQUAL(R(1,2), 60);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dMixedOrientations)

template <class Space>
void TestArray2dResize(void)
{
    cusp::array2d<float, Space> A;

    A.resize(3, 2);

    ASSERT_EQUAL(A.num_rows,       3);
    ASSERT_EQUAL(A.num_cols,       2);
    ASSERT_EQUAL(A.pitch,          2);
    ASSERT_EQUAL(A.num_entries,    6);
    ASSERT_EQUAL(A.values.size(),  6);

    A.resize(3, 2, 4);

    ASSERT_EQUAL(A.num_rows,       3);
    ASSERT_EQUAL(A.num_cols,       2);
    ASSERT_EQUAL(A.pitch,          4);
    ASSERT_EQUAL(A.num_entries,    6);
    ASSERT_EQUAL(A.values.size(), 12);

    ASSERT_THROWS(A.resize(3,2,1), cusp::invalid_input_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dResize)

template <class Space>
void TestArray2dSwap(void)
{
    cusp::array2d<float, Space> A(2,2);
    cusp::array2d<float, Space> B(3,1);

    A(0,0) = 10;
    A(0,1) = 20;
    A(1,0) = 30;
    A(1,1) = 40;

    B(0,0) = 50;
    B(1,0) = 60;
    B(2,0) = 70;

    cusp::array2d<float, Space> A_copy(A);
    cusp::array2d<float, Space> B_copy(B);

    A.swap(B);

    ASSERT_EQUAL(A.num_rows,    B_copy.num_rows);
    ASSERT_EQUAL(A.num_cols,    B_copy.num_cols);
    ASSERT_EQUAL(A.num_entries, B_copy.num_entries);
    ASSERT_EQUAL(A.values,      B_copy.values);

    ASSERT_EQUAL(B.num_rows,    A_copy.num_rows);
    ASSERT_EQUAL(B.num_cols,    A_copy.num_cols);
    ASSERT_EQUAL(B.num_entries, A_copy.num_entries);
    ASSERT_EQUAL(B.values,      A_copy.values);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dSwap)

void TestArray2dRebind(void)
{
    typedef cusp::array2d<float, cusp::host_memory>       HostMatrix;
    typedef HostMatrix::rebind<cusp::device_memory>::type DeviceMatrix;

    HostMatrix   h_A(10,10);
    DeviceMatrix d_A(h_A);

    ASSERT_EQUAL(h_A.num_entries, d_A.num_entries);
}
DECLARE_UNITTEST(TestArray2dRebind);

template <typename MemorySpace>
void TestArray2dEquality(void)
{
    cusp::array2d<float, MemorySpace, cusp::row_major>    A(2,2);
    cusp::array2d<float, MemorySpace, cusp::column_major> B(2,2);
    cusp::array2d<float, MemorySpace, cusp::row_major>    C(2,2);
    C.resize(2,2,5);
    thrust::fill(C.values.begin(), C.values.end(), -1);
    cusp::array2d<float, MemorySpace, cusp::row_major>    D(2,3);
    cusp::array2d<float, MemorySpace, cusp::column_major> E(2,3);
    cusp::array2d<float, MemorySpace, cusp::column_major> F(2,3);
    F.resize(2,3,4);
    thrust::fill(F.values.begin(), F.values.end(), -1);

    // start with A == B == C and D == E == F
    A(0,0) = 1;
    A(0,1) = 2;
    A(1,0) = 4;
    A(1,1) = 5;

    B(0,0) = 1;
    B(0,1) = 2;
    B(1,0) = 4;
    B(1,1) = 5;

    C(0,0) = 1;
    C(0,1) = 2;
    C(1,0) = 4;
    C(1,1) = 5;

    D(0,0) = 1;
    D(0,1) = 2;
    D(0,2) = 3;
    D(1,0) = 7;
    D(1,1) = 5;
    D(1,2) = 6;

    E(0,0) = 1;
    E(0,1) = 2;
    E(0,2) = 3;
    E(1,0) = 7;
    E(1,1) = 5;
    E(1,2) = 6;

    F(0,0) = 1;
    F(0,1) = 2;
    F(0,2) = 3;
    F(1,0) = 7;
    F(1,1) = 5;
    F(1,2) = 6;

    ASSERT_EQUAL(A == A,  true);
    ASSERT_EQUAL(B == A,  true);
    ASSERT_EQUAL(C == A,  true);
    ASSERT_EQUAL(D == A, false);
    ASSERT_EQUAL(E == A, false);
    ASSERT_EQUAL(F == A, false);
    ASSERT_EQUAL(A == B,  true);
    ASSERT_EQUAL(B == B,  true);
    ASSERT_EQUAL(C == B,  true);
    ASSERT_EQUAL(D == B, false);
    ASSERT_EQUAL(E == B, false);
    ASSERT_EQUAL(F == B, false);
    ASSERT_EQUAL(A == C,  true);
    ASSERT_EQUAL(B == C,  true);
    ASSERT_EQUAL(C == C,  true);
    ASSERT_EQUAL(D == C, false);
    ASSERT_EQUAL(E == C, false);
    ASSERT_EQUAL(F == C, false);
    ASSERT_EQUAL(A == D, false);
    ASSERT_EQUAL(B == D, false);
    ASSERT_EQUAL(C == D, false);
    ASSERT_EQUAL(D == D,  true);
    ASSERT_EQUAL(E == D,  true);
    ASSERT_EQUAL(F == D,  true);
    ASSERT_EQUAL(A == E, false);
    ASSERT_EQUAL(B == E, false);
    ASSERT_EQUAL(C == E, false);
    ASSERT_EQUAL(D == E,  true);
    ASSERT_EQUAL(E == E,  true);
    ASSERT_EQUAL(F == E,  true);
    ASSERT_EQUAL(A == F, false);
    ASSERT_EQUAL(B == F, false);
    ASSERT_EQUAL(C == F, false);
    ASSERT_EQUAL(D == F,  true);
    ASSERT_EQUAL(E == F,  true);
    ASSERT_EQUAL(F == F,  true);

    // peturb B and E
    B(1,0) = 9;
    E(1,0) = 9;

    ASSERT_EQUAL(A == A,  true);
    ASSERT_EQUAL(B == A, false);
    ASSERT_EQUAL(C == A,  true);
    ASSERT_EQUAL(D == A, false);
    ASSERT_EQUAL(E == A, false);
    ASSERT_EQUAL(F == A, false);
    ASSERT_EQUAL(A == B, false);
    ASSERT_EQUAL(B == B,  true);
    ASSERT_EQUAL(C == B, false);
    ASSERT_EQUAL(D == B, false);
    ASSERT_EQUAL(E == B, false);
    ASSERT_EQUAL(F == B, false);
    ASSERT_EQUAL(A == C,  true);
    ASSERT_EQUAL(B == C, false);
    ASSERT_EQUAL(C == C,  true);
    ASSERT_EQUAL(D == C, false);
    ASSERT_EQUAL(E == C, false);
    ASSERT_EQUAL(F == C, false);
    ASSERT_EQUAL(A == D, false);
    ASSERT_EQUAL(B == D, false);
    ASSERT_EQUAL(C == D, false);
    ASSERT_EQUAL(D == D,  true);
    ASSERT_EQUAL(E == D, false);
    ASSERT_EQUAL(F == D,  true);
    ASSERT_EQUAL(A == E, false);
    ASSERT_EQUAL(B == E, false);
    ASSERT_EQUAL(C == E, false);
    ASSERT_EQUAL(D == E, false);
    ASSERT_EQUAL(E == E,  true);
    ASSERT_EQUAL(F == E, false);
    ASSERT_EQUAL(A == F, false);
    ASSERT_EQUAL(B == F, false);
    ASSERT_EQUAL(C == F, false);
    ASSERT_EQUAL(D == F,  true);
    ASSERT_EQUAL(E == F, false);
    ASSERT_EQUAL(F == F,  true);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dEquality)

template <class Space>
void TestArray2dCopySemantics(void)
{
    // check that destination .pitch is respected
    cusp::array2d<float, Space> A(3, 2);
    A(0,0) = 1;
    A(0,1) = 2;
    A(1,0) = 3;
    A(1,1) = 4;
    A(2,0) = 5;
    A(2,1) = 6;

    cusp::array2d<float, Space> B;
    B.resize(3, 2, 4);

    B = A;

    ASSERT_EQUAL_QUIET(A, B);
    ASSERT_EQUAL(B.pitch, 4);

    cusp::array2d<float, Space> C;
    C.resize(3, 2, 4);

    cusp::copy(A, C);

    ASSERT_EQUAL_QUIET(A, C);
    ASSERT_EQUAL(C.pitch, 4);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dCopySemantics)

template <class Space>
void TestArray2dRowView(void)
{
    // row view of row major matrix
    {
        cusp::array2d<float, Space, cusp::row_major> A(3, 2, -1);

        ASSERT_EQUAL(A.row(0).size(), 2);

        for (size_t i = 0; i < A.num_rows; i++)
            cusp::blas::fill(A.row(i), i);

        ASSERT_EQUAL(A(0,0), 0);
        ASSERT_EQUAL(A(0,1), 0);
        ASSERT_EQUAL(A(1,0), 1);
        ASSERT_EQUAL(A(1,1), 1);
        ASSERT_EQUAL(A(2,0), 2);
        ASSERT_EQUAL(A(2,1), 2);
    }

    // row view of column major matrix
    {
        cusp::array2d<float, Space, cusp::column_major> A(3, 2, -1);

        ASSERT_EQUAL(A.row(0).size(), 2);

        for (size_t i = 0; i < A.num_rows; i++)
            cusp::blas::fill(A.row(i), i);

        ASSERT_EQUAL(A(0,0), 0);
        ASSERT_EQUAL(A(0,1), 0);
        ASSERT_EQUAL(A(1,0), 1);
        ASSERT_EQUAL(A(1,1), 1);
        ASSERT_EQUAL(A(2,0), 2);
        ASSERT_EQUAL(A(2,1), 2);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dRowView)

template <class Space>
void TestArray2dColumnView(void)
{
    // column view of column major matrix
    {
        cusp::array2d<float, Space, cusp::column_major> A(3, 2, -1);

        ASSERT_EQUAL(A.column(0).size(), 3);

        for (size_t i = 0; i < A.num_cols; i++)
            cusp::blas::fill(A.column(i), i);

        ASSERT_EQUAL(A(0,0), 0);
        ASSERT_EQUAL(A(1,0), 0);
        ASSERT_EQUAL(A(2,0), 0);
        ASSERT_EQUAL(A(0,1), 1);
        ASSERT_EQUAL(A(1,1), 1);
        ASSERT_EQUAL(A(2,1), 1);
    }

    // column view of row major matrix
    {
        cusp::array2d<float, Space, cusp::row_major> A(3, 2, -1);

        ASSERT_EQUAL(A.column(0).size(), 3);

        for (size_t i = 0; i < A.num_cols; i++)
            cusp::blas::fill(A.column(i), i);

        ASSERT_EQUAL(A(0,0), 0);
        ASSERT_EQUAL(A(1,0), 0);
        ASSERT_EQUAL(A(2,0), 0);
        ASSERT_EQUAL(A(0,1), 1);
        ASSERT_EQUAL(A(1,1), 1);
        ASSERT_EQUAL(A(2,1), 1);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dColumnView)


#include <unittest/unittest.h>
#include <cusp/array2d.h>
#include <cusp/gallery/poisson.h>
#include <cusp/lapack/lapack.h>

template<typename ValueType>
void TestGETRF(void)
{
    cusp::array2d<ValueType, cusp::host_memory> A;
    cusp::array1d<int, cusp::host_memory> piv;

    cusp::gallery::poisson5pt(A, 4, 4);

    cusp::lapack::getrf(A, piv);

    cusp::array2d<ValueType, cusp::host_memory> B(A.num_rows, 10);
    cusp::lapack::getrs(A, piv, B);
}
DECLARE_NUMERIC_UNITTEST(TestGETRF);

template<typename Array2d, typename Array1d>
void getrf(my_system& system, Array2d& A, Array1d& piv )
{
    system.validate_dispatch();
    return;
}

void TestGETRFDispatch()
{
    // initialize testing variables
    cusp::array2d<float, cusp::host_memory> A;
    cusp::array1d<float, cusp::host_memory> piv;

    my_system sys(0);

    // call with explicit dispatching
    cusp::lapack::getrf(sys, A, piv);

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestGETRFDispatch);

template<typename ValueType>
void TestPOTRF(void)
{
    cusp::array2d<ValueType, cusp::host_memory> A;

    cusp::gallery::poisson5pt(A, 4, 4);

    cusp::lapack::potrf(A);

    cusp::array2d<ValueType, cusp::host_memory> B(A.num_rows, 10);
    cusp::lapack::potrs(A, B);
}
DECLARE_NUMERIC_UNITTEST(TestPOTRF);

template<typename Array2d>
void potrf(my_system& system, Array2d& A, char uplo)
{
    system.validate_dispatch();
    return;
}

void TestPOTRFDispatch()
{
    // initialize testing variables
    cusp::array2d<float, cusp::host_memory> A;

    my_system sys(0);

    // call with explicit dispatching
    cusp::lapack::potrf(sys, A);

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestPOTRFDispatch);

template<typename ValueType>
void TestSYTRF(void)
{
    cusp::array2d<ValueType, cusp::host_memory> A;
    cusp::array1d<int, cusp::host_memory> piv;

    cusp::gallery::poisson5pt(A, 4, 4);

    cusp::lapack::sytrf(A, piv);

    cusp::array2d<ValueType, cusp::host_memory> B(A.num_rows, 10);
    cusp::lapack::sytrs(A, piv, B);
}
DECLARE_NUMERIC_UNITTEST(TestSYTRF);

template<typename Array2d, typename Array1d>
void sytrf(my_system& system, Array2d& A, Array1d& piv, char uplo)
{
    system.validate_dispatch();
    return;
}

void TestSYTRFDispatch()
{
    // initialize testing variables
    cusp::array2d<float, cusp::host_memory> A;
    cusp::array1d<int, cusp::host_memory> piv;

    my_system sys(0);

    // call with explicit dispatching
    cusp::lapack::sytrf(sys, A, piv);

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestSYTRFDispatch);

template<typename ValueType>
void TestGESV(void)
{
    cusp::array2d<ValueType, cusp::host_memory> A;
    cusp::array1d<int, cusp::host_memory> piv;

    cusp::gallery::poisson5pt(A, 4, 4);

    cusp::array2d<ValueType, cusp::host_memory> B(A.num_rows, 10);
    cusp::lapack::gesv(A, B, piv);
}
DECLARE_NUMERIC_UNITTEST(TestGESV);

template<typename Array2d, typename Array1d>
void gesv(my_system& system, const Array2d& A, Array2d& B, Array1d& piv)
{
    system.validate_dispatch();
    return;
}

void TestGESVDispatch()
{
    // initialize testing variables
    cusp::array2d<float, cusp::host_memory> A;
    cusp::array1d<int, cusp::host_memory> piv;

    my_system sys(0);

    // call with explicit dispatching
    cusp::lapack::gesv(sys, A, A, piv);

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestGESVDispatch);

template<typename ValueType>
void TestTRTRS(void)
{
    cusp::array2d<ValueType, cusp::host_memory> A;

    cusp::gallery::poisson5pt(A, 4, 4);

    cusp::array2d<ValueType, cusp::host_memory> B(A.num_rows, 10);
    cusp::lapack::trtrs(A, B);
}
DECLARE_NUMERIC_UNITTEST(TestTRTRS);

template<typename Array2d>
void trtrs(my_system& system, const Array2d& A, Array2d& B,
           char uplo, char trans, char diag)
{
    system.validate_dispatch();
    return;
}

void TestTRTRSDispatch()
{
    // initialize testing variables
    cusp::array2d<float, cusp::host_memory> A;

    my_system sys(0);

    // call with explicit dispatching
    cusp::lapack::trtrs(sys, A, A);

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestTRTRSDispatch);

template<typename ValueType>
void TestTRTRI(void)
{
    cusp::array2d<ValueType, cusp::host_memory> A;

    cusp::gallery::poisson5pt(A, 4, 4);

    cusp::lapack::trtri(A);
}
DECLARE_NUMERIC_UNITTEST(TestTRTRI);

template<typename Array2d>
void trtri(my_system& system, Array2d& A, char uplo, char diag)
{
    system.validate_dispatch();
    return;
}

void TestTRTRIDispatch()
{
    // initialize testing variables
    cusp::array2d<float, cusp::host_memory> A;

    my_system sys(0);

    // call with explicit dispatching
    cusp::lapack::trtri(sys, A);

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestTRTRIDispatch);

template<typename ValueType>
void TestSYEV(void)
{
    cusp::array2d<ValueType, cusp::host_memory> A;

    cusp::gallery::poisson5pt(A, 4, 4);

    cusp::array1d<ValueType, cusp::host_memory> eigvals;
    cusp::array2d<ValueType, cusp::host_memory> B(A.num_rows, A.num_cols);
    cusp::lapack::syev(A, eigvals, B);
}
DECLARE_REAL_UNITTEST(TestSYEV);

template<typename Array2d, typename Array1d>
void syev(my_system& system, const Array2d& A,
          Array1d& eigvals, Array2d& eigvecs, char uplo)
{
    system.validate_dispatch();
    return;
}

void TestSYEVDispatch()
{
    // initialize testing variables
    cusp::array2d<float, cusp::host_memory> A;
    cusp::array1d<float, cusp::host_memory> eigvals;
    cusp::array2d<float, cusp::host_memory> eigvecs;

    my_system sys(0);

    // call with explicit dispatching
    cusp::lapack::syev(sys, A, eigvals, eigvecs);

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestSYEVDispatch);


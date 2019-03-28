#include <unittest/unittest.h>

#include <cusp/print.h>

#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

#include <sstream>

template <typename Matrix>
void TestPrintMatrix(void)
{
    typedef typename Matrix::value_type ValueType;

    // initialize a 2x3 matrix
    cusp::array2d<ValueType, cusp::host_memory> A(2,3);
    A(0,0) = 42;
    A(0,1) =  0;
    A(0,2) = 53;
    A(1,0) =  0;
    A(1,1) = 71;
    A(1,2) =  0;

    Matrix M(A);

    std::ostringstream oss;

    cusp::print(M, oss);

    // ensure certain substrings are present in the output
    ASSERT_EQUAL(oss.str().length() > 0, true);
    ASSERT_EQUAL(oss.str().find("<2, 3>") != std::string::npos, true);
    ASSERT_EQUAL(oss.str().find("42") != std::string::npos, true);
    ASSERT_EQUAL(oss.str().find("53") != std::string::npos, true);
    ASSERT_EQUAL(oss.str().find("71") != std::string::npos, true);
}
DECLARE_MATRIX_UNITTEST(TestPrintMatrix);


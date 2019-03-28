#include <unittest/unittest.h>

#include <cusp/graph/symmetric_rcm.h>

#include <cusp/csr_matrix.h>
#include <cusp/permutation_matrix.h>

template <typename MatrixType, typename PermutationType>
void symmetric_rcm(my_system& system, const MatrixType& G, PermutationType& P)
{
    system.validate_dispatch();
    return;
}

void TestSymmetricRCMDispatch()
{
    // initialize testing variables
    cusp::csr_matrix<int, float, cusp::device_memory> A;
    cusp::permutation_matrix<int,cusp::device_memory> P;

    my_system sys(0);

    // call with explicit dispatching
    cusp::graph::symmetric_rcm(sys, A, P);

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestSymmetricRCMDispatch);


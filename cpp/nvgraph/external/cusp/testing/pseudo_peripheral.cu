#include <unittest/unittest.h>

#include <cusp/graph/pseudo_peripheral.h>

#include <cusp/csr_matrix.h>

template <typename MatrixType>
typename MatrixType::index_type
pseudo_peripheral_vertex(my_system& system, const MatrixType& G)
{
    system.validate_dispatch();
    return 0;
}

void TestPseudoPeripheralDispatch()
{
    // initialize testing variables
    cusp::csr_matrix<int, float, cusp::device_memory> A;

    my_system sys(0);

    // call with explicit dispatching
    cusp::graph::pseudo_peripheral_vertex(sys, A);

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestPseudoPeripheralDispatch);


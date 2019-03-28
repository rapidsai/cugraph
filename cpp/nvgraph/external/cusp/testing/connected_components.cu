#include <unittest/unittest.h>

#include <cusp/graph/connected_components.h>

#include <cusp/csr_matrix.h>

template <typename MatrixType, typename ArrayType>
size_t connected_components(my_system& system,
                            const MatrixType& G,
                            ArrayType& components)
{
    system.validate_dispatch();
    return 0;
}

void TestConnectedComponentsDispatch()
{
    // initialize testing variables
    cusp::csr_matrix<int, float, cusp::device_memory> A;
    cusp::array1d<int, cusp::device_memory> components;

    my_system sys(0);

    // call with explicit dispatching
    cusp::graph::connected_components(sys, A, components);

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestConnectedComponentsDispatch);


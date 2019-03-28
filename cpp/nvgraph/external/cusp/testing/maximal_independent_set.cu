#include <unittest/unittest.h>

#include <cusp/graph/maximal_independent_set.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/multiply.h>
#include <cusp/gallery/poisson.h>

// check whether the MIS is valid
template <typename MatrixType, typename ArrayType>
bool is_valid_mis(MatrixType& A, ArrayType& stencil)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    // convert matrix to CSR format on host
    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> csr(A);

    // copy mis array to host
    cusp::array1d<int,cusp::host_memory> mis(stencil);

    for (size_t i = 0; i < csr.num_rows; i++)
    {
        size_t num_mis_neighbors = 0;

        for(IndexType jj = csr.row_offsets[i]; jj < csr.row_offsets[i + 1]; jj++)
        {
            size_t j = csr.column_indices[jj];

            // XXX if/when MIS code filters explicit zeros we need to do that here too

            if (i != j && mis[j])
                num_mis_neighbors++;
        }

        if (mis[i])
        {
            if(num_mis_neighbors > 0)
            {
                std::cout << "Node " << i << " conflicts with another node" << std::endl;
                return false;
            }
        }
        else
        {
            if (num_mis_neighbors == 0)
            {
                std::cout << "Node " << i << " is not in the MIS and has no MIS neighbors" << std::endl;
                return false;
            }
        }
    }

    return true;
}

template <typename TestMatrix, typename ExampleMatrix>
void _TestMaximalIndependentSet(const ExampleMatrix& example_matrix)
{
    typedef typename TestMatrix::value_type   ValueType;
    typedef typename TestMatrix::memory_space MemorySpace;

    // initialize test matrix
    TestMatrix test_matrix(example_matrix);

    // allocate storage for MIS result
    cusp::array1d<int, MemorySpace> stencil(test_matrix.num_rows);

    {
        // compute MIS(0)
        size_t num_nodes = cusp::graph::maximal_independent_set(test_matrix, stencil, 0);

        // check MIS(0)
        ASSERT_EQUAL(thrust::count(stencil.begin(), stencil.end(), 1), num_nodes);
        ASSERT_EQUAL(num_nodes, test_matrix.num_rows);
    }

    {
        // compute MIS(1)
        size_t num_nodes = cusp::graph::maximal_independent_set(test_matrix, stencil);

        // check MIS for default k=1
        ASSERT_EQUAL(is_valid_mis(test_matrix, stencil), true);
        ASSERT_EQUAL(thrust::count(stencil.begin(), stencil.end(), 1), num_nodes);
    }

    {
        // compute MIS(2)
        size_t num_nodes = cusp::graph::maximal_independent_set(test_matrix, stencil, 2);

        // check MIS(2)
        cusp::coo_matrix<int,ValueType,MemorySpace> A(example_matrix);
        cusp::coo_matrix<int,ValueType,MemorySpace> A2;
        cusp::multiply(A, A, A2);

        ASSERT_EQUAL(is_valid_mis(A2, stencil), true);
        ASSERT_EQUAL(thrust::count(stencil.begin(), stencil.end(), 1), num_nodes);
    }
}

template <typename TestMatrix>
void TestMaximalIndependentSet(void)
{
    typedef typename TestMatrix::value_type   ValueType;

    // note: examples should be {0,1} matrices with 1s on the diagonal

    // two components of two nodes
    cusp::array2d<ValueType,cusp::host_memory> A(4,4);
    A(0,0) = 1;
    A(0,1) = 1;
    A(0,2) = 0;
    A(0,3) = 0;
    A(1,0) = 1;
    A(1,1) = 1;
    A(1,2) = 0;
    A(1,3) = 0;
    A(2,0) = 0;
    A(2,1) = 0;
    A(2,2) = 1;
    A(2,3) = 1;
    A(3,0) = 0;
    A(3,1) = 0;
    A(3,2) = 1;
    A(3,3) = 1;

    // linear graph
    cusp::array2d<ValueType,cusp::host_memory> B(4,4);
    B(0,0) = 1;
    B(0,1) = 1;
    B(0,2) = 0;
    B(0,3) = 0;
    B(1,0) = 1;
    B(1,1) = 1;
    B(1,2) = 1;
    B(1,3) = 0;
    B(2,0) = 0;
    B(2,1) = 1;
    B(2,2) = 1;
    B(2,3) = 1;
    B(3,0) = 0;
    B(3,1) = 0;
    B(3,2) = 1;
    B(3,3) = 1;

    // complete graph
    cusp::array2d<ValueType,cusp::host_memory> C(6,6,1);

    // empty graph
    cusp::array2d<ValueType,cusp::host_memory> D(6,6,0);

    cusp::coo_matrix<int,ValueType,cusp::host_memory> E;
    cusp::gallery::poisson5pt(E, 3, 3);
    thrust::fill(E.values.begin(), E.values.end(), 1.0f);

    cusp::coo_matrix<int,ValueType,cusp::host_memory> F;
    cusp::gallery::poisson5pt(F, 13, 17);
    thrust::fill(F.values.begin(), F.values.end(), 1.0f);

    cusp::coo_matrix<int,ValueType,cusp::host_memory> G;
    cusp::gallery::poisson5pt(G, 23, 24);
    thrust::fill(G.values.begin(), G.values.end(), 1.0f);

    cusp::coo_matrix<int,ValueType,cusp::host_memory> H;
    cusp::gallery::poisson5pt(H, 105, 107);
    thrust::fill(H.values.begin(), H.values.end(), 1.0f);

    _TestMaximalIndependentSet<TestMatrix>(A);
    _TestMaximalIndependentSet<TestMatrix>(B);
    _TestMaximalIndependentSet<TestMatrix>(C);
    _TestMaximalIndependentSet<TestMatrix>(D);
    _TestMaximalIndependentSet<TestMatrix>(E);
    _TestMaximalIndependentSet<TestMatrix>(F);
    _TestMaximalIndependentSet<TestMatrix>(G);
    _TestMaximalIndependentSet<TestMatrix>(H);
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestMaximalIndependentSet);


#include <unittest/unittest.h>

#include <cusp/graph/breadth_first_search.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

#include <cusp/gallery/poisson.h>

// check whether the MIS is valid
template <typename MatrixType, typename ArrayType1, typename ArrayType2>
bool is_valid_level_set(const MatrixType& A, const ArrayType1& tree, const ArrayType2& levels)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    // convert matrix to CSR format on host
    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> csr_graph(A);

    // copy mis array to host
    cusp::array1d<IndexType, cusp::host_memory> h_labels(tree);
    cusp::array1d<IndexType, cusp::host_memory> reference_labels(levels);

    // Verify plausibility of parent markings
    bool correct = true;

    for (size_t node = 0; node < A.num_rows; node++) {
        IndexType parent = h_labels[node];

        // Check that parentless nodes have zero or unvisited source distance
        IndexType node_dist = reference_labels[node];
        if (parent < 0) {
            if (reference_labels[node] > 0) {
                printf("INCORRECT: parentless node %lld (parent %lld) has positive distance distance %lld",
                       (long long) node, (long long) parent, (long long) node_dist);
                correct = false;
                break;
            }
            continue;
        }

        // Check that parent has iteration one less than node
        IndexType parent_dist = reference_labels[parent];
        if (parent_dist + 1 != node_dist) {
            printf("INCORRECT: parent %lld has distance %lld, node %lld has distance %lld",
                   (long long) parent, (long long) parent_dist, (long long) node, (long long) node_dist);
            correct = false;
            break;
        }

        // Check that parent is in fact a parent
        bool found = false;
        for (IndexType neighbor_offset = csr_graph.row_offsets[parent];
                neighbor_offset < csr_graph.row_offsets[parent + 1];
                neighbor_offset++)
        {
            if (csr_graph.column_indices[neighbor_offset] == IndexType(node)) {
                found = true;
                break;
            }
        }
        if (!found) {
            printf("INCORRECT: %lld is not a neighbor of %lld",
                   (long long) parent, (long long) node);
            correct = false;
            break;
        }
    }

    return correct;
}

template <typename TestMatrix, typename ExampleMatrix>
void _TestBreadthFirstSearch(const ExampleMatrix& example_matrix)
{
    typedef typename TestMatrix::index_type   IndexType;
    typedef typename TestMatrix::value_type   ValueType;
    typedef typename TestMatrix::memory_space MemorySpace;

    // initialize test matrix
    TestMatrix test_matrix(example_matrix);

    // allocate storage for MIS result
    cusp::array1d<IndexType, MemorySpace> tree(test_matrix.num_rows, -1);

    IndexType src = 0;

    if(sizeof(IndexType) <= 4)
    {
        // compute MIS(1)
        cusp::graph::breadth_first_search(test_matrix, src, tree, false);

        // compute MIS(1)
        cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> h_test_matrix(test_matrix);
        cusp::array1d<int, cusp::host_memory> levels(test_matrix.num_rows);
        cusp::graph::breadth_first_search(h_test_matrix, src, levels, true);

        // check MIS for default k=1
        ASSERT_EQUAL(is_valid_level_set(test_matrix, tree, levels), true);
    }
}

template <typename TestMatrix>
void TestBreadthFirstSearch(void)
{
    typedef typename TestMatrix::value_type ValueType;

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

    _TestBreadthFirstSearch<TestMatrix>(B);
    _TestBreadthFirstSearch<TestMatrix>(C);
    _TestBreadthFirstSearch<TestMatrix>(D);
    _TestBreadthFirstSearch<TestMatrix>(E);
    _TestBreadthFirstSearch<TestMatrix>(F);
    _TestBreadthFirstSearch<TestMatrix>(G);
    _TestBreadthFirstSearch<TestMatrix>(H);
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestBreadthFirstSearch)

template <typename MatrixType, typename ArrayType>
void breadth_first_search(my_system& system,
                          const MatrixType& G,
                          const typename MatrixType::index_type src,
                          ArrayType& labels,
                          const bool mark_levels)
{
    system.validate_dispatch();
    return;
}

void TestBreadthFirstSearchDispatch()
{
    // initialize testing variables
    cusp::csr_matrix<int, float, cusp::device_memory> A;
    cusp::array1d<int, cusp::device_memory> labels;

    my_system sys(0);

    // call with explicit dispatching
    cusp::graph::breadth_first_search(sys, A, 0, labels, true);

    // check if dispatch policy was used
    ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestBreadthFirstSearchDispatch);


#include <cusp/coo_matrix.h>
#include <cusp/print.h>

#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>

// Construct a sparse matrix from a list of unordered (i,j,v) triplets
// where duplicate entries are summed together.

int main(void)
{
    // dimensions of the matrix
    int num_rows = 3;
    int num_cols = 3;

    // number of (i,j,v) triplets
    int num_triplets = 10;

    // allocate storage for unordered triplets
    cusp::array1d<int,   cusp::device_memory> I(num_triplets);  // row indices
    cusp::array1d<int,   cusp::device_memory> J(num_triplets);  // column indices
    cusp::array1d<float, cusp::device_memory> V(num_triplets);  // values

    // fill triplet arrays
    I[0] = 2; J[0] = 0; V[0] = 10;
    I[1] = 0; J[1] = 2; V[1] = 10;
    I[2] = 1; J[2] = 1; V[2] = 10;
    I[3] = 2; J[3] = 0; V[3] = 10;
    I[4] = 1; J[4] = 1; V[4] = 10;
    I[5] = 0; J[5] = 0; V[5] = 10;
    I[6] = 2; J[6] = 2; V[6] = 10;
    I[7] = 0; J[7] = 0; V[7] = 10;
    I[8] = 1; J[8] = 0; V[8] = 10;
    I[9] = 0; J[9] = 0; V[9] = 10;

    // sort triplets by (i,j) index using two stable sorts (first by J, then by I)
    thrust::stable_sort_by_key(J.begin(), J.end(), thrust::make_zip_iterator(thrust::make_tuple(I.begin(), V.begin())));
    thrust::stable_sort_by_key(I.begin(), I.end(), thrust::make_zip_iterator(thrust::make_tuple(J.begin(), V.begin())));

    // compute unique number of nonzeros in the output
    int num_entries = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                                            thrust::make_zip_iterator(thrust::make_tuple(I.end (),  J.end()))   - 1,
                                            thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())) + 1,
                                            int(0),
                                            thrust::plus<int>(),
                                            thrust::not_equal_to< thrust::tuple<int,int> >()) + 1;

    // allocate output matrix
    cusp::coo_matrix<int, float, cusp::device_memory> A(num_rows, num_cols, num_entries);

    // sum values with the same (i,j) index
    thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(I.end(),   J.end())),
                          V.begin(),
                          thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
                          A.values.begin(),
                          thrust::equal_to< thrust::tuple<int,int> >(),
                          thrust::plus<float>());

    // print matrix
    cusp::print(A);

    return 0;
}


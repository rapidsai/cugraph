#include <cusp/csr_matrix.h>
#include <cusp/print.h>

int main(void)
{
    // define array container types
    typedef cusp::array1d<int,   cusp::device_memory> IndexArray;
    typedef cusp::array1d<float, cusp::device_memory> ValueArray;

    // define array view types
    typedef typename cusp::array1d_view<IndexArray::iterator> IndexArrayView;
    typedef typename cusp::array1d_view<ValueArray::iterator> ValueArrayView;

    // matrix dimensions
    int num_rows    = 4;
    int num_cols    = 3;
    int num_entries = 6;

    // allocate storage for (4,3) matrix with 6 nonzeros
    IndexArray row_offsets(num_rows + 1);
    IndexArray column_indices(num_entries);
    ValueArray values(num_entries);

    // initialize matrix data
    row_offsets[0] = 0;  // first offset is always zero
    row_offsets[1] = 2;
    row_offsets[2] = 2;
    row_offsets[3] = 3;
    row_offsets[4] = 6;  // last offset is always num_entries

    column_indices[0] = 0;
    values[0] = 10;
    column_indices[1] = 2;
    values[1] = 20;
    column_indices[2] = 2;
    values[2] = 30;
    column_indices[3] = 0;
    values[3] = 40;
    column_indices[4] = 1;
    values[4] = 50;
    column_indices[5] = 2;
    values[5] = 60;

    // define csr_matrix_view type
    typedef cusp::csr_matrix_view<IndexArrayView, IndexArrayView, ValueArrayView> View;

    // create a csr_matrix_view
    View A(num_rows, num_cols, num_entries,
           cusp::make_array1d_view(row_offsets),
           cusp::make_array1d_view(column_indices),
           cusp::make_array1d_view(values));

    // create a csr_matrix_view like A but with different values
    ValueArray other_values(6);
    other_values[0] = 60;
    other_values[1] = 50;
    other_values[2] = 40;
    other_values[3] = 30;
    other_values[4] = 20;
    other_values[5] = 10;

    View B(num_rows, num_cols, num_entries,
           cusp::make_array1d_view(row_offsets),
           cusp::make_array1d_view(column_indices),
           cusp::make_array1d_view(other_values));

    // note that A and B are views to the same row_offsets and column_indices arrays

    // A now represents the following matrix
    //    [10  0 20]
    //    [ 0  0  0]
    //    [ 0  0 30]
    //    [40 50 60]

    // B now represents the following matrix
    //    [60  0 50]
    //    [ 0  0  0]
    //    [ 0  0 40]
    //    [30 20 10]

    // print matrix entries
    cusp::print(A);
    cusp::print(B);

    return 0;
}



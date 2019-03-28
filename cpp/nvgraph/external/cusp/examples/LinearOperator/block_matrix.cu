#include <cusp/multiply.h>
#include <cusp/linear_operator.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>

#include <thrust/functional.h>

template<typename MatrixType>
class block_matrix : public cusp::linear_operator<typename MatrixType::value_type, typename MatrixType::memory_space>
{
private:
    typedef typename MatrixType::index_type   IndexType;
    typedef typename MatrixType::value_type   ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    typedef cusp::linear_operator<ValueType,MemorySpace> Parent;
    typedef std::vector<MatrixType> MatrixList;

    typedef cusp::detail::plus_value<IndexType>                             ShiftOp;
    typedef typename MatrixType::row_indices_array_type::const_iterator     RowIterator;
    typedef typename MatrixType::column_indices_array_type::const_iterator  ColumnIterator;

    typedef thrust::transform_iterator<ShiftOp,RowIterator>    RowTransformIterator;
    typedef thrust::transform_iterator<ShiftOp,ColumnIterator> ColumnTransformIterator;

    typedef cusp::array1d_view<RowTransformIterator>            RowView;
    typedef cusp::array1d_view<ColumnTransformIterator>         ColumnView;
    typedef typename MatrixType::values_array_type::const_view  ValueView;
    typedef cusp::coo_matrix_view<RowView,ColumnView,ValueView> ShiftedViewType;

    const MatrixList& C_list;
    const MatrixList& D_list;

    std::vector<ShiftedViewType> shifted_C_list;
    std::vector<ShiftedViewType> shifted_D_list;

    cusp::array1d<unsigned int, MemorySpace> indices;

public:
    block_matrix(MatrixList& D_list, MatrixList& C_list)
        : Parent(), C_list(C_list), D_list(D_list)
    {
        size_t num_rows = 0;
        size_t num_C_entries = 0;
        size_t num_D_entries = 0;

        // Compute number of row and entries in diagonal blocks
        for(size_t i = 0; i < D_list.size(); i++)
        {
            num_rows += D_list[i].num_rows;
            num_D_entries += D_list[i].num_entries;
        }

        // Add entries from block column
        for(size_t i = 0; i < C_list.size(); i++)
            num_C_entries += C_list[i].num_entries;

        size_t num_cols = num_rows - D_list.back().num_rows;
        size_t num_entries = num_D_entries + (2 * num_C_entries);

        // Resize parent to correct size of concatenated matrices
        Parent::resize(num_rows, num_rows, num_entries);

        // resize indices to total number of nonzeros
        indices.resize(num_entries, -1);

        size_t row = 0;
        size_t col = 0;

        size_t C_offset = 0;
        size_t D_offset = 0;

        for(size_t i = 0; i < C_list.size(); i++)
        {
            const MatrixType& A = C_list[i];
            ShiftedViewType S_C(row + A.num_rows, num_cols + A.num_cols, A.num_entries,
                                cusp::make_array1d_view(thrust::make_transform_iterator(A.row_indices.cbegin(), ShiftOp(row)),
                                        thrust::make_transform_iterator(A.row_indices.cbegin(), ShiftOp(row)) + A.num_entries),
                                cusp::make_array1d_view(thrust::make_transform_iterator(A.column_indices.cbegin(), ShiftOp(num_cols)),
                                        thrust::make_transform_iterator(A.column_indices.cbegin(), ShiftOp(num_cols)) + A.num_entries),
                                cusp::make_array1d_view(A.values));
            shifted_C_list.push_back(S_C);

            const MatrixType& B = D_list[i];
            ShiftedViewType S_D(row + B.num_rows, row + B.num_cols, B.num_entries,
                                cusp::make_array1d_view(thrust::make_transform_iterator(B.row_indices.cbegin(), ShiftOp(row)),
                                        thrust::make_transform_iterator(B.row_indices.cbegin(), ShiftOp(row)) + B.num_entries),
                                cusp::make_array1d_view(thrust::make_transform_iterator(B.column_indices.cbegin(), ShiftOp(col)),
                                        thrust::make_transform_iterator(B.column_indices.cbegin(), ShiftOp(col)) + B.num_entries),
                                cusp::make_array1d_view(B.values));

            // compute row lengths of C matrix
            cusp::array1d<IndexType,MemorySpace> C_row_lengths(C_list[i].num_rows + 1, 0);
            thrust::reduce_by_key(C_list[i].row_indices.begin(), C_list[i].row_indices.end(),
                                  thrust::constant_iterator<IndexType>(1),
                                  thrust::make_discard_iterator(),
                                  C_row_lengths.begin());

            // compute row lengths of D matrix
            cusp::array1d<IndexType,MemorySpace> D_row_lengths(D_list[i].num_rows + 1, 0);
            thrust::reduce_by_key(D_list[i].row_indices.begin(), D_list[i].row_indices.end(),
                                  thrust::constant_iterator<IndexType>(1),
                                  thrust::make_discard_iterator(),
                                  D_row_lengths.begin());

            // compute combined operator offsets
            cusp::array1d<IndexType,MemorySpace> K_row_offsets(D_list[i].num_rows + 1, 0);
            thrust::transform(C_row_lengths.begin(), C_row_lengths.end(),
                              D_row_lengths.begin(), K_row_offsets.begin(),
                              thrust::plus<IndexType>());
            thrust::exclusive_scan(K_row_offsets.begin(), K_row_offsets.end(), K_row_offsets.begin(), 0);

            // transform D_row_lengths to D_row_offsets for scattering into D_map
            thrust::exclusive_scan(D_row_lengths.begin(), D_row_lengths.end(), D_row_lengths.begin(), 0);

            // allocate array of ones for mapping D nonzeros to K operator
            cusp::array1d<IndexType,MemorySpace> D_map(D_list[i].num_entries, 1);
            // scatter starting offsets with respect to K operator into D_map
            thrust::scatter(K_row_offsets.begin(), K_row_offsets.end(), D_row_lengths.begin(), D_map.begin());
            // run segmented scan over indices to construct running offsets of D matrix nonzeros
            thrust::inclusive_scan_by_key(D_list[i].row_indices.begin(),
                                          D_list[i].row_indices.end(),
                                          D_map.begin(),
                                          D_map.begin());
            // scatter final index offsets into indices array
            thrust::scatter(thrust::counting_iterator<IndexType>(D_offset),
                            thrust::counting_iterator<IndexType>(D_offset + D_list[i].num_entries),
                            D_map.begin(),
                            indices.begin() + C_offset + D_offset);

            // transform C_row_lengths to C_row_offsets for scattering into C_map
            thrust::exclusive_scan(C_row_lengths.begin(), C_row_lengths.end(), C_row_lengths.begin(), 0);

            // shift K_row_offsets by D_row_offsets
            thrust::reduce_by_key(D_list[i].row_indices.begin(), D_list[i].row_indices.end(),
                                  thrust::constant_iterator<IndexType>(1),
                                  thrust::make_discard_iterator(),
                                  D_row_lengths.begin());
            thrust::transform(D_row_lengths.begin(), D_row_lengths.end(),
                              K_row_offsets.begin(), K_row_offsets.begin(),
                              thrust::plus<IndexType>());

            // allocate array of ones for mapping C nonzeros to K operator
            cusp::array1d<IndexType,MemorySpace> C_map(C_list[i].num_entries, 1);
            // scatter starting offsets with respect to K operator into C_map
            thrust::scatter(K_row_offsets.begin(), K_row_offsets.end(), C_row_lengths.begin(), C_map.begin());
            // run segmented scan over indices to construct running offsets of D matrix nonzeros
            thrust::inclusive_scan_by_key(C_list[i].row_indices.begin(),
                                          C_list[i].row_indices.end(),
                                          C_map.begin(),
                                          C_map.begin());
            // scatter final index offsets into indices array
            thrust::scatter(thrust::counting_iterator<IndexType>(num_D_entries + C_offset),
                            thrust::counting_iterator<IndexType>(C_offset + num_D_entries + C_list[i].num_entries),
                            C_map.begin(),
                            indices.begin() + C_offset + D_offset);

            // increment C and D matrix starting offsets
            row += D_list[i].num_rows;
            col += D_list[i].num_cols;
            C_offset += C_list[i].num_entries;
            D_offset += D_list[i].num_entries;
        }

        {
            cusp::counting_array<IndexType> C_indices(num_C_entries);
            auto B_join_indices = cusp::make_array1d_view(cusp::make_join_iterator(shifted_C_list[0].num_entries, shifted_C_list[1].num_entries,
                                  shifted_C_list[0].column_indices.begin(), shifted_C_list[1].column_indices.begin(), C_indices.begin()),
                                  cusp::make_join_iterator(shifted_C_list[0].num_entries, shifted_C_list[1].num_entries,
                                          shifted_C_list[0].column_indices.begin(), shifted_C_list[1].column_indices.begin(), C_indices.begin()) + num_C_entries);

            cusp::array1d<IndexType,MemorySpace> C_t_indices(num_C_entries);
            thrust::sequence(C_t_indices.begin(), C_t_indices.end());

            cusp::array1d<IndexType,MemorySpace> B_column_indices(B_join_indices);
            thrust::stable_sort_by_key(B_column_indices.begin(), B_column_indices.end(), C_t_indices.begin());

            auto B_t = cusp::make_coo_matrix_view(num_rows, num_cols, num_C_entries,
                                                  cusp::make_array1d_view(cusp::make_join_iterator(shifted_C_list[0].num_entries, shifted_C_list[1].num_entries,
                                                          shifted_C_list[0].column_indices.begin(), shifted_C_list[1].column_indices.begin(), C_t_indices.begin()),
                                                          cusp::make_join_iterator(shifted_C_list[0].num_entries, shifted_C_list[1].num_entries,
                                                                  shifted_C_list[0].column_indices.begin(), shifted_C_list[1].column_indices.begin(), C_t_indices.begin()) + num_C_entries),
                                                  cusp::make_array1d_view(cusp::make_join_iterator(shifted_C_list[0].num_entries, shifted_C_list[1].num_entries,
                                                          shifted_C_list[0].row_indices.begin(), shifted_C_list[1].row_indices.begin(), C_t_indices.begin()),
                                                          cusp::make_join_iterator(shifted_C_list[0].num_entries, shifted_C_list[1].num_entries,
                                                                  shifted_C_list[0].row_indices.begin(), shifted_C_list[1].row_indices.begin(), C_t_indices.begin()) + num_C_entries),
                                                  cusp::make_array1d_view(cusp::make_join_iterator(shifted_C_list[0].num_entries, shifted_C_list[1].num_entries,
                                                          shifted_C_list[0].values.begin(), shifted_C_list[1].values.begin(), C_t_indices.begin()),
                                                          cusp::make_join_iterator(shifted_C_list[0].num_entries, shifted_C_list[1].num_entries,
                                                                  shifted_C_list[0].values.begin(), shifted_C_list[1].values.begin(), C_t_indices.begin()) + num_C_entries));

            cusp::array1d<IndexType,MemorySpace> B_t_row_offsets(num_rows + 1);
            cusp::indices_to_offsets(B_t.row_indices, B_t_row_offsets);

            const MatrixType& E = D_list.back();

            // compute row lengths of D matrix
            cusp::array1d<IndexType,MemorySpace> E_row_lengths(num_rows + 1, 0);
            thrust::reduce_by_key(E.row_indices.begin(), E.row_indices.end(),
                                  thrust::constant_iterator<IndexType>(1),
                                  thrust::make_discard_iterator(),
                                  E_row_lengths.begin() + row);

            // compute row lengths of D matrix
            cusp::array1d<IndexType,MemorySpace> B_t_row_lengths(num_rows + 1, 0);
            thrust::reduce_by_key(B_t.row_indices.begin(), B_t.row_indices.end(),
                                  thrust::constant_iterator<IndexType>(1),
                                  thrust::make_discard_iterator(),
                                  B_t_row_lengths.begin() + row);

            // compute combined operator offsets
            cusp::array1d<IndexType,MemorySpace> K_row_offsets(num_rows + 1, 0);
            thrust::transform(E_row_lengths.begin(), E_row_lengths.end(),
                              B_t_row_lengths.begin(), K_row_offsets.begin(),
                              thrust::plus<IndexType>());
            thrust::exclusive_scan(K_row_offsets.begin(), K_row_offsets.end(), K_row_offsets.begin(), 0);

            // transform D_row_lengths to D_row_offsets for scattering into D_map
            thrust::exclusive_scan(B_t_row_lengths.begin(), B_t_row_lengths.end(), B_t_row_lengths.begin(), 0);

            // allocate array of ones for mapping D nonzeros to K operator
            cusp::array1d<IndexType,MemorySpace> B_t_map(B_t.num_entries, 1);
            // scatter starting offsets with respect to K operator into D_map
            thrust::scatter(K_row_offsets.begin(), K_row_offsets.end(), B_t_row_lengths.begin(), B_t_map.begin());
            // run segmented scan over indices to construct running offsets of D matrix nonzeros
            thrust::inclusive_scan_by_key(B_t.row_indices.begin(),
                                          B_t.row_indices.end(),
                                          B_t_map.begin(),
                                          B_t_map.begin());

            cusp::print(K_row_offsets);
            cusp::print(B_t_row_lengths);
            cusp::print(B_t_map);
            // scatter final index offsets into indices array
            thrust::scatter(thrust::counting_iterator<IndexType>(C_offset + D_offset),
                            thrust::counting_iterator<IndexType>(C_offset + D_offset + B_t.num_entries),
                            B_t_map.begin(),
                            indices.begin() + C_offset + D_offset);

            // shift K_row_offsets by D_row_offsets
            /* cusp::blas::fill(E_row_lengths, 0); */
            /* thrust::reduce_by_key(E.row_indices.begin(), E.row_indices.end(), */
            /*                       thrust::constant_iterator<IndexType>(1), */
            /*                       thrust::make_discard_iterator(), */
            /*                       E_row_lengths.begin()); */
            /* thrust::transform(B_t_row_lengths.begin(), B_t_row_lengths.end(), */
            /*                   K_row_offsets.begin(), K_row_offsets.begin(), */
            /*                   thrust::plus<IndexType>()); */
            /*  */
            /* // transform C_row_lengths to C_row_offsets for scattering into C_map */
            /* thrust::exclusive_scan(E_row_lengths.begin(), E_row_lengths.end(), E_row_lengths.begin(), 0); */
            /*  */
            /* // allocate array of ones for mapping C nonzeros to K operator */
            /* cusp::array1d<IndexType,MemorySpace> C_map(E.num_entries, 1); */
            /* // scatter starting offsets with respect to K operator into C_map */
            /* thrust::scatter(K_row_offsets.begin(), K_row_offsets.end(), E_row_lengths.begin(), C_map.begin()); */
            /* // run segmented scan over indices to construct running offsets of D matrix nonzeros */
            /* thrust::inclusive_scan_by_key(E.row_indices.begin(), */
            /*                               E.row_indices.end(), */
            /*                               C_map.begin(), */
            /*                               C_map.begin()); */
            /* // scatter final index offsets into indices array */
            /* thrust::scatter(thrust::counting_iterator<IndexType>(C_offset + D_offset), */
            /*                 thrust::counting_iterator<IndexType>(C_offset + D_offset + E.num_entries), */
            /*                 C_map.begin(), */
            /*                 indices.begin() + C_offset + D_offset + E.num_entries); */
        }
    }
};

int main(void)
{
    typedef cusp::device_memory MemorySpace;
    typedef cusp::coo_matrix<int,float,MemorySpace> MatrixType;
    MatrixType A;

    cusp::gallery::poisson5pt(A, 5, 5);
    std::cout << "Generated base operator with shape ("  << A.num_rows << "," << A.num_cols << ") and "
              << A.num_entries << " entries" << "\n\n";

    std::vector<MatrixType> D_list;
    std::vector<MatrixType> C_list;

    D_list.push_back(A);
    D_list.push_back(A);
    D_list.push_back(A);

    C_list.push_back(A);
    C_list.push_back(A);

    auto K = block_matrix<MatrixType>(D_list, C_list);

    std::cout << "Generated concatenated operator with shape ("  << K.num_rows << "," << K.num_cols << ") and "
              << K.num_entries << " entries" << "\n\n";

    return 0;
}




/*
 *  Copyright 2008-2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/format_utils.h>

#include <cusp/detail/temporary_array.h>

#include <cusp/system/detail/sequential/execution_policy.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{
namespace detail
{

template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType1,
          typename ArrayType2>
void standard_aggregate(thrust::cpp::execution_policy<DerivedPolicy> &exec,
                        const MatrixType& A,
                              ArrayType1& aggregates,
                              ArrayType2& roots,
                              cusp::csr_format)
{
    typedef typename MatrixType::index_type IndexType;

    IndexType next_aggregate = 1; // number of aggregates + 1

    // initialize aggregates to 0
    thrust::fill(exec, aggregates.begin(), aggregates.end(), 0);

    IndexType n_row = A.num_rows;

    //Pass #1
    for (IndexType i = 0; i < n_row; i++)
    {
        if (aggregates[i]) {
            continue;    //already marked
        }

        const IndexType row_start = A.row_offsets[i];
        const IndexType row_end   = A.row_offsets[i+1];

        //Determine whether all neighbors of this node are free (not already aggregates)
        bool has_aggregated_neighbors = false;
        bool has_neighbors            = false;

        for (IndexType jj = row_start; jj < row_end; jj++)
        {
            const IndexType j = A.column_indices[jj];
            if ( i != j )
            {
                has_neighbors = true;
                if ( aggregates[j] )
                {
                    has_aggregated_neighbors = true;
                    break;
                }
            }
        }

        if (!has_neighbors)
        {
            //isolated node, do not aggregate
            aggregates[i] = -n_row;
        }
        else if (!has_aggregated_neighbors)
        {
            //Make an aggregate out of this node and its neighbors
            aggregates[i] = next_aggregate;
            roots[next_aggregate-1] = i;
            for (IndexType jj = row_start; jj < row_end; jj++) {
                aggregates[A.column_indices[jj]] = next_aggregate;
            }
            next_aggregate++;
        }
    }

    //Pass #2
    // Add unaggregated nodes to any neighboring aggregate
    for (IndexType i = 0; i < n_row; i++) {
        if (aggregates[i]) {
            continue;    //already marked
        }

        for (IndexType jj = A.row_offsets[i]; jj < A.row_offsets[i+1]; jj++) {
            const IndexType j = A.column_indices[jj];

            const IndexType tj = aggregates[j];
            if (tj > 0) {
                aggregates[i] = -tj;
                break;
            }
        }
    }

    next_aggregate--;

    //Pass #3
    for (IndexType i = 0; i < n_row; i++) {
        const IndexType ti = aggregates[i];

        if (ti != 0) {
            // node i has been aggregated
            if (ti > 0)
                aggregates[i] = ti - 1;
            else if (ti == -n_row)
                aggregates[i] = -1;
            else
                aggregates[i] = -ti - 1;
            continue;
        }

        // node i has not been aggregated
        const IndexType row_start = A.row_offsets[i];
        const IndexType row_end   = A.row_offsets[i+1];

        aggregates[i] = next_aggregate;
        roots[next_aggregate] = i;

        for (IndexType jj = row_start; jj < row_end; jj++) {
            const IndexType j = A.column_indices[jj];

            if (aggregates[j] == 0) { //unmarked neighbors
                aggregates[j] = next_aggregate;
            }
        }
        next_aggregate++;
    }

    if ( next_aggregate == 0 ) {
        thrust::fill(exec, aggregates.begin(), aggregates.end(), 0);
    }
}

template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType1,
          typename ArrayType2>
void standard_aggregate(thrust::cpp::execution_policy<DerivedPolicy> &exec,
                        const MatrixType& A,
                              ArrayType1& aggregates,
                              ArrayType2& roots,
                        cusp::known_format)
{
    typedef typename MatrixType::index_type          IndexType;
    typedef typename MatrixType::const_coo_view_type CooView;

    CooView A_coo(A);

    cusp::detail::temporary_array<IndexType, DerivedPolicy> row_offsets(exec, A.num_rows + 1);
    cusp::indices_to_offsets(exec, A_coo.row_indices, row_offsets);

    standard_aggregate(exec,
                       cusp::make_csr_matrix_view(A.num_rows, A.num_cols, A.num_entries,
                                                  cusp::make_array1d_view(row_offsets),
                                                  cusp::make_array1d_view(A_coo.column_indices),
                                                  cusp::make_array1d_view(A_coo.values)),
                       aggregates,
                       roots,
                       cusp::csr_format());
}

template <typename DerivedPolicy,
          typename MatrixType,
          typename ArrayType1,
          typename ArrayType2>
void standard_aggregate(thrust::cpp::execution_policy<DerivedPolicy> &exec,
                        const MatrixType& A,
                              ArrayType1& aggregates,
                              ArrayType2& roots)
{
    typedef typename MatrixType::format Format;

    Format format;

    standard_aggregate(exec, A, aggregates, roots, format);
}

} // end namespace detail
} // end namespace aggregation
} // end namespace precond
} // end namespace cusp


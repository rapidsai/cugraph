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

#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/format_utils.h>
#include <cusp/sort.h>

#include <cusp/detail/temporary_array.h>

#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

#include <list>

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3,
          typename ArrayType1,
          typename ArrayType2,
          typename BinaryFunction1,
          typename BinaryFunction2>
void coo_spmm_helper(cuda::execution_policy<DerivedPolicy>& exec,
                     size_t workspace_size,
                     size_t begin_row,
                     size_t end_row,
                     size_t begin_segment,
                     size_t end_segment,
                     const MatrixType1& A,
                     const MatrixType2& B,
                     MatrixType3& C,
                     const ArrayType1& B_row_offsets,
                     const ArrayType1& segment_lengths,
                     const ArrayType1& output_ptr,
                     ArrayType1& A_gather_locations,
                     ArrayType1& B_gather_locations,
                     ArrayType1& I,
                     ArrayType1& J,
                     ArrayType2& V,
                     BinaryFunction1 combine,
                     BinaryFunction2 reduce)
{
    typedef typename ArrayType1::value_type IndexType;
    typedef typename ArrayType2::value_type ValueType;

    A_gather_locations.resize(workspace_size);
    B_gather_locations.resize(workspace_size);
    I.resize(workspace_size);
    J.resize(workspace_size);
    V.resize(workspace_size);

    // nothing to do
    if (workspace_size == 0)
    {
        C.resize(A.num_rows, B.num_cols, 0);
        return;
    }

    // compute gather locations of intermediate format
    thrust::fill(exec, A_gather_locations.begin(), A_gather_locations.end(), 0);
    thrust::scatter_if(exec,
                       thrust::counting_iterator<IndexType>(begin_segment), thrust::counting_iterator<IndexType>(end_segment),
                       output_ptr.begin() + begin_segment,
                       segment_lengths.begin() + begin_segment,
                       A_gather_locations.begin() - output_ptr[begin_segment]);
    thrust::inclusive_scan(exec, A_gather_locations.begin(), A_gather_locations.end(), A_gather_locations.begin(), thrust::maximum<IndexType>());

    // compute gather locations of intermediate format
    thrust::fill(exec, B_gather_locations.begin(), B_gather_locations.end(), 1);
    thrust::scatter_if(exec,
                       thrust::make_permutation_iterator(B_row_offsets.begin(), A.column_indices.begin()) + begin_segment,
                       thrust::make_permutation_iterator(B_row_offsets.begin(), A.column_indices.begin()) + end_segment,
                       output_ptr.begin() + begin_segment,
                       segment_lengths.begin() + begin_segment,
                       B_gather_locations.begin() - output_ptr[begin_segment]);
    thrust::inclusive_scan_by_key(exec,
                                  A_gather_locations.begin(), A_gather_locations.end(),
                                  B_gather_locations.begin(),
                                  B_gather_locations.begin());

    thrust::gather(exec,
                   A_gather_locations.begin(), A_gather_locations.end(),
                   A.row_indices.begin(),
                   I.begin());
    thrust::gather(exec,
                   B_gather_locations.begin(), B_gather_locations.end(),
                   B.column_indices.begin(),
                   J.begin());

    thrust::transform(exec,
                      thrust::make_permutation_iterator(A.values.begin(), A_gather_locations.begin()),
                      thrust::make_permutation_iterator(A.values.begin(), A_gather_locations.end()),
                      thrust::make_permutation_iterator(B.values.begin(), B_gather_locations.begin()),
                      V.begin(),
                      combine);

    // sort (I,J,V) tuples by (I,J)
    cusp::sort_by_row_and_column(exec, I, J, V);

    // compute unique number of nonzeros in the output
    IndexType NNZ = thrust::inner_product(exec,
                                          thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                                          thrust::make_zip_iterator(thrust::make_tuple(I.end (),  J.end()))   - 1,
                                          thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())) + 1,
                                          IndexType(0),
                                          thrust::plus<IndexType>(),
                                          thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >()) + 1;

    // allocate space for output
    C.resize(A.num_rows, B.num_cols, NNZ);

    // sum values with the same (i,j)
    thrust::reduce_by_key
    (exec,
     thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
     thrust::make_zip_iterator(thrust::make_tuple(I.end(),   J.end())),
     V.begin(),
     thrust::make_zip_iterator(thrust::make_tuple(C.row_indices.begin(), C.column_indices.begin())),
     C.values.begin(),
     thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
     reduce);
}

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void multiply(cuda::execution_policy<DerivedPolicy>& exec,
              const MatrixType1& A,
              const MatrixType2& B,
              MatrixType3& C,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce,
              cusp::coo_format,
              cusp::coo_format,
              cusp::coo_format)
{
    typedef typename MatrixType3::index_type   IndexType;
    typedef typename MatrixType3::value_type   ValueType;
    typedef typename MatrixType3::memory_space MemorySpace;

    // check whether matrices are empty
    if (A.num_entries == 0 || B.num_entries == 0)
    {
        C.resize(A.num_rows, B.num_cols, 0);
        return;
    }

    // compute row offsets for B
#if THRUST_VERSION >= 100800
    cusp::detail::temporary_array<IndexType, DerivedPolicy> B_row_offsets(exec, B.num_rows + 1);
#else
    cusp::array1d<IndexType, MemorySpace> B_row_offsets(B.num_rows + 1);
#endif

    cusp::indices_to_offsets(exec, B.row_indices, B_row_offsets);

    // compute row lengths for B
#if THRUST_VERSION >= 100800
    cusp::detail::temporary_array<IndexType, DerivedPolicy> B_row_lengths(exec, B.num_rows);
#else
    cusp::array1d<IndexType, MemorySpace> B_row_lengths(B.num_rows);
#endif

    thrust::transform(exec, B_row_offsets.begin() + 1, B_row_offsets.end(), B_row_offsets.begin(), B_row_lengths.begin(), thrust::minus<IndexType>());

    // for each element A(i,j) compute the number of nonzero elements in B(j,:)
#if THRUST_VERSION >= 100800
    cusp::detail::temporary_array<IndexType, DerivedPolicy> segment_lengths(exec, A.num_entries);
#else
    cusp::array1d<IndexType, MemorySpace> segment_lengths(A.num_entries);
#endif

    thrust::gather(exec,
                   A.column_indices.begin(), A.column_indices.end(),
                   B_row_lengths.begin(),
                   segment_lengths.begin());

    // output pointer
#if THRUST_VERSION >= 100800
    cusp::detail::temporary_array<IndexType, DerivedPolicy> output_ptr(exec, A.num_entries + 1);
#else
    cusp::array1d<IndexType, MemorySpace> output_ptr(A.num_entries + 1);
#endif

    thrust::exclusive_scan(exec,
                           segment_lengths.begin(), segment_lengths.end(),
                           output_ptr.begin(),
                           IndexType(0));
    output_ptr[A.num_entries] = output_ptr[A.num_entries - 1] + segment_lengths[A.num_entries - 1]; // XXX is this necessary?

    size_t coo_num_nonzeros = output_ptr[A.num_entries];

    size_t workspace_capacity = thrust::min<size_t>(coo_num_nonzeros, 16 << 20);

    {
        size_t free, total;
        cudaMemGetInfo(&free, &total);

        // divide free bytes by the size of each workspace unit
        size_t max_workspace_capacity = free / (4 * sizeof(IndexType) + sizeof(ValueType));

        // use at most one third of the remaining capacity
        workspace_capacity = thrust::min<size_t>(max_workspace_capacity / 3, workspace_capacity);
    }

    // workspace arrays
#if THRUST_VERSION >= 100800
    cusp::detail::temporary_array<IndexType, DerivedPolicy> A_gather_locations(exec);
    cusp::detail::temporary_array<IndexType, DerivedPolicy> B_gather_locations(exec);
    cusp::detail::temporary_array<IndexType, DerivedPolicy> I(exec);
    cusp::detail::temporary_array<IndexType, DerivedPolicy> J(exec);
    cusp::detail::temporary_array<ValueType, DerivedPolicy> V(exec);
#else
    cusp::array1d<IndexType, MemorySpace> A_gather_locations;
    cusp::array1d<IndexType, MemorySpace> B_gather_locations;
    cusp::array1d<IndexType, MemorySpace> I;
    cusp::array1d<IndexType, MemorySpace> J;
    cusp::array1d<ValueType, MemorySpace> V;
#endif

    if (coo_num_nonzeros <= workspace_capacity)
    {
        // compute C = A * B in one step
        size_t begin_row      = 0;
        size_t end_row        = A.num_rows;
        size_t begin_segment  = 0;
        size_t end_segment    = A.num_entries;
        size_t workspace_size = coo_num_nonzeros;

        coo_spmm_helper(exec,
                        workspace_size,
                        begin_row, end_row,
                        begin_segment, end_segment,
                        A, B, C,
                        B_row_offsets,
                        segment_lengths, output_ptr,
                        A_gather_locations, B_gather_locations,
                        I, J, V,
                        combine, reduce);
    }
    else
    {
        // decompose C = A * B into several C[slice,:] = A[slice,:] * B operations
        typedef typename cusp::coo_matrix<IndexType,ValueType,MemorySpace> Container;
        typedef typename std::list<Container> ContainerList;

        // storage for C[slice,:] partial results
        ContainerList slices;

        // compute row offsets for A
#if THRUST_VERSION >= 100800
        cusp::detail::temporary_array<IndexType, DerivedPolicy> A_row_offsets(exec, A.num_rows + 1);
#else
        cusp::array1d<IndexType, MemorySpace> A_row_offsets(A.num_rows + 1);
#endif

        cusp::indices_to_offsets(exec, A.row_indices, A_row_offsets);

        // compute worspace requirements for each row
#if THRUST_VERSION >= 100800
        cusp::detail::temporary_array<IndexType, DerivedPolicy> cummulative_row_workspace(exec, A.num_rows);
#else
        cusp::array1d<IndexType, MemorySpace> cummulative_row_workspace(A.num_rows);
#endif

        thrust::gather(exec,
                       A_row_offsets.begin() + 1, A_row_offsets.end(),
                       output_ptr.begin(),
                       cummulative_row_workspace.begin());

        size_t begin_row = 0;
        size_t total_work = 0;

        while (begin_row < size_t(A.num_rows))
        {
            Container C_slice;

            // find largest end_row such that the capacity of [begin_row, end_row) fits in the workspace_capacity
            size_t end_row = thrust::upper_bound(exec,
                                                 cummulative_row_workspace.begin() + begin_row, cummulative_row_workspace.end(),
                                                 IndexType(total_work + workspace_capacity)) - cummulative_row_workspace.begin();

            size_t begin_segment = A_row_offsets[begin_row];
            size_t end_segment   = A_row_offsets[end_row];

            // TODO throw exception signaling that there is insufficient memory (not necessarily bad_alloc)
            //if (begin_row == end_row)
            //    // workspace wasn't large enough, throw cusp::memory_allocation_failure?

            size_t workspace_size = output_ptr[end_segment] - output_ptr[begin_segment];

            total_work += workspace_size;

            // TODO remove these when an exception is in place
            assert(end_row > begin_row);
            assert(workspace_size <= workspace_capacity);

            coo_spmm_helper(exec,
                            workspace_size,
                            begin_row, end_row,
                            begin_segment, end_segment,
                            A, B, C_slice,
                            B_row_offsets,
                            segment_lengths, output_ptr,
                            A_gather_locations, B_gather_locations,
                            I, J, V,
                            combine, reduce);

            slices.push_back(Container());
            slices.back().swap(C_slice);

            begin_row = end_row;
        }

        // deallocate workspace
        // A_gather_locations.clear();
        // A_gather_locations.shrink_to_fit();
        // B_gather_locations.clear();
        // B_gather_locations.shrink_to_fit();
        // I.clear();
        // I.shrink_to_fit();
        // J.clear();
        // J.shrink_to_fit();
        // V.clear();
        // V.shrink_to_fit();

        // compute total output size
        size_t C_num_entries = 0;
        for(typename ContainerList::iterator iter = slices.begin(); iter != slices.end(); ++iter)
            C_num_entries += iter->num_entries;

        // resize output
        C.resize(A.num_rows, B.num_cols, C_num_entries);

        // copy slices into output
        size_t base = 0;
        for(typename ContainerList::iterator iter = slices.begin(); iter != slices.end(); ++iter)
        {
            thrust::copy(exec, iter->row_indices.begin(),    iter->row_indices.end(),    C.row_indices.begin()    + base);
            thrust::copy(exec, iter->column_indices.begin(), iter->column_indices.end(), C.column_indices.begin() + base);
            thrust::copy(exec, iter->values.begin(),         iter->values.end(),         C.values.begin()         + base);
            base += iter->num_entries;
        }
    }
}

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp


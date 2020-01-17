// -*-c++-*-

/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Bitonic sort implementation
// Author: Chuck Hastings charlesh@nvidia.com

// TODO:  Read a paper (Hagen Peters 2011) that suggests some
//        ways to optimize this.  Need to shift into a kernel
//        and then organize to support multiple passes in
//        a single kernel call.  This should reduce kernel
//        launch overhead and the number of memory references,
//        which should drive down the overall time.
//

#ifndef BITONIC_SORT_H
#define BITONIC_SORT_H

#include <thrust/for_each.h>
#include <thrust/scan.h>

#include "rmm_utils.h"


namespace cugraph { 
namespace sort {

  namespace bitonic {
    /*
     *  This implementation is based upon the bitonic sort technique.
     *  This should be pretty efficient in a SIMT environment.
     */
    namespace detail {
      /**
       * @brief Compare two items, if the compare functor returns true
       *        then swap them.
       *
       * @param a - reference to the first item
       * @param b - reference to the second item
       * @param compare - reference to a comparison functor
       */
      template <typename ValueT, typename CompareT>
      inline void __device__ compareAndSwap(ValueT &a, ValueT &b, CompareT &compare) {
        if (!compare(a,b)) {
          thrust::swap(a,b);
        }
      }

      /*
       * @brief perform repartitioning of two sorted partitions.  This
       *        is analagous to the bitonic merge step.  But it only
       *        performs the compare and swap portion of the bitonic
       *        merge.  The subsequent sorts are handled externally.
       *
       *        The repartition assumes that the data is segregated
       *        into partitions of binSize.  So if there are 8 elements
       *        and a bin size of 2 then the array will be partitioned
       *        into 4 bins of size 2.  Each bin is assumed to be
       *        sorted.  The repartition takes consecutive bins and
       *        repartitions them so that the first bin contains the
       *        low elements and the second bin contains the high elements.
       *
       * @param array - the array containing the data we need to repartition
       * @param count - the number of elements in the array
       * @param binSize - the size of the bin
       * @param compare - comparison functor
       */
      template <typename ValueT, typename CompareT>
      void repartition(ValueT *array, int count, int binSize, CompareT &compare) {

        thrust::for_each(thrust::make_counting_iterator<int>(0),
                         thrust::make_counting_iterator<int>(count / 2),

                         [array, count, binSize, compare]
                         __device__ (int idx) {
                           //
                           // Identify which elements in which partition
                           // we are responsible for comparing and swapping
                           //
                           // We're running count/2 iterations.  Each iteration
                           // needs to operate on a pair of elements.  Consider
                           // the pairs of partitions, this will let us determine
                           // which elements we compare.
                           //
                           int bi_partition = idx / binSize;

                           //
                           // bi_partition identifies which pair of partitions
                           // we're operating on.  Out of each bin we're only
                           // going to do binSize comparisons, so the first
                           // element in the comparison will be based on
                           // idx % binSize.
                           //
                           int offset = idx % binSize;

                           //
                           // First element is easy.
                           // Second element is "easy" but we'll fix
                           //   special cases below.
                           //
                           int i = bi_partition * (binSize * 2) + offset;
                           int j = (bi_partition + 1) * (binSize * 2) - 1 - offset;

                           //
                           // The last partition pair is the problem.
                           // There are several cases:
                           //    1) Both partitions are full.  This
                           //       is the easy case, we can just
                           //       compare and swap elements
                           //    2) First partition is full, the second
                           //       partition is not full (possibly
                           //       empty).  In this case, we only
                           //       compare some of the elements.
                           //    3) First partition is not full, there
                           //       is no second partition.  In this
                           //       case we actually don't have any
                           //       work to do.
                           //
                           // This should be a simple check.  If the
                           // second element is beyond the end of
                           // the array then there is nothing to compare
                           // and swap.  Note that if the first
                           // element is beyond the end of the array
                           // there is also nothing to compare and swap,
                           // but if the first element is beyond the
                           // end of the array then the second element
                           // will also be beyond the end of the array.
                           //
                           if (j < count)
                             compareAndSwap(array[i], array[j], compare);
                         });
  
      }

      /*
       * @brief perform shuffles.  After the repartition we need
       *        to perform shuffles of the halves to get things in
       *        order.
       *
       * @param array - the array containing the data we need to repartition
       * @param count - the number of elements in the array
       * @param binSize - the size of the bin
       * @param compare - comparison functor
       */
      template <typename ValueT, typename CompareT>
      void shuffles(ValueT *array, int count, int binSize, CompareT &compare) {

        thrust::for_each(thrust::make_counting_iterator<int>(0),
                         thrust::make_counting_iterator<int>((count + 1) / 2),
                         [array, count, binSize, compare]
                         __device__ (int idx) {
                           //
                           // Identify which elements in which partition
                           // we are responsible for comparing and swapping
                           //
                           // We're running count/2 iterations.  Each iteration
                           // needs to operate on a pair of elements.  Consider
                           // the pairs of partitions, this will let us determine
                           // which elements we compare.
                           //
                           int bi_partition = idx / binSize;

                           //
                           // bi_partition identifies which pair of partitions
                           // we're operating on.  Out of each bin we're only
                           // going to do binSize comparisons, so the first
                           // element in the comparison will be based on
                           // idx % binSize.
                           //
                           int offset = idx % binSize;

                           //
                           // First element is easy.
                           // Second element is "easy" i + binSize.
                           //
                           int i = bi_partition * (binSize * 2) + offset;
                           int j = i + binSize;

                           //
                           // If the second element is beyond the end of
                           // the array then there is nothing to compare
                           // and swap.
                           //
                           if (j < count)
                             compareAndSwap(array[i], array[j], compare);
                         });
  
      }

      /*
       * @brief perform repartitioning of two sorted partitions in the
       *        segmented sort case.
       *
       *        The repartition assumes that the data is segregated
       *        into partitions of binSize.  So if there are 8 elements
       *        and a bin size of 2 then the array will be partitioned
       *        into 4 bins of size 2.  Each bin is assumed to be
       *        sorted.  The repartition takes consecutive bins and
       *        repartitions them so that the first bin contains the
       *        low elements and the second bin contains the high elements.
       *
       * @param array - the array containing the data we need to repartition
       * @param count - the number of elements in the array
       * @param binSize - the size of the bin
       * @param compare - comparison functor
       */
      template <typename IndexT, typename ValueT, typename CompareT>
      void repartition_segmented(const IndexT *d_begin_offsets,
                                 const IndexT *d_end_offsets,
                                 ValueT *d_items,
                                 IndexT start,
                                 IndexT stop,
                                 IndexT *d_grouped_bins,
                                 int binSize,
                                 int max_count,
                                 int bin_pairs,
                                 CompareT &compare) {

        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator<int>(0),
                         thrust::make_counting_iterator<int>(max_count/2),
                         [d_begin_offsets, d_end_offsets, d_items, start,
                          stop, d_grouped_bins, bin_pairs, binSize, compare]
                         __device__ (int idx) {
                           //
                           //  idx needs to be mapped into the correct place
                           //
                           int entry = idx / bin_pairs;
                           int entry_idx = idx % bin_pairs;
                           int base = d_begin_offsets[d_grouped_bins[start + entry]];
                           int count = d_end_offsets[d_grouped_bins[start + entry]] - base;

                           //
                           // Identify which elements in which partition
                           // we are responsible for comparing and swapping
                           //
                           // We're running count/2 iterations.  Each iteration
                           // needs to operate on a pair of elements.  Consider
                           // the pairs of partitions, this will let us determine
                           // which elements we compare.
                           //
                           int bi_partition = entry_idx / binSize;

                           //
                           // bi_partition identifies which pair of partitions
                           // we're operating on.  Out of each bin we're only
                           // going to do binSize comparisons, so the first
                           // element in the comparison will be based on
                           // idx % binSize.
                           //
                           int offset = entry_idx % binSize;

                           //
                           // First element is easy.
                           // Second element is "easy" but we'll fix
                           //   special cases below.
                           //
                           int i = bi_partition * (binSize * 2) + offset;
                           int j = (bi_partition + 1) * (binSize * 2) - 1 - offset;

                           //
                           // The last partition pair is the problem.
                           // There are several cases:
                           //    1) Both partitions are full.  This
                           //       is the easy case, we can just
                           //       compare and swap elements
                           //    2) First partition is full, the second
                           //       partition is not full (possibly
                           //       empty).  In this case, we only
                           //       compare some of the elements.
                           //    3) First partition is not full, there
                           //       is no second partition.  In this
                           //       case we actually don't have any
                           //       work to do.
                           //
                           // This should be a simple check.  If the
                           // second element is beyond the end of
                           // the array then there is nothing to compare
                           // and swap.  Note that if the first
                           // element is beyond the end of the array
                           // there is also nothing to compare and swap,
                           // but if the first element is beyond the
                           // end of the array then the second element
                           // will also be beyond the end of the array.
                           //
                           if (j < count) {
                             compareAndSwap(d_items[base + i], d_items[base + j], compare);
                           }
                         });
      }

      /*
       * @brief perform shuffles.  After the repartition we need
       *        to perform shuffles of the halves to get things in
       *        order.
       *
       * @param rowOffsets - the row offsets identifying the segments
       * @param colIndices - the values to sort within the segments
       * @param start - position within the grouped bins where we
       *                start this pass
       * @param stop - position within the grouped bins where we stop
       *               this pass
       * @param d_grouped_bins - lrb grouped bins.  All bins between
       *                         start and stop are in the same lrb bin
       * @param binSize - the bitonic bin size for this pass of the shuffles
       * @param max_count - maximum number of elements possible for
       *                    this call
       * @param bin_pairs - the number of bin pairs
       * @param compare - the comparison functor
       */
      template <typename IndexT, typename ValueT, typename CompareT>
      void shuffles_segmented(const IndexT *d_begin_offsets,
                              const IndexT *d_end_offsets,
                              ValueT *d_items,
                              IndexT start,
                              IndexT stop,
                              IndexT *d_grouped_bins,
                              int binSize,
                              long max_count,
                              int bin_pairs,
                              CompareT &compare) {

        thrust::for_each(thrust::make_counting_iterator<int>(0),
                         thrust::make_counting_iterator<int>(max_count / 2),
                         [d_begin_offsets, d_end_offsets, d_items,
                          start, stop, d_grouped_bins,
                          compare, max_count, bin_pairs, binSize]
                         __device__ (int idx) {
                           //
                           //  idx needs to be mapped into the correct place
                           //
                           int entry = idx / bin_pairs;
                           int entry_idx = idx % bin_pairs;
                           int base = d_begin_offsets[d_grouped_bins[start + entry]];
                           int count = d_end_offsets[d_grouped_bins[start + entry]] - base;

                           //
                           // Identify which elements in which partition
                           // we are responsible for comparing and swapping
                           //
                           // We're running count/2 iterations.  Each iteration
                           // needs to operate on a pair of elements.  Consider
                           // the pairs of partitions, this will let us determine
                           // which elements we compare.
                           //
                           int bi_partition = entry_idx / binSize;

                           //
                           // bi_partition identifies which pair of partitions
                           // we're operating on.  Out of each bin we're only
                           // going to do binSize comparisons, so the first
                           // element in the comparison will be based on
                           // idx % binSize.
                           //
                           int offset = entry_idx % binSize;

                           //
                           // First element is easy.
                           // Second element is "easy" i + binSize.
                           //
                           int i = bi_partition * (binSize * 2) + offset;
                           int j = i + binSize;

                           //
                           // If the second element is beyond the end of
                           // the array then there is nothing to compare
                           // and swap.
                           //
                           if (j < count)
                             compareAndSwap(d_items[base + i], d_items[base + j], compare);
                         });
      }
    }

    template <typename ValueT, typename CompareT>
    void sort(ValueT *array, int count, CompareT &compare) {
      for (int i = 1 ; i < count ; i *= 2) {
        detail::repartition(array, count, i, compare);

        for (int j = i / 2 ; j > 0 ; j /= 2) {
          detail::shuffles(array, count, j, compare);
        }
      }
    }

    /**
     * @brief Perform a segmented sort.  This function performs a sort
     *        on each segment of the specified input.  This sort is done
     *        in place, so the d_items array is modified during this call.
     *        Sort is done according to the (optionally) specified
     *        comparison function.
     *
     *        Note that this function uses O(num_segments) temporary
     *        memory during execution.
     *
     * @param [in] num_segments - the number of segments that the items array is divided into
     * @param [in] num_items - the number of items in the array
     * @param [in] d_begin_offsets - device array containing the offset denoting the start
     *                               of each segment
     * @param [in] d_end_offsets - device array containing the offset denoting the end
     *                               of each segment.
     * @param [in/out] d_items - device array containing the items to sort
     * @param [in] compare - [optional] comparison function.  Default is thrust::less<ValueT>.
     * @param [in] stream - [optional] CUDA stream to launch kernels with.  Default is stream 0.
     *
     * @return error code
     */
    template <typename IndexT, typename ValueT, typename CompareT>
    void segmented_sort(IndexT num_segments, IndexT num_items,
                             const IndexT *d_begin_offsets,
                             const IndexT *d_end_offsets,
                             ValueT *d_items,
                             CompareT compare = thrust::less<ValueT>(),
                             cudaStream_t stream = nullptr) {
      
      //
      //  NOTE: This should probably be computed somehow.  At the moment
      //        we are limited to 32 bits because of memory sizes.
      //
      int     lrb_size = 32;
      IndexT  lrb[lrb_size + 1];
      IndexT *d_lrb;
      IndexT *d_grouped_bins;

      ALLOC_TRY(&d_lrb, (lrb_size + 1) * sizeof(IndexT), stream);
      ALLOC_TRY(&d_grouped_bins, (num_segments + 1) * sizeof(IndexT), stream);

      CUDA_TRY(cudaMemset(d_lrb, 0, (lrb_size + 1) * sizeof(IndexT)));

      //
      //  First we'll count how many entries go in each bin
      //
      thrust::for_each(thrust::make_counting_iterator<int>(0),
                       thrust::make_counting_iterator<int>(num_segments),
                       [d_begin_offsets, d_end_offsets, d_lrb]
                       __device__ (int idx) {
                         int size = d_end_offsets[idx] - d_begin_offsets[idx];
                         //
                         // NOTE: If size is 0 or 1 then no
                         //       sorting is required, so we'll
                         //       eliminate those bins here
                         //
                         if (size > 1)
                           atomicAdd(d_lrb + __clz(size), 1);
                       });
      
      //
      //  Exclusive sum will identify where each bin begins
      //
      thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                             d_lrb, d_lrb + (lrb_size + 1), d_lrb);

      //
      //  Copy the start of each bin to local memory
      //
      CUDA_TRY(cudaMemcpy(lrb, d_lrb, (lrb_size + 1) * sizeof(IndexT), cudaMemcpyDeviceToHost));

      //
      //  Now we'll populate grouped_bins.  This will corrupt
      //  d_lrb, but we've already copied it locally.
      //
      thrust::for_each(thrust::make_counting_iterator<int>(0),
                       thrust::make_counting_iterator<int>(num_segments),
                       [d_begin_offsets, d_end_offsets, d_lrb, d_grouped_bins]
                       __device__ (int idx) {
                         int size = d_end_offsets[idx] - d_begin_offsets[idx];
                         if (size > 1) {
                           int pos = atomicAdd(d_lrb + __clz(size), 1);
                           d_grouped_bins[pos] = idx;
                         }
                       });

      //
      //  At this point, d_grouped_bins contains the index of the
      //  different segments, ordered into log2 bins.
      //

      //
      //  Now we're ready to go.
      //
      //  For simplicity (at least for now), let's just
      //  iterate over each lrb bin.  Note that the larger
      //  the index i, the smaller the size of each bin... but
      //  there will likely be many more inhabitants of that bin.
      //
      for (int i = 0 ; i < lrb_size ; ++i) {
        int size = lrb[i+1] - lrb[i];
        if (size > 0) {
          //
          //  There are inhabitants of this lrb range
          //
          //  max_count will be used to drive the bitonic
          //  passes (1, 2, 4, 8, ... up to max_count)
          //
          int max_count = 1 << (lrb_size - i);

          for (int j = 1 ; j < max_count ; j *= 2) {
            detail::repartition_segmented(d_begin_offsets,
                                          d_end_offsets,
                                          d_items,
                                          lrb[i],
                                          lrb[i+1],
                                          d_grouped_bins,
                                          j,
                                          size * max_count,
                                          max_count / 2,
                                          compare);

            for (int k = j / 2 ; k > 0 ; k /= 2) {
              detail::shuffles_segmented(d_begin_offsets,
                                         d_end_offsets,
                                         d_items,
                                         lrb[i],
                                         lrb[i+1],
                                         d_grouped_bins,
                                         k,
                                         size * max_count,
                                         max_count / 2,
                                         compare);
            }
          }
        }
      }

      ALLOC_FREE_TRY(d_grouped_bins, stream);
      ALLOC_FREE_TRY(d_lrb, stream);
      
    }

} } } //namespace

#endif

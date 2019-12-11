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

// Utilities to treat array as a heap
// Author: Chuck Hastings charlesh@nvidia.com

#ifndef HEAP_H
#define HEAP_H

namespace cugraph { 
namespace detail {

  namespace heap {
    /*
     *  Our goal here is to treat a C-style array indexed
     *  from 0 to n-1 as a heap.  The heap is a binary tress
     *  structure where the root of each tree is the smallest
     *  (or largest) value in that subtree.
     *
     *  This is a completely serial implementation.  The intention
     *  from a parallelism perspective would be to use this on
     *  a block of data assigned to a particular GPU (or CPU) thread.
     *
     *  These functions will allow you to use an existing
     *  c-style array (host or device side) and manipulate
     *  it as a heap.
     *
     *  Note, the heap will be represented like this - the
     *  shape indicates the binary tree structure, the element
     *  indicates the index of the array that is associated
     *  with the element.  This diagram will help understand
     *  the parent/child calculations defined below.
     *
     *                    0
     *              1           2
     *           3     4     5     6
     *          7  8  9 10 11 12 13 14
     *
     *   So element 0 is the root of the tree, element 1 is the
     *   left child of 0, element 2 is the right child of 0, etc.
     */

    namespace detail {
      /**
       * @brief Identify the parent index of the specified index.
       *        NOTE: This function does no bounds checking, so
       *        the parent of 0 is 0.
       *
       *   See the above documentation for a picture to describe
       *   the tree.
       *
       *   IndexT is a templated integer type of the index
       *
       * @param[in]  index - the current array index
       * @return     the index of the parent of the current index
       */
      template <typename IndexT>
      inline IndexT __host__ __device__ parent(IndexT index) {
	static_assert(std::is_integral<IndexT>::value, "Index must be of an integral type");
      
	return ((index + 1) / 2) - 1;
      }

      /**
       * @brief Identify the left child index of the specified index.
       *        NOTE: This function does no bounds checking, so
       *        the left child computed might be out of bounds.
       *
       *   See the above documentation for a picture to describe
       *   the tree.
       *
       *   IndexT is a templated integer type of the index
       *
       * @param[in]  index - the current array index
       * @return     the index of the left child of the current index
       */
      template <typename IndexT>
      inline IndexT __host__ __device__ left_child(IndexT index) {
	static_assert(std::is_integral<IndexT>::value, "Index must be of an integral type");
      
	return ((index + 1) * 2 - 1);
      }

      /**
       * @brief Identify the right child index of the specified index.
       *        NOTE: This function does no bounds checking, so
       *        the right child computed might be out of bounds.
       *
       *   See the above documentation for a picture to describe
       *   the tree.
       *
       *   IndexT is a templated integer type of the index
       *
       * @param[in]  index - the current array index
       * @return     the index of the right child of the current index
       */
      template <typename IndexT>
      inline IndexT __host__ __device__ right_child(IndexT index) {
	static_assert(std::is_integral<IndexT>::value, "Index must be of an integral type");
      
	return (index + 1) * 2;
      }
    }
      
    /**
     * @brief Reorder an existing array of elements into a heap
     *
     *   ArrayT is a templated type of the array elements
     *   IndexT is a templated integer type of the index
     *   CompareT is a templated compare function
     *
     * @param[in, out]   array   - the existing array
     * @param[in]        size    - the number of elements in the existing array
     * @param[in]        compare - the comparison function to use
     *
     */
    template <typename ArrayT, typename IndexT, typename CompareT>
    inline void __host__ __device__ heapify(ArrayT *array, IndexT size, CompareT compare) {
      static_assert(std::is_integral<IndexT>::value, "Index must be of an integral type");

      //
      // We want to order ourselves as a heap.  This is accomplished by starting
      // at the end and for each element, compare with its parent and
      // swap if necessary.  We repeat this until there are no more swaps
      // (should take no more than log2(size) iterations).
      //
      IndexT count_swaps = 1;
      while (count_swaps > 0) {
	count_swaps = 0;
	for (IndexT i = size - 1 ; i > 0 ; --i) {
	  IndexT p = detail::parent(i);

	  if (compare(array[i], array[p])) {
	    thrust::swap(array[i], array[p]);
	    ++count_swaps;
	  }
	}
      }
    }

    /**
     * @brief Pop the top element off of the heap.  Note that the caller
     *        should decrement the size - the last element in the
     *        array is no longer used.
     *
     *   ArrayT is a templated type of the array elements
     *   IndexT is a templated integer type of the index
     *   CompareT is a templated compare function
     *
     * @return - the top of the heap.
     */
    template <typename ArrayT, typename IndexT, typename CompareT>
    inline ArrayT __host__ __device__ heap_pop(ArrayT *array, IndexT size, CompareT compare) {
      static_assert(std::is_integral<IndexT>::value, "Index must be of an integral type");

      //
      //  Swap the top of the array with the last element
      //
      --size;
      thrust::swap(array[0], array[size]);

      //
      //  Now top element is no longer the smallest (largest), so we need
      //  to sift it down to the proper location.
      //
      for (IndexT i = 0 ; i < size ; ) {
	IndexT lc = detail::left_child(i);
	IndexT rc = detail::right_child(i);
	IndexT smaller = i;

	//
	//  We can go out of bounds, let's check the simple cases
	//
	if (rc < size) {
	  //
	  //  Both children exist in tree, pick the smaller (lerger)
	  //  one.
	  //
	  smaller = (compare(array[lc], array[rc])) ? lc : rc;
	} else if (lc < size) {
	  smaller = lc;
	}

	if ((smaller != i) && (compare(array[smaller], array[i]))) {
	  thrust::swap(array[i], array[smaller]);
	  i = smaller;
	} else {
	  //
	  //  If we don't swap then we can stop checking, break out of the loop
	  //
	  i = size;
	}
      }
      
      return array[size];
    }
  }
  
} } //namespace

#endif

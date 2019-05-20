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

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <iostream>

#ifndef RANGE_VIEW_HXX
#define RANGE_VIEW_HXX

// This example demonstrates the use of a view: a non-owning wrapper for an
// iterator range which presents a container-like interface to the user.
//
// For example, a view of a device_vector's data can be helpful when we wish to
// access that data from a device function. Even though device_vectors are not
// accessible from device functions, the range_view class allows us to access
// and manipulate its data as if we were manipulating a real container.
//

// This example demonstrate use of range_view with for_each algorithm which is
// dispatch from GPU
//

template<class Iterator>
class range_view
{
public:
  typedef Iterator iterator;
  typedef typename thrust::iterator_traits<iterator>::value_type value_type;
  typedef typename thrust::iterator_traits<iterator>::pointer pointer;
  typedef typename thrust::iterator_traits<iterator>::difference_type difference_type;
  typedef typename thrust::iterator_traits<iterator>::reference reference;

private:
  const iterator first;
  const iterator last;


public:
  __host__ __device__
  range_view(Iterator first, Iterator last)
      : first(first), last(last) {}
  __host__ __device__
  ~range_view() {}

  __host__ __device__
  difference_type size() const { return thrust::distance(first, last); }


  __host__ __device__
  reference operator[](difference_type n)
  {
    return *(first + n);
  }
  __host__ __device__
  const reference operator[](difference_type n) const
  {
    return *(first + n);
  }

  __host__ __device__
  iterator begin() 
  {
    return first;
  }
  __host__ __device__
  const iterator cbegin() const
  {
    return first;
  }
  __host__ __device__
  iterator end() 
  {
    return last;
  }
  __host__ __device__
  const iterator cend() const
  {
    return last;
  }


  __host__ __device__
  thrust::reverse_iterator<iterator> rbegin()
  {
    return thrust::reverse_iterator<iterator>(end());
  }
  __host__ __device__
  const thrust::reverse_iterator<const iterator> crbegin() const 
  {
    return thrust::reverse_iterator<const iterator>(cend());
  }
  __host__ __device__
  thrust::reverse_iterator<iterator> rend()
  {
    return thrust::reverse_iterator<iterator>(begin());
  }
  __host__ __device__
  const thrust::reverse_iterator<const iterator> crend() const 
  {
    return thrust::reverse_iterator<const iterator>(cbegin());
  }
  __host__ __device__
  reference front() 
  {
    return *begin();
  }
  __host__ __device__
  const reference front()  const
  {
    return *cbegin();
  }

  __host__ __device__
  reference back() 
  {
    return *end();
  }
  __host__ __device__
  const reference back()  const
  {
    return *cend();
  }

  __host__ __device__
  bool empty() const 
  {
    return size() == 0;
  }

};

// This helper function creates a range_view from iterator and the number of
// elements
template <class Iterator, class Size>
range_view<Iterator>
__host__ __device__
make_range_view(Iterator first, Size n)
{
  return range_view<Iterator>(first, first+n);
}

// This helper function creates a range_view from a pair of iterators
template <class Iterator>
range_view<Iterator>
__host__ __device__
make_range_view(Iterator first, Iterator last)
{
  return range_view<Iterator>(first, last);
}

// This helper function creates a range_view from a Vector
template <class Vector>
range_view<typename Vector::iterator>
__host__
make_range_view(Vector& v)
{
  return range_view<typename Vector::iterator>(v.begin(), v.end());
}

#endif

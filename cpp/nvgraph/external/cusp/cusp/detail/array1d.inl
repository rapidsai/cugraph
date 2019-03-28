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

#include <thrust/swap.h>

namespace cusp
{
namespace detail
{

template <typename Array1, typename Array2>
bool array1d_equal(const Array1& lhs, const Array2& rhs)
{
    return lhs.size() == rhs.size() && thrust::detail::vector_equal(lhs.begin(), lhs.end(), rhs.begin());
}

} // end namespace detail

template<typename T, typename MemorySpace>
typename array1d<T,MemorySpace>::view
array1d<T,MemorySpace>
::subarray(size_type start_index, size_type num_entries)
{
    return view(Parent::begin() + start_index, Parent::begin() + start_index + num_entries);
} // end array1d::subarray

template<typename T, typename MemorySpace>
typename array1d<T,MemorySpace>::const_view
array1d<T,MemorySpace>
::subarray(size_type start_index, size_type num_entries) const
{
    return const_view(Parent::begin() + start_index, Parent::begin() + start_index + num_entries);
} // end array1d::subarray

template<typename RandomAccessIterator>
array1d_view<RandomAccessIterator>&
array1d_view<RandomAccessIterator>
::operator=(const array1d_view& v)
{
    this->base_reference()  = v.begin();
    m_size                  = v.size();
    m_capacity              = v.capacity();
    return *this;
} // end array1d_view::operator=

template<typename RandomAccessIterator>
typename array1d_view<RandomAccessIterator>::reference
array1d_view<RandomAccessIterator>
::front(void) const
{
    return *begin();
} // end array1d_view::front

template<typename RandomAccessIterator>
typename array1d_view<RandomAccessIterator>::reference
array1d_view<RandomAccessIterator>
::back(void) const
{
    return *(begin() + (size() - 1));
} // end array1d_view::back

template<typename RandomAccessIterator>
typename array1d_view<RandomAccessIterator>::reference
array1d_view<RandomAccessIterator>
::operator[](size_type n) const
{
    return *(begin() + n);
} // end array1d_view::operator[]

template<typename RandomAccessIterator>
typename array1d_view<RandomAccessIterator>::iterator
array1d_view<RandomAccessIterator>
::begin(void) const
{
    return this->base();
} // end array1d_view::begin

template<typename RandomAccessIterator>
typename array1d_view<RandomAccessIterator>::iterator
array1d_view<RandomAccessIterator>
::end(void) const
{
    return begin() + m_size;
} // end array1d_view::end

template<typename RandomAccessIterator>
typename array1d_view<RandomAccessIterator>::size_type
array1d_view<RandomAccessIterator>
::size(void) const
{
    return m_size;
} // end array1d_view::size

template<typename RandomAccessIterator>
typename array1d_view<RandomAccessIterator>::size_type
array1d_view<RandomAccessIterator>
::capacity(void) const
{
    return m_capacity;
} // end array1d_view::capacity

template<typename RandomAccessIterator>
void
array1d_view<RandomAccessIterator>
::resize(size_type new_size)
{
    if (new_size <= m_capacity)
        m_size = new_size;
    else
    {
        throw cusp::not_implemented_exception("array1d_view cannot resize() larger than capacity()");
    }
} // end array1d_view::resize

template<typename RandomAccessIterator>
void
array1d_view<RandomAccessIterator>
::swap(array1d_view &v)
{
    // TODO: cost of swap_ranges vs swap?
    // thrust::swap(this->base_reference(), v.base_reference());
    thrust::swap_ranges(this->begin(), this->end(), v.begin());

    thrust::swap(m_size,     v.m_size);
    thrust::swap(m_capacity, v.m_capacity);
} // end array1d_view::swap

template<typename RandomAccessIterator>
typename array1d_view<RandomAccessIterator>::view
array1d_view<RandomAccessIterator>
::subarray(size_type start_index, size_type num_entries)
{
    return view(begin() + start_index, begin() + start_index + num_entries);
} // end array1d_view::subarray

////////////////////////
// Equality Operators //
////////////////////////

// containers
template<typename T, typename Alloc,
         typename Array>
bool operator==(const array1d<T,Alloc>& lhs,
                const Array&            rhs)
{
    return cusp::detail::array1d_equal(lhs, rhs);
}

template<typename T, typename Alloc,
         typename Array>
bool operator==(const Array&            lhs,
                const array1d<T,Alloc>& rhs)
{
    return cusp::detail::array1d_equal(lhs, rhs);
}

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator==(const array1d<T1,Alloc1>& lhs,
                const array1d<T2,Alloc2>& rhs)
{
    return cusp::detail::array1d_equal(lhs, rhs);
}

template<typename T, typename Alloc,
         typename Array>
bool operator!=(const array1d<T,Alloc>& lhs,
                const Array&            rhs)
{
    return !(lhs == rhs);
}

template<typename T, typename Alloc,
         typename Array>
bool operator!=(const Array&            lhs,
                const array1d<T,Alloc>& rhs)
{
    return !(lhs == rhs);
}

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator!=(const array1d<T1,Alloc1>& lhs,
                const array1d<T2,Alloc2>& rhs)
{
    return !(lhs == rhs);
}

// views
template<typename I,
         typename Array>
bool operator==(const array1d_view<I>& lhs,
                const Array&           rhs)
{
    return cusp::detail::array1d_equal(lhs, rhs);
}

template<typename I,
         typename Array>
bool operator==(const Array&           lhs,
                const array1d_view<I>& rhs)
{
    return cusp::detail::array1d_equal(lhs, rhs);
}

template<typename I1,
         typename I2>
bool operator==(const array1d_view<I1>& lhs,
                const array1d_view<I2>& rhs)
{
    return cusp::detail::array1d_equal(lhs, rhs);
}

template<typename I,
         typename Array>
bool operator!=(const array1d_view<I>& lhs,
                const Array&           rhs)
{
    return !(lhs == rhs);
}

template<typename I,
         typename Array>
bool operator!=(const Array&           lhs,
                const array1d_view<I>& rhs)
{
    return !(lhs == rhs);
}

template<typename I1,
         typename I2>
bool operator!=(const array1d_view<I1>& lhs,
                const array1d_view<I2>& rhs)
{
    return !(lhs == rhs);
}


// mixed containers and views (to resolve ambiguity)
template<typename I,
         typename T, typename Alloc>
bool operator==(const array1d_view<I>&  lhs,
                const array1d<T,Alloc>& rhs)
{
    return cusp::detail::array1d_equal(lhs, rhs);
}

template<typename I,
         typename T, typename Alloc>
bool operator==(const array1d<T,Alloc>& lhs,
                const array1d_view<I>&  rhs)
{
    return cusp::detail::array1d_equal(lhs, rhs);
}

template<typename I,
         typename T, typename Alloc>
bool operator!=(const array1d_view<I>&  lhs,
                const array1d<T,Alloc>& rhs)
{
    return !(lhs == rhs);
}

template<typename I,
         typename T, typename Alloc>
bool operator!=(const array1d<T,Alloc>& lhs,
                const array1d_view<I>&  rhs)
{
    return !(lhs == rhs);
}

} // end namespace cusp



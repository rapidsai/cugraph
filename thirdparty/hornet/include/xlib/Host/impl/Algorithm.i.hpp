/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date June, 2017
 * @version v1.3
 *
 * @copyright Copyright © 2017 XLib. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 */
#include "Host/Basic.hpp"           //ERROR
#include "Device/Util/CudaUtil.cuh" //xlib::numeric_limits
#include <algorithm>                //std::transform, std::sort, std::equal
#include <cassert>                  //assert
#include <thread>                   //std::thread
#include <cmath>                   //std::thread

namespace xlib {

//template<typename T>
//struct numeric_limits;


//------------------------------------------------------------------------------

template<typename T, typename R>
R UniqueMap<T, R>::insert(const T& key) noexcept {
    const auto& it = this->find(key);
    if (it == this->end()) {
        auto id = static_cast<R>(this->size());
        this->std::unordered_map<T, R>::insert(std::pair<T, R>(key, id));
        return id;
    }
    return it->second;
}

template<class Iterator1, class Iterator2>
bool equal_sorted(Iterator1 start1, Iterator1 end1, Iterator2 start2) noexcept {
    return xlib::equal_sorted(start1, end1, start2,
                              start2 + std::distance(start1, end1));
}

template<class Iterator1, class Iterator2>
bool equal_sorted(Iterator1 start1, Iterator1 end1,
                  Iterator2 start2, Iterator2 end2) noexcept {
    using T = typename std::iterator_traits<Iterator1>::value_type;
    using R = typename std::iterator_traits<Iterator2>::value_type;
    auto size1 = std::distance(start1, end1);
    auto size2 = std::distance(start1, end1);
    //if (size1 != size2)
    //    ERROR("Objects must constains the same numbers of parameters")
    auto tmp_array1 = new T[size1];
    auto tmp_array2 = new R[size1];

    std::copy(start1, end1, tmp_array1);
    std::copy(start2, end2, tmp_array2);
    std::sort(tmp_array1, tmp_array1 + size1);
    std::sort(tmp_array2, tmp_array2 + size1);

    bool flag = std::equal(tmp_array1, tmp_array1 + size1, tmp_array2);

    delete[] tmp_array1;
    delete[] tmp_array2;
    return flag;
}

/*
template<class FUN_T, typename... T>
inline void Funtion_TO_multiThreads(bool MultiCore, FUN_T FUN, T... Args) {
    if (MultiCore) {
        auto concurrency =static_cast<int>(std::thread::hardware_concurrency());
        std::thread threadArray[32];

        for (int i = 0; i < concurrency; i++)
            threadArray[i] = std::thread(FUN, Args..., i, concurrency);
        for (int i = 0; i < concurrency; i++)
            threadArray[i].join();
    } else
        FUN(Args..., 0, 1);
}*/

namespace detail {

template<typename S, typename R>
void sort_by_key_aux3(const S* indexes, size_t size, R* data) {
    auto tmp = new R[size];
    std::copy(data, data + size, tmp);
    std::transform(indexes, indexes + size, data,
                    [&](S index) { return tmp[index]; });
    delete[] tmp;
}

template<typename S>
void sort_by_key_aux2(const S*, size_t) {};

template<typename S, typename R, typename... RArgs>
void sort_by_key_aux2(const S* indexes, size_t size, R* data,
                      RArgs... data_packed) {
    sort_by_key_aux3(indexes, size, data);
    sort_by_key_aux2(indexes, size, data_packed...);
}

template<typename S, typename T, typename... RArgs>
void sort_by_key_aux1(T* start, T* end, RArgs... data_packed) {
    auto    size = static_cast<size_t>(std::distance(start, end));
    auto indexes = new S[size];
    std::iota(indexes, indexes + size, 0);

    auto lambda = [&](S i, S j) { return start[i] < start[j]; };
    std::sort(indexes, indexes + size, lambda);

    sort_by_key_aux3(indexes, size, start);
    sort_by_key_aux2(indexes, size, data_packed...);
    delete[] indexes;
}

} // namespace detail

/**
 * required auxilary space: O(|end -start| * 2)
 */
template<typename T, typename... RArgs>
void sort_by_key(T* start, T* end, RArgs... data_packed) {
    if (std::distance(start, end) < std::numeric_limits<int>::max())
        detail::sort_by_key_aux1<int>(start, end, data_packed...);
    else
        detail::sort_by_key_aux1<int64_t>(start, end, data_packed...);
}


/**
 * required auxilary space: O(|end -start| * 2)
 */
template<typename T, typename R>
void sort_by_key(T* start, T* end, R* data) {
    auto   size = static_cast<size_t>(std::distance(start, end));
    auto  pairs = new std::pair<T, R>[size];
    for (size_t i = 0; i < size; i++)
        pairs[i] = std::make_pair(start[i], data[i]);

    const auto& lambda = [](const auto& a, const auto& b) {
                                return a.first < b.first;
                            };
    std::sort(pairs, pairs + size, lambda);
    for (size_t i = 0; i < size; i++) {
        start[i] = pairs[i].first;
        data[i]  = pairs[i].second;
    }
    delete[] pairs;
}

template<typename T, typename S>
HOST_DEVICE
void merge(const T* left, S size_left, const T* right, S size_right, T* merge) {
    S i = 0, j = 0, k = 0;
    while (i < size_left && j < size_right)
        merge[k++] = left[i] <= right[j] ? left[i++] : right[j++];

    if (i < size_left) {
        for (S p = i; p < size_left; p++)
            merge[k++] = left[p];
    }
    else {
        for (S p = j; p < size_right; p++)
            merge[k++] = right[p];
    }
}

template<typename T, typename S>
HOST_DEVICE
void inplace_merge(T* left, S size_left, const T* right, S size_right) {
    S i = size_left - 1, j = size_right - 1, k = size_left + size_right - 1;
    while (i >= 0 && j >= 0)
        left[k--] = (left[i] <= right[j]) ? right[j--] : left[i--];
    while (j >= 0)
        left[k--] = right[j--];
}

//==============================================================================


/**
 * The following algorithms return the leftmost place where the given element
 * can be correctly inserted (and still maintain the sorted order).
 * the lowest index where the element is equal to the given value or
 * the highest index where the element is less than the given value
 *
 * mem: {0, 3, 5, 5, 8, 8, 8, 18, 36}
 *
 * RIGHT: searched: 5    return 2
 *  LEFT: searched: 5    return 2
 *
 * RIGHT: searched: 2    return 1
 *  LEFT: searched: 2    return 0
 *
 * RIGHT: searched: -1    return 0
 *  LEFT: searched: -1    return -1
 *
 * RIGHT: searched: 40    return 9
 *  LEFT: searched: 40    return 8
 * $\Theta(log(n))$
 */
template<bool RIGHT, typename T, typename R>
HOST_DEVICE
R lower_bound(const T* mem, R size, T searched) {
    R start = 0, end = size;
    bool flag = false;
    assert(size < xlib::numeric_limits<R>::max / 2 && "May overflow");

    while (start < end) {        // mid = low + (high - low) / 2u avoid overflow
        R mid = (start + end) / 2u;
        T tmp = mem[mid];
        if (searched <= tmp) {
            end  = mid;
            flag = searched == tmp;
        }
        else
            start = mid + 1;
    }
    return (RIGHT || flag) ? start : start - 1;
}

template<typename T, typename R>
HOST_DEVICE
R lower_bound_left(const T* mem, R size, T searched) {
    return lower_bound<false>(mem, size, searched);
}
template<typename T, typename R>
HOST_DEVICE
R lower_bound_right(const T* mem, R size, T searched) {
    return lower_bound<true>(mem, size, searched);
}

/**
 * The following algorithms return the rightmost place where the given element
 * can be correctly inserted (and still maintain the sorted order).
 *
 * mem: {0, 3, 5, 5, 8, 8, 8, 18, 36}
 *
 * RIGHT : searched: 5    return 4
 *  LEFT : searched: 5    return 3
 *
 * RIGHT : searched: 2    return 1
 *  LEFT : searched: 2    return 0
 *
 * RIGHT : searched: -1    return 0
 *  LEFT : searched: -1    return -1
 *
 * RIGHT : searched: 40    return 9
 *  LEFT : searched: 40    return 8
 */
//first greater
template<typename T, typename R>
HOST_DEVICE
R upper_bound_left(const T* mem, R size, T searched) {
    return upper_bound_right(mem, size, searched) - 1;
}

template<typename T, typename R>
HOST_DEVICE
R upper_bound_right(const T* mem, R size, T searched) {
    R start = 0, end = size, mid;
    assert(size < xlib::numeric_limits<R>::max / 2 && "May overflow");

    while (start < end) {
        mid = (start + end) / 2u; // mid = low + (high - low) / 2 avoid overflow
        if (searched >= mem[mid])
            start = mid + 1;
        else
            end = mid;
    }
    return start;
}

template<typename T, typename R>
HOST_DEVICE
R binary_search(const T* mem, R size, T searched) {
    assert(size != 0 || std::is_signed<R>::value);
    R start = 0, end = size - 1;
    while (start <= end) {
        R mid = (start + end) / 2u;
        if (mem[mid] > searched)
            end = mid - 1;
        else if (mem[mid] < searched)
            start = mid + 1;
        else
            return mid;
    }
    return size; // indicate not found
}

//==============================================================================
//==============================================================================
//==============================================================================

#if defined(__NVCC__)
    #define RET_TYPE int2
#else
    #define RET_TYPE std::pair<int,int>
#endif

template<typename itA_t, typename itB_t>
HOST_DEVICE
RET_TYPE merge_path_search(const itA_t& A, int A_size,
                           const itB_t& B, int B_size,
                           int diagonal) {
#if defined(__CUDA_ARCH__)
    int x_min = ::max(diagonal - B_size, 0);
    int x_max = ::min(diagonal, A_size);
#else
    int x_min = std::max(diagonal - B_size, 0);
    int x_max = std::min(diagonal, A_size);
#endif

    while (x_min < x_max) {
        int pivot = (x_max + x_min) / 2u;
        if (A[pivot] <= B[diagonal - pivot - 1])
            x_min = pivot + 1;
        else
            x_max = pivot;
    }
#if defined(__CUDA_ARCH__)
    return { ::min(x_min, A_size), diagonal - x_min };
#else
    return { std::min(x_min, A_size), diagonal - x_min };
#endif
}

#undef RET_TYPE

//------------------------------------------------------------------------------

HOST_DEVICE
NaturalIterator::NaturalIterator() : _start(0) {}

HOST_DEVICE
NaturalIterator::NaturalIterator(int start) : _start(start) {}

HOST_DEVICE
int NaturalIterator::operator[](int index) const {
    return _start + index;
}

} // namespace xlib

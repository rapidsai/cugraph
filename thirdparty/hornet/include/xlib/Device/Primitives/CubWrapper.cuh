/**
 * @internal
 * @brief Vec-Tree interface
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
 * @version v2
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
 *
 * @file
 */
#pragma once

#include <limits>
#include <utility>

#if defined(CUB_WRAPPER)

#include "Host/Basic.hpp"//xlib::byte_t

namespace xlib {

class CubWrapper {
protected:
    explicit CubWrapper() = default;
    explicit CubWrapper(const int num_items) noexcept;
    ~CubWrapper() noexcept;

    void initialize(const int num_items) noexcept;

    void release(void) noexcept;

    byte_t*  _d_temp_storage     { nullptr };
    size_t _temp_storage_bytes { 0 };
    int    _num_items          { 0 };
};

//==============================================================================

template<typename T>
class CubUnique : public CubWrapper {
public:
    CubUnique(const T* d_in, size_t num_items, T*& d_unique_batch);
    ~CubUnique() noexcept;
    int run() noexcept;
private:
    const T* _d_in;
    T*&      _d_unique_batch;
    int*     _d_unique_egdes;
};

template<typename T>
class CubReduce : public CubWrapper {
public:
    explicit CubReduce(const T* d_in, size_t num_items) noexcept;
    ~CubReduce() noexcept;
    T run() noexcept;
private:
    const T* _d_in  { nullptr };
    T*       _d_out { nullptr };
};

template<typename T>
class CubPartitionFlagged : public CubWrapper {
public:
    CubPartitionFlagged(const T* d_in, const bool* d_flag, size_t num_items,
                        T*& d_out) noexcept;
    ~CubPartitionFlagged() noexcept;
    int run() noexcept;
    void run_no_copy() noexcept;
private:
    const T*    _d_in;
    T*&         _d_out;
    const bool* _d_flag;
    int*        _d_num_selected_out;
};

template<typename T>
class CubSegmentedReduce : public CubWrapper {
public:
    CubSegmentedReduce(int* _d_offsets, const T* d_in, int _num_segments,
                       T*& d_out);
    ~CubSegmentedReduce() noexcept;
    void run() noexcept;
private:
    int*  _d_offsets;
    const T*    _d_in;
    T*&         _d_out;
};

template<typename T>
class CubSpMV : public CubWrapper {
public:
    CubSpMV(T* d_value, int* d_row_offsets, int* d_column_indices,
            T* d_vector_x, T* d_vector_y,
            int num_rows, int num_cols, int num_nonzeros);
    //~CubSpMV() noexcept;
    void run() noexcept;
private:
    int*  _d_row_offsets;
    int*  _d_column_indices;
    T*    _d_vector_x;
    T*    _d_vector_y;
    T*    _d_values;
    int   _num_rows, _num_cols, _num_nonzeros;
};

template<typename T>
class CubArgMax : public CubWrapper {
public:
    explicit CubArgMax(const T* d_in, size_t num_items) noexcept;
    typename std::pair<int, T> run() noexcept;
private:
    const T* _d_in;
    void*    _d_out;
};

//==============================================================================

template<typename T>
class CubSortByValue : public CubWrapper {
public:
    explicit CubSortByValue() = default;

    explicit CubSortByValue(const int max_items) noexcept;

    void initialize(const int max_items) noexcept;

    void resize(const int num_items) noexcept;

    void run(const T* d_in, int num_items, T* d_sorted,
             T d_in_max = std::numeric_limits<T>::max()) noexcept;

    static void srun(const T* d_in, int num_items, T* d_sorted,
                     T d_in_max = std::numeric_limits<T>::max()) noexcept;
private:
    byte_t* _d_temp_storage { nullptr };
};

//==============================================================================

namespace cub_sort_by_key {

template<typename T, typename R>
extern void run(const T* d_key,
                const R* d_data_in,
                int      num_items,
                T*       d_key_sorted,
                R*       d_data_out,
                T        d_key_max = std::numeric_limits<T>::max());

/*
template<typename T, typename R>
void run(const T* d_key,
         const R* d_data_in,
         int      num_items,
         T*       d_key_sorted,
         R*       d_data_out,
         T*       d_key_tmp,
         R*       d_data_tmp,
         T        d_key_max = std::numeric_limits<T>::max()) noexcept;*/

} // namespace cub_sort_by_key

//------------------------------------------------------------------------------

template<typename T, typename R>
class CubSortByKey : public CubWrapper {
public:
    explicit CubSortByKey() = default;

    explicit CubSortByKey(const int max_items) noexcept;

    void initialize(const int max_items) noexcept;

    void resize(const int max_items) noexcept;

    void shrink_to_fit(const int max_items) noexcept;

    void run(const T* d_key, const R* d_data_in, const int num_items,
             T* d_key_sorted, R* d_data_out,
             T d_key_max = std::numeric_limits<T>::max()) noexcept;

    static void srun(const T* d_key, const R* d_data_in, const int num_items,
                     T* d_key_sorted, R* d_data_out,
                     T d_key_max = std::numeric_limits<T>::max()) noexcept;
};

//==============================================================================

namespace cub_sort_pair {

template<typename T, typename R>
static void run(T* d_in1, R* d_in2, const int num_items,
                T  d_in1_max = std::numeric_limits<T>::max(),
                R  d_in2_max = std::numeric_limits<R>::max()) noexcept;

template<typename T, typename R>
static void run(T* d_in1,     R* d_in2, const int num_items,
                T* d_in1_tmp, R* d_in2_tmp,
                T d_in1_max = std::numeric_limits<T>::max(),
                R d_in2_max = std::numeric_limits<R>::max()) noexcept;

} // namespace cub_sort_pair

//------------------------------------------------------------------------------

template<typename T, typename R>
class CubSortPairs2 : public CubWrapper {
public:
    explicit CubSortPairs2() = default;

    explicit CubSortPairs2(const int max_items, const bool internal_allocation = true)
                           noexcept;

    ~CubSortPairs2() noexcept;

    void initialize(const int max_items, const bool internal_allocation = true) noexcept;

    void resize(const int max_items) noexcept;

    void release(void) noexcept;

    void shrink_to_fit(const int max_items) noexcept;

    void run(T* d_in1, R* d_in2, const int num_items,
             T d_in1_max = std::numeric_limits<T>::max(),
             R d_in2_max = std::numeric_limits<R>::max()) noexcept;

    void run(T* d_in1, R* d_in2, const int num_items,
             T* d_in1_tmp, R* d_in2_tmp,
             T d_in1_max = std::numeric_limits<T>::max(),
             R d_in2_max = std::numeric_limits<R>::max()) noexcept;

    static void srun(T* d_in1, R* d_in2, const int num_items,
                     T d_in1_max = std::numeric_limits<T>::max(),
                     R d_in2_max = std::numeric_limits<R>::max()) noexcept;

    static void srun(T* d_in1, R* d_in2, const int num_items,
                     T* d_in1_tmp, R* d_in2_tmp,
                     T d_in1_max = std::numeric_limits<T>::max(),
                     R d_in2_max = std::numeric_limits<R>::max()) noexcept;
private:
    T*    _d_in1_tmp      { nullptr };
    R*    _d_in2_tmp      { nullptr };
    bool  _internal_alloc { true };
};

//==============================================================================

namespace cub_runlength {

template<typename T>
extern int run(const T* d_in, int num_items, T* d_unique_out,
               int* d_counts_out);

} // namespace cub_runlength

//------------------------------------------------------------------------------

template<typename T>
class CubRunLengthEncode : public CubWrapper {
public:
    explicit CubRunLengthEncode() = default;

    explicit CubRunLengthEncode(const int max_items) noexcept;

    ~CubRunLengthEncode() noexcept;

    void initialize(const int max_items) noexcept;

    void resize(const int max_items) noexcept;

    void release(void) noexcept;

    void shrink_to_fit(const int max_items) noexcept;

    int run(const T* d_in, const int num_items, T* d_unique_out, int* d_counts_out)
            noexcept;

    static int srun(const T* d_in, const int num_items, T* d_unique_out,
                    int* d_counts_out) noexcept;
private:
    int*  _d_num_runs_out { nullptr };
};

//==============================================================================

namespace cub_exclusive_sum {

template<typename T>
extern void run(const T* d_in, int num_items, T* d_out);

template<typename T>
extern void run(T* d_in_out, int num_items);

} // namespace cub_exclusive_sum

//------------------------------------------------------------------------------

template<typename T>
class CubExclusiveSum : public CubWrapper {
public:
    explicit CubExclusiveSum() noexcept = default;

    explicit CubExclusiveSum(const int max_items) noexcept;

    void initialize(const int max_items) noexcept;

    void resize(const int max_items) noexcept;

    void shrink_to_fit(const int max_items) noexcept;

    void run(const T* d_in, const int num_items, T* d_out) const noexcept;

    void run(T* d_in_out, const int num_items) const noexcept;

    static void srun(const T* d_in, const int num_items, T* d_out) noexcept;

    static void srun(T* d_in_out, const int num_items) noexcept;
};

//==============================================================================

template<typename T>
class CubSelectFlagged : public CubWrapper {
public:
    explicit CubSelectFlagged() noexcept = default;

    explicit CubSelectFlagged(const int max_items) noexcept;

    ~CubSelectFlagged() noexcept;

    void initialize(const int max_items) noexcept;

    void resize(const int max_items) noexcept;

    void release(void) noexcept;

    void shrink_to_fit(const int max_items) noexcept;

    int run(const T* d_in, const int num_items, const bool* d_flags, T* d_out)
            noexcept;

    int run(T* d_in_out, const int num_items, const bool* d_flags) noexcept;

    static int srun(const T* d_in, const int num_items, const bool* d_flags, T* d_out)
                    noexcept;

    static int srun(T* d_in_out, const int num_items, const bool* d_flags) noexcept;
private:
    int*  _d_num_selected_out { nullptr };
};

//==============================================================================

} // namespace xlib

#endif


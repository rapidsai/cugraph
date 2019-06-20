/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
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

namespace hornets_nest {

/**
 * @internal
 * @brief aggregation of two pointers
 * @tparam type of the pointers
 */
template<typename T>
struct ptr2_t {
    T* first;
    T*       second;

    ///@internal @brief swap the two pointers
    void swap() noexcept;
};

/**
 * @brief The class implements a two-levels generic type host-device queue
 * @details All elements of the input queue are discarted between different
 *         iterations
 * @tparam T type of objects stored in the queue
 */
template<typename T>
class TwoLevelQueue {
public:
    TwoLevelQueue() = default;

    /**
     * @brief Default costructor
     * @param[in] hornet reference to the hornet instance
     * @param[in] enable_traverse if `true` enable the traverse of the vertices
     *            stored in the queue
     * @param[in] max_allocated_items number of allocated items for a single
     *            level of the queue. Default value: V * 2
     */
    template<typename HornetClass>
    explicit TwoLevelQueue(const HornetClass& hornet,
                           const float work_factor = 2.0f) noexcept;

    explicit TwoLevelQueue(size_t max_allocated_items) noexcept;

    template<typename HornetClass>
    void initialize(const HornetClass& hornet,
                   const float work_factor = 2.0f) noexcept;

    void initialize(size_t max_allocated_items) noexcept;

    HOST_DEVICE
    TwoLevelQueue(const TwoLevelQueue<T>& obj) noexcept;

    //TwoLevelQueue(TwoLevelQueue<T>&& obj) noexcept;

    HOST_DEVICE
    ~TwoLevelQueue() noexcept;

    /**
     * @brief insert an item in the queue
     * @param[in] item item to insert
     * @remark the method can be called on both host and device
     * @remark the method may expensive on host, cheap on device
     */
    __host__ __device__ __forceinline__
    void insert(const T& item) noexcept;

    /**
     * @brief insert a set of items in the queue
     * @param[in] items_array array of items to insert
     * @param[in] num_items number of items in the queue
     * @remark the method can be called only on the host
     * @remark the method may be expensive
     */
    void insert(const T* items_array, int num_items) noexcept;

    /**
     * @brief swap input and output queue
     * @detail the queue output counter is set to zero
     * @remark the method is sycnhronized
     */
    void swap() noexcept;

    /**
     * @brief reset the queue
     * @details the queue counter is set to zero
     */
    void clear() noexcept;

    /**
     * @brief swap input and output queue
     * @remark the queue counter is also set to zero
     */
    void sync() const noexcept;

    /**
     * @brief size of the queue at the input queue
     * @return actual number of queue items at the input queue
     * @remark the method is cheap
     * @warning the method is NOT sycnhronized
     */
    int size() const noexcept;

    int size_sync_in()  const noexcept;
    int size_sync_out() const noexcept;


    /**
     * @warning the method is NOT sycnhronized
     */
    int output_size() const noexcept;

    /**
     * @brief device pointer of the input queue
     * @return constant device pointer to the start of the input queue
     * @remark the method is cheap
     */
    const T* device_input_ptr() const noexcept;

    /**
     * @brief device pointer of the output queue
     * @return constant device pointer to the start of the output queue
     * @remark the method is cheap
     */
    const T* device_output_ptr() const noexcept;

    /**
     * @brief host pointer of the data stored in the output device queue
     * @return constant host pointer to the start of the output queue
     * @remark the method may be expensive
     */
    //const T* host_data() noexcept;

    /**
     * @brief print the items stored at the output queue
     * @remark the method may be expensive
     */
    void print() const noexcept;

    /**
     * @brief print the items stored at the output queue
     * @remark the method may be expensive
     */
    void print_output() const noexcept;

    /**
     * @brief total enqueue items
     */
    int enqueue_items() const noexcept;

    void set_positions(int2 &h_positions);

private:
    ///@internal @brief input and output queue pointers
    ptr2_t<T>    _d_queue_ptrs        { nullptr, nullptr };

    size_t       _max_allocated_items { 0 };
    ///@internal @brief device counter of the queue for `traverse_edges()`
    int2*        _d_counters          { nullptr };
    mutable int2 _h_counters          { 0, 0 };
    bool         _kernel_copy         { false };
    int          _enqueue_items       { 0 };

    /**
     * @brief Internal function to initialize max allocated items
     */
    void _initialize() noexcept;
};

} // namespace hornets_nest

#include "Queue/TwoLevelQueue.i.cuh"

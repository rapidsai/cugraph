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

#include <vector>

namespace custinger_alg {

///TO IMPROVE
/**
 * @brief The class implements a multi-levels generic type host-device queue
 * @details All elements of the queue are not discarted between different
 *         iterations
 * @tparam T type of objects stored in the queue
 */
template<typename T>
class MultiLevelQueue {
static const bool is_vid = std::is_same<T, custinger::vid_t>::value;
using     EnableTraverse = typename std::enable_if< is_vid >::type;
public:
    /**
     * @brief Default costructor
     * @param[in] max_allocated_items total host-device allocated items.
     *            The allocated space must be sufficient to store all items
     *            among different iterations
     */
    explicit MultiLevelQueue(size_t max_allocated_items) noexcept;

    /**
     * @brief Decostructor
     */
    ~MultiLevelQueue() noexcept;

    /**
     * @brief insert an item in the queue
     * @param[in] item item to insert
     * @remark the method can be called on both host and device
     * @remark the method may expensive on host, cheap on device
     */
    __host__ __device__ void insert(const T& item) noexcept;

    /**
     * @brief insert a set of items in the queue
     * @param[in] items_array array of items to insert
     * @param[in] num_items number of items in the queue
     * @remark the method can be called only on the host
     * @remark the method may be expensive
     */
    __host__ void insert(const T* items_array, int num_items) noexcept;

    /**
     * @brief Advance in the queue
     * @details After the method call the start of the queue is set at the
     *          end of the previous iteration
     */
    __host__ void next() noexcept;

    /**
     * @brief size of the queue at the last level
     * @return actual number of queue items at the last level
     * @remark the method is cheap
     */
    __host__ int size() const noexcept;

    /**
     * @brief size of the queue at a selected level
     * @param[in] level selected level of the queue
     * @return number of queue items at the selected level
     * @remark the method is cheap
     */
    __host__ int size(int level) const noexcept;

    /**
     * @brief device pointer of the last level of the queue
     * @return constant device pointer to the start of the last level queue
     * @remark the method is cheap
     */
    __host__ const T* device_ptr() const noexcept;

    /**
     * @brief device pointer of a selected level of the queue
     * @return constant device pointer to the start of the selected level of the
     *         queue
     * @remark the method is cheap
     */
    __host__ const T* device_ptr(int level) const noexcept;

    /**
     * @brief host pointer of the data stored in the last level of the device
     *        queue
     * @return constant host pointer to the start of the last level of the
     *         queue
     * @remark the method may be expensive
     */
    __host__ const T* host_data() noexcept;

    /**
     * @brief host pointer of the data stored at the selected level of the
     *        device queue
     * @return constant host pointer to the start of the selected level of the
     *         queue
     * @remark the method may be expensive
     */
    __host__ const T* host_data(int level) noexcept;

    /**
     * @brief print the items stored at the last level of the device queue
     * @remark the method may be expensive
     */
    __host__ void print() const noexcept;

    /**
     * @brief print the items stored at the selected level of the device queue
     * @remark the method may be expensive
     */
    __host__ void print(int level) const noexcept;

    // NOT IMPLEMENTED!!!
    /**
     * @brief traverse the edges of queue vertices
     * @tparam    Operator typename of the operator (deduced)
     * @param[in] op struct/lambda expression that implements the operator
     * @remark the Operator typename must implement the method
     *         `void operator()(Vertex, Edge)` or the lambda expression
     *         `[=](Vertex, Edge){}`
     * @remark the method is enabled only if the queue type is `vid_t`
     */
    template<typename Operator>
    __host__ EnableTraverse
    traverse_edges(Operator op) noexcept;

private:
    ///@internal @brief prefixsum of the level sizes
    std::vector<int> _level_sizes;
    ///@internal @brief start of input queue and output queue
    ptr2_t<T>        _d_queue_ptrs    { nullptr, nullptr };
    ///@internal @brief actual couter of the queue
    int*             _d_queue_counter { nullptr };
    ///@internal @brief  device pointer of the whole queue
    T*               _d_multiqueue    { nullptr };
    ///@internal @brief host pointer used by `host_data()` method
    T*               _host_data       { nullptr };
    size_t           _max_allocated_items;
    int              _current_level   { 0 };

    // NOT IMPLEMENTED!!!
    template<typename Operator>
    __host__ EnableTraverse
    work_evaluate(const custinger::eoff_t* csr_offsets) noexcept;
};

} // namespace custinger_alg

#include "MultiLevelQueue.i.cuh"

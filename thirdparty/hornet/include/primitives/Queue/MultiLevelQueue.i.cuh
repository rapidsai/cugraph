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
 */
#include <Device/PrintExt.cuh>      //cu::printArray
#include <Device/SafeCudaAPI.cuh>   //cuMemcpyToDeviceAsync

namespace custinger_alg {

template<typename T>
MultiLevelQueue<T>::MultiLevelQueue(size_t max_allocated_items) noexcept :
                                     _level_sizes(32768),
                                     _max_allocated_items(max_allocated_items) {
    hornets_nest::gpu::allocate(_d_multiqueue, max_allocated_items);
    _d_queue_ptrs.first  = _d_multiqueue;
    _d_queue_ptrs.second = _d_multiqueue;
    hornets_nest::gpu::allocate(_d_queue_counter, 1);
    cuMemcpyToDeviceAsync(0, _d_queue_counter);
    _level_sizes.push_back(0);
    _level_sizes.push_back(0);
}

template<typename T>
inline MultiLevelQueue<T>::~MultiLevelQueue() noexcept {
    hornets_nest::gpu::allocate(_d_multiqueue, _d_queue_counter);
    delete[] _host_data;
}

template<typename T>
__host__ void MultiLevelQueue<T>::insert(const T& item) noexcept {
#if defined(__CUDA_ARCH__)
    unsigned       ballot = __ballot(true);
    unsigned elected_lane = xlib::__msb(ballot);
    int warp_offset;
    if (xlib::lane_id() == elected_lane)
        warp_offset = atomicAdd(_d_queue_counter, __popc(ballot));
    int offset = __popc(ballot & xlib::LaneMaskLT()) +
                 __shfl(warp_offset, elected_lane);
    _d_queue_ptrs.second[offset] = item;
#else
    cuMemcpyToDeviceAsync(item, _d_queue_ptrs.second);
    _d_queue_ptrs.second++;
    _level_sizes[_current_level + 1]++;
#endif
}

template<typename T>
__host__ inline void MultiLevelQueue<T>
::insert(const T* items_array, int num_items) noexcept {
    cuMemcpyToDeviceAsync(items_array, num_items, _d_queue_ptrs.second);
    _d_queue_ptrs.second += num_items;
    _level_sizes[_current_level + 1] += num_items;
}

template<typename T>
__host__ void MultiLevelQueue<T>::next() noexcept {
    int h_queue_counter;
    cuMemcpyToHostAsync(_d_queue_counter, h_queue_counter);
    _d_queue_ptrs.first   = _d_queue_ptrs.second;
    _d_queue_ptrs.second += h_queue_counter;
    cuMemcpyToDeviceAsync(0, _d_queue_counter);

    _level_sizes.push_back(_level_sizes[_current_level + 1] + h_queue_counter);
    _current_level++;
}

template<typename T>
__host__ int MultiLevelQueue<T>::size() const noexcept {
    return size(_current_level);
}

template<typename T>
__host__ int MultiLevelQueue<T>::size(int level) const noexcept {
    if (level < 0 || level > _current_level)
        ERROR("Level out-of-bound. level < 0 || level > current")
    return _level_sizes[level + 1] - _level_sizes[level];
}

template<typename T>
__host__ const T* MultiLevelQueue<T>::device_ptr() const noexcept {
    return device_ptr(_current_level);
}

template<typename T>
__host__ const T* MultiLevelQueue<T>::device_ptr(int level) const noexcept {
    return _d_multiqueue + _level_sizes[level];
}

template<typename T>
__host__ const T* MultiLevelQueue<T>::host_data() noexcept {
    return host_data(_current_level);
}

template<typename T>
__host__ const T* MultiLevelQueue<T>::host_data(int level) noexcept {
    if (_host_data == nullptr)
        _host_data = new T[_max_allocated_items];
    cuMemcpyToHost(_d_multiqueue + _level_sizes[level], size(level),_host_data);
    return _host_data;
}

template<typename T>
__host__ void MultiLevelQueue<T>::print() const noexcept {
    cu::printArray(_d_queue_ptrs.first, size());
}

template<typename T>
__host__ void MultiLevelQueue<T>::print(int level) const noexcept {
    cu::printArray(_d_multiqueue + _level_sizes[level], size(level));
}

} // namespace custinger_alg

/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date April, 2017
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
#include "Host/Basic.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>

namespace xlib {

namespace detail {

template<typename T>
QueueBase<T>::QueueBase(size_t size) noexcept : _size(size) {
    try {
        _array = new T[_size];
    } catch (const std::bad_alloc&) {
        ERROR_LINE
    }
}

template<typename T>
QueueBase<T>::~QueueBase() noexcept {
    delete[] _array;
}

template<typename T>
void QueueBase<T>::init(size_t size) noexcept {
    assert(_array == nullptr);
     _size = size;
    try {
        _array = new T[_size];
    } catch (const std::bad_alloc&) {
        ERROR_LINE
    }
}

template<typename T>
void QueueBase<T>::free() noexcept {
    delete[] _array;
    _array = nullptr;
}

template<typename T>
void QueueBase<T>::clear() noexcept {
     _left = 0;
    _right = 0;
}

template<typename T>
void QueueBase<T>::insert(T value) noexcept {
    assert(_right < _size && "Queue::insert(T) : right < size");
    _array[_right++] = value;
}

template<typename T>
void QueueBase<T>::sort() noexcept {
    assert(_right - _left > 0);
    std::sort(_array + _left, _array + _right);
}

template<typename T>
bool QueueBase<T>::empty() const noexcept {
    return _left >= _right;
}

template<typename T>
size_t QueueBase<T>::size() const noexcept {
    return _right - _left;
}

template<typename T>
T& QueueBase<T>::at(size_t index) noexcept {
    assert(index < _right && "Queue::at(i) : i < right");
    return _array[index];
}

template<typename T>
const T& QueueBase<T>::at(size_t index) const noexcept {
    assert(index < _right && "Queue::at(i) : i >= 0 && i < right");
    return _array[index];
}

template<typename T>
void QueueBase<T>::print() const noexcept {
    std::cout << "Queue:\n";
    for (size_t i = _left; i < _right; i++)
        std::cout << _array[i] << ' ';
    std::cout << std::endl;
}

} // namespace detail

//==============================================================================

template<typename T>
inline Queue<T, QueuePolicy::FIFO>::Queue(size_t size) noexcept :
                        detail::QueueBase<T>(size) {}

template<typename T>
inline T& Queue<T, QueuePolicy::FIFO>::tail() noexcept {
    assert(_right - _left > 0);
    return _array[_right - 1];
}

template<typename T>
inline const T& Queue<T, QueuePolicy::FIFO>::tail() const noexcept {
    assert(_right - _left > 0);
    return _array[_right - 1];
}

template<typename T>
inline T& Queue<T, QueuePolicy::FIFO>::extract() noexcept {
    assert(_right - _left > 0);
    return _array[_left++];
}

template<typename T>
inline size_t Queue<T, QueuePolicy::FIFO>::getTotalEnqueueItems()
                                                    const noexcept {
    return _right;
}

//------------------------------------------------------------------------------

template<typename T>
inline Queue<T, QueuePolicy::LIFO>::Queue(size_t size) noexcept :
        detail::QueueBase<T>(size) {}

template<typename T>
inline T& Queue<T, QueuePolicy::LIFO>::top() noexcept {
    assert(_right - _left > 0);
    return _array[_right - 1];
}

template<typename T>
inline const T& Queue<T, QueuePolicy::LIFO>::top() const noexcept {
    assert(_right - _left > 0);
    return _array[_right - 1];
}

template<typename T>
inline T& Queue<T, QueuePolicy::LIFO>::pop() noexcept {
    assert(_right - _left > 0);
    return _array[--_right];
}

template<typename T>
inline T& Queue<T, QueuePolicy::LIFO>::extract() noexcept {
    return pop();
}

//==============================================================================
//==============================================================================

namespace detail {

template<typename T, size_t SIZE>
void QueueStackBase<T, SIZE>::clear() noexcept {
     _left = 0;
    _right = 0;
}

template<typename T, size_t SIZE>
void QueueStackBase<T, SIZE>::insert(T value) noexcept {
    assert(_right < SIZE && "Queue::insert(T) : right < size");
    _array[_right++] = value;
}

template<typename T, size_t SIZE>
void QueueStackBase<T, SIZE>::sort() noexcept {
    std::sort(_array + _left, _array + _right);
}

template<typename T, size_t SIZE>
bool QueueStackBase<T, SIZE>::empty() const noexcept {
    return _left >= _right;
}

template<typename T, size_t SIZE>
size_t QueueStackBase<T, SIZE>::size() const noexcept {
    return _right - _left;
}

template<typename T, size_t SIZE>
T& QueueStackBase<T, SIZE>::at(size_t index) noexcept {
    assert(index < _right && "Queue::at(i) : i >= 0 && i < size");
    return _array[index];
}

template<typename T, size_t SIZE>
const T& QueueStackBase<T, SIZE>::at(size_t index) const noexcept {
    assert(index < _right && "Queue::at(i) : i >= 0 && i < size");
    return _array[index];
}

template<typename T, size_t SIZE>
void QueueStackBase<T, SIZE>::print() const noexcept {
    std::cout << "Queue:\n";
    for (size_t i = _left; i < _right; i++)
        std::cout << _array[i] << ' ';
    std::cout << std::endl;
}

} // namespace detail

//==============================================================================

template<typename T, size_t SIZE>
inline QueueStack<T, SIZE, QueuePolicy::FIFO>::QueueStack() noexcept :
                    detail::QueueStackBase<T, SIZE>() {}

template<typename T, size_t SIZE>
inline T& QueueStack<T, SIZE, QueuePolicy::FIFO>::head() noexcept {
    assert(_right - _left > 0);
    return _array[_left];
}

template<typename T, size_t SIZE>
inline const T& QueueStack<T, SIZE, QueuePolicy::FIFO>::head() const noexcept {
    assert(_right - _left > 0);
    return _array[_left];
}

template<typename T, size_t SIZE>
inline T& QueueStack<T, SIZE, QueuePolicy::FIFO>::tail() noexcept {
    assert(_right - _left > 0);
    return _array[_right - 1];
}

template<typename T, size_t SIZE>
inline const T& QueueStack<T, SIZE, QueuePolicy::FIFO>::tail() const noexcept {
    assert(_right - _left > 0);
    return _array[_right - 1];
}

template<typename T, size_t SIZE>
inline T& QueueStack<T, SIZE, QueuePolicy::FIFO>::extract() noexcept {
    assert(_right - _left > 0);
    return _array[_left++];
}

//------------------------------------------------------------------------------

template<typename T, size_t SIZE>
inline QueueStack<T, SIZE, QueuePolicy::LIFO>::QueueStack() noexcept :
                    detail::QueueStackBase<T, SIZE>() {}

template<typename T, size_t SIZE>
inline T& QueueStack<T, SIZE, QueuePolicy::LIFO>::top() noexcept {
    assert(_right - _left > 0);
    return _array[_right - 1];
}

template<typename T, size_t SIZE>
inline const T& QueueStack<T, SIZE, QueuePolicy::LIFO>::top() const noexcept {
    assert(_right - _left > 0);
    return _array[_right - 1];
}

template<typename T, size_t SIZE>
inline T& QueueStack<T, SIZE, QueuePolicy::LIFO>::pop() noexcept {
    assert(_right - _left > 0);
    return _array[--_right];
}

template<typename T, size_t SIZE>
inline T& QueueStack<T, SIZE, QueuePolicy::LIFO>::extract() noexcept {
    return pop();
}

} // namespace xlib

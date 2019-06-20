/**
 * @internal
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
 *
 * @file
 */
#pragma once

namespace xlib {

enum class QueuePolicy { FIFO, LIFO };

namespace detail {

template<typename T>
class QueueBase {
public:
    explicit QueueBase()                 = default;
    explicit QueueBase(const QueueBase&) = delete;
    void operator=(const QueueBase&)     = delete;
protected:
    size_t _left  { 0 };
    size_t _right { 0 };
    size_t _items { 0 };
    size_t _size  { 0 };
    T*     _array { nullptr };

    explicit QueueBase(size_t size) noexcept;
    virtual ~QueueBase()            noexcept;

    virtual void init(size_t size)  noexcept final;
    virtual void free()             noexcept final;

    virtual T&   extract()          noexcept = 0;
    virtual void insert(T value)    noexcept final;
    virtual void clear()            noexcept final;
    virtual void sort()             noexcept final;

    virtual bool     empty()          const noexcept final;
    virtual size_t   size()           const noexcept final;
    virtual T&       at(size_t index) noexcept final;
    virtual const T& at(size_t index) const noexcept final;
    virtual void     print()          const noexcept final;
};

} // namespace detail

template<typename T, QueuePolicy P = QueuePolicy::FIFO>
class Queue;

template<typename T>
class Queue<T, QueuePolicy::FIFO> final : public detail::QueueBase<T> {
public:
    explicit Queue()            = default;
    explicit Queue(size_t size) noexcept;

    T&        tail()                 noexcept;
    const T&  tail()                 const noexcept;
    size_t    getTotalEnqueueItems() const noexcept;
    T&        extract()              noexcept override;

    using detail::QueueBase<T>::init;
    using detail::QueueBase<T>::free;
    using detail::QueueBase<T>::clear;
    using detail::QueueBase<T>::insert;
    using detail::QueueBase<T>::sort;
    using detail::QueueBase<T>::empty;
    using detail::QueueBase<T>::size;
    using detail::QueueBase<T>::at;
    using detail::QueueBase<T>::print;
private:
    using detail::QueueBase<T>::_left;
    using detail::QueueBase<T>::_right;
    using detail::QueueBase<T>::_items;
    using detail::QueueBase<T>::_size;
    using detail::QueueBase<T>::_array;
};

template<typename T>
class Queue<T, QueuePolicy::LIFO> final : public detail::QueueBase<T> {
public:
    explicit Queue()            = default;
    explicit Queue(size_t size) noexcept;

    T&       top() noexcept;
    const T& top() const noexcept;
    T&       pop() noexcept;
    T&       extract() noexcept override;

    using detail::QueueBase<T>::init;
    using detail::QueueBase<T>::free;
    using detail::QueueBase<T>::clear;
    using detail::QueueBase<T>::insert;
    using detail::QueueBase<T>::sort;
    using detail::QueueBase<T>::empty;
    using detail::QueueBase<T>::size;
    using detail::QueueBase<T>::at;
    using detail::QueueBase<T>::print;
private:
    using detail::QueueBase<T>::_left;
    using detail::QueueBase<T>::_right;
    using detail::QueueBase<T>::_items;
    using detail::QueueBase<T>::_size;
    using detail::QueueBase<T>::_array;
};

//==============================================================================
//==============================================================================

namespace detail {

template<typename T, size_t SIZE>
class QueueStackBase {
public:
    QueueStackBase(const QueueStackBase&) = delete;
    void operator=(const QueueStackBase&) = delete;
protected:
    size_t _left  { 0 };
    size_t _right { 0 };
    T _array[SIZE];

    //explicit QueueStackBase() noexcept = default;

    void             clear()         noexcept;
    void             insert(T value) noexcept;
    virtual T&       extract()       noexcept = 0;
    void             sort()          noexcept;

    bool     empty()          const noexcept;
    size_t   size()           const noexcept;
    T&       at(size_t index) noexcept;
    const T& at(size_t index) const noexcept;
    void     print()          const noexcept;
};

} // namespace detail

template<typename T, size_t SIZE, QueuePolicy P = QueuePolicy::FIFO>
class QueueStack;

template<typename T, size_t SIZE>
class QueueStack<T, SIZE, QueuePolicy::FIFO> :
                     public detail::QueueStackBase<T, SIZE> {
public:
    explicit QueueStack() noexcept;

    T&       head()    noexcept;
    const T& head()    const noexcept;
    T&       tail()    noexcept;
    const T& tail()    const noexcept;
    T&       extract() noexcept final;                                  //NOLINT
private:
    using detail::QueueStackBase<T, SIZE>::_left;
    using detail::QueueStackBase<T, SIZE>::_array;
    using detail::QueueStackBase<T, SIZE>::_right;
};


template<typename T, size_t SIZE>
class QueueStack<T, SIZE, QueuePolicy::LIFO> :
                    public detail::QueueStackBase<T, SIZE> {
public:
    explicit QueueStack() noexcept;

    T&       top()     noexcept;
    const T& top()     const noexcept;
    T&       pop()     noexcept;
    T&       extract() noexcept final;                                  //NOLINT
private:
    using detail::QueueStackBase<T, SIZE>::_array;
    using detail::QueueStackBase<T, SIZE>::_left;
    using detail::QueueStackBase<T, SIZE>::_right;
};

} // namespace xlib

#include "impl/Queue.i.hpp"

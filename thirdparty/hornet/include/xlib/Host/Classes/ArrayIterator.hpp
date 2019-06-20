/*------------------------------------------------------------------------------
Copyright © 2016 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
/**
 * @version 1.3
 */
#pragma once

#include <cassert>
#include <iterator>

namespace xlib {

template<typename T>
class ArrayIterator final :
                    public std::iterator<std::bidirectional_iterator_tag, T> {
public:
    explicit ArrayIterator(T* ptr) noexcept;

    ArrayIterator& operator++() noexcept;
    ArrayIterator& operator--() noexcept;
    T& operator* ()             noexcept;

    template<class R>
    bool operator== (const ArrayIterator<R>& itr) const noexcept;

    template<class R>
    bool operator!= (const ArrayIterator<R>& itr) const noexcept;
private:
    T* _ptr;
};

template<typename T>
class ArrayRevIterator final :
                      public std::iterator<std::bidirectional_iterator_tag, T> {
public:
    explicit ArrayRevIterator(T* ptr) noexcept;

    ArrayRevIterator& operator++() noexcept;
    ArrayRevIterator& operator--() noexcept;
    T& operator* ()                noexcept;

    template<class R>
    bool operator== (const ArrayRevIterator<R>& itr) const noexcept;

    template<class R>
    bool operator!= (const ArrayRevIterator<R>& itr) const noexcept;
private:
    T* _ptr;
};

//==============================================================================

template<typename T>
class ArrayWrapper final {
public:
    explicit ArrayWrapper(T* ptr, size_t size) noexcept;

    ArrayIterator<T> begin() const noexcept;
    ArrayIterator<T> end()   const noexcept;

    ArrayWrapper(const ArrayWrapper& it)   = delete;
    void operator=(const ArrayWrapper& it) = delete;
private:
    T*           _ptr;
    const size_t _size;
};

} // namespace xlib

#include "impl/ArrayIterator.i.hpp"

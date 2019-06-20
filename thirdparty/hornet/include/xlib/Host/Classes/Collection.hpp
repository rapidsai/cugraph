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

#include "Base/Host/Classes/ArrayIterator.hpp"

namespace xlib {

template<typename T>
class Collection final {
public:
    explicit Collection()                noexcept = default;
    explicit Collection(size_t max_size) noexcept;
    ~Collection()                        noexcept;

    const T        operator[](size_t index)                 const noexcept;
    Collection<T>& operator= (const Collection& collection) noexcept;
    void operator+=(const Collection& collection)           noexcept;
    void operator+=(const T& element)                       noexcept;
    size_t size()                                           const noexcept;

    ArrayIterator<T> begin() const noexcept;
    ArrayIterator<T> end()   const noexcept;
    void reverse()           noexcept;

    Collection(const Collection<T>&) = delete;
private:
    T*           _array    { nullptr };
    const size_t _max_size { 0 };
    size_t       _size     { 0 };
};

} // namespace xlib

#include "impl/Collection.i.hpp"

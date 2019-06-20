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
#include "Base/Host/Basic.hpp"
#include <cassert>
#include <stdexcept>

namespace xlib {

template<typename T>
Collection<T>::Collection(size_t max_size) noexcept : _max_size(max_size) {
    try {
        _array = new T[max_size];
    } catch (const std::bad_alloc&) {
        ERROR_LINE
    }
}

template<typename T>
Collection<T>::~Collection() noexcept {
    delete[] _array;
}

template<typename T>
inline const T Collection<T>::operator[](size_t index) const noexcept {
    assert(index < _size && "Collection::operator[]");
    return _array[index];
}

template<typename T>
inline void Collection<T>::operator+=(const Collection& collection) noexcept {
    assert(_size + collection._size < _max_size && "Collection::operator+=");
    for (const auto& it : collection)
        _array[_size++] = it;
}

template<typename T>
inline void Collection<T>::operator+=(const T& element) noexcept {
    assert(_size < _max_size && "Collection::operator+=");
    _array[_size++] = element;
}

template<typename T>
inline Collection<T>& Collection<T>::operator=(const Collection& collection)
                                               noexcept {
    assert(collection._size < _max_size && "Collection::operator=");
    for (size_t i = 0; i < collection._size; i++)
        _array[i] = collection._array[i];
    _size = collection._size;
    return *this;
}

template<typename T>
inline size_t Collection<T>::size() const noexcept {
    return _size;
}

template<typename T>
inline ArrayIterator<T> Collection<T>::begin() const noexcept {
    return ArrayIterator<T>(_array);
}

template<typename T>
inline ArrayIterator<T> Collection<T>::end() const noexcept {
    return ArrayIterator<T>(_array + _size);
}

template<typename T>
inline void Collection<T>::reverse() noexcept {
    std::reverse(_array, _array + _size);
}

} // namespace xlib

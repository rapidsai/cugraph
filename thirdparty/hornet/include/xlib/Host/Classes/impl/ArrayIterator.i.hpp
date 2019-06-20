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

namespace xlib {

template<typename T>
inline ArrayIterator<T>::ArrayIterator(T* ptr) noexcept : _ptr(ptr) {}

template<typename T>
inline ArrayIterator<T>& ArrayIterator<T>::operator++() noexcept {
    assert(_ptr != nullptr && "Out-of-bounds iterator increment!");
    _ptr++;
    return *this;
}

template<typename T>
inline ArrayIterator<T>& ArrayIterator<T>::operator--() noexcept {
    assert(_ptr != nullptr && "Out-of-bounds iterator increment!");
    _ptr--;
    return *this;
}

template<typename T>
template<class R>
inline bool ArrayIterator<T>::operator== (const ArrayIterator<R>& itr)
                                        const noexcept {
    return _ptr == itr._ptr;
}

template<typename T>
template<class R>
inline bool ArrayIterator<T>::operator!= (const ArrayIterator<R>& itr)
                                        const noexcept {
    return _ptr != itr._ptr;
}

template<typename T>
inline T& ArrayIterator<T>::operator* () noexcept {
    assert(_ptr != nullptr && "Invalid iterator dereference!");
    return *_ptr;
}

//==============================================================================

template<typename T>
inline ArrayRevIterator<T>::ArrayRevIterator(T* ptr) noexcept : _ptr(ptr - 1) {}

template<typename T>
inline ArrayRevIterator<T>& ArrayRevIterator<T>::operator++() noexcept {
    assert(_ptr != nullptr && "Out-of-bounds iterator increment!");
    _ptr--;
    return *this;
}

template<typename T>
inline ArrayRevIterator<T>& ArrayRevIterator<T>::operator--() noexcept {
    assert(_ptr != nullptr && "Out-of-bounds iterator increment!");
    _ptr++;
    return *this;
}

template<typename T>
template<class R>
inline bool ArrayRevIterator<T>::operator== (const ArrayRevIterator<R>& itr)
                                             const noexcept {
    return _ptr == itr._ptr;
}

template<typename T>
template<class R>
inline bool ArrayRevIterator<T>::operator!= (const ArrayRevIterator<R>& itr)
                                             const noexcept {
    return _ptr != itr._ptr;
}

template<typename T>
inline T& ArrayRevIterator<T>::operator* () noexcept {
    assert(_ptr != nullptr && "Invalid iterator dereference!");
    return *_ptr;
}

//==============================================================================

template<typename T>
inline ArrayWrapper<T>::ArrayWrapper(T* ptr, size_t size) noexcept :
                                    _ptr(ptr), _size(size) {}

template<typename T>
inline ArrayIterator<T> ArrayWrapper<T>::begin() const noexcept {
    return ArrayIterator<T>(_ptr);
}

template<typename T>
inline ArrayIterator<T> ArrayWrapper<T>::end() const noexcept {
    return ArrayIterator<T>(_ptr + _size);
}

} // namespace xlib

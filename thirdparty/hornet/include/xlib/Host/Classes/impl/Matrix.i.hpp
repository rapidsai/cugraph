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
#include <cassert>

namespace xlib {

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols) noexcept :
                        _rows(rows), _cols(cols),
                        _shift(static_cast<unsigned>(xlib::ceil_log2(cols))) {
    size_t size = _rows << _shift;
    try {
        _matrix = new T[size];
    } catch (const std::bad_alloc&) {
        ERROR_LINE
    }
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& obj) noexcept :
                                        _rows(obj._rows), _cols(obj._cols),
                                        _shift(obj._shift),
                                        _matrix(obj._matrix) {}

template<typename T>
Matrix<T>::Matrix(Matrix<T>&& obj) noexcept :
                                        _rows(obj._rows), _cols(obj._cols),
                                        _shift(obj._shift),
                                        _matrix(obj._matrix) {
    obj._matrix = nullptr;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& obj) noexcept {
    delete[] _matrix;
    _rows   = obj._rows;
    _cols   = obj._cols;
    _shift  = obj._shift;
    _matrix = obj._matrix;
    obj._matrix = nullptr;
    return *this;
}

template<typename T>
void Matrix<T>::init(size_t rows, size_t cols) noexcept {
    _rows  = rows;
    _cols  = cols;
    _shift = xlib::ceil_log2(_cols);
    size_t size = _rows << _shift;
    try {
        _matrix = new T[size];
    } catch (const std::bad_alloc&) {
        ERROR_LINE
    }
}

template<typename T>
void Matrix<T>::deepCopy(const Matrix<T>& obj) noexcept {
    _rows  = obj._rows;
    _cols  = obj._cols;
    _shift = obj._shift;
    size_t size = _rows << _shift;
    try {
        _matrix = new T[size];
    } catch (const std::bad_alloc&) {
        ERROR_LINE
    }
    std::copy(obj._matrix, obj._matrix + size, _matrix);
}

template<typename T>
Matrix<T>::~Matrix() noexcept {
    delete[] _matrix;
}

template<typename T>
inline T* Matrix<T>::operator[](size_t row_index) noexcept {
    assert(row_index < _rows);
    return _matrix + (row_index << _shift);
}

template<typename T>
inline const T* Matrix<T>::operator[](size_t row_index) const noexcept {
    assert(row_index < _rows);
    return _matrix + (row_index << _shift);
}

template<typename T>
void Matrix<T>::print() const noexcept {
    for (size_t i = 0; i < _rows; i++) {
        for (size_t j = 0; j < _cols; j++)
            std::cout << _matrix[(i << _shift) + j] << " ";
        std::cout << "\n";
    }
    std::cout << std::endl;
}

//------------------------------------------------------------------------------

inline Matrix<bool>::Matrix(size_t rows, size_t cols) noexcept :
         _rows(rows), _cols(cols),
         _shift( static_cast<unsigned>(xlib::ceil_log2(ceil_div<32>(_cols))) ) {

    size_t size = _rows << _shift;
    try {
        _matrix = new unsigned[size]();
    } catch (const std::bad_alloc&) {
        ERROR_LINE
    }
}

inline void Matrix<bool>::deepCopy(const Matrix<bool>& obj) noexcept {
    _rows  = obj._rows;
    _cols  = obj._cols;
    _shift = obj._shift;
    size_t size = _rows << _shift;
    try {
        _matrix = new unsigned[size];
    } catch (const std::bad_alloc&) {
        ERROR_LINE
    }
    std::copy(obj._matrix, obj._matrix + size, _matrix);
}

inline Matrix<bool>::Matrix(const Matrix<bool>& obj) noexcept {
    deepCopy(obj);
}

inline Matrix<bool>::Matrix(Matrix<bool>&& obj) noexcept :
                                        _rows(obj._rows),
                                        _cols(obj._cols),
                                        _matrix(obj._matrix),
                                        _shift(obj._shift) {
    obj._matrix = nullptr;
}

inline Matrix<bool>::~Matrix() noexcept {
    delete[] _matrix;
}

inline size_t Matrix<bool>::rows() const noexcept {
    return _rows;
}

inline void Matrix<bool>::print(const std::string& str) const {
    std::cout << str << "  (" << _rows << " x " << _cols << ")\n\n";

    size_t nnz = 0;
    for (size_t i = 0; i < _rows; i++) {
        size_t row_index = i << _shift;
        for (size_t j = 0; j < _cols; j++) {
            unsigned value = _matrix[row_index + (j / 32u)];
            unsigned  mask = 1u << (j % 32u);
            bool bit_value = static_cast<bool>(value & mask);
            if (bit_value)
                nnz++;
            std::cout << bit_value << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\nnnz: " << nnz << "\n" << std::endl;
}

inline void Matrix<bool>::reset() noexcept {
    std::fill(_matrix, _matrix + (_rows << _shift), 0);
}

inline void Matrix<bool>::row_reset(size_t row_index) noexcept {
    assert(row_index < _rows);
    size_t start = row_index << _shift;
    size_t   end = (row_index + 1) << _shift;
    std::fill(_matrix + start, _matrix + end, 0);
}

inline size_t Matrix<bool>::get_nnz() const noexcept {
    size_t  size = _rows << _shift;
    size_t count = 0;
    for (size_t i = 0; i < size; i++)
        count += static_cast<size_t>(__builtin_popcount(_matrix[i]));
    return count;
}

inline Row<bool> Matrix<bool>::operator[](size_t row_index) noexcept {
    assert(row_index < _rows);
    return Row<bool>(_matrix + (row_index << _shift));
}

inline const Row<bool> Matrix<bool>::operator[](size_t row_index)
                                                const noexcept {
    assert(row_index < _rows);
    return Row<bool>(_matrix + (row_index << _shift));
}

//==============================================================================

template<typename T>
inline Row<T>::Row(T* row_ptr) noexcept : _row_ptr(row_ptr) {}

template<typename T>
inline T& Row<T>::operator[](size_t index) noexcept {
    return &_row_ptr[index];
}

template<typename T>
inline const T& Row<T>::operator[](size_t index) const noexcept {
    return &_row_ptr[index];
}

//------------------------------------------------------------------------------

inline Row<bool>::Row(unsigned* row_ptr) noexcept : _row_ptr(row_ptr) {}

inline BitRef Row<bool>::operator[](size_t index) noexcept {
    return BitRef(_row_ptr[index / 32u], 1u << (index % 32u));
}

inline const BitRef Row<bool>::operator[](size_t index) const noexcept {
    return BitRef(_row_ptr[index / 32u], 1u << (index % 32u));
}

} // namespace xlib

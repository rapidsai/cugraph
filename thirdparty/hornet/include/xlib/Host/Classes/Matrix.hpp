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

#include "Base/Host/Classes/BitRef.hpp"
#include "Base/Host/Numeric.hpp"

namespace xlib {

template<typename T> class Matrix;
template<>           class Matrix<bool>;

template<typename T>
class Row {
friend class Matrix<T>;
public:
    T&       operator[](size_t index) noexcept;
    const T& operator[](size_t index) const noexcept;
private:
    T* _row_ptr;
    explicit Row(T* row_ptr) noexcept;
};

template<>
class Row<bool> {
friend class Matrix<bool>;
public:
    BitRef       operator[](size_t index) noexcept;
    const BitRef operator[](size_t index) const noexcept;
private:
    unsigned* _row_ptr;
    explicit Row(unsigned* row_ptr) noexcept;
};

//------------------------------------------------------------------------------

template<typename T>
class Matrix {
public:
    explicit Matrix()                         noexcept = default;
    explicit Matrix(size_t rows, size_t cols) noexcept;
    Matrix(const Matrix<T>& obj)              noexcept; //shallow copy
    Matrix(Matrix<T>&& obj)                   noexcept;
    ~Matrix()                                 noexcept;

    void deepCopy(const Matrix<T>& obj)       noexcept;
    void init(size_t rows, size_t cols)       noexcept;
    T* operator[](size_t row_index)           noexcept;
    const T* operator[](size_t row_index)     const noexcept;
    void print()                              const noexcept;
    size_t rows()                             const noexcept;

    Matrix<T>& operator=(Matrix<T>&& obj)     noexcept;
    void operator=(const Matrix<T>&) = delete;
private:
    size_t    _rows   { 0 };
    size_t    _cols   { 0 };
    T*        _matrix { nullptr };
    unsigned _shift   { 0 };
};


template<>
class Matrix<bool> final {
public:
    explicit Matrix()                         noexcept = default;
    explicit Matrix(size_t rows, size_t cols) noexcept;
    Matrix(const Matrix<bool>& obj)           noexcept;
    Matrix(Matrix<bool>&& obj)                noexcept;
    ~Matrix()                                 noexcept;

    void deepCopy(const Matrix<bool>& obj)    noexcept;

    Row<bool>       operator[](size_t row_index) noexcept;
    const Row<bool> operator[](size_t row_index) const noexcept;

    size_t rows()                      const noexcept;
    void reset()                       noexcept;
    void row_reset(size_t row_index)   noexcept;
    size_t get_nnz()                   const noexcept;
    void print(const std::string& str) const;

private:
    size_t    _rows   { 0 };
    size_t    _cols   { 0 };
    unsigned* _matrix { nullptr };
    unsigned  _shift  { 0 };
};

} // namespace xlib

#include "impl/Matrix.i.hpp"

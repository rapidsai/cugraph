/*
 *  Copyright 2008-2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#pragma once

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/convert.h>
#include <cusp/exception.h>
#include <cusp/io/matrix_market.h>

#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace cusp
{
namespace io
{
namespace detail
{

template <typename IndexType, typename ValueType, typename Stream>
void read_binary_stream(cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& coo, Stream& input)
{
    size_t num_rows, num_cols, num_entries;
    IndexType ientry;
    ValueType ventry;

    input.read(reinterpret_cast<char *>(&num_rows), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&num_cols), sizeof(size_t));
    input.read(reinterpret_cast<char *>(&num_entries), sizeof(size_t));

    coo.resize(num_rows, num_cols, num_entries);

    size_t index = 0;
    while( index < num_entries )
    {
        input.read(reinterpret_cast<char *>(&ientry), sizeof(IndexType));
        coo.row_indices[index++] = ientry;
    }

    index = 0;
    while( index < num_entries )
    {
        input.read(reinterpret_cast<char *>(&ientry), sizeof(IndexType));
        coo.column_indices[index++] = ientry;
    }

    index = 0;
    while( index < num_entries )
    {
        input.read(reinterpret_cast<char *>(&ventry), sizeof(ValueType));
        coo.values[index++] = ventry;
    }

    // sort indices by (row,column)
    coo.sort_by_row_and_column();
}

template <typename Matrix, typename Stream, typename Format>
void read_binary_stream(Matrix& mtx, Stream& input, Format)
{
    // general case
    typedef typename Matrix::index_type IndexType;
    typedef typename Matrix::value_type ValueType;

    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> temp;

    read_binary_stream(temp, input);

    cusp::convert(temp, mtx);
}

template <typename IndexType, typename ValueType, typename Stream>
void write_binary_stream(const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& coo,
                         Stream& output)
{
    output.write(reinterpret_cast<const char *>(&coo.num_rows), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&coo.num_cols), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&coo.num_entries), sizeof(size_t));
    output.write(reinterpret_cast<const char *>(&coo.row_indices[0]), coo.num_entries*sizeof(IndexType));
    output.write(reinterpret_cast<const char *>(&coo.column_indices[0]), coo.num_entries*sizeof(IndexType));
    output.write(reinterpret_cast<const char *>(&coo.values[0]), coo.num_entries*sizeof(ValueType));
}

template <typename Matrix, typename Stream>
void write_binary_stream(const Matrix& mtx, Stream& output, cusp::sparse_format)
{
    // general sparse case
    typedef typename Matrix::index_type IndexType;
    typedef typename Matrix::value_type ValueType;

    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> coo(mtx);

    cusp::io::detail::write_binary_stream(coo, output);
}

} // end namespace detail


template <typename Matrix>
void read_binary_file(Matrix& mtx, const std::string& filename)
{
    std::ifstream file(filename.c_str(), std::ios::binary);

    if (!file)
        throw cusp::io_exception(std::string("unable to open file \"") + filename + std::string("\" for reading"));

#ifdef __APPLE__
    // WAR OSX-specific issue using rdbuf
    std::stringstream file_string (std::stringstream::in | std::stringstream::out | std::ios::binary);
    std::vector<char> buffer(file.rdbuf()->pubseekoff(0, std::ios::end,std::ios::in));
    file.rdbuf()->pubseekpos(0, std::ios::in);
    file.rdbuf()->sgetn(&buffer[0], buffer.size());
    file_string.write(&buffer[0], buffer.size());

    cusp::io::read_binary_stream(mtx, file_string);
#else
    cusp::io::read_binary_stream(mtx, file);
#endif
}

template <typename Matrix, typename Stream>
void read_binary_stream(Matrix& mtx, Stream& input)
{
    return cusp::io::detail::read_binary_stream(mtx, input, typename Matrix::format());
}

template <typename Matrix>
void write_binary_file(const Matrix& mtx, const std::string& filename)
{
    std::ofstream file(filename.c_str(), std::ios::binary);

    if (!file)
        throw cusp::io_exception(std::string("unable to open file \"") + filename + std::string("\" for writing"));

#ifdef __APPLE__
    // WAR OSX-specific issue using rdbuf
    std::stringstream file_string (std::stringstream::in | std::stringstream::out | std::ios::binary);

    cusp::io::write_binary_stream(mtx, file_string);

    file.rdbuf()->sputn(file_string.str().c_str(), file_string.str().size());
#else
    cusp::io::write_binary_stream(mtx, file);
#endif
}

template <typename Matrix, typename Stream>
void write_binary_stream(const Matrix& mtx, Stream& output)
{
    cusp::io::detail::write_binary_stream(mtx, output, typename Matrix::format());
}

} //end namespace io
} //end namespace cusp


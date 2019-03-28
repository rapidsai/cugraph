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
thrust::tuple<IndexType,IndexType>
read_dimacs_stream(cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& coo, Stream& input)
{
    // read file contents line by line
    std::string line;

    // skip over banner and comments
    do
    {
        std::getline(input, line);
    } while (line[0] == 'c');

    // line contains [num_rows num_columns num_entries]
    std::vector<std::string> tokens;
    detail::tokenize(tokens, line);

    if (tokens.size() != 4)
        throw cusp::io_exception("invalid Dimacs coordinate format");

    size_t num_verts, num_entries;

    std::istringstream(tokens[2]) >> num_verts;
    std::istringstream(tokens[3]) >> num_entries;

    coo.resize(num_verts, num_verts, num_entries);

    size_t num_entries_read = 0;
    IndexType src = -1;
    IndexType snk = -1;

    while(num_entries_read < coo.num_entries && !input.eof())
    {
        double real;

        input >> line;

        if(line[0] == 'a')
        {
            input >> coo.row_indices[num_entries_read];
            input >> coo.column_indices[num_entries_read];
            input >> real;

            coo.values[num_entries_read++] = real;
        }
        else if(line[0] == 'n')
        {
            IndexType vertex;

            input >> vertex;
            input >> line;

            if(line[0] == 's')
                src = vertex - 1;
            else if(line[0] == 't')
                snk = vertex - 1;
            else
                throw cusp::io_exception("unexpected terminal vertex specified");
        }
        else
        {
            throw cusp::io_exception("unexpected edge type specified");
        }
    }

    if(num_entries_read != coo.num_entries)
        throw cusp::io_exception("unexpected EOF while reading Dimacs entries");

    // check validity of row and column index data
    if (coo.num_entries > 0)
    {
        size_t min_row_index = *std::min_element(coo.row_indices.begin(), coo.row_indices.end());
        size_t max_row_index = *std::max_element(coo.row_indices.begin(), coo.row_indices.end());
        size_t min_col_index = *std::min_element(coo.column_indices.begin(), coo.column_indices.end());
        size_t max_col_index = *std::max_element(coo.column_indices.begin(), coo.column_indices.end());

        if (min_row_index < 1)            throw cusp::io_exception("found invalid row index (index < 1)");
        if (min_col_index < 1)            throw cusp::io_exception("found invalid column index (index < 1)");
        if (max_row_index > coo.num_rows) throw cusp::io_exception("found invalid row index (index > num_rows)");
        if (max_col_index > coo.num_cols) throw cusp::io_exception("found invalid column index (index > num_columns)");
    }

    // convert base-1 indices to base-0
    for(size_t n = 0; n < coo.num_entries; n++)
    {
        coo.row_indices[n]    -= 1;
        coo.column_indices[n] -= 1;
    }

    // sort indices by (row,column)
    coo.sort_by_row_and_column();

    return thrust::tie(src,snk);
}

template <typename Matrix, typename Stream, typename Format>
thrust::tuple<typename Matrix::index_type, typename Matrix::index_type>
read_dimacs_stream(Matrix& mtx, Stream& input, Format)
{
    // general case
    typedef typename Matrix::index_type IndexType;
    typedef typename Matrix::value_type ValueType;

    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> temp;

    thrust::tuple<IndexType,IndexType> ret = read_dimacs_stream(temp, input);

    cusp::convert(temp, mtx);

    return ret;
}

template <typename IndexType, typename ValueType, typename Stream>
void write_dimacs_stream(const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& coo,
                         const thrust::tuple<IndexType,IndexType>& t,
                         Stream& output)
{
    output << "p max " << coo.num_rows << " " << coo.num_entries << std::endl;
    output << "n " << (thrust::get<0>(t) + 1) << " s" << std::endl;
    output << "n " << (thrust::get<1>(t) + 1) << " t" << std::endl;

    for(size_t i = 0; i < coo.num_entries; i++)
    {
        output << "a ";
        output << (coo.row_indices[i]    + 1) << " ";
        output << (coo.column_indices[i] + 1) << " ";

        int val = coo.values[i];
        output << val << std::endl;
    }
}

template <typename Matrix, typename Stream>
void write_dimacs_stream(const Matrix& mtx,
                         const thrust::tuple<typename Matrix::index_type,typename Matrix::index_type>& t,
                         Stream& output,
                         cusp::sparse_format)
{
    // general sparse case
    typedef typename Matrix::index_type IndexType;
    typedef typename Matrix::value_type ValueType;

    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> coo(mtx);

    cusp::io::detail::write_dimacs_stream(coo, t, output);
}

} // end namespace detail


template <typename Matrix>
thrust::tuple<typename Matrix::index_type, typename Matrix::index_type>
read_dimacs_file(Matrix& mtx, const std::string& filename)
{
    std::ifstream file(filename.c_str());

    if (!file)
        throw cusp::io_exception(std::string("unable to open file \"") + filename + std::string("\" for reading"));

#ifdef __APPLE__
    // WAR OSX-specific issue using rdbuf
    std::stringstream file_string (std::stringstream::in | std::stringstream::out);
    std::vector<char> buffer(file.rdbuf()->pubseekoff(0, std::ios::end,std::ios::in));
    file.rdbuf()->pubseekpos(0, std::ios::in);
    file.rdbuf()->sgetn(&buffer[0], buffer.size());
    file_string.write(&buffer[0], buffer.size());

    return cusp::io::read_dimacs_stream(mtx, file_string);
#else
    return cusp::io::read_dimacs_stream(mtx, file);
#endif
}

template <typename Matrix, typename Stream>
thrust::tuple<typename Matrix::index_type, typename Matrix::index_type>
read_dimacs_stream(Matrix& mtx, Stream& input)
{
    return cusp::io::detail::read_dimacs_stream(mtx, input, typename Matrix::format());
}

template <typename Matrix>
void write_dimacs_file(const Matrix& mtx,
                       const thrust::tuple<typename Matrix::index_type,typename Matrix::index_type>& t,
                       const std::string& filename)
{
    std::ofstream file(filename.c_str());

    if (!file)
        throw cusp::io_exception(std::string("unable to open file \"") + filename + std::string("\" for writing"));

#ifdef __APPLE__
    // WAR OSX-specific issue using rdbuf
    std::stringstream file_string (std::stringstream::in | std::stringstream::out);

    cusp::io::write_dimacs_stream(mtx, t, file_string);

    file.rdbuf()->sputn(file_string.str().c_str(), file_string.str().size());
#else
    cusp::io::write_dimacs_stream(mtx, t, file);
#endif
}

template <typename Matrix, typename Stream>
void write_dimacs_stream(const Matrix& mtx,
                         const thrust::tuple<typename Matrix::index_type,typename Matrix::index_type>& t,
                         Stream& output)
{
    cusp::io::detail::write_dimacs_stream(mtx, t, output, typename Matrix::format());
}

} //end namespace io
} //end namespace cusp


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
#include <cusp/complex.h>
#include <cusp/convert.h>
#include <cusp/exception.h>

#include <thrust/sort.h>

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

inline
void tokenize(std::vector<std::string>& tokens,
              const std::string& str,
              const std::string& delimiters = "\n\r\t ")
{
    // Skip delimiters at beginning.
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

    while (std::string::npos != pos || std::string::npos != lastPos)
    {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }
}

struct matrix_market_banner
{
    std::string storage;    // "array" or "coordinate"
    std::string symmetry;   // "general", "symmetric", "hermitian", or "skew-symmetric"
    std::string type;       // "complex", "real", "integer", or "pattern"
};

template <typename Stream>
void read_matrix_market_banner(matrix_market_banner& banner, Stream& input)
{
    std::string line;
    std::vector<std::string> tokens;

    // read first line
    std::getline(input, line);
    detail::tokenize(tokens, line);

    if (tokens.size() != 5 || tokens[0] != "%%MatrixMarket" || tokens[1] != "matrix")
        throw cusp::io_exception("invalid MatrixMarket banner");

    banner.storage  = tokens[2];
    banner.type     = tokens[3];
    banner.symmetry = tokens[4];

    if (banner.storage != "array" && banner.storage != "coordinate")
        throw cusp::io_exception("invalid MatrixMarket storage format [" + banner.storage + "]");

    if (banner.type != "complex" && banner.type != "real"
            && banner.type != "integer" && banner.type != "pattern")
        throw cusp::io_exception("invalid MatrixMarket data type [" + banner.type + "]");

    if (banner.symmetry != "general" && banner.symmetry != "symmetric"
            && banner.symmetry != "hermitian" && banner.symmetry != "skew-symmetric")
        throw cusp::io_exception("invalid MatrixMarket symmetry [" + banner.symmetry + "]");
}



template <typename ScalarType>
void assign_complex(ScalarType& value, double real, double imag)
{
    value = real;
}

template <typename ScalarType>
void assign_complex(cusp::complex<ScalarType>& value, double real, double imag)
{
    value.real(real);
    value.imag(imag);
}

template <typename Stream, typename ScalarType>
void write_value(Stream& output, const ScalarType& value)
{
    output << value;
}

template <typename Stream, typename ScalarType>
void write_value(Stream& output, const cusp::complex<ScalarType>& value)
{
    output << value.real() << " " << value.imag();
}

template<typename Stream>
thrust::tuple<size_t,size_t,size_t>
read_input_size(Stream& input)
{
    // read file contents line by line
    std::string line;

    // skip over banner and comments
    do
    {
        std::getline(input, line);
    } while (line[0] == '%');

    // line contains [num_rows num_columns num_entries]
    std::vector<std::string> tokens;
    detail::tokenize(tokens, line);

    if (tokens.size() != 3)
        throw cusp::io_exception("invalid MatrixMarket coordinate format");

    size_t num_rows, num_cols, num_entries;

    std::istringstream(tokens[0]) >> num_rows;
    std::istringstream(tokens[1]) >> num_cols;
    std::istringstream(tokens[2]) >> num_entries;

    return thrust::tie(num_rows, num_cols, num_entries);
}

template <typename MatrixType, typename Stream>
void read_coordinate_stream(MatrixType& coo,
                            Stream& input,
                            const matrix_market_banner& banner,
                            cusp::host_memory,
                            cusp::coo_format)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    size_t num_rows = coo.num_rows;
    size_t num_cols = coo.num_cols;

    size_t num_entries_read = 0;

    // read file contents
    if (banner.type == "pattern")
    {
        while(num_entries_read < coo.num_entries && !input.eof())
        {
            input >> coo.row_indices[num_entries_read];
            input >> coo.column_indices[num_entries_read];
            num_entries_read++;
        }

        std::fill(coo.values.begin(), coo.values.end(), ValueType(1));
    }
    else if (banner.type == "real" || banner.type == "integer")
    {
        while(num_entries_read < coo.num_entries && !input.eof())
        {
            double real;

            input >> coo.row_indices[num_entries_read];
            input >> coo.column_indices[num_entries_read];
            input >> real;

            coo.values[num_entries_read] = real;
            num_entries_read++;
        }
    }
    else if (banner.type == "complex")
    {
        while(num_entries_read < coo.num_entries && !input.eof())
        {
            double real, imag;

            input >> coo.row_indices[num_entries_read];
            input >> coo.column_indices[num_entries_read];
            input >> real;
            input >> imag;

            assign_complex(coo.values[num_entries_read], real, imag);

            num_entries_read++;
        }
    }
    else
    {
        throw cusp::io_exception("invalid MatrixMarket data type");
    }

    if(num_entries_read != coo.num_entries)
    {
        std::cerr << " Read " << num_entries_read << " out of " << coo.num_entries << " expected entries!" << std::endl;
        throw cusp::io_exception("unexpected EOF while reading MatrixMarket entries");
    }

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

    // expand symmetric formats to "general" format
    if (banner.symmetry != "general")
    {
        size_t off_diagonals = 0;

        for (size_t n = 0; n < coo.num_entries; n++)
            if(coo.row_indices[n] != coo.column_indices[n])
                off_diagonals++;

        size_t general_num_entries = coo.num_entries + off_diagonals;

        cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> general(num_rows, num_cols, general_num_entries);

        if (banner.symmetry == "symmetric")
        {
            size_t nnz = 0;

            for (size_t n = 0; n < coo.num_entries; n++)
            {
                // copy entry over
                general.row_indices[nnz]    = coo.row_indices[n];
                general.column_indices[nnz] = coo.column_indices[n];
                general.values[nnz]         = coo.values[n];
                nnz++;

                // duplicate off-diagonals
                if (coo.row_indices[n] != coo.column_indices[n])
                {
                    general.row_indices[nnz]    = coo.column_indices[n];
                    general.column_indices[nnz] = coo.row_indices[n];
                    general.values[nnz]         = coo.values[n];
                    nnz++;
                }
            }
        }
        else if (banner.symmetry == "hermitian")
        {
            throw cusp::not_implemented_exception("MatrixMarket I/O does not currently support hermitian matrices");
            //TODO
        }
        else if (banner.symmetry == "skew-symmetric")
        {
            //TODO
            throw cusp::not_implemented_exception("MatrixMarket I/O does not currently support skew-symmetric matrices");
        }

        // store full matrix in coo
        coo.swap(general);
    } // if (banner.symmetry != "general")

    // sort indices by (row,column)
    coo.sort_by_row_and_column();
}

template <typename IndexType, typename ValueType, typename Stream>
void read_coordinate_stream(cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& csr, Stream& input, const matrix_market_banner& banner)
{
    typedef typename cusp::array1d<IndexType,cusp::host_memory>::view IndexView;
    typedef typename cusp::array1d<ValueType,cusp::host_memory>::view ValueView;
    typedef cusp::coo_matrix_view<IndexView,IndexView,ValueView> CooView;

    size_t num_rows, num_cols, num_entries;
    thrust::tie(num_rows, num_cols, num_entries) = read_input_size(input);

    cusp::array1d<IndexType,cusp::host_memory> row_indices(num_entries);
    csr.resize(num_rows, num_cols, num_entries);

    CooView coo(csr.num_rows, csr.num_cols, csr.num_entries,
                cusp::make_array1d_view(row_indices),
                cusp::make_array1d_view(csr.column_indices),
                cusp::make_array1d_view(csr.values));

    read_coordinate_stream(coo, input, banner, cusp::host_memory(), cusp::coo_format());
    cusp::indices_to_offsets(row_indices, csr.row_offsets);
}

template <typename IndexType, typename ValueType, typename Stream>
void read_coordinate_stream(cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& coo, Stream& input, const matrix_market_banner& banner)
{
    size_t num_rows, num_cols, num_entries;
    thrust::tie(num_rows, num_cols, num_entries) = read_input_size(input);

    coo.resize(num_rows, num_cols, num_entries);
    read_coordinate_stream(coo, input, banner, cusp::host_memory(), cusp::coo_format());
}

template <typename Matrix, typename Stream>
void read_coordinate_stream(Matrix& mtx, Stream& input, const matrix_market_banner& banner)
{
    typedef typename Matrix::index_type IndexType;
    typedef typename Matrix::value_type ValueType;

    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> temp;

    read_coordinate_stream(temp, input, banner);

    cusp::convert(temp, mtx);
}

template <typename ValueType, typename Stream>
void read_array_stream(cusp::array2d<ValueType,cusp::host_memory>& mtx, Stream& input, const matrix_market_banner& banner)
{
    // read file contents line by line
    std::string line;

    // skip over banner and comments
    do
    {
        std::getline(input, line);
    } while (line[0] == '%');

    std::vector<std::string> tokens;
    detail::tokenize(tokens, line);

    if (tokens.size() != 2)
        throw cusp::io_exception("invalid MatrixMarket array format");

    size_t num_rows, num_cols;

    std::istringstream(tokens[0]) >> num_rows;
    std::istringstream(tokens[1]) >> num_cols;

    cusp::array2d<ValueType,cusp::host_memory,cusp::column_major> dense(num_rows, num_cols);

    size_t num_entries = num_rows * num_cols;

    size_t num_entries_read = 0;

    // read file contents
    if (banner.type == "pattern")
    {
        throw cusp::not_implemented_exception("pattern array MatrixMarket format is not supported");
    }
    else if (banner.type == "real" || banner.type == "integer")
    {
        while(num_entries_read < num_entries && !input.eof())
        {
            double real;

            input >> real;

            dense.values[num_entries_read] = real;

            num_entries_read++;
        }
    }
    else if (banner.type == "complex")
    {
        while(num_entries_read < num_entries && !input.eof())
        {
            double real, imag;

            input >> real;
            input >> imag;

            assign_complex(dense.values[num_entries_read], real, imag);

            num_entries_read++;
        }
    }
    else
    {
        throw cusp::io_exception("invalid MatrixMarket data type");
    }

    if(num_entries_read != num_entries)
    {
        std::cerr << " Read " << num_entries_read << " out of " << num_entries << " expected entries!" << std::endl;
        throw cusp::io_exception("unexpected EOF while reading MatrixMarket entries");
    }

    if (banner.symmetry != "general")
        throw cusp::not_implemented_exception("only general array symmetric MatrixMarket format is supported");

    cusp::copy(dense, mtx);
}



template <typename IndexType, typename ValueType, typename Stream>
void write_coordinate_stream(const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& coo, Stream& output)
{
    bool is_complex = thrust::detail::is_same<ValueType, cusp::complex<typename cusp::norm_type<ValueType>::type> >::value;

    if (is_complex)
        output << "%%MatrixMarket matrix coordinate complex general\n";
    else
        output << "%%MatrixMarket matrix coordinate real general\n";

    output << "\t" << coo.num_rows << "\t" << coo.num_cols << "\t" << coo.num_entries << "\n";

    for(size_t i = 0; i < coo.num_entries; i++)
    {
        output << (coo.row_indices[i]    + 1) << " ";
        output << (coo.column_indices[i] + 1) << " ";
        cusp::io::detail::write_value(output, coo.values[i]);
        output << "\n";
    }
}


template <typename Matrix, typename Stream, typename Format>
void read_matrix_market_stream(Matrix& mtx, Stream& input, Format)
{
    // general case
    typedef typename Matrix::value_type ValueType;

    // read banner
    matrix_market_banner banner;
    read_matrix_market_banner(banner, input);

    if (banner.storage == "coordinate")
    {
        read_coordinate_stream(mtx, input, banner);
    }
    else // banner.storage == "array"
    {
        cusp::array2d<ValueType,cusp::host_memory> temp;

        read_array_stream(temp, input, banner);

        cusp::convert(temp, mtx);
    }
}

template <typename Matrix, typename Stream>
void read_matrix_market_stream(Matrix& mtx, Stream& input, cusp::array1d_format)
{
    // array1d case
    typedef typename Matrix::value_type ValueType;

    cusp::array2d<ValueType,cusp::host_memory> temp;

    cusp::io::read_matrix_market_stream(temp, input);

    cusp::convert(temp, mtx);
}

template <typename Matrix, typename Stream>
void write_matrix_market_stream(const Matrix& mtx, Stream& output, cusp::sparse_format)
{
    // general sparse case
    typedef typename Matrix::index_type IndexType;
    typedef typename Matrix::value_type ValueType;

    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> coo(mtx);

    cusp::io::detail::write_coordinate_stream(coo, output);
}

template <typename Matrix, typename Stream>
void write_matrix_market_stream(const Matrix& mtx, Stream& output, cusp::array1d_format)
{
    typedef typename Matrix::value_type ValueType;

    bool is_complex = thrust::detail::is_same<ValueType, cusp::complex<typename cusp::norm_type<ValueType>::type> >::value;

    if (is_complex)
        output << "%%MatrixMarket matrix array complex general\n";
    else
        output << "%%MatrixMarket matrix array real general\n";

    output << "\t" << mtx.size() << "\t1\n";

    for(size_t i = 0; i < mtx.size(); i++)
    {
        write_value(output, mtx[i]);
        output << "\n";
    }
}

template <typename Matrix, typename Stream>
void write_matrix_market_stream(const Matrix& mtx, Stream& output, cusp::array2d_format)
{
    typedef typename Matrix::value_type ValueType;

    bool is_complex = thrust::detail::is_same<ValueType, cusp::complex<typename cusp::norm_type<ValueType>::type> >::value;

    if (is_complex)
        output << "%%MatrixMarket matrix array complex general\n";
    else
        output << "%%MatrixMarket matrix array real general\n";

    output << "\t" << mtx.num_rows << "\t" << mtx.num_cols << "\n";

    for(size_t j = 0; j < mtx.num_cols; j++)
    {
        for(size_t i = 0; i < mtx.num_rows; i++)
        {
            write_value(output, mtx(i,j));
            output << "\n";
        }
    }
}

} // end namespace detail


template <typename Matrix>
void read_matrix_market_file(Matrix& mtx, const std::string& filename)
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

    cusp::io::read_matrix_market_stream(mtx, file_string);
#else
    cusp::io::read_matrix_market_stream(mtx, file);
#endif
}

template <typename Matrix, typename Stream>
void read_matrix_market_stream(Matrix& mtx, Stream& input)
{
    cusp::io::detail::read_matrix_market_stream(mtx, input, typename Matrix::format());
}

template <typename Matrix>
void write_matrix_market_file(const Matrix& mtx, const std::string& filename)
{
    std::ofstream file(filename.c_str());

    if (!file)
        throw cusp::io_exception(std::string("unable to open file \"") + filename + std::string("\" for writing"));

#ifdef __APPLE__
    // WAR OSX-specific issue using rdbuf
    std::stringstream file_string (std::stringstream::in | std::stringstream::out);

    cusp::io::write_matrix_market_stream(mtx, file_string);

    file.rdbuf()->sputn(file_string.str().c_str(), file_string.str().size());
#else
    cusp::io::write_matrix_market_stream(mtx, file);
#endif
}

template <typename Matrix, typename Stream>
void write_matrix_market_stream(const Matrix& mtx, Stream& output)
{
    cusp::io::detail::write_matrix_market_stream(mtx, output, typename Matrix::format());
}

} //end namespace io
} //end namespace cusp


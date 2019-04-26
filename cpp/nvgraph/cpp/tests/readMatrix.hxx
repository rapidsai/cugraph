/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <fstream>
#include <sstream> //stringstream
#include <string.h>
#include <vector>
#include <cstdlib> 
#include <iomanip> 
#include <algorithm>
#include <cfloat>

//Matrix Market COO reader-requires a call to sort in the test file
template<typename IndexType_, typename ValueType_>
struct Mat
{

	IndexType_ i;
	IndexType_ j;
	ValueType_ val;
	bool transpose;
	Mat() {
	} //default cosntructor
	Mat(bool transpose) :
			transpose(transpose) {
	} //pass in when comapring rows or columns
	bool operator()(const Mat<IndexType_, ValueType_> &x1, const Mat<IndexType_, ValueType_> &x2)
							{
		if (!transpose)
		{
			if (x1.i == x2.i)
				return x1.j < x2.j; //if rows equal sort by column index
			return x1.i < x2.i;
		}
		else
		{
			if (x1.j == x2.j)
				return x1.i < x2.i; //if rows equal sort by column index
			return x1.j < x2.j;
		}
	}
};
template<typename ValueType_>
void dump_host_dense_mat(std::vector<ValueType_>& v, int ld)
									{
	std::stringstream ss;
	ss.str(std::string());
	ss << std::setw(10);
	ss.precision(3);
	for (int i = 0; i < ld; ++i)
			{
		for (int j = 0; j < ld; ++j)
				{
			ss << v[i * ld + j] << std::setw(10);
		}
		ss << std::endl;
	}
	std::cout << ss.str();
}

/**
 * Reads in graphs given in the "network" format. This format consists a
 * row for each edge in the graph, giving its source and destination. There
 * is no header or comment lines.
 * @param filename The name of the file to read in.
 * @param nnz The number of edges given in the file.
 * @param src Vector to write out the sources to.
 * @param dest Vector to write out the destinations to.
 */
template<typename IndexType>
void readNetworkFile(const char * filename,
							size_t nnz,
							std::vector<IndexType>& src,
							std::vector<IndexType>& dest) {
	std::ifstream infile;
	infile.open(filename);
	src.resize(nnz);
	dest.resize(nnz);
	for (size_t i = 0; i < nnz; i++) {
		infile >> src[i];
		infile >> dest[i];
	}
	infile.close();
	std::cout << "Read in " <<  nnz << " rows from: " << filename << "\n";
}

//reads the Matrix Market format from the florida collection of sparse matrices assuming
//the first lines are comments beginning with %
template<typename IndexType_, typename ValueType_>
void readMatrixMarketFile(const char * filename,
                          IndexType_ &m,
                          IndexType_ &n,
                          IndexType_ &nnz,
                          std::vector<Mat<IndexType_, ValueType_> > &matrix,
                          bool edges_only) {
	std::ifstream infile;
	infile.open(filename);
	std::string line;
	std::stringstream params;
	while (1)
	{
		std::getline(infile, line);
		//ignore initial comments that begin with %
		if (line[0] != '%')
				{
			//first line without % for comments will have matrix size
			params << line;
			params >> n;
			params >> m;
			params >> nnz;
			break; //break and then read in COO format
		}
	}
	//COO format
	matrix.resize(nnz);
	//remaining file lines are tuples of row ind, col ind and possibly value
	//sometimes value assumed to be one
	for (int k = 0; k < nnz; ++k)
			{
		infile >> matrix[k].i;
		infile >> matrix[k].j;
		if (edges_only)
			matrix[k].val = 1.0;
		else
			infile >> matrix[k].val;
	}

	infile.close();
}
//binary matrix reader functions
void printUsageAndExit()
{
	printf("%s", "Usage:./csrmv_pl matrix_csr.bin\n");
	printf("%s", "M is square, in Amgx binary format\n");

	exit(0);
}

int read_header_amgx_csr_bin(FILE* fpin,
										int & n,
										int & nz
										)
										{
	char text_header[255];
	unsigned int system_flags[9];
	size_t is_read1, is_read2;

	is_read1 = fread(text_header, sizeof(char), strlen("%%NVAMGBinary\n"), fpin);
	is_read2 = fread(system_flags, sizeof(unsigned int), 9, fpin);
	if (!is_read1 || !is_read2)
			{
		printf("%s", "I/O fail\n");
		return 1;
	}

	// We assume that system_flags [] = { 1, 1, whatever, 0, 0, 1, 1, n, nz };
	/*
	 bool is_mtx = system_flags[0];
	 bool is_rhs = system_flags[1];
	 bool is_soln = system_flags[2];
	 unsigned idx_t matrix_format = system_flags[3];
	 bool diag = system_flags[4];
	 unsigned idx_t block_dimx = system_flags[5];
	 unsigned idx_t block_dimy = system_flags[6];
	 */

	if (system_flags[0] != 1 || system_flags[1] != 1 ||
			system_flags[3] != 0 || system_flags[4] != 0 || system_flags[5] != 1 ||
			system_flags[6] != 1 || system_flags[7] < 1 || system_flags[8] < 1)

			{
		printf(	"Wrong format : system_flags [] != { 1(%d), 1(%d), 0(%d), 0(%d), 0(%d), 1(%d), 1(%d), n(%d), nz(%d) }\n\n",
					system_flags[0],
					system_flags[1],
					system_flags[2],
					system_flags[3],
					system_flags[4],
					system_flags[5],
					system_flags[6],
					system_flags[7],
					system_flags[8]);
		return 1;
	}

	n = system_flags[7];
	nz = system_flags[8];
	return 0;
}

//reader is for ints and double
template<typename I>
int read_csr_bin(FILE* fpin,
						I &n,
						I &nz,
						std::vector<I> &row_ptr,
						std::vector<I> &col_ind
						)
						{
	size_t is_read1, is_read2, is_read3, is_read4;
	is_read1 = fread(&n, sizeof(I), 1, fpin);
	is_read2 = fread(&nz, sizeof(I), 1, fpin);
	if (!is_read1 || !is_read2)
			{
		printf("%s", "I/O fail\n");
		return 1;
	}
	row_ptr.resize(n + 1);
	col_ind.resize(nz);
	is_read3 = fread(&row_ptr[0], sizeof(I), n + 1, fpin);
	is_read4 = fread(&col_ind[0], sizeof(I), nz, fpin);

	if (!is_read3 || !is_read4)
			{
		printf("%s", "I/O fail\n");
		return 1;
	}
	return 0;
}

//reader is for ints and double
int read_data_amgx_csr_bin(FILE* fpin,
									int n,
									int nz,
									std::vector<int> & row_ptr,
									std::vector<int> & col_ind,
									std::vector<double>& val
									)
									{
	size_t is_read1, is_read2, is_read3;
	is_read1 = fread(&row_ptr[0], sizeof(std::vector<int>::value_type), n + 1, fpin);
	is_read2 = fread(&col_ind[0], sizeof(std::vector<int>::value_type), nz, fpin);
	is_read3 = fread(&val[0], sizeof(std::vector<double>::value_type), nz, fpin);

	if (!is_read1 || !is_read2 || !is_read3)
			{
		printf("%s", "I/O fail\n");
		return 1;
	}
	return 0;
}

int read_data_amgx_csr_bin_rhs(FILE* fpin,
											int n,
											int nz,
											std::vector<int> & row_ptr,
											std::vector<int> & col_ind,
											std::vector<double>& val,
											std::vector<double>& rhs
											)
											{
	size_t is_read1, is_read2, is_read3, is_read4;
	is_read1 = fread(&row_ptr[0], sizeof(std::vector<int>::value_type), n + 1, fpin);
	is_read2 = fread(&col_ind[0], sizeof(std::vector<int>::value_type), nz, fpin);
	is_read3 = fread(&val[0], sizeof(std::vector<double>::value_type), nz, fpin);
	is_read4 = fread(&rhs[0], sizeof(std::vector<double>::value_type), n, fpin);

	if (!is_read1 || !is_read2 || !is_read3 || !is_read4)
			{
		printf("%s", "I/O fail\n");
		return 1;
	}
	return 0;
}

//reader is for ints and double
int read_data_amgx_csr_bin(FILE* fpin,
									int n,
									int nz,
									std::vector<int> & row_ptr,
									std::vector<int> & col_ind,
									std::vector<float>& val
									)
									{
	size_t is_read1, is_read2, is_read3;
	is_read1 = fread(&row_ptr[0], sizeof(std::vector<int>::value_type), n + 1, fpin);
	is_read2 = fread(&col_ind[0], sizeof(std::vector<int>::value_type), nz, fpin);

	double* t_storage = new double[std::max(n, nz)];
	is_read3 = fread(t_storage, sizeof(double), nz, fpin);
	for (int i = 0; i < nz; i++)
			{
		val[i] = static_cast<float>(t_storage[i]);
	}
	delete[] t_storage;

	if (!is_read1 || !is_read2 || !is_read3)
			{
		printf("%s", "I/O fail\n");
		return 1;
	}
	return 0;
}

int read_data_amgx_csr_bin_rhs(FILE* fpin,
											int n,
											int nz,
											std::vector<int> & row_ptr,
											std::vector<int> & col_ind,
											std::vector<float>& val,
											std::vector<float>& rhs
											)
											{
	size_t is_read1, is_read2, is_read3, is_read4;
	is_read1 = fread(&row_ptr[0], sizeof(std::vector<int>::value_type), n + 1, fpin);
	is_read2 = fread(&col_ind[0], sizeof(std::vector<int>::value_type), nz, fpin);
	double* t_storage = new double[std::max(n, nz)];
	is_read3 = fread(t_storage, sizeof(double), nz, fpin);
	for (int i = 0; i < nz; i++)
			{
		val[i] = static_cast<float>(t_storage[i]);
	}
	is_read4 = fread(t_storage, sizeof(double), n, fpin);
	for (int i = 0; i < n; i++)
			{
		rhs[i] = static_cast<float>(t_storage[i]);
	}
	delete[] t_storage;

	if (!is_read1 || !is_read2 || !is_read3 || !is_read4)
			{
		printf("%s", "I/O fail\n");
		return 1;
	}
	return 0;
}

//read binary vector from file
int read_binary_vector(FILE* fpin,
								int n,
								std::vector<float>& val
								)
								{
	size_t is_read1;

	double* t_storage = new double[n];
	is_read1 = fread(t_storage, sizeof(double), n, fpin);
	for (int i = 0; i < n; i++)
			{
		if (t_storage[i] == DBL_MAX)
			val[i] = FLT_MAX;
		else if (t_storage[i] == -DBL_MAX)
			val[i] = -FLT_MAX;
		else
			val[i] = static_cast<float>(t_storage[i]);
	}
	delete[] t_storage;

	if (is_read1 != (size_t) n)
			{
		printf("%s", "I/O fail\n");
		return 1;
	}
	return 0;
}

int read_binary_vector(FILE* fpin,
								int n,
								std::vector<double>& val
								)
								{
	size_t is_read1;

	is_read1 = fread(&val[0], sizeof(double), n, fpin);

	if (is_read1 != (size_t) n)
			{
		printf("%s", "I/O fail\n");
		return 1;
	}
	return 0;
}

int read_binary_vector(FILE* fpin,
								int n,
								std::vector<int>& val
								)
								{
	size_t is_read1;

	is_read1 = fread(&val[0], sizeof(int), n, fpin);

	if (is_read1 != (size_t) n)
			{
		printf("%s", "I/O fail\n");
		return 1;
	}
	return 0;
}

//read in as one based
template<typename IndexType_, typename ValueType_>
void init_MatrixMarket(IndexType_ base,
								const char *filename,
								bool edges_only, //assumes value is 1
								bool transpose, //parameter to run on A or A'
								IndexType_ &n,
								IndexType_ &m,
								IndexType_ &nnz,
								std::vector<ValueType_> &csrVal,
								std::vector<IndexType_> &csrColInd,
								std::vector<IndexType_> &csrRowInd)
								{
	FILE *inputFile = fopen(filename, "r");
	if (inputFile == NULL)
	{
		std::cerr << "ERROR: File path not valid!" << std::endl;
		exit(EXIT_FAILURE);
	}
	std::vector<Mat<IndexType_, ValueType_> > matrix;
	readMatrixMarketFile<IndexType_, ValueType_>(filename, m, n, nnz,
																matrix,
																edges_only);

	Mat<IndexType_, ValueType_> compare(transpose);
	std::sort(matrix.begin(), matrix.end(), compare);
	csrVal.resize(nnz);
	csrColInd.resize(nnz);
	csrRowInd.resize(nnz);
	for (int k = 0; k < nnz; ++k)
			{
		csrVal[k] = matrix[k].val;
		csrColInd[k] = (transpose) ? matrix[k].i : matrix[k].j; //doing the transpose
		csrRowInd[k] = (transpose) ? matrix[k].j : matrix[k].i;
	}
	if (base == 0) //always give base 0
			{
		for (int i = 0; i < nnz; ++i)
				{
			csrColInd[i] -= 1; //get zero based
			csrRowInd[i] -= 1;
		}
	}
	fclose(inputFile);
}
/*template<typename val_t>
 bool almost_equal (std::vector<val_t> & a, std::vector<val_t> & b, val_t epsilon)
 {
 if (a.size() != b.size()) return false;
 bool passed = true;
 std::vector<val_t>::iterator itb=b.begin();
 for (std::vector<val_t>::iterator ita = a.begin() ; ita != a.end(); ++ita)
 {
 if (fabs(*ita - *itb) > epsilon)
 {
 printf("At ( %ld ) : x1=%lf | x2=%lf\n",ita-a.begin(), *ita,*itb);
 passed = false;
 }
 ++itb;
 }
 return passed;
 }*/


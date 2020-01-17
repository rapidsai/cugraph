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
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "include/nvgraph_error.hxx"
#include "include/nvgraph_vector_kernels.hxx"

#include "include/debug_macros.h"

namespace nvgraph
{

void check_size(size_t sz)
{
	if (sz>INT_MAX) FatalError("Vector larger than INT_MAX", NVGRAPH_ERR_BAD_PARAMETERS);
}
template <typename ValueType_>
void nrm1_raw_vec (ValueType_* vec, size_t n, ValueType_* res, cudaStream_t stream)
{
    thrust::device_ptr<ValueType_> dev_ptr(vec);
    *res = thrust::reduce(dev_ptr, dev_ptr+n);
    cudaCheckError();
}

template <typename ValueType_>
void fill_raw_vec (ValueType_* vec, size_t n , ValueType_ value, cudaStream_t stream)
{
    thrust::device_ptr<ValueType_> dev_ptr(vec);
    thrust::fill(dev_ptr, dev_ptr + n, value);
    cudaCheckError();
}

template <typename ValueType_>
void dump_raw_vec (ValueType_* vec, size_t n, int offset, cudaStream_t stream)
{
#ifdef DEBUG
    thrust::device_ptr<ValueType_> dev_ptr(vec);
    COUT().precision(15);
    COUT() << "sample size = "<< n << ", offset = "<< offset << std::endl;
    thrust::copy(dev_ptr+offset,dev_ptr+offset+n, std::ostream_iterator<ValueType_>(COUT(), " "));
    cudaCheckError();
    COUT() << std::endl;
#endif
}

template <typename ValueType_>
__global__ void flag_zeroes_kernel(int num_vertices, ValueType_* vec, int* flags)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int r = tidx; r < num_vertices; r += blockDim.x * gridDim.x)
    {
        if (vec[r] != 0.0)
            flags[r] = 1; // NOTE 2 : alpha*0 + (1-alpha)*1 = (1-alpha)
        else
            flags[r] = 0;
    }
}
template <typename ValueType_> 
 __global__ void dmv0_kernel(const ValueType_ * __restrict__ D, const ValueType_ * __restrict__ x, ValueType_ * __restrict__ y, int n) 
 {
   //y=D*x
   int tidx = blockIdx.x*blockDim.x + threadIdx.x ;
   for (int i = tidx; i < n; i += blockDim.x * gridDim.x)
        y[i] = D[i]*x[i];
}
template <typename ValueType_> 
 __global__ void dmv1_kernel(const ValueType_ * __restrict__ D, const ValueType_ * __restrict__ x, ValueType_ * __restrict__ y, int n) 
 {
   // y+=D*x
   int tidx = blockIdx.x*blockDim.x + threadIdx.x ;
   for (int i = tidx; i < n; i += blockDim.x * gridDim.x)
        y[i] += D[i]*x[i];
}
template<typename ValueType_>
void copy_vec(ValueType_ *vec1, size_t n, ValueType_ *res, cudaStream_t stream)
{
    thrust::device_ptr<ValueType_> dev_ptr(vec1);
    thrust::device_ptr<ValueType_> res_ptr(res);
#ifdef DEBUG
    //COUT() << "copy "<< n << " elements" << std::endl;
#endif
    thrust::copy_n(dev_ptr, n, res_ptr);
    cudaCheckError();
    //dump_raw_vec (res, n, 0);
}

template <typename ValueType_>
void flag_zeros_raw_vec(size_t num_vertices, ValueType_* vec, int* flags, cudaStream_t stream)
{
    int items_per_thread = 4;
    int num_threads = 128;
    int max_grid_size = 4096;
    check_size(num_vertices);
    int n = static_cast<int>(num_vertices);
    int num_blocks = std::min(max_grid_size, (n/(items_per_thread*num_threads))+1);
    flag_zeroes_kernel<<<num_blocks, num_threads, 0, stream>>>(num_vertices, vec, flags);
    cudaCheckError();
}

template <typename ValueType_>
void dmv (size_t num_vertices, ValueType_ alpha, ValueType_* D, ValueType_* x, ValueType_ beta, ValueType_* y, cudaStream_t stream)
{
    int items_per_thread = 4;
    int num_threads = 128;
    int max_grid_size = 4096;
    check_size(num_vertices);
    int n = static_cast<int>(num_vertices);
    int num_blocks = std::min(max_grid_size, (n/(items_per_thread*num_threads))+1);
    if (alpha ==1.0 && beta == 0.0)
        dmv0_kernel<<<num_blocks, num_threads, 0, stream>>>(D, x, y, n);
    else if (alpha ==1.0 && beta == 1.0)
        dmv1_kernel<<<num_blocks, num_threads, 0, stream>>>(D, x, y, n);
    else
        FatalError("Not implemented case of y = D*x", NVGRAPH_ERR_BAD_PARAMETERS);

    cudaCheckError();
}

template <typename IndexType_, typename ValueType_>
void set_connectivity( size_t n, IndexType_ root, ValueType_ self_loop_val, ValueType_ unreachable_val, ValueType_* res, cudaStream_t stream)
{
    fill_raw_vec(res, n, unreachable_val);
    cudaMemcpy(&res[root], &self_loop_val, sizeof(self_loop_val), cudaMemcpyHostToDevice);
    cudaCheckError();        
}

template void nrm1_raw_vec <float> (float* vec, size_t n, float* res, cudaStream_t stream);
template void nrm1_raw_vec <double> (double* vec, size_t n, double* res, cudaStream_t stream);

template void dmv <float>(size_t num_vertices, float alpha, float* D, float* x, float beta, float* y, cudaStream_t stream);
template void dmv <double>(size_t num_vertices, double alpha, double* D, double* x, double beta, double* y, cudaStream_t stream);

template void set_connectivity <int, float> (size_t n, int root, float self_loop_val, float unreachable_val, float* res, cudaStream_t stream);
template void set_connectivity <int, double>(size_t n, int root, double self_loop_val, double unreachable_val, double* res, cudaStream_t stream);

template void flag_zeros_raw_vec <float>(size_t num_vertices, float* vec, int* flags, cudaStream_t stream);
template void flag_zeros_raw_vec <double>(size_t num_vertices, double* vec, int* flags, cudaStream_t stream);

template void fill_raw_vec<float> (float* vec, size_t n, float value, cudaStream_t stream);
template void fill_raw_vec<double> (double* vec, size_t n, double value, cudaStream_t stream);
template void fill_raw_vec<int> (int* vec, size_t n, int value, cudaStream_t stream);
template void fill_raw_vec<char> (char* vec, size_t n, char value, cudaStream_t stream);

template void copy_vec<float>(float * vec1, size_t n, float *res, cudaStream_t stream);
template void copy_vec<double>(double * vec1, size_t n, double *res, cudaStream_t stream);
template void copy_vec<int>(int * vec1, size_t n, int *res, cudaStream_t stream);
template void copy_vec<char>(char * vec1, size_t n, char *res, cudaStream_t stream);

template void dump_raw_vec<float> (float* vec, size_t n, int off, cudaStream_t stream);
template void dump_raw_vec<double> (double* vec, size_t n, int off, cudaStream_t stream);
template void dump_raw_vec<int> (int* vec, size_t n, int off, cudaStream_t stream);
template void dump_raw_vec<char> (char* vec, size_t n, int off, cudaStream_t stream);
} // end namespace nvgraph


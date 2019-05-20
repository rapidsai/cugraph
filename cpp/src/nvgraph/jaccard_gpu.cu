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
// Jaccard symilarity edge weights
// Author: Alexandre Fender afender@nvidia.com and Maxim Naumov.

#include "include/graph_utils.cuh"
#include "include/jaccard_gpu.cuh"

namespace nvlouvain 
{

//#define CUDA_MAX_BLOCKS 65535
//#define CUDA_MAX_KERNEL_THREADS 256  //kernel will launch at most 256 threads per block
//#define DEFAULT_MASK 0xffffffff

// Volume of neighboors (*weight_s)
template<bool weighted, typename T>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
jaccard_row_sum(int n, int e, int *csrPtr, int *csrInd, T *v, T *work) {
    int row,start,end,length;
    T sum;

    for (row=threadIdx.y+blockIdx.y*blockDim.y; row<n; row+=gridDim.y*blockDim.y) {
        start = csrPtr[row];
        end   = csrPtr[row+1];
        length= end-start;
        //compute row sums 
        if (weighted) {
            sum = parallel_prefix_sum(length, csrInd + start, v); 
            if (threadIdx.x == 0) work[row] = sum;
        }
        else {
            work[row] = (T)length;
        }
    }
}

// Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
template<bool weighted, typename T>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
jaccard_is(int n, int e, int *csrPtr, int *csrInd, T *v, T *work, T *weight_i, T *weight_s) {
    int i,j,row,col,Ni,Nj;
    int ref,cur,ref_col,cur_col,match;
    T ref_val;

    for (row=threadIdx.z+blockIdx.z*blockDim.z; row<n; row+=gridDim.z*blockDim.z) {  
        for (j=csrPtr[row]+threadIdx.y+blockIdx.y*blockDim.y; j<csrPtr[row+1]; j+=gridDim.y*blockDim.y) { 
            col = csrInd[j];
            //find which row has least elements (and call it reference row)
            Ni = csrPtr[row+1] - csrPtr[row];
            Nj = csrPtr[col+1] - csrPtr[col];
            ref= (Ni < Nj) ? row : col;
            cur= (Ni < Nj) ? col : row;

            //compute new sum weights
            weight_s[j] = work[row] + work[col];

            //compute new intersection weights 
            //search for the element with the same column index in the reference row
            for (i=csrPtr[ref]+threadIdx.x+blockIdx.x*blockDim.x; i<csrPtr[ref+1]; i+=gridDim.x*blockDim.x) {
                match  =-1;           
                ref_col = csrInd[i];
                if (weighted) {
                    ref_val = v[ref_col];
                }
                else {
                    ref_val = 1.0;
                }
         
                //binary search (column indices are sorted within each row)
                int left = csrPtr[cur]; 
                int right= csrPtr[cur+1]-1; 
                while(left <= right){
                    int middle = (left+right)>>1; 
                    cur_col= csrInd[middle];
                    if (cur_col > ref_col) {
                        right=middle-1;
                    }
                    else if (cur_col < ref_col) {
                        left=middle+1;
                    }
                    else {
                        match = middle; 
                        break; 
                    }
                }            

                //if the element with the same column index in the reference row has been found
                if (match != -1){
                    atomicAdd(&weight_i[j],ref_val);
                }
            }
        }
    }
}

//Jaccard  weights (*weight)
template<bool weighted, typename T>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
jaccard_jw(int n, int e, int *csrPtr, int *csrInd, T *csrVal, T *v, T gamma, T *weight_i, T *weight_s, T *weight_j) {
    int j;
    T Wi,Ws,Wu;

    for (j=threadIdx.x+blockIdx.x*blockDim.x; j<e; j+=gridDim.x*blockDim.x) {  
        Wi =  weight_i[j];
        Ws =  weight_s[j];
        Wu =  Ws - Wi;
        weight_j[j] = (gamma*csrVal[j])* (Wi/Wu); 
    }
}
template<bool weighted, typename T>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
jaccard_jw(int n, int e, int *csrPtr, int *csrInd, T *v, T *weight_i, T *weight_s, T *weight_j) {
    int j;
    T Wi,Ws,Wu;

    for (j=threadIdx.x+blockIdx.x*blockDim.x; j<e; j+=gridDim.x*blockDim.x) {  
        Wi =  weight_i[j];
        Ws =  weight_s[j];
        Wu =  Ws - Wi;
        weight_j[j] = (Wi/Wu); 
    }
}


template <bool weighted, typename T> 
int jaccard(int n, int e, int *csrPtr, int *csrInd, T * csrVal, T *v, T *work, T gamma, T *weight_i, T *weight_s, T *weight_j) {
    dim3 nthreads, nblocks;
    int y=4;
    
    //setup launch configuration
    nthreads.x = 32/y; 
    nthreads.y = y; 
    nthreads.z = 1; 
    nblocks.x  = 1; 
    nblocks.y  = min((n + nthreads.y - 1)/nthreads.y,CUDA_MAX_BLOCKS); 
    nblocks.z  = 1; 
    //launch kernel
    jaccard_row_sum<weighted,T><<<nblocks,nthreads>>>(n,e,csrPtr,csrInd,v,work);
    fill(e,weight_i,(T)0.0);
    //setup launch configuration
    nthreads.x = 32/y;
    nthreads.y = y;
    nthreads.z = 8;
    nblocks.x  = 1;
    nblocks.y  = 1;
    nblocks.z  = min((n + nthreads.z - 1)/nthreads.z,CUDA_MAX_BLOCKS); //1; 
    //launch kernel
    jaccard_is<weighted,T><<<nblocks,nthreads>>>(n,e,csrPtr,csrInd,v,work,weight_i,weight_s);

    //setup launch configuration
    nthreads.x = min(e,CUDA_MAX_KERNEL_THREADS); 
    nthreads.y = 1; 
    nthreads.z = 1;  
    nblocks.x  = min((e + nthreads.x - 1)/nthreads.x,CUDA_MAX_BLOCKS); 
    nblocks.y  = 1; 
    nblocks.z  = 1;
    //launch kernel
    if (csrVal != NULL)
        jaccard_jw<weighted,T><<<nblocks,nthreads>>>(n,e,csrPtr,csrInd,csrVal,v,gamma,weight_i,weight_s,weight_j);
    else
        jaccard_jw<weighted,T><<<nblocks,nthreads>>>(n,e,csrPtr,csrInd,v,weight_i,weight_s,weight_j);
       
    return 0;
}

//template int jaccard<true, half>  ( int n, int e, int *csrPtr, int *csrInd, half *csrVal, half *v, half *work, half gamma, half *weight_i, half *weight_s, half *weight_j);
//template int jaccard<false, half> ( int n, int e, int *csrPtr, int *csrInd, half *csrVal, half *v, half *work, half gamma, half *weight_i, half *weight_s, half *weight_j);

template int jaccard<true, float>  ( int n, int e, int *csrPtr, int *csrInd, float *csrVal, float *v, float *work, float gamma, float *weight_i, float *weight_s, float *weight_j);
template int jaccard<false, float> ( int n, int e, int *csrPtr, int *csrInd, float *csrVal, float *v, float *work, float gamma, float *weight_i, float *weight_s, float *weight_j);

template int jaccard<true, double>  (int n, int e, int *csrPtr, int *csrInd, double *csrVal, double *v, double *work, double gamma, double *weight_i, double *weight_s, double *weight_j);
template int jaccard<false, double> (int n, int e, int *csrPtr, int *csrInd, double *csrVal, double *v, double *work, double gamma, double *weight_i, double *weight_s, double *weight_j);

} //namespace nvga

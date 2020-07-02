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
namespace nvgraph {
//This file contains the atomic operations for floats and doubles from cusparse/src/cusparse_atomics.h

static __inline__ __device__ double atomicFPAdd(double *addr, double val)
{
// atomicAdd for double starts with sm_60
#if __CUDA_ARCH__ >= 600
    return atomicAdd( addr, val );
#else
    unsigned long long old = __double_as_longlong( addr[0] ), assumed;

    do
    {
        assumed = old;
        old = atomicCAS( (unsigned long long *) addr, assumed, __double_as_longlong( val + __longlong_as_double( assumed ) ) );
    }
    while ( assumed != old );

    return old;
#endif
} 

// atomicAdd for float starts with sm_20
static __inline__ __device__ float atomicFPAdd(float *addr, float val)
{
    return atomicAdd( addr, val );
}

static __inline__ __device__ double atomicFPMin(double *addr, double val)
{
    double old, assumed;
    old=*addr; 
    do{
        assumed = old;
        old     = __longlong_as_double(atomicCAS((unsigned long long int *)addr, __double_as_longlong(assumed),
                                                 __double_as_longlong(min(val,assumed))));
    } while (__double_as_longlong(assumed) != __double_as_longlong(old));
    return old;
} 

/* atomic addition: based on Nvidia Research atomic's tricks from cusparse */
static __inline__ __device__ float atomicFPMin(float *addr, float val)
{       
    float old, assumed;
    old=*addr;
    do{
        assumed = old;
        old     = int_as_float(atomicCAS((int *)addr, float_as_int(assumed),float_as_int(min(val,assumed))));
    } while (float_as_int(assumed) != float_as_int(old));

    return old;
}

static __inline__ __device__ double atomicFPMax(double *addr, double val)
{
    double old, assumed;
    old=*addr; 
    do{
        assumed = old;
        old     = __longlong_as_double(atomicCAS((unsigned long long int *)addr, __double_as_longlong(assumed),
                                                 __double_as_longlong(max(val,assumed))));
    } while (__double_as_longlong(assumed) != __double_as_longlong(old));
    return old;
} 

/* atomic addition: based on Nvidia Research atomic's tricks from cusparse */
static __inline__ __device__ float atomicFPMax(float *addr, float val)
{       
    float old, assumed;
    old=*addr;
    do{
        assumed = old;
        old     = int_as_float(atomicCAS((int *)addr, float_as_int(assumed),float_as_int(max(val,assumed))));
    } while (float_as_int(assumed) != float_as_int(old));

    return old;
}

static __inline__ __device__ double atomicFPOr(double *addr, double val)
{
    double old, assumed;
    old=*addr; 
    do{
        assumed = old;
        old     = __longlong_as_double(atomicCAS((unsigned long long int *)addr, __double_as_longlong(assumed),
                                                 __double_as_longlong((bool)val | (bool)assumed)));
    } while (__double_as_longlong(assumed) != __double_as_longlong(old));
    return old;
} 

/* atomic addition: based on Nvidia Research atomic's tricks from cusparse */
static __inline__ __device__ float atomicFPOr(float *addr, float val)
{       
    float old, assumed;
    old=*addr;
    do{
        assumed = old;
        old     = int_as_float(atomicCAS((int *)addr, float_as_int(assumed),float_as_int((bool)val | (bool)assumed)));
    } while (float_as_int(assumed) != float_as_int(old));

    return old;
}

static __inline__ __device__ double atomicFPLog(double *addr, double val)
{
    double old, assumed;
    old=*addr; 
    do{
        assumed = old;
        old     = __longlong_as_double(atomicCAS((unsigned long long int *)addr, __double_as_longlong(assumed),
                                                 __double_as_longlong(-log(exp(-val)+exp(-assumed)))));
    } while (__double_as_longlong(assumed) != __double_as_longlong(old));
    return old;
} 

/* atomic addition: based on Nvidia Research atomic's tricks from cusparse */
static __inline__ __device__ float atomicFPLog(float *addr, float val)
{       
    float old, assumed;
    old=*addr;
    do{
        assumed = old;
        old     = int_as_float(atomicCAS((int *)addr, float_as_int(assumed),float_as_int(-logf(expf(-val)+expf(-assumed)))));
    } while (float_as_int(assumed) != float_as_int(old));

    return old;
}

} //end anmespace nvgraph


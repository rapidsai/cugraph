#pragma once

///#define DEBUG_NO_RMM

#include <sstream>
#include <stdexcept>

#define RMM_TRY_THROW( call )  if ((call)!=RMM_SUCCESS) \
    {                                                   \
      std::stringstream ss;                             \
      ss << "ERROR: RMM runtime call  " << #call        \
         << cudaGetErrorString(cudaGetLastError());     \
      throw std::runtime_error(ss.str());               \
    }

#ifdef DEBUG_NO_RMM

#include <thrust/device_malloc_allocator.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <thrust/execution_policy.h>

template<typename T>
//using rmm_allocator = thrust::device_malloc_allocator<T>;
class rmm_allocator : public thrust::device_malloc_allocator<T>
{
  public:
    using value_type = T;

    rmm_allocator(cudaStream_t stream = 0) : stream(stream) {}
    ~rmm_allocator() {}

private:
  	cudaStream_t stream;
};

using rmm_temp_allocator = rmm_allocator<char>; // Use this alias for thrust::cuda::par(allocator).on(stream)

#define ALLOC_TRY(ptr, sz, stream){            \
    if (stream == nullptr) ;                      \
    cudaMalloc((ptr), (sz));                   \
}

#define ALLOC_MANAGED_TRY(ptr, sz, stream){    \
    if (stream == nullptr) ;                      \
    cudaMallocManaged((ptr), (sz));            \
}

  //#define REALLOC_TRY(ptr, new_sz, stream)

#define ALLOC_FREE_TRY(ptr, stream){                \
    if (stream == nullptr) ;                      \
    cudaFree( (ptr) );                              \
}
#else

#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

using rmm_temp_allocator = rmm_allocator<char>;

#define ALLOC_TRY( ptr, sz, stream ){                   \
      RMM_TRY_THROW( RMM_ALLOC((ptr), (sz), (stream)) ) \
    }

//TODO: change this when RMM alloc managed will be available !!!!!
#define ALLOC_MANAGED_TRY(ptr, sz, stream){         \
  RMM_TRY_THROW( RMM_ALLOC((ptr), (sz), (stream)) ) \
}

#define REALLOC_TRY(ptr, new_sz, stream){             \
  RMM_TRY_THROW( RMM_REALLOC((ptr), (sz), (stream)) ) \
}

#define ALLOC_FREE_TRY(ptr, stream){                \
  RMM_TRY_THROW( RMM_FREE( (ptr), (stream) ) )  \
}

#endif


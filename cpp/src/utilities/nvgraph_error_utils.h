#ifndef NVGRAPH_ERRORUTILS_H
#define NVGRAPH_ERRORUTILS_H

#include <nvgraph/nvgraph.h>

#ifdef VERBOSE
#define NVG_TRY(call)                                           \
{                                                               \
  nvgraphStatus_t err_code = (call);                            \
  if (err_code != NVGRAPH_STATUS_SUCCESS) {                     \
      switch (err_code) {                                       \
        case NVGRAPH_STATUS_SUCCESS:                            \
          return GDF_SUCCESS;                                   \
        case NVGRAPH_STATUS_NOT_INITIALIZED:                    \
          return GDF_INVALID_API_CALL;                          \
        case NVGRAPH_STATUS_INVALID_VALUE:                      \
          return GDF_INVALID_API_CALL;                          \
        case NVGRAPH_STATUS_TYPE_NOT_SUPPORTED:                 \
          return GDF_UNSUPPORTED_DTYPE;                         \
        case NVGRAPH_STATUS_GRAPH_TYPE_NOT_SUPPORTED:           \
          return GDF_INVALID_API_CALL;                          \
        default:                                                \
          return GDF_CUDA_ERROR;                                \
      }                                                         \
  }                                                             \
}
#else
#define NVG_TRY(call)                                           \
{                                                               \
  nvgraphStatus_t err_code = (call);                            \
  if (err_code != NVGRAPH_STATUS_SUCCESS) {                     \
      switch (err_code) {                                       \
        case NVGRAPH_STATUS_NOT_INITIALIZED:                    \
          std::cerr << "nvGRAPH not initialized";               \
          return GDF_CUDA_ERROR;                                \
        case NVGRAPH_STATUS_ALLOC_FAILED:                       \
          std::cerr << "nvGRAPH alloc failed";                  \
          return GDF_CUDA_ERROR;                                \
        case NVGRAPH_STATUS_INVALID_VALUE:                      \
          std::cerr << "nvGRAPH invalid value";                 \
          return GDF_CUDA_ERROR;                                \
        case NVGRAPH_STATUS_ARCH_MISMATCH:                      \
          std::cerr << "nvGRAPH arch mismatch";                 \
          return GDF_CUDA_ERROR;                                \
        case NVGRAPH_STATUS_MAPPING_ERROR:                      \
          std::cerr << "nvGRAPH mapping error";                 \
          return GDF_CUDA_ERROR;                                \
        case NVGRAPH_STATUS_EXECUTION_FAILED:                   \
          std::cerr << "nvGRAPH execution failed";              \
          return GDF_CUDA_ERROR;                                \
        case NVGRAPH_STATUS_INTERNAL_ERROR:                     \
          std::cerr << "nvGRAPH internal error";                \
          return GDF_CUDA_ERROR;                                \
        case NVGRAPH_STATUS_TYPE_NOT_SUPPORTED:                 \
          std::cerr << "nvGRAPH type not supported";            \
          return GDF_CUDA_ERROR;                                \
        case NVGRAPH_STATUS_NOT_CONVERGED:                      \
          std::cerr << "nvGRAPH algorithm failed to converge";  \
          return GDF_CUDA_ERROR;                                \
        case NVGRAPH_STATUS_GRAPH_TYPE_NOT_SUPPORTED:           \
          std::cerr << "nvGRAPH graph type not supported";      \
          return GDF_CUDA_ERROR;                                \
        default:                                                \
          std::cerr << "Unknown nvGRAPH Status";                \
          return GDF_CUDA_ERROR;                                \
      }                                                         \
    }                                                           \
}
#endif

#endif

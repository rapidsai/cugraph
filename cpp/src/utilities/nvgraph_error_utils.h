#ifndef NVGRAPH_ERRORUTILS_H
#define NVGRAPH_ERRORUTILS_H

#include <nvgraph/nvgraph.h>

#define NVG_TRY(call)                                           \
{                                                               \
  nvgraphStatus_t err_code = (call);                            \
  if (err_code != NVGRAPH_STATUS_SUCCESS) {                     \
      switch (err_code) {                                       \
        case NVGRAPH_STATUS_NOT_INITIALIZED:                    \
          CUGRAPH_FAIL("nvGRAPH not initialized");               \
        case NVGRAPH_STATUS_ALLOC_FAILED:                       \
          CUGRAPH_FAIL("nvGRAPH alloc failed");                  \
        case NVGRAPH_STATUS_INVALID_VALUE:                      \
          CUGRAPH_FAIL("nvGRAPH invalid value");                \
        case NVGRAPH_STATUS_ARCH_MISMATCH:                      \
          CUGRAPH_FAIL("nvGRAPH arch mismatch");                 \
        case NVGRAPH_STATUS_MAPPING_ERROR:                      \
          CUGRAPH_FAIL("nvGRAPH mapping error");                 \
        case NVGRAPH_STATUS_EXECUTION_FAILED:                   \
          CUGRAPH_FAIL("nvGRAPH execution failed");              \
        case NVGRAPH_STATUS_INTERNAL_ERROR:                     \
          CUGRAPH_FAIL("nvGRAPH internal error");                \
        case NVGRAPH_STATUS_TYPE_NOT_SUPPORTED:                 \
          CUGRAPH_FAIL("nvGRAPH type not supported");            \
        case NVGRAPH_STATUS_NOT_CONVERGED:                      \
          CUGRAPH_FAIL("nvGRAPH algorithm failed to converge");  \
        case NVGRAPH_STATUS_GRAPH_TYPE_NOT_SUPPORTED:           \
          CUGRAPH_FAIL("nvGRAPH graph type not supported");      \
        default:                                                \
          CUGRAPH_FAIL("Unknown nvGRAPH Status");                \
      }                                                         \
    }                                                           \
}

#endif

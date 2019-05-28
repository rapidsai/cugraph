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

#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cfloat>
#include <vector>

#include <cusolverDn.h>

#include <nvgraph/nvgraph.h>   // public header **This is NVGRAPH C API**

#include "include/nvlouvain.cuh"
#include "include/jaccard_gpu.cuh"
#include "include/nvgraph_error.hxx"
#include "include/rmm_shared_ptr.hxx"
#include "include/valued_csr_graph.hxx"
#include "include/multi_valued_csr_graph.hxx"
#include "include/nvgraph_vector.hxx"
#include "include/nvgraph_cusparse.hxx"
#include "include/nvgraph_cublas.hxx"
#include "include/nvgraph_csrmv.hxx"
#include "include/pagerank.hxx"
#include "include/arnoldi.hxx"
#include "include/sssp.hxx"
#include "include/widest_path.hxx"
#include "include/partition.hxx"
#include "include/nvgraph_convert.hxx"
#include "include/size2_selector.hxx"
#include "include/modularity_maximization.hxx"
#include "include/bfs.hxx"
#include "include/triangles_counting.hxx"
#include "include/csrmv_cub.h"
#include "include/nvgraphP.h"  // private header, contains structures, and potentially other things, used in the public C API that should never be exposed.
#include "include/nvgraph_experimental.h"  // experimental header, contains hidden API entries, can be shared only under special circumstances without reveling internal things
#include "include/debug_macros.h"
#include "include/2d_partitioning.h"
#include "include/bfs2d.hxx"

static inline int check_context(const nvgraphHandle_t h) {
    int ret = 0;
    if (h == NULL || !h->nvgraphIsInitialized)
        ret = 1;
    return ret;
}

static inline int check_graph(const nvgraphGraphDescr_t d) {
    int ret = 0;
    if (d == NULL || d->graphStatus == IS_EMPTY)
        ret = 1;
    return ret;
}
static inline int check_topology(const nvgraphGraphDescr_t d) {
    int ret = 0;
    if (d->graphStatus == IS_EMPTY)
        ret = 1;
    return ret;
}

static inline int check_int_size(size_t sz) {
    int ret = 0;
    if (sz >= INT_MAX)
        ret = 1;
    return ret;
}

static inline int check_int_ptr(const int* p) {
    int ret = 0;
    if (!p)
        ret = 1;
    return ret;
}

static inline int check_uniform_type_array(const cudaDataType_t * t, size_t sz) {
    int ret = 0;
    cudaDataType_t uniform_type = t[0];
    for (size_t i = 1; i < sz; i++)
            {
        if (t[i] != uniform_type)
            ret = 1;
    }
    return ret;
}

template<typename T>
bool check_ptr(const T* p) {
    bool ret = false;
    if (!p)
        ret = true;
    return ret;
}

namespace nvgraph
{

//TODO: make those template functions in a separate header to be included by both
//graph_extractor.cu and nvgraph.cpp;
//right now this header does not exist and including graph_concrete_visitors.hxx
//doesn't compile because of the Thrust code;
//
    extern CsrGraph<int>* extract_subgraph_by_vertices(CsrGraph<int>& graph,
                                                       int* pV,
                                                       size_t n,
                                                       cudaStream_t stream);
    extern MultiValuedCsrGraph<int, float>* extract_subgraph_by_vertices(MultiValuedCsrGraph<int,
                                                                         float>& graph,
                                                                         int* pV,
                                                                         size_t n,
                                                                         cudaStream_t stream);
    extern MultiValuedCsrGraph<int, double>* extract_subgraph_by_vertices(MultiValuedCsrGraph<int,
                                                                          double>& graph,
                                                                          int* pV,
                                                                          size_t n,
                                                                          cudaStream_t stream);

    extern CsrGraph<int>* extract_subgraph_by_edges(CsrGraph<int>& graph,
                                                    int* pV,
                                                    size_t n,
                                                    cudaStream_t stream);
    extern MultiValuedCsrGraph<int, float>* extract_subgraph_by_edges(MultiValuedCsrGraph<int, float>& graph,
                                                                      int* pV,
                                                                      size_t n,
                                                                      cudaStream_t stream);
    extern MultiValuedCsrGraph<int, double>* extract_subgraph_by_edges(MultiValuedCsrGraph<int,
                                                                       double>& graph,
                                                                       int* pV,
                                                                       size_t n,
                                                                       cudaStream_t stream);

    nvgraphStatus_t getCAPIStatusForError(NVGRAPH_ERROR err) {
        nvgraphStatus_t ret = NVGRAPH_STATUS_SUCCESS;

        switch (err) {
            case NVGRAPH_OK:
                ret = NVGRAPH_STATUS_SUCCESS;
                break;
            case NVGRAPH_ERR_BAD_PARAMETERS:
                ret = NVGRAPH_STATUS_INVALID_VALUE;
                break;
            case NVGRAPH_ERR_UNKNOWN:
                ret = NVGRAPH_STATUS_INTERNAL_ERROR;
                break;
            case NVGRAPH_ERR_CUDA_FAILURE:
                ret = NVGRAPH_STATUS_EXECUTION_FAILED;
                break;
            case NVGRAPH_ERR_THRUST_FAILURE:
                ret = NVGRAPH_STATUS_EXECUTION_FAILED;
                break;
            case NVGRAPH_ERR_IO:
                ret = NVGRAPH_STATUS_INTERNAL_ERROR;
                break;
            case NVGRAPH_ERR_NOT_IMPLEMENTED:
                ret = NVGRAPH_STATUS_INVALID_VALUE;
                break;
            case NVGRAPH_ERR_NO_MEMORY:
                ret = NVGRAPH_STATUS_ALLOC_FAILED;
                break;
            case NVGRAPH_ERR_NOT_CONVERGED:
                ret = NVGRAPH_STATUS_NOT_CONVERGED;
                break;
            default:
                ret = NVGRAPH_STATUS_INTERNAL_ERROR;
        }
        return ret;
    }

    extern "C" {
        const char* nvgraphStatusGetString(nvgraphStatus_t status) {
            switch (status) {
                case NVGRAPH_STATUS_SUCCESS:
                    return "Success";
                case NVGRAPH_STATUS_NOT_INITIALIZED:
                    return "nvGRAPH not initialized";
                case NVGRAPH_STATUS_ALLOC_FAILED:
                    return "nvGRAPH alloc failed";
                case NVGRAPH_STATUS_INVALID_VALUE:
                    return "nvGRAPH invalid value";
                case NVGRAPH_STATUS_ARCH_MISMATCH:
                    return "nvGRAPH arch mismatch";
                case NVGRAPH_STATUS_MAPPING_ERROR:
                    return "nvGRAPH mapping error";
                case NVGRAPH_STATUS_EXECUTION_FAILED:
                    return "nvGRAPH execution failed";
                case NVGRAPH_STATUS_INTERNAL_ERROR:
                    return "nvGRAPH internal error";
                case NVGRAPH_STATUS_TYPE_NOT_SUPPORTED:
                    return "nvGRAPH type not supported";
                case NVGRAPH_STATUS_NOT_CONVERGED:
                    return "nvGRAPH algorithm failed to converge";
                case NVGRAPH_STATUS_GRAPH_TYPE_NOT_SUPPORTED:
                    return "nvGRAPH graph type not supported";
                default:
                    return "Unknown nvGRAPH Status";
            }
        }
    }

    static nvgraphStatus_t nvgraphCreateMulti_impl(struct nvgraphContext **outCtx,
                                                   int numDevices,
                                                   int* _devices) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            // First, initialize NVGraph's context

            auto ctx = static_cast<struct nvgraphContext*>(calloc(1, sizeof(struct nvgraphContext)));
            if (ctx == nullptr) {
                FatalError("Cannot allocate NVGRAPH context.", NVGRAPH_ERR_UNKNOWN);
            }

            auto option = rmmOptions_t{};
            if (rmmIsInitialized(&option) == true) {
                if ((option.allocation_mode & PoolAllocation) != 0) {
                    FatalError("RMM does not support multi-GPUs with pool allocation, yet.", NVGRAPH_ERR_UNKNOWN);
                }
            }
            // if RMM is unintialized, RMM_ALLOC/RMM_FREE are just aliases for cudaMalloc/cudaFree

            ctx->stream = nullptr;
            ctx->nvgraphIsInitialized = true;

             if (outCtx != nullptr) {
                 *outCtx = ctx;
             }

            // Second, initialize Cublas and Cusparse (get_handle() creates a new handle
            // if there is no existing handle).

            nvgraph::Cusparse::get_handle();
            nvgraph::Cublas::get_handle();
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    static nvgraphStatus_t nvgraphCreate_impl(struct nvgraphContext **outCtx) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            // First, initialize NVGraph's context

            auto ctx = static_cast<struct nvgraphContext*>(calloc(1, sizeof(struct nvgraphContext)));
            if (ctx == nullptr) {
                FatalError("Cannot allocate NVGRAPH context.", NVGRAPH_ERR_UNKNOWN);
            }

            // Now NVGraph assumes that RMM is initialized outside NVGraph
            // if RMM is unintialized, RMM_ALLOC/RMM_FREE are just aliases for cudaMalloc/cudaFree

            ctx->stream = nullptr;
            ctx->nvgraphIsInitialized = true;

             if (outCtx != nullptr) {
                 *outCtx = ctx;
             }

            // Second, initialize Cublas and Cusparse (get_handle() creates a new handle
            // if there is no existing handle).

            nvgraph::Cusparse::get_handle();
            nvgraph::Cublas::get_handle();
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    static nvgraphStatus_t nvgraphDestroy_impl(nvgraphHandle_t handle) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle))
                FatalError("Cannot initialize memory manager.", NVGRAPH_ERR_NO_MEMORY);

            // First, destroy Cublas and Cusparse

            nvgraph::Cusparse::destroy_handle();
            nvgraph::Cublas::destroy_handle();

            // Second, destroy NVGraph's context

            free(handle);
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    static nvgraphStatus_t nvgraphCreateGraphDescr_impl(nvgraphHandle_t handle,
                                                        struct nvgraphGraphDescr **outGraphDescr) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            struct nvgraphGraphDescr *descrG = NULL;
            descrG = (struct nvgraphGraphDescr*) malloc(sizeof(*descrG));
            if (!descrG)
            {
                FatalError("Cannot allocate graph descriptor.", NVGRAPH_ERR_UNKNOWN);
            }
            descrG->graphStatus = IS_EMPTY;
            if (outGraphDescr)
            {
                *outGraphDescr = descrG;
            }
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    static nvgraphStatus_t nvgraphDestroyGraphDescr_impl(nvgraphHandle_t handle,
                                                         struct nvgraphGraphDescr *descrG) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG) {
                if (descrG->TT == NVGRAPH_2D_32I_32I) {
                    switch (descrG->T) {
                        case CUDA_R_32I: {
                            nvgraph::Matrix2d<int32_t, int32_t, int32_t>* m =
                                    static_cast<nvgraph::Matrix2d<int32_t, int32_t, int32_t>*>(descrG->graph_handle);
                            delete m;
                            break;
                        }
                        default:
                            return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
                    }
                }
                else {
                    switch (descrG->graphStatus) {
                        case IS_EMPTY: {
                            break;
                        }
                        case HAS_TOPOLOGY: {
                            nvgraph::CsrGraph<int> *CSRG =
                                    static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
                            delete CSRG;
                            break;
                        }
                        case HAS_VALUES: {
                            if (descrG->T == CUDA_R_32F) {
                                nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                                        static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                                delete MCSRG;
                            }
                            else if (descrG->T == CUDA_R_64F) {
                                nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                                        static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                                delete MCSRG;
                            }
                            else if (descrG->T == CUDA_R_32I) {
                                nvgraph::MultiValuedCsrGraph<int, int> *MCSRG =
                                        static_cast<nvgraph::MultiValuedCsrGraph<int, int>*>(descrG->graph_handle);
                                delete MCSRG;
                            }
                            else
                                return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
                            break;
                        }
                        default:
                            return NVGRAPH_STATUS_INVALID_VALUE;
                    }
                }
                free(descrG);
            }
            else
                return NVGRAPH_STATUS_INVALID_VALUE;
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphSetStream_impl(nvgraphHandle_t handle, cudaStream_t stream) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
            // nvgraph handle
            handle->stream = stream;
            //Cublas and Cusparse
            nvgraph::Cublas::setStream(stream);
            nvgraph::Cusparse::setStream(stream);
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphSetGraphStructure_impl(nvgraphHandle_t handle,
                                                              nvgraphGraphDescr_t descrG,
                                                              void* topologyData,
                                                              nvgraphTopologyType_t TT) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
            if (descrG->graphStatus != IS_EMPTY)
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
            if (check_ptr(topologyData))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (TT == NVGRAPH_CSR_32 || TT == NVGRAPH_CSC_32)
                    {
                int v = 0, e = 0, *neighborhood = NULL, *edgedest = NULL;
                switch (TT)
                {
                    case NVGRAPH_CSR_32:
                        {
                        nvgraphCSRTopology32I_t t = static_cast<nvgraphCSRTopology32I_t>(topologyData);
                        if (!t->nvertices || !t->nedges || check_ptr(t->source_offsets)
                                || check_ptr(t->destination_indices))
                            return NVGRAPH_STATUS_INVALID_VALUE;
                        v = t->nvertices;
                        e = t->nedges;
                        neighborhood = t->source_offsets;
                        edgedest = t->destination_indices;
                        break;
                    }
                    case NVGRAPH_CSC_32:
                        {
                        nvgraphCSCTopology32I_t t = static_cast<nvgraphCSCTopology32I_t>(topologyData);
                        if (!t->nvertices || !t->nedges || check_ptr(t->destination_offsets)
                                || check_ptr(t->source_indices))
                            return NVGRAPH_STATUS_INVALID_VALUE;
                        v = t->nvertices;
                        e = t->nedges;
                        neighborhood = t->destination_offsets;
                        edgedest = t->source_indices;
                        break;
                    }
                    default:
                        return NVGRAPH_STATUS_INVALID_VALUE;
                }

                descrG->TT = TT;

                // Create the internal CSR representation
                nvgraph::CsrGraph<int> * CSRG = new nvgraph::CsrGraph<int>(v, e, handle->stream);

                CHECK_CUDA(cudaMemcpy(CSRG->get_raw_row_offsets(),
                                      neighborhood,
                                      (size_t )((CSRG->get_num_vertices() + 1) * sizeof(int)),
                                      cudaMemcpyDefault));

                CHECK_CUDA(cudaMemcpy(CSRG->get_raw_column_indices(),
                                      edgedest,
                                      (size_t )((CSRG->get_num_edges()) * sizeof(int)),
                                      cudaMemcpyDefault));

                // Set the graph handle
                descrG->graph_handle = CSRG;
                descrG->graphStatus = HAS_TOPOLOGY;
            }
            else if (TT == NVGRAPH_2D_32I_32I) {
                nvgraph2dCOOTopology32I_t td = static_cast<nvgraph2dCOOTopology32I_t>(topologyData);
                switch (td->valueType) {
                    case CUDA_R_32I: {
                        if (!td->nvertices || !td->nedges || !td->source_indices
                                || !td->destination_indices || !td->numDevices || !td->devices
                                || !td->blockN)
                            return NVGRAPH_STATUS_INVALID_VALUE;
                        descrG->TT = TT;
                        descrG->graphStatus = HAS_TOPOLOGY;
                        if (td->values)
                            descrG->graphStatus = HAS_VALUES;
                        descrG->T = td->valueType;
                        std::vector<int32_t> devices;
                        for (int32_t i = 0; i < td->numDevices; i++)
                            devices.push_back(td->devices[i]);
                        nvgraph::MatrixDecompositionDescription<int32_t, int32_t> description(td->nvertices,
                                                                                              td->blockN,
                                                                                              td->nedges,
                                                                                              devices);
                        nvgraph::Matrix2d<int32_t, int32_t, int32_t>* m = new nvgraph::Matrix2d<int32_t,
                                int32_t, int32_t>();
                        *m = nvgraph::COOto2d(description,
                                              td->source_indices,
                                              td->destination_indices,
                                              (int32_t*) td->values);
                        descrG->graph_handle = m;
                        break;
                    }
                    default: {
                        return NVGRAPH_STATUS_INVALID_VALUE;
                    }
                }
            }
            else
            {
                return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
            }

        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);

    }

    nvgraphStatus_t NVGRAPH_API nvgraphAttachGraphStructure_impl(nvgraphHandle_t handle,
                                                            nvgraphGraphDescr_t descrG,
                                                            void* topologyData,
                                                            nvgraphTopologyType_t TT) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
            if (descrG->graphStatus != IS_EMPTY)
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
            if (check_ptr(topologyData))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (TT == NVGRAPH_CSR_32 || TT == NVGRAPH_CSC_32)
                    {
                int v = 0, e = 0, *neighborhood = NULL, *edgedest = NULL;
                switch (TT)
                {
                    case NVGRAPH_CSR_32:
                        {
                        nvgraphCSRTopology32I_t t = static_cast<nvgraphCSRTopology32I_t>(topologyData);
                        if (!t->nvertices || !t->nedges || check_ptr(t->source_offsets)
                                || check_ptr(t->destination_indices))
                            return NVGRAPH_STATUS_INVALID_VALUE;
                        v = t->nvertices;
                        e = t->nedges;
                        neighborhood = t->source_offsets;
                        edgedest = t->destination_indices;
                        break;
                    }
                    case NVGRAPH_CSC_32:
                        {
                        nvgraphCSCTopology32I_t t = static_cast<nvgraphCSCTopology32I_t>(topologyData);
                        if (!t->nvertices || !t->nedges || check_ptr(t->destination_offsets)
                                || check_ptr(t->source_indices))
                            return NVGRAPH_STATUS_INVALID_VALUE;
                        v = t->nvertices;
                        e = t->nedges;
                        neighborhood = t->destination_offsets;
                        edgedest = t->source_indices;
                        break;
                    }
                    default:
                        return NVGRAPH_STATUS_INVALID_VALUE;
                }

                descrG->TT = TT;

                // Create the internal CSR representation
                nvgraph::CsrGraph<int> * CSRG = new nvgraph::CsrGraph<int>(v, e, handle->stream);

                CSRG->set_raw_row_offsets(neighborhood);
                CSRG->set_raw_column_indices(edgedest);

                // Set the graph handle
                descrG->graph_handle = CSRG;
                descrG->graphStatus = HAS_TOPOLOGY;
            }
            else if (TT == NVGRAPH_2D_32I_32I) {
                nvgraph2dCOOTopology32I_t td = static_cast<nvgraph2dCOOTopology32I_t>(topologyData);
                switch (td->valueType) {
                    case CUDA_R_32I: {
                        if (!td->nvertices || !td->nedges || !td->source_indices
                                || !td->destination_indices || !td->numDevices || !td->devices
                                || !td->blockN)
                            return NVGRAPH_STATUS_INVALID_VALUE;
                        descrG->TT = TT;
                        descrG->graphStatus = HAS_TOPOLOGY;
                        if (td->values)
                            descrG->graphStatus = HAS_VALUES;
                        descrG->T = td->valueType;
                        std::vector<int32_t> devices;
                        for (int32_t i = 0; i < td->numDevices; i++)
                            devices.push_back(td->devices[i]);
                        nvgraph::MatrixDecompositionDescription<int32_t, int32_t> description(td->nvertices,
                                                                                              td->blockN,
                                                                                              td->nedges,
                                                                                              devices);
                        nvgraph::Matrix2d<int32_t, int32_t, int32_t>* m = new nvgraph::Matrix2d<int32_t,
                                int32_t, int32_t>();
                        *m = nvgraph::COOto2d(description,
                                              td->source_indices,
                                              td->destination_indices,
                                              (int32_t*) td->values);
                        descrG->graph_handle = m;
                        break;
                    }
                    default: {
                        return NVGRAPH_STATUS_INVALID_VALUE;
                    }
                }
            }
            else
            {
                return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
            }

        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);

    }

    nvgraphStatus_t NVGRAPH_API nvgraphGetGraphStructure_impl(nvgraphHandle_t handle,
                                                              nvgraphGraphDescr_t descrG,
                                                              void* topologyData,
                                                              nvgraphTopologyType_t* TT) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_topology(descrG))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            nvgraphTopologyType_t graphTType = descrG->TT;

            if (TT != NULL)
                *TT = graphTType;

            if (topologyData != NULL) {
                nvgraph::CsrGraph<int> *CSRG =
                        static_cast<nvgraph::CsrGraph<int> *>(descrG->graph_handle);
                int v = static_cast<int>(CSRG->get_num_vertices());
                int e = static_cast<int>(CSRG->get_num_edges());
                int *neighborhood = NULL, *edgedest = NULL;

                switch (graphTType)
                {
                    case NVGRAPH_CSR_32:
                        {
                        nvgraphCSRTopology32I_t t = static_cast<nvgraphCSRTopology32I_t>(topologyData);
                        t->nvertices = static_cast<int>(v);
                        t->nedges = static_cast<int>(e);
                        neighborhood = t->source_offsets;
                        edgedest = t->destination_indices;
                        break;
                    }
                    case NVGRAPH_CSC_32:
                        {
                        nvgraphCSCTopology32I_t t = static_cast<nvgraphCSCTopology32I_t>(topologyData);
                        t->nvertices = static_cast<int>(v);
                        t->nedges = static_cast<int>(e);
                        neighborhood = t->destination_offsets;
                        edgedest = t->source_indices;
                        break;
                    }
                    default:
                        return NVGRAPH_STATUS_INTERNAL_ERROR;
                }

                if (neighborhood != NULL) {
                    CHECK_CUDA(cudaMemcpy(neighborhood,
                                          CSRG->get_raw_row_offsets(),
                                          (size_t )((v + 1) * sizeof(int)),
                                          cudaMemcpyDefault));
                }

                if (edgedest != NULL) {
                    CHECK_CUDA(cudaMemcpy(edgedest,
                                          CSRG->get_raw_column_indices(),
                                          (size_t )((e) * sizeof(int)),
                                          cudaMemcpyDefault));
                }

            }
        }
        NVGRAPH_CATCHES(rc)
        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphAllocateVertexData_impl(nvgraphHandle_t handle,
                                                               nvgraphGraphDescr_t descrG,
                                                               size_t numsets,
                                                               cudaDataType_t *settypes) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_size(numsets)
                    || check_ptr(settypes))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
            if (check_uniform_type_array(settypes, numsets))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->graphStatus == HAS_TOPOLOGY) // need to convert CsrGraph to MultiValuedCsrGraph first
                    {
                if (*settypes == CUDA_R_32F)
                        {
                    nvgraph::CsrGraph<int> *CSRG =
                            static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
                    nvgraph::MultiValuedCsrGraph<int, float> *MCSRG = new nvgraph::MultiValuedCsrGraph<
                            int, float>(*CSRG);
                    descrG->graph_handle = MCSRG;
                }
                else if (*settypes == CUDA_R_64F)
                        {
                    nvgraph::CsrGraph<int> *CSRG =
                            static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
                    nvgraph::MultiValuedCsrGraph<int, double> *MCSRG = new nvgraph::MultiValuedCsrGraph<
                            int, double>(*CSRG);
                    descrG->graph_handle = MCSRG;
                }
                else if (*settypes == CUDA_R_32I)
                        {
                    nvgraph::CsrGraph<int> *CSRG =
                            static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
                    nvgraph::MultiValuedCsrGraph<int, int> *MCSRG = new nvgraph::MultiValuedCsrGraph<int,
                            int>(*CSRG);
                    descrG->graph_handle = MCSRG;
                }
                else
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
                descrG->T = *settypes;
                descrG->graphStatus = HAS_VALUES;
            }
            else if (descrG->graphStatus == HAS_VALUES) // Already in MultiValuedCsrGraph, just need to check the type
                    {
                if (*settypes != descrG->T)
                    return NVGRAPH_STATUS_INVALID_VALUE;
            }
            else
                return NVGRAPH_STATUS_INVALID_VALUE;

            // Allocate and transfer
            if (*settypes == CUDA_R_32F)
                    {
                nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                MCSRG->allocateVertexData(numsets, NULL);
            }
            else if (*settypes == CUDA_R_64F)
                    {
                nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                MCSRG->allocateVertexData(numsets, NULL);
            }
            else if (*settypes == CUDA_R_32I)
                    {
                nvgraph::MultiValuedCsrGraph<int, int> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, int>*>(descrG->graph_handle);
                MCSRG->allocateVertexData(numsets, NULL);
            }
            else
                return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphAttachVertexData_impl(nvgraphHandle_t handle,
                                                             nvgraphGraphDescr_t descrG,
                                                             size_t setnum,
                                                             cudaDataType_t settype,
                                                             void *vertexData) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_size(setnum))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->graphStatus == HAS_TOPOLOGY) // need to convert CsrGraph to MultiValuedCsrGraph first
                    {
                if (settype == CUDA_R_32F)
                        {
                    nvgraph::CsrGraph<int> *CSRG =
                            static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
                    nvgraph::MultiValuedCsrGraph<int, float> *MCSRG = new nvgraph::MultiValuedCsrGraph<
                            int, float>(*CSRG);
                    descrG->graph_handle = MCSRG;
                }
                else if (settype == CUDA_R_64F)
                        {
                    nvgraph::CsrGraph<int> *CSRG =
                            static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
                    nvgraph::MultiValuedCsrGraph<int, double> *MCSRG = new nvgraph::MultiValuedCsrGraph<
                            int, double>(*CSRG);
                    descrG->graph_handle = MCSRG;
                }
                else if (settype == CUDA_R_32I)
                        {
                    nvgraph::CsrGraph<int> *CSRG =
                            static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
                    nvgraph::MultiValuedCsrGraph<int, int> *MCSRG = new nvgraph::MultiValuedCsrGraph<int,
                            int>(*CSRG);
                    descrG->graph_handle = MCSRG;
                }
                else
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
                descrG->T = settype;
                descrG->graphStatus = HAS_VALUES;
            }
            else if (descrG->graphStatus == HAS_VALUES) // Already in MultiValuedCsrGraph, just need to check the type
                    {
                if (settype != descrG->T)
                    return NVGRAPH_STATUS_INVALID_VALUE;
            }
            else
                return NVGRAPH_STATUS_INVALID_VALUE;

            // transfer
            if (settype == CUDA_R_32F)
                    {
                nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                MCSRG->attachVertexData(setnum, (float*)vertexData, NULL);
            }
            else if (settype == CUDA_R_64F)
                    {
                nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                MCSRG->attachVertexData(setnum, (double*)vertexData, NULL);
            }
            else if (settype == CUDA_R_32I)
                    {
                nvgraph::MultiValuedCsrGraph<int, int> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, int>*>(descrG->graph_handle);
                MCSRG->attachVertexData(setnum, (int*)vertexData, NULL);
            }
            else
                return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }
    nvgraphStatus_t NVGRAPH_API nvgraphAllocateEdgeData_impl(nvgraphHandle_t handle,
                                                             nvgraphGraphDescr_t descrG,
                                                             size_t numsets,
                                                             cudaDataType_t *settypes) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_size(numsets)
                    || check_ptr(settypes))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
            if (check_uniform_type_array(settypes, numsets))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
            // Look at what kind of graph we have
            if (descrG->graphStatus == HAS_TOPOLOGY) // need to convert CsrGraph to MultiValuedCsrGraph first
                    {
                if (*settypes == CUDA_R_32F)
                        {
                    nvgraph::CsrGraph<int> *CSRG =
                            static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
                    nvgraph::MultiValuedCsrGraph<int, float> *MCSRG = new nvgraph::MultiValuedCsrGraph<
                            int, float>(*CSRG);
                    descrG->graph_handle = MCSRG;
                }
                else if (*settypes == CUDA_R_64F)
                        {
                    nvgraph::CsrGraph<int> *CSRG =
                            static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
                    nvgraph::MultiValuedCsrGraph<int, double> *MCSRG = new nvgraph::MultiValuedCsrGraph<
                            int, double>(*CSRG);
                    descrG->graph_handle = MCSRG;
                }
                else if (*settypes == CUDA_R_32I)
                        {
                    nvgraph::CsrGraph<int> *CSRG =
                            static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
                    nvgraph::MultiValuedCsrGraph<int, int> *MCSRG = new nvgraph::MultiValuedCsrGraph<int,
                            int>(*CSRG);
                    descrG->graph_handle = MCSRG;
                }
                else
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
                descrG->T = *settypes;
                descrG->graphStatus = HAS_VALUES;
            }
            else if (descrG->graphStatus == HAS_VALUES) // Already in MultiValuedCsrGraph, just need to check the type
                    {
                if (*settypes != descrG->T)
                    return NVGRAPH_STATUS_INVALID_VALUE;
            }
            else
                return NVGRAPH_STATUS_INVALID_VALUE;

            // Allocate and transfer
            if (*settypes == CUDA_R_32F)
                    {
                nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                MCSRG->allocateEdgeData(numsets, NULL);
            }
            else if (*settypes == CUDA_R_64F)
                    {
                nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                MCSRG->allocateEdgeData(numsets, NULL);
            }
            else if (*settypes == CUDA_R_32I)
                    {
                nvgraph::MultiValuedCsrGraph<int, int> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, int>*>(descrG->graph_handle);
                MCSRG->allocateEdgeData(numsets, NULL);
            }
            else
                return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;

        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphAttachEdgeData_impl(nvgraphHandle_t handle,
                                                           nvgraphGraphDescr_t descrG,
                                                           size_t setnum,
                                                           cudaDataType_t settype,
                                                           void *edgeData) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_size(setnum))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
            // Look at what kind of graph we have
            if (descrG->graphStatus == HAS_TOPOLOGY) // need to convert CsrGraph to MultiValuedCsrGraph first
                    {
                if (settype == CUDA_R_32F)
                        {
                    nvgraph::CsrGraph<int> *CSRG =
                            static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
                    nvgraph::MultiValuedCsrGraph<int, float> *MCSRG = new nvgraph::MultiValuedCsrGraph<
                            int, float>(*CSRG);
                    descrG->graph_handle = MCSRG;
                }
                else if (settype == CUDA_R_64F)
                        {
                    nvgraph::CsrGraph<int> *CSRG =
                            static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
                    nvgraph::MultiValuedCsrGraph<int, double> *MCSRG = new nvgraph::MultiValuedCsrGraph<
                            int, double>(*CSRG);
                    descrG->graph_handle = MCSRG;
                }
                else if (settype == CUDA_R_32I)
                        {
                    nvgraph::CsrGraph<int> *CSRG =
                            static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
                    nvgraph::MultiValuedCsrGraph<int, int> *MCSRG = new nvgraph::MultiValuedCsrGraph<int,
                            int>(*CSRG);
                    descrG->graph_handle = MCSRG;
                }
                else
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
                descrG->T = settype;
                descrG->graphStatus = HAS_VALUES;
            }
            else if (descrG->graphStatus == HAS_VALUES) // Already in MultiValuedCsrGraph, just need to check the type
                    {
                if (settype != descrG->T)
                    return NVGRAPH_STATUS_INVALID_VALUE;
            }
            else
                return NVGRAPH_STATUS_INVALID_VALUE;

            // Allocate and transfer
            if (settype == CUDA_R_32F)
                    {
                nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                MCSRG->attachEdgeData(setnum, (float*)edgeData, NULL);
            }
            else if (settype == CUDA_R_64F)
                    {
                nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                MCSRG->attachEdgeData(setnum, (double*)edgeData, NULL);
            }
            else if (settype == CUDA_R_32I)
                    {
                nvgraph::MultiValuedCsrGraph<int, int> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, int>*>(descrG->graph_handle);
                MCSRG->attachEdgeData(setnum, (int*)edgeData, NULL);
            }
            else
                return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;

        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphSetVertexData_impl(nvgraphHandle_t handle,
                                                          nvgraphGraphDescr_t descrG,
                                                          void *vertexData,
                                                          size_t setnum) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_size(setnum)
                    || check_ptr(vertexData))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
                FatalError("Graph should have allocated values.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->T == CUDA_R_32F)
                    {
                nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
                    return NVGRAPH_STATUS_INVALID_VALUE;
                cudaMemcpy(MCSRG->get_raw_vertex_dim(setnum),
                           (float*) vertexData,
                           (size_t) ((MCSRG->get_num_vertices()) * sizeof(float)),
                           cudaMemcpyDefault);
            }
            else if (descrG->T == CUDA_R_64F)
                    {
                nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
                    return NVGRAPH_STATUS_INVALID_VALUE;
                cudaMemcpy(MCSRG->get_raw_vertex_dim(setnum),
                           (double*) vertexData,
                           (size_t) ((MCSRG->get_num_vertices()) * sizeof(double)),
                           cudaMemcpyDefault);
            }
            else if (descrG->T == CUDA_R_32I)
                    {
                nvgraph::MultiValuedCsrGraph<int, int> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, int>*>(descrG->graph_handle);
                if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
                    return NVGRAPH_STATUS_INVALID_VALUE;
                cudaMemcpy(MCSRG->get_raw_vertex_dim(setnum),
                           (int*) vertexData,
                           (size_t) ((MCSRG->get_num_vertices()) * sizeof(int)),
                           cudaMemcpyDefault);
            }
            else
                return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;

            cudaCheckError();
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphGetVertexData_impl(nvgraphHandle_t handle,
                                                          nvgraphGraphDescr_t descrG,
                                                          void *vertexData,
                                                          size_t setnum) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_size(setnum)
                    || check_ptr(vertexData))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
                FatalError("Graph should have values.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->T == CUDA_R_32F)
                    {
                nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
                    return NVGRAPH_STATUS_INVALID_VALUE;
                cudaMemcpy((float*) vertexData,
                                MCSRG->get_raw_vertex_dim(setnum),
                                (size_t) ((MCSRG->get_num_vertices()) * sizeof(float)),
                                cudaMemcpyDefault);
            }
            else if (descrG->T == CUDA_R_64F)
                    {
                nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
                    return NVGRAPH_STATUS_INVALID_VALUE;
                cudaMemcpy((double*) vertexData,
                                MCSRG->get_raw_vertex_dim(setnum),
                                (size_t) ((MCSRG->get_num_vertices()) * sizeof(double)),
                                cudaMemcpyDefault);
            }
            else if (descrG->T == CUDA_R_32I)
                    {
                nvgraph::MultiValuedCsrGraph<int, int> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, int>*>(descrG->graph_handle);
                if (setnum >= MCSRG->get_num_vertex_dim()) // base index is 0
                    return NVGRAPH_STATUS_INVALID_VALUE;
                cudaMemcpy((int*) vertexData,
                                MCSRG->get_raw_vertex_dim(setnum),
                                (size_t) ((MCSRG->get_num_vertices()) * sizeof(int)),
                                cudaMemcpyDefault);
            }
            else
                return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;

            cudaCheckError();
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphConvertTopology_impl(nvgraphHandle_t handle,
                                                            nvgraphTopologyType_t srcTType,
                                                            void *srcTopology,
                                                            void *srcEdgeData,
                                                            cudaDataType_t *dataType,
                                                            nvgraphTopologyType_t dstTType,
                                                            void *dstTopology,
                                                            void *dstEdgeData) {

        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_ptr(dstEdgeData) || check_ptr(srcEdgeData))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            size_t sizeT;
            if (*dataType == CUDA_R_32F)
                sizeT = sizeof(float);
            else if (*dataType == CUDA_R_64F)
                sizeT = sizeof(double);
            else
                return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;

            // Trust me, this better than nested if's.
            if (srcTType == NVGRAPH_CSR_32 && dstTType == NVGRAPH_CSR_32) {                  // CSR2CSR
                nvgraphCSRTopology32I_t srcT = static_cast<nvgraphCSRTopology32I_t>(srcTopology);
                nvgraphCSRTopology32I_t dstT = static_cast<nvgraphCSRTopology32I_t>(dstTopology);
                dstT->nvertices = srcT->nvertices;
                dstT->nedges = srcT->nedges;
                CHECK_CUDA(cudaMemcpy(dstT->source_offsets,
                                                srcT->source_offsets,
                                                (srcT->nvertices + 1) * sizeof(int),
                                                cudaMemcpyDefault));
                CHECK_CUDA(cudaMemcpy(dstT->destination_indices,
                                                srcT->destination_indices,
                                                srcT->nedges * sizeof(int),
                                                cudaMemcpyDefault));
                CHECK_CUDA(cudaMemcpy(dstEdgeData,
                                                srcEdgeData,
                                                srcT->nedges * sizeT,
                                                cudaMemcpyDefault));
            } else if (srcTType == NVGRAPH_CSR_32 && dstTType == NVGRAPH_CSC_32) {           // CSR2CSC
                nvgraphCSRTopology32I_t srcT = static_cast<nvgraphCSRTopology32I_t>(srcTopology);
                nvgraphCSCTopology32I_t dstT = static_cast<nvgraphCSCTopology32I_t>(dstTopology);
                dstT->nvertices = srcT->nvertices;
                dstT->nedges = srcT->nedges;
                csr2csc(srcT->nvertices, srcT->nvertices, srcT->nedges,
                            srcEdgeData,
                            srcT->source_offsets, srcT->destination_indices,
                            dstEdgeData,
                            dstT->source_indices, dstT->destination_offsets,
                            CUSPARSE_ACTION_NUMERIC,
                            CUSPARSE_INDEX_BASE_ZERO, dataType);
            } else if (srcTType == NVGRAPH_CSR_32 && dstTType == NVGRAPH_COO_32) {           // CSR2COO
                nvgraphCSRTopology32I_t srcT = static_cast<nvgraphCSRTopology32I_t>(srcTopology);
                nvgraphCOOTopology32I_t dstT = static_cast<nvgraphCOOTopology32I_t>(dstTopology);
                dstT->nvertices = srcT->nvertices;
                dstT->nedges = srcT->nedges;
                if (dstT->tag == NVGRAPH_SORTED_BY_SOURCE || dstT->tag == NVGRAPH_DEFAULT
                        || dstT->tag == NVGRAPH_UNSORTED) {
                    csr2coo(srcT->source_offsets,
                                srcT->nedges,
                                srcT->nvertices,
                                dstT->source_indices,
                                CUSPARSE_INDEX_BASE_ZERO);
                    CHECK_CUDA(cudaMemcpy(dstT->destination_indices,
                                                    srcT->destination_indices,
                                                    srcT->nedges * sizeof(int),
                                                    cudaMemcpyDefault));
                    CHECK_CUDA(cudaMemcpy(dstEdgeData,
                                                    srcEdgeData,
                                                    srcT->nedges * sizeT,
                                                    cudaMemcpyDefault));
                } else if (dstT->tag == NVGRAPH_SORTED_BY_DESTINATION) {
                    // Step 1: Convert to COO_Source
                    csr2coo(srcT->source_offsets,
                                srcT->nedges,
                                srcT->nvertices,
                                dstT->source_indices,
                                CUSPARSE_INDEX_BASE_ZERO);
                    // Step 2: Convert to COO_Destination
                    cooSortByDestination(srcT->nvertices, srcT->nvertices, srcT->nedges,
                                                srcEdgeData,
                                                dstT->source_indices, srcT->destination_indices,
                                                dstEdgeData,
                                                dstT->source_indices, dstT->destination_indices,
                                                CUSPARSE_INDEX_BASE_ZERO,
                                                dataType);
                } else {
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
                }
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////
            } else if (srcTType == NVGRAPH_CSC_32 && dstTType == NVGRAPH_CSR_32) {           // CSC2CSR
                nvgraphCSCTopology32I_t srcT = static_cast<nvgraphCSCTopology32I_t>(srcTopology);
                nvgraphCSRTopology32I_t dstT = static_cast<nvgraphCSRTopology32I_t>(dstTopology);
                dstT->nvertices = srcT->nvertices;
                dstT->nedges = srcT->nedges;
                csc2csr(srcT->nvertices, srcT->nvertices, srcT->nedges,
                            srcEdgeData,
                            srcT->source_indices, srcT->destination_offsets,
                            dstEdgeData,
                            dstT->source_offsets, dstT->destination_indices,
                            CUSPARSE_ACTION_NUMERIC,
                            CUSPARSE_INDEX_BASE_ZERO, dataType);
            } else if (srcTType == NVGRAPH_CSC_32 && dstTType == NVGRAPH_CSC_32) {           // CSC2CSC
                nvgraphCSCTopology32I_t srcT = static_cast<nvgraphCSCTopology32I_t>(srcTopology);
                nvgraphCSCTopology32I_t dstT = static_cast<nvgraphCSCTopology32I_t>(dstTopology);
                dstT->nvertices = srcT->nvertices;
                dstT->nedges = srcT->nedges;
                CHECK_CUDA(cudaMemcpy(dstT->destination_offsets,
                                                srcT->destination_offsets,
                                                (srcT->nvertices + 1) * sizeof(int),
                                                cudaMemcpyDefault));
                CHECK_CUDA(cudaMemcpy(dstT->source_indices,
                                                srcT->source_indices,
                                                srcT->nedges * sizeof(int),
                                                cudaMemcpyDefault));
                CHECK_CUDA(cudaMemcpy(dstEdgeData,
                                                srcEdgeData,
                                                srcT->nedges * sizeT,
                                                cudaMemcpyDefault));
            } else if (srcTType == NVGRAPH_CSC_32 && dstTType == NVGRAPH_COO_32) {           // CSC2COO
                nvgraphCSCTopology32I_t srcT = static_cast<nvgraphCSCTopology32I_t>(srcTopology);
                nvgraphCOOTopology32I_t dstT = static_cast<nvgraphCOOTopology32I_t>(dstTopology);
                dstT->nvertices = srcT->nvertices;
                dstT->nedges = srcT->nedges;
                if (dstT->tag == NVGRAPH_SORTED_BY_SOURCE) {
                    // Step 1: Convert to COO_Destination
                    csr2coo(srcT->destination_offsets,
                                srcT->nedges,
                                srcT->nvertices,
                                dstT->destination_indices,
                                CUSPARSE_INDEX_BASE_ZERO);
                    // Step 2: Convert to COO_Source
                    cooSortBySource(srcT->nvertices, srcT->nvertices, srcT->nedges,
                                            srcEdgeData,
                                            srcT->source_indices, dstT->destination_indices,
                                            dstEdgeData,
                                            dstT->source_indices, dstT->destination_indices,
                                            CUSPARSE_INDEX_BASE_ZERO,
                                            dataType);
                } else if (dstT->tag == NVGRAPH_SORTED_BY_DESTINATION || dstT->tag == NVGRAPH_DEFAULT
                        || dstT->tag == NVGRAPH_UNSORTED) {
                    csr2coo(srcT->destination_offsets,
                                srcT->nedges,
                                srcT->nvertices,
                                dstT->destination_indices,
                                CUSPARSE_INDEX_BASE_ZERO);
                    CHECK_CUDA(cudaMemcpy(dstT->source_indices,
                                                    srcT->source_indices,
                                                    srcT->nedges * sizeof(int),
                                                    cudaMemcpyDefault));
                    CHECK_CUDA(cudaMemcpy(dstEdgeData,
                                                    srcEdgeData,
                                                    srcT->nedges * sizeT,
                                                    cudaMemcpyDefault));
                } else {
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
                }
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////
            } else if (srcTType == NVGRAPH_COO_32 && dstTType == NVGRAPH_CSR_32) {           // COO2CSR
                nvgraphCOOTopology32I_t srcT = static_cast<nvgraphCOOTopology32I_t>(srcTopology);
                nvgraphCSRTopology32I_t dstT = static_cast<nvgraphCSRTopology32I_t>(dstTopology);
                dstT->nvertices = srcT->nvertices;
                dstT->nedges = srcT->nedges;
                if (srcT->tag == NVGRAPH_SORTED_BY_SOURCE) {
                    coo2csr(srcT->source_indices,
                                srcT->nedges,
                                srcT->nvertices,
                                dstT->source_offsets,
                                CUSPARSE_INDEX_BASE_ZERO);
                    CHECK_CUDA(cudaMemcpy(dstT->destination_indices,
                                                    srcT->destination_indices,
                                                    srcT->nedges * sizeof(int),
                                                    cudaMemcpyDefault));
                    CHECK_CUDA(cudaMemcpy(dstEdgeData,
                                                    srcEdgeData,
                                                    srcT->nedges * sizeT,
                                                    cudaMemcpyDefault));
                } else if (srcT->tag == NVGRAPH_SORTED_BY_DESTINATION) {
                    cood2csr(srcT->nvertices, srcT->nvertices, srcT->nedges,
                                srcEdgeData,
                                srcT->source_indices, srcT->destination_indices,
                                dstEdgeData,
                                dstT->source_offsets, dstT->destination_indices,
                                CUSPARSE_INDEX_BASE_ZERO,
                                dataType);
                } else if (srcT->tag == NVGRAPH_DEFAULT || srcT->tag == NVGRAPH_UNSORTED) {
                    coou2csr(srcT->nvertices, srcT->nvertices, srcT->nedges,
                                srcEdgeData,
                                srcT->source_indices, srcT->destination_indices,
                                dstEdgeData,
                                dstT->source_offsets, dstT->destination_indices,
                                CUSPARSE_INDEX_BASE_ZERO,
                                dataType);
                } else {
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
                }
            } else if (srcTType == NVGRAPH_COO_32 && dstTType == NVGRAPH_CSC_32) {           // COO2CSC
                nvgraphCOOTopology32I_t srcT = static_cast<nvgraphCOOTopology32I_t>(srcTopology);
                nvgraphCSCTopology32I_t dstT = static_cast<nvgraphCSCTopology32I_t>(dstTopology);
                dstT->nvertices = srcT->nvertices;
                dstT->nedges = srcT->nedges;
                if (srcT->tag == NVGRAPH_SORTED_BY_SOURCE) {
                    coos2csc(srcT->nvertices, srcT->nvertices, srcT->nedges,
                                srcEdgeData,
                                srcT->source_indices, srcT->destination_indices,
                                dstEdgeData,
                                dstT->source_indices, dstT->destination_offsets,
                                CUSPARSE_INDEX_BASE_ZERO,
                                dataType);
                } else if (srcT->tag == NVGRAPH_SORTED_BY_DESTINATION) {
                    coo2csr(srcT->destination_indices,
                                srcT->nedges,
                                srcT->nvertices,
                                dstT->destination_offsets,
                                CUSPARSE_INDEX_BASE_ZERO);
                    CHECK_CUDA(cudaMemcpy(dstT->source_indices,
                                                    srcT->source_indices,
                                                    srcT->nedges * sizeof(int),
                                                    cudaMemcpyDefault));
                    CHECK_CUDA(cudaMemcpy(dstEdgeData,
                                                    srcEdgeData,
                                                    srcT->nedges * sizeT,
                                                    cudaMemcpyDefault));
                } else if (srcT->tag == NVGRAPH_DEFAULT || srcT->tag == NVGRAPH_UNSORTED) {
                    coou2csc(srcT->nvertices, srcT->nvertices, srcT->nedges,
                                srcEdgeData,
                                srcT->source_indices, srcT->destination_indices,
                                dstEdgeData,
                                dstT->source_indices, dstT->destination_offsets,
                                CUSPARSE_INDEX_BASE_ZERO,
                                dataType);
                } else {
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
                }
            } else if (srcTType == NVGRAPH_COO_32 && dstTType == NVGRAPH_COO_32) {           // COO2COO
                nvgraphCOOTopology32I_t srcT = static_cast<nvgraphCOOTopology32I_t>(srcTopology);
                nvgraphCOOTopology32I_t dstT = static_cast<nvgraphCOOTopology32I_t>(dstTopology);
                dstT->nvertices = srcT->nvertices;
                dstT->nedges = srcT->nedges;
                if (srcT->tag == dstT->tag || dstT->tag == NVGRAPH_DEFAULT
                        || dstT->tag == NVGRAPH_UNSORTED) {
                    CHECK_CUDA(cudaMemcpy(dstT->source_indices,
                                                    srcT->source_indices,
                                                    srcT->nedges * sizeof(int),
                                                    cudaMemcpyDefault));
                    CHECK_CUDA(cudaMemcpy(dstT->destination_indices,
                                                    srcT->destination_indices,
                                                    srcT->nedges * sizeof(int),
                                                    cudaMemcpyDefault));
                    CHECK_CUDA(cudaMemcpy(dstEdgeData,
                                                    srcEdgeData,
                                                    srcT->nedges * sizeT,
                                                    cudaMemcpyDefault));
                } else if (dstT->tag == NVGRAPH_SORTED_BY_SOURCE) {
                    cooSortBySource(srcT->nvertices, srcT->nvertices, srcT->nedges,
                                            srcEdgeData,
                                            srcT->source_indices, srcT->destination_indices,
                                            dstEdgeData,
                                            dstT->source_indices, dstT->destination_indices,
                                            CUSPARSE_INDEX_BASE_ZERO,
                                            dataType);
                } else if (dstT->tag == NVGRAPH_SORTED_BY_DESTINATION) {
                    cooSortByDestination(srcT->nvertices, srcT->nvertices, srcT->nedges,
                                                srcEdgeData,
                                                srcT->source_indices, srcT->destination_indices,
                                                dstEdgeData,
                                                dstT->source_indices, dstT->destination_indices,
                                                CUSPARSE_INDEX_BASE_ZERO,
                                                dataType);
                } else {
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
                }

                ///////////////////////////////////////////////////////////////////////////////////////////////////////////
            } else {
                return NVGRAPH_STATUS_INVALID_VALUE;
            }

        }
        NVGRAPH_CATCHES(rc)
        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphConvertGraph_impl(nvgraphHandle_t handle,
                                                         nvgraphGraphDescr_t srcDescrG,
                                                         nvgraphGraphDescr_t dstDescrG,
                                                         nvgraphTopologyType_t dstTType) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        nvgraphStatus_t status = NVGRAPH_STATUS_SUCCESS;
        try
        {
            if (check_context(handle) || check_graph(srcDescrG))  // Graph must have a topology
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (dstDescrG->graphStatus != IS_EMPTY) // dst Graph must be empty
                return NVGRAPH_STATUS_INVALID_VALUE;

            // graphs can only have CSR or CSC topology (EL is for storage only)
            if (srcDescrG->TT != NVGRAPH_CSR_32 && srcDescrG->TT != NVGRAPH_CSC_32)
                return NVGRAPH_STATUS_INTERNAL_ERROR; // invalid state, you can only create graph with CSR/CSC
            if (dstTType != NVGRAPH_CSR_32 && dstTType != NVGRAPH_CSC_32)
                return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED; // only conversion to CSR/CSC is allowed

            int nvertices, nedges;
            int *srcOffsets = NULL, *srcIndices = NULL, *dstOffsets = NULL, *dstIndices = NULL;
            std::shared_ptr<int> permutation, offsets, indices;

            // Step 1: get source graph structure
            nvgraph::CsrGraph<int> *CSRG =
                    static_cast<nvgraph::CsrGraph<int> *>(srcDescrG->graph_handle);
            nvertices = static_cast<int>(CSRG->get_num_vertices());
            nedges = static_cast<int>(CSRG->get_num_edges());
            srcOffsets = CSRG->get_raw_row_offsets();
            srcIndices = CSRG->get_raw_column_indices();

            // Step 2: convert topology and get permutation array.
            if (srcDescrG->TT != dstTType) { // Otherwise conversion is not needed, only copy.
                offsets = allocateDevice<int>(nvertices + 1, NULL);
                indices = allocateDevice<int>(nedges, NULL);
                permutation = allocateDevice<int>(nedges, NULL);
                csr2cscP(nvertices, nvertices, nedges,
                            srcOffsets,
                            srcIndices,
                            indices.get(),
                            offsets.get(), permutation.get(), CUSPARSE_INDEX_BASE_ZERO);
                dstOffsets = offsets.get();
                dstIndices = indices.get();
            } else {
                dstOffsets = srcOffsets;
                dstIndices = srcIndices;
            }

            // Step 3: Set dst graph structure
            if (dstTType == NVGRAPH_CSR_32) {
                nvgraphCSRTopology32I_st dstTopology;
                dstTopology.nedges = nedges;
                dstTopology.nvertices = nvertices;
                dstTopology.source_offsets = dstOffsets;
                dstTopology.destination_indices = dstIndices;
                status = nvgraphSetGraphStructure(handle, dstDescrG, &dstTopology, dstTType);
            } else if (dstTType == NVGRAPH_CSC_32) {
                nvgraphCSCTopology32I_st dstTopology;
                dstTopology.nedges = nedges;
                dstTopology.nvertices = nvertices;
                dstTopology.destination_offsets = dstOffsets;
                dstTopology.source_indices = dstIndices;
                status = nvgraphSetGraphStructure(handle, dstDescrG, &dstTopology, dstTType);
            } else
                return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
            if (status != NVGRAPH_STATUS_SUCCESS)
                return NVGRAPH_STATUS_INTERNAL_ERROR;
            offsets.reset();
            indices.reset();

            // Step 4: Allocate, convert and set edge+vertex data on the new graph
            if (srcDescrG->graphStatus == HAS_VALUES) {
                if (srcDescrG->T == CUDA_R_32F) {
                    nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(srcDescrG->graph_handle);
                    size_t vertexDim = MCSRG->get_num_vertex_dim();
                    size_t edgesDim = MCSRG->get_num_edge_dim();
                    // Step 4.1: allocate and set vertex data (no need for convert)
                    if (vertexDim > 0) {
                        std::vector<cudaDataType_t> vertexDataType(vertexDim);
                        std::fill(vertexDataType.begin(), vertexDataType.end(), srcDescrG->T);
                        status = nvgraphAllocateVertexData(handle,
                                                           dstDescrG,
                                                           vertexDim,
                                                           vertexDataType.data());
                        if (status != NVGRAPH_STATUS_SUCCESS)
                            return NVGRAPH_STATUS_INTERNAL_ERROR;
                        for (size_t i = 0; i < vertexDim; ++i) {
                            void *vertexData = MCSRG->get_raw_vertex_dim(i);
                            status = nvgraphSetVertexData(handle, dstDescrG, vertexData, i);
                            if (status != NVGRAPH_STATUS_SUCCESS)
                                return NVGRAPH_STATUS_INTERNAL_ERROR;
                        }
                    }
                    // Step 4.2: allocate and set vertex data
                    if (edgesDim > 0) {
                        void *dstEdgeData = NULL;
                        std::shared_ptr<float> dstEdgeDataSP;

                        std::vector<cudaDataType_t> edgeDataType(edgesDim);
                        std::fill(edgeDataType.begin(), edgeDataType.end(), srcDescrG->T);
                        status = nvgraphAllocateEdgeData(handle,
                                                         dstDescrG,
                                                         edgesDim,
                                                         edgeDataType.data());
                        if (status != NVGRAPH_STATUS_SUCCESS)
                            return NVGRAPH_STATUS_INTERNAL_ERROR;
                        // allocate edge data memory (if there is a need)
                        if (edgesDim > 0 && srcDescrG->TT != dstTType) {
                            dstEdgeDataSP = allocateDevice<float>(nedges, NULL);
                            dstEdgeData = dstEdgeDataSP.get();
                        }
                        // Convert and set edge data (using permutation array)
                        for (size_t i = 0; i < edgesDim; ++i) {
                            void *srcEdgeData = (void*) (MCSRG->get_raw_edge_dim((int) i));
                            if (srcDescrG->TT != dstTType) // Convert using permutation array
                                gthrX(nedges,
                                        srcEdgeData,
                                        dstEdgeData,
                                        permutation.get(),
                                        CUSPARSE_INDEX_BASE_ZERO,
                                        &(srcDescrG->T));
                            else
                                dstEdgeData = srcEdgeData;
                            // set edgedata
                            status = nvgraphSetEdgeData(handle, dstDescrG, dstEdgeData, i);
                            if (status != NVGRAPH_STATUS_SUCCESS)
                                return NVGRAPH_STATUS_INTERNAL_ERROR;
                        }
                    }
                } else if (srcDescrG->T == CUDA_R_64F) {
                    nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(srcDescrG->graph_handle);
                    size_t vertexDim = MCSRG->get_num_vertex_dim();
                    size_t edgesDim = MCSRG->get_num_edge_dim();
                    // Step 4.1: allocate and set vertex data (no need for convert)
                    if (vertexDim > 0) {
                        std::vector<cudaDataType_t> vertexDataType(vertexDim);
                        std::fill(vertexDataType.begin(), vertexDataType.end(), srcDescrG->T);
                        status = nvgraphAllocateVertexData(handle,
                                                           dstDescrG,
                                                           vertexDim,
                                                           vertexDataType.data());
                        if (status != NVGRAPH_STATUS_SUCCESS)
                            return NVGRAPH_STATUS_INTERNAL_ERROR;
                        for (size_t i = 0; i < vertexDim; ++i) {
                            void *vertexData = MCSRG->get_raw_vertex_dim(i);
                            status = nvgraphSetVertexData(handle, dstDescrG, vertexData, i);
                            if (status != NVGRAPH_STATUS_SUCCESS)
                                return NVGRAPH_STATUS_INTERNAL_ERROR;
                        }
                    }
                    // Step 4.2: allocate and set vertex data
                    if (edgesDim > 0) {
                        void *dstEdgeData = NULL;
                        std::shared_ptr<double> dstEdgeDataSP;

                        std::vector<cudaDataType_t> edgeDataType(edgesDim);
                        std::fill(edgeDataType.begin(), edgeDataType.end(), srcDescrG->T);
                        status = nvgraphAllocateEdgeData(handle,
                                                         dstDescrG,
                                                         edgesDim,
                                                         edgeDataType.data());
                        if (status != NVGRAPH_STATUS_SUCCESS)
                            return NVGRAPH_STATUS_INTERNAL_ERROR;
                        // allocate edge data memory (if there is a need)
                        if (edgesDim > 0 && srcDescrG->TT != dstTType) {
                            dstEdgeDataSP = allocateDevice<double>(nedges, NULL);
                            dstEdgeData = dstEdgeDataSP.get();
                        }
                        // Convert and set edge data (using permutation array)
                        for (size_t i = 0; i < edgesDim; ++i) {
                            void *srcEdgeData = (void*) (MCSRG->get_raw_edge_dim((int) i));
                            if (srcDescrG->TT != dstTType) // Convert using permutation array
                                gthrX(nedges,
                                        srcEdgeData,
                                        dstEdgeData,
                                        permutation.get(),
                                        CUSPARSE_INDEX_BASE_ZERO,
                                        &(srcDescrG->T));
                            else
                                dstEdgeData = srcEdgeData;
                            // set edgedata
                            status = nvgraphSetEdgeData(handle, dstDescrG, dstEdgeData, i);
                            if (status != NVGRAPH_STATUS_SUCCESS)
                                return NVGRAPH_STATUS_INTERNAL_ERROR;
                        }
                    }
                } else
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
            }
        }
        NVGRAPH_CATCHES(rc)
        return getCAPIStatusForError(rc);

    }

    nvgraphStatus_t NVGRAPH_API nvgraphSetEdgeData_impl(nvgraphHandle_t handle,
                                                        nvgraphGraphDescr_t descrG,
                                                        void *edgeData,
                                                        size_t setnum) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_size(setnum)
                    || check_ptr(edgeData))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (descrG->T == CUDA_R_32F)
                    {
                nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                if (setnum >= MCSRG->get_num_edge_dim()) // base index is 0
                    return NVGRAPH_STATUS_INVALID_VALUE;
                cudaMemcpy(MCSRG->get_raw_edge_dim(setnum),
                                (float*) edgeData,
                                (size_t) ((MCSRG->get_num_edges()) * sizeof(float)),
                                cudaMemcpyDefault);
            }
            else if (descrG->T == CUDA_R_64F)
                    {
                nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                if (setnum >= MCSRG->get_num_edge_dim()) // base index is 0
                    return NVGRAPH_STATUS_INVALID_VALUE;
                cudaMemcpy(MCSRG->get_raw_edge_dim(setnum),
                                (double*) edgeData,
                                (size_t) ((MCSRG->get_num_edges()) * sizeof(double)),
                                cudaMemcpyDefault);
            }
            else if (descrG->T == CUDA_R_32I)
                    {
                nvgraph::MultiValuedCsrGraph<int, int> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, int>*>(descrG->graph_handle);
                if (setnum >= MCSRG->get_num_edge_dim()) // base index is 0
                    return NVGRAPH_STATUS_INVALID_VALUE;
                cudaMemcpy(MCSRG->get_raw_edge_dim(setnum),
                                (int*) edgeData,
                                (size_t) ((MCSRG->get_num_edges()) * sizeof(int)),
                                cudaMemcpyDefault);
            }
            else
                return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;

            cudaCheckError();
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphGetEdgeData_impl(nvgraphHandle_t handle,
                                                        nvgraphGraphDescr_t descrG,
                                                        void *edgeData,
                                                        size_t setnum) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_size(setnum)
                    || check_ptr(edgeData))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (descrG->T == CUDA_R_32F)
                    {
                nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                if (setnum >= MCSRG->get_num_edge_dim()) // base index is 0
                    return NVGRAPH_STATUS_INVALID_VALUE;
                cudaMemcpy((float*) edgeData,
                                MCSRG->get_raw_edge_dim(setnum),
                                (size_t) ((MCSRG->get_num_edges()) * sizeof(float)),
                                cudaMemcpyDefault);
            }
            else if (descrG->T == CUDA_R_64F)
                    {
                nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                        static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                if (setnum >= MCSRG->get_num_edge_dim()) // base index is 0
                    return NVGRAPH_STATUS_INVALID_VALUE;
                cudaMemcpy((double*) edgeData,
                                MCSRG->get_raw_edge_dim(setnum),
                                (size_t) ((MCSRG->get_num_edges()) * sizeof(double)),
                                cudaMemcpyDefault);
            }
            else
                return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;

            cudaCheckError();
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphSrSpmv_impl_cub(nvgraphHandle_t handle,
                                                       const nvgraphGraphDescr_t descrG,
                                                       const size_t weight_index,
                                                       const void *alpha,
                                                       const size_t x,
                                                       const void *beta,
                                                       const size_t y,
                                                       const nvgraphSemiring_t SR) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;

        try
        {
            // some basic checks
            if (check_context(handle) || check_graph(descrG) || check_int_size(weight_index))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
            rc = SemiringAPILauncher(handle, descrG, weight_index, alpha, x, beta, y, SR);
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphSssp_impl(nvgraphHandle_t handle,
                                                 const nvgraphGraphDescr_t descrG,
                                                 const size_t weight_index,
                                                 const int *source_vert,
                                                 const size_t sssp) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_size(weight_index)
                    || check_int_ptr(source_vert))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->TT != NVGRAPH_CSC_32) // supported topologies
                return NVGRAPH_STATUS_INVALID_VALUE;
//        cudaError_t cuda_status;

            if (descrG->graphStatus != HAS_VALUES)
                return NVGRAPH_STATUS_INVALID_VALUE;

            switch (descrG->T)
            {
                case CUDA_R_32F:
                    {
                    nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim() || sssp >= MCSRG->get_num_vertex_dim()) // base index is 0
                        return NVGRAPH_STATUS_INVALID_VALUE;

                    int n = static_cast<int>(MCSRG->get_num_vertices());
                    nvgraph::Vector<float> co(n, handle->stream);
                    nvgraph::Sssp<int, float> sssp_solver(*MCSRG->get_valued_csr_graph(weight_index));
                    nvgraph::set_connectivity<int, float>(n, *source_vert, 0.0, FLT_MAX, co.raw());
                    MCSRG->get_vertex_dim(sssp).copy(co);
                    rc = sssp_solver.solve(*source_vert, co, MCSRG->get_vertex_dim(sssp));
                    break;
                }
                case CUDA_R_64F:
                    {
                    nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim() || sssp >= MCSRG->get_num_vertex_dim()) // base index is 0
                        return NVGRAPH_STATUS_INVALID_VALUE;

                    int n = static_cast<int>(MCSRG->get_num_vertices());
                    nvgraph::Vector<double> co(n, handle->stream);
                    nvgraph::Sssp<int, double> sssp_solver(*MCSRG->get_valued_csr_graph(weight_index));
                    nvgraph::set_connectivity<int, double>(n, *source_vert, 0.0, DBL_MAX, co.raw());
                    MCSRG->get_vertex_dim(sssp).copy(co);
                    rc = sssp_solver.solve(*source_vert, co, MCSRG->get_vertex_dim(sssp));
                    break;
                }
                default:
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
            }
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphTraversal_impl(nvgraphHandle_t handle,
                                                      const nvgraphGraphDescr_t descrG,
                                                      const nvgraphTraversal_t traversalT,
                                                      const int *source_vertex_ptr,
                                                      const nvgraphTraversalParameter_t params) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_ptr(source_vertex_ptr))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph (storing results)
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (descrG->TT != NVGRAPH_CSR_32) // supported topologies
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (descrG->T != CUDA_R_32I) //results are ints
                return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;

            //Results (bfs distances, predecessors..) are written in dimension in mvcsrg
            nvgraph::MultiValuedCsrGraph<int, int> *MCSRG = static_cast<nvgraph::MultiValuedCsrGraph<
                    int, int>*>(descrG->graph_handle);

            //
            //Computing traversal parameters
            //

            size_t distancesIndex, predecessorsIndex, edgeMaskIndex;
            size_t undirectedFlagParam;
            size_t alpha_ul, beta_ul;

            int *distances = NULL, *predecessors = NULL, *edge_mask = NULL;

            nvgraphTraversalGetDistancesIndex(params, &distancesIndex);
            nvgraphTraversalGetPredecessorsIndex(params, &predecessorsIndex);
            nvgraphTraversalGetEdgeMaskIndex(params, &edgeMaskIndex);
            nvgraphTraversalGetUndirectedFlag(params, &undirectedFlagParam);
            nvgraphTraversalGetAlpha(params, &alpha_ul);
            nvgraphTraversalGetBeta(params, &beta_ul);

            int alpha = static_cast<int>(alpha_ul);
            int beta = static_cast<int>(beta_ul);

            //If distances_index was set by user, then use it
            if (distancesIndex <= MCSRG->get_num_vertex_dim()) {
                distances = MCSRG->get_vertex_dim(distancesIndex).raw();
            }

            //If predecessors_index was set by user, then use it
            if (predecessorsIndex <= MCSRG->get_num_vertex_dim()) {
                predecessors = MCSRG->get_vertex_dim(predecessorsIndex).raw();
            }

            //If edgemask_index was set by user, then use it
            if (edgeMaskIndex <= MCSRG->get_num_vertex_dim()) {
                edge_mask = MCSRG->get_edge_dim(edgeMaskIndex).raw();
            }

            int source_vertex = *source_vertex_ptr;

            int n = static_cast<int>(MCSRG->get_num_vertices());
            int nnz = static_cast<int>(MCSRG->get_num_edges());
            int *row_offsets = MCSRG->get_raw_row_offsets();
            int *col_indices = MCSRG->get_raw_column_indices();

            bool undirected = (bool) undirectedFlagParam;

            if (source_vertex < 0 || source_vertex >= n) {
                return NVGRAPH_STATUS_INVALID_VALUE;
            }

            //Calling corresponding implementation
            switch (traversalT) {
                case NVGRAPH_TRAVERSAL_BFS:
                    nvgraph::Bfs<int> bfs_solver(n,
                                                 nnz,
                                                 row_offsets,
                                                 col_indices,
                                                 !undirected,
                                                 alpha,
                                                 beta,
                                                 handle->stream);

                    //To easily implement multi source with single source,
                    //loop on those two
                    rc = bfs_solver.configure(distances, predecessors, edge_mask);
                    rc = bfs_solver.traverse(source_vertex);
                    break;
            };

        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    /**
     * CAPI Method for calling 2d BFS algorithm.
     * @param handle Nvgraph context handle.
     * @param descrG Graph handle (must be 2D partitioned)
     * @param source_vert The source vertex ID
     * @param distances Pointer to memory allocated to store the distances.
     * @param predecessors Pointer to memory allocated to store the predecessors
     * @return Status code.
     */
    nvgraphStatus_t NVGRAPH_API nvgraph2dBfs_impl(nvgraphHandle_t handle,
                                                  const nvgraphGraphDescr_t descrG,
                                                  const int32_t source_vert,
                                                  int32_t* distances,
                                                  int32_t* predecessors) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try {
            if (check_context(handle) || check_graph(descrG))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
            if (descrG->graphStatus == IS_EMPTY)
                return NVGRAPH_STATUS_INVALID_VALUE;
            if (descrG->TT != NVGRAPH_2D_32I_32I)
                return NVGRAPH_STATUS_INVALID_VALUE;
            if (descrG->T != CUDA_R_32I)
                return NVGRAPH_STATUS_INVALID_VALUE;
            nvgraph::Matrix2d<int32_t, int32_t, int32_t>* m = static_cast<nvgraph::Matrix2d<int32_t,
                    int32_t, int32_t>*>(descrG->graph_handle);
//            std::cout << m->toString();
            nvgraph::Bfs2d<int32_t, int32_t, int32_t> bfs(m, true, 0, 0);
            rc = bfs.configure(distances, predecessors);
            rc = bfs.traverse(source_vert);
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphWidestPath_impl(nvgraphHandle_t handle,
                                                       const nvgraphGraphDescr_t descrG,
                                                       const size_t weight_index,
                                                       const int *source_vert,
                                                       const size_t widest_path) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_size(weight_index)
                    || check_int_ptr(source_vert))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (descrG->TT != NVGRAPH_CSC_32) // supported topologies
                return NVGRAPH_STATUS_INVALID_VALUE;

//        cudaError_t cuda_status;

            switch (descrG->T)
            {
                case CUDA_R_32F:
                    {
                    nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim()
                            || widest_path >= MCSRG->get_num_vertex_dim()) // base index is 0
                        return NVGRAPH_STATUS_INVALID_VALUE;

                    int n = static_cast<int>(MCSRG->get_num_vertices());
                    nvgraph::Vector<float> co(n, handle->stream);
                    nvgraph::WidestPath<int, float> widest_path_solver(*MCSRG->get_valued_csr_graph(weight_index));
                    nvgraph::set_connectivity<int, float>(n, *source_vert, FLT_MAX, -FLT_MAX, co.raw());
                    MCSRG->get_vertex_dim(widest_path).copy(co);
                    rc = widest_path_solver.solve(*source_vert, co, MCSRG->get_vertex_dim(widest_path));
                    break;
                }
                case CUDA_R_64F:
                    {
                    nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim()
                            || widest_path >= MCSRG->get_num_vertex_dim()) // base index is 0
                        return NVGRAPH_STATUS_INVALID_VALUE;

                    int n = static_cast<int>(MCSRG->get_num_vertices());
                    nvgraph::Vector<double> co(n, handle->stream);
                    nvgraph::WidestPath<int, double> widest_path_solver(*MCSRG->get_valued_csr_graph(weight_index));
                    nvgraph::set_connectivity<int, double>(n, *source_vert, DBL_MAX, -DBL_MAX, co.raw());
                    MCSRG->get_vertex_dim(widest_path).copy(co);
                    rc = widest_path_solver.solve(*source_vert, co, MCSRG->get_vertex_dim(widest_path));
                    break;
                }
                default:
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
            }
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphPagerank_impl(nvgraphHandle_t handle,
                                                     const nvgraphGraphDescr_t descrG,
                                                     const size_t weight_index,
                                                     const void *alpha,
                                                     const size_t bookmark,
                                                     const int has_guess,
                                                     const size_t rank,
                                                     const float tolerance,
                                                     const int max_iter) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_size(weight_index)
                    || check_ptr(alpha))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (descrG->TT != NVGRAPH_CSC_32) // supported topologies
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (!(has_guess == 0 || has_guess == 1))
                return NVGRAPH_STATUS_INVALID_VALUE;

            int max_it;
            float tol;

            if (max_iter > 0)
                max_it = max_iter;
            else
                max_it = 500;

            if (tolerance == 0.0f)
                tol = 1.0E-6f;
            else if (tolerance < 1.0f && tolerance > 0.0f)
                tol = tolerance;
            else
                return NVGRAPH_STATUS_INVALID_VALUE;

            switch (descrG->T)
            {
                case CUDA_R_32F:
                    {
                    float alphaT = *static_cast<const float*>(alpha);
                    if (alphaT <= 0.0f || alphaT >= 1.0f)
                        return NVGRAPH_STATUS_INVALID_VALUE;
                    nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim()
                            || bookmark >= MCSRG->get_num_vertex_dim()
                            || rank >= MCSRG->get_num_vertex_dim()) // base index is 0
                        return NVGRAPH_STATUS_INVALID_VALUE;

                    int n = static_cast<int>(MCSRG->get_num_vertices());
                    nvgraph::Vector<float> guess(n, handle->stream);
                    nvgraph::Vector<float> bm(n, handle->stream);
                    if (has_guess)
                        guess.copy(MCSRG->get_vertex_dim(rank));
                    else
                        guess.fill(static_cast<float>(1.0 / n));
                    bm.copy(MCSRG->get_vertex_dim(bookmark));
                    nvgraph::Pagerank<int, float> pagerank_solver(*MCSRG->get_valued_csr_graph(weight_index), bm);
                    rc = pagerank_solver.solve(alphaT, guess, MCSRG->get_vertex_dim(rank), tol, max_it);
                    break;
                }
                case CUDA_R_64F:
                    {
                    double alphaT = *static_cast<const double*>(alpha);
                    if (alphaT <= 0.0 || alphaT >= 1.0)
                        return NVGRAPH_STATUS_INVALID_VALUE;

                    nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim()
                            || bookmark >= MCSRG->get_num_vertex_dim()
                            || rank >= MCSRG->get_num_vertex_dim()) // base index is 0
                        return NVGRAPH_STATUS_INVALID_VALUE;

                    int n = static_cast<int>(MCSRG->get_num_vertices());
                    nvgraph::Vector<double> guess(n, handle->stream);
                    nvgraph::Vector<double> bm(n, handle->stream);
                    bm.copy(MCSRG->get_vertex_dim(bookmark));
                    if (has_guess)
                        guess.copy(MCSRG->get_vertex_dim(rank));
                    else
                        guess.fill(static_cast<float>(1.0 / n));
                    nvgraph::Pagerank<int, double> pagerank_solver(*MCSRG->get_valued_csr_graph(weight_index), bm);
                    rc = pagerank_solver.solve(alphaT, guess, MCSRG->get_vertex_dim(rank), tol, max_it);
                    break;
                }
                default:
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
            }
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphKrylovPagerank_impl(nvgraphHandle_t handle,
                                                           const nvgraphGraphDescr_t descrG,
                                                           const size_t weight_index,
                                                           const void *alpha,
                                                           const size_t bookmark,
                                                           const float tolerance,
                                                           const int max_iter,
                                                           const int subspace_size,
                                                           const int has_guess,
                                                           const size_t rank) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_size(weight_index)
                    || check_ptr(alpha))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (descrG->TT != NVGRAPH_CSC_32) // supported topologies
                return NVGRAPH_STATUS_INVALID_VALUE;

//        cudaError_t cuda_status;
            int max_it;
            int ss_sz;
            float tol;

            if (max_iter > 0)
                max_it = max_iter;
            else
                max_it = 500;

            if (subspace_size > 0)
                ss_sz = subspace_size;
            else
                ss_sz = 8;

            if (tolerance == 0.0f)
                tol = 1.0E-6f;
            else if (tolerance < 1.0f && tolerance > 0.0f)
                tol = tolerance;
            else
                return NVGRAPH_STATUS_INVALID_VALUE;

            switch (descrG->T)
            {
                case CUDA_R_32F:
                    {
                    float alphaT = *static_cast<const float*>(alpha);
                    if (alphaT <= 0.0f || alphaT >= 1.0f)
                        return NVGRAPH_STATUS_INVALID_VALUE;
                    nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim()
                            || bookmark >= MCSRG->get_num_vertex_dim()
                            || rank >= MCSRG->get_num_vertex_dim()) // base index is 0
                        return NVGRAPH_STATUS_INVALID_VALUE;

                    int n = static_cast<int>(MCSRG->get_num_vertices());
                    nvgraph::Vector<float> guess(n, handle->stream), eigVals(1, handle->stream);
                    if (has_guess)
                        guess.copy(MCSRG->get_vertex_dim(rank));
                    else
                        guess.fill(static_cast<float>(1.0 / n));
                    nvgraph::ImplicitArnoldi<int, float> iram_solver(*MCSRG->get_valued_csr_graph(weight_index),
                                                                     MCSRG->get_vertex_dim(bookmark),
                                                                     tol,
                                                                     max_it,
                                                                     alphaT);
                    rc = iram_solver.solve(ss_sz, 1, guess, eigVals, MCSRG->get_vertex_dim(rank));
                    break;
                }
                case CUDA_R_64F:
                    {
                    // curently iram solver accept float for alpha
                    double alphaTemp = *static_cast<const double*>(alpha);
                    float alphaT = static_cast<float>(alphaTemp);
                    if (alphaT <= 0.0f || alphaT >= 1.0f)
                        return NVGRAPH_STATUS_INVALID_VALUE;
                    nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim()
                            || bookmark >= MCSRG->get_num_vertex_dim()
                            || rank >= MCSRG->get_num_vertex_dim()) // base index is 0
                        return NVGRAPH_STATUS_INVALID_VALUE;

                    int n = static_cast<int>(MCSRG->get_num_vertices());
                    nvgraph::Vector<double> guess(n, handle->stream), eigVals(1, handle->stream);
                    if (has_guess)
                        guess.copy(MCSRG->get_vertex_dim(rank));
                    else
                        guess.fill(static_cast<float>(1.0 / n));
                    nvgraph::ImplicitArnoldi<int, double> iram_solver(*MCSRG->get_valued_csr_graph(weight_index),
                                                                      MCSRG->get_vertex_dim(bookmark),
                                                                      tol,
                                                                      max_it,
                                                                      alphaT);
                    rc = iram_solver.solve(ss_sz, 1, guess, eigVals, MCSRG->get_vertex_dim(rank));
                    break;
                }
                default:
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
            }
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphExtractSubgraphByVertex_impl(nvgraphHandle_t handle,
                                                                    nvgraphGraphDescr_t descrG,
                                                                    nvgraphGraphDescr_t subdescrG,
                                                                    int *subvertices,
                                                                    size_t numvertices) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        typedef int IndexType;

        try
        {
            if (check_context(handle) ||
                    check_graph(descrG) ||
                    !subdescrG ||
                    check_int_size(numvertices) ||
                    check_ptr(subvertices))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (!numvertices)
                return NVGRAPH_STATUS_INVALID_VALUE;

            subdescrG->TT = descrG->TT;
            subdescrG->T = descrG->T;

            switch (descrG->graphStatus)
            {
                case HAS_TOPOLOGY: //CsrGraph
                {
                    nvgraph::CsrGraph<int> *CSRG =
                            static_cast<nvgraph::CsrGraph<IndexType>*>(descrG->graph_handle);

                    Graph<IndexType>* subgraph = extract_subgraph_by_vertices(*CSRG,
                                                                              subvertices,
                                                                              numvertices,
                                                                              handle->stream);

                    subdescrG->graph_handle = subgraph;
                    subdescrG->graphStatus = HAS_TOPOLOGY;
                }
                    break;

                case HAS_VALUES: //MultiValuedCsrGraph
                    if (descrG->T == CUDA_R_32F)
                            {
                        nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                                static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);

                        nvgraph::MultiValuedCsrGraph<int, float>* subgraph =
                                extract_subgraph_by_vertices(*MCSRG,
                                                             subvertices,
                                                             numvertices,
                                                             handle->stream);

                        subdescrG->graph_handle = subgraph;
                        subdescrG->graphStatus = HAS_VALUES;
                    }
                    else if (descrG->T == CUDA_R_64F)
                            {
                        nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                                static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);

                        nvgraph::MultiValuedCsrGraph<int, double>* subgraph =
                                extract_subgraph_by_vertices(*MCSRG,
                                                             subvertices,
                                                             numvertices,
                                                             handle->stream);

                        subdescrG->graph_handle = subgraph;
                        subdescrG->graphStatus = HAS_VALUES;
                    }
                    else
                        return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
                    break;

                default:
                    return NVGRAPH_STATUS_INVALID_VALUE;
            }
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphExtractSubgraphByEdge_impl(nvgraphHandle_t handle,
                                                                  nvgraphGraphDescr_t descrG,
                                                                  nvgraphGraphDescr_t subdescrG,
                                                                  int *subedges,
                                                                  size_t numedges) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        //TODO: extract handle->stream info, from handler/nvgraphContext (?)
        typedef int IndexType;

        try
        {
            if (check_context(handle) ||
                    check_graph(descrG) ||
                    !subdescrG ||
                    check_int_size(numedges) ||
                    check_ptr(subedges))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (!numedges)
                return NVGRAPH_STATUS_INVALID_VALUE;

            subdescrG->TT = descrG->TT;
            subdescrG->T = descrG->T;

            switch (descrG->graphStatus)
            {
                case HAS_TOPOLOGY: //CsrGraph
                {
                    nvgraph::CsrGraph<int> *CSRG =
                            static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
                    Graph<IndexType>* subgraph = extract_subgraph_by_edges(*CSRG,
                                                                           subedges,
                                                                           numedges,
                                                                           handle->stream);

                    subdescrG->graph_handle = subgraph;
                    subdescrG->graphStatus = HAS_TOPOLOGY;
                }
                    break;

                case HAS_VALUES: //MultiValuedCsrGraph
                    if (descrG->T == CUDA_R_32F)
                            {
                        nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                                static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);

                        nvgraph::MultiValuedCsrGraph<int, float>* subgraph =
                                extract_subgraph_by_edges(*MCSRG, subedges, numedges, handle->stream);

                        subdescrG->graph_handle = subgraph;
                        subdescrG->graphStatus = HAS_VALUES;
                    }
                    else if (descrG->T == CUDA_R_64F)
                            {
                        nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                                static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);

                        nvgraph::MultiValuedCsrGraph<int, double>* subgraph =
                                extract_subgraph_by_edges(*MCSRG, subedges, numedges, handle->stream);

                        subdescrG->graph_handle = subgraph;
                        subdescrG->graphStatus = HAS_VALUES;
                    }
                    else
                        return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
                    break;

                default:
                    return NVGRAPH_STATUS_INVALID_VALUE;
            }
        }
        NVGRAPH_CATCHES(rc)

        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphBalancedCutClustering_impl(nvgraphHandle_t handle,
                                                                  const nvgraphGraphDescr_t descrG,
                                                                  const size_t weight_index,
                                                                  const int n_clusters,
                                                                  const int n_eig_vects,
                                                                  const int evs_type,
                                                                  const float evs_tolerance,
                                                                  const int evs_max_iter,
                                                                  const float kmean_tolerance,
                                                                  const int kmean_max_iter,
                                                                  int* clustering,
                                                                  void* eig_vals,
                                                                  void* eig_vects) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_size(weight_index))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (descrG->TT != NVGRAPH_CSR_32) // supported topologies
                return NVGRAPH_STATUS_INVALID_VALUE;

            int evs_max_it, kmean_max_it;
            int iters_lanczos, iters_kmeans;
            float evs_tol, kmean_tol;

            if (evs_max_iter > 0)
                evs_max_it = evs_max_iter;
            else
                evs_max_it = 4000;

            if (evs_tolerance == 0.0f)
                evs_tol = 1.0E-3f;
            else if (evs_tolerance < 1.0f && evs_tolerance > 0.0f)
                evs_tol = evs_tolerance;
            else
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (kmean_max_iter > 0)
                kmean_max_it = kmean_max_iter;
            else
                kmean_max_it = 200;

            if (kmean_tolerance == 0.0f)
                kmean_tol = 1.0E-2f;
            else if (kmean_tolerance < 1.0f && kmean_tolerance > 0.0f)
                kmean_tol = kmean_tolerance;
            else
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (n_clusters < 2)
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (n_eig_vects > n_clusters)
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (!(evs_type == 0 || evs_type == 1))
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (clustering == NULL || eig_vals == NULL || eig_vects == NULL)
                return NVGRAPH_STATUS_INVALID_VALUE;

            switch (descrG->T)
            {
                case CUDA_R_32F:
                    {
                    nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim()
                            || n_clusters > static_cast<int>(MCSRG->get_num_vertices())) // base index is 0
                        return NVGRAPH_STATUS_INVALID_VALUE;
                    nvgraph::ValuedCsrGraph<int, float> network =
                            *MCSRG->get_valued_csr_graph(weight_index);
                    Vector<int> clust(MCSRG->get_num_vertices(), handle->stream);
                    Vector<float> eigVals(n_eig_vects, handle->stream);
                    Vector<float> eigVecs(MCSRG->get_num_vertices() * n_eig_vects, handle->stream);

                    if (evs_type == 0)
                            {
                        int restartIter_lanczos = 15 + n_eig_vects;
                        rc = partition<int, float>(network,
                                                   n_clusters,
                                                   n_eig_vects,
                                                   evs_max_it,
                                                   restartIter_lanczos,
                                                   evs_tol,
                                                   kmean_max_it,
                                                   kmean_tol,
                                                   clust.raw(),
                                                   eigVals,
                                                   eigVecs,
                                                   iters_lanczos,
                                                   iters_kmeans);
                    }
                    else
                    {
                        cusolverDnHandle_t cusolverHandle;
                        cusolverDnCreate(&cusolverHandle);
                        rc = partition_lobpcg<int, float>(network,
                                                          NULL, // preconditioner
                                                          cusolverHandle,
                                                          n_clusters,
                                                          n_eig_vects,
                                                          evs_max_it,
                                                          evs_tol,
                                                          kmean_max_it,
                                                          kmean_tol,
                                                          clust.raw(),
                                                          eigVals,
                                                          eigVecs,
                                                          iters_lanczos,
                                                          iters_kmeans);
                    }
                    // give a copy of results to the user
                    if (rc == NVGRAPH_OK)
                            {
                        CHECK_CUDA(cudaMemcpy((int* )clustering,
                                                        clust.raw(),
                                                        (size_t )(MCSRG->get_num_vertices() * sizeof(int)),
                                                        cudaMemcpyDefault));
                        CHECK_CUDA(cudaMemcpy((float* )eig_vals,
                                                        eigVals.raw(),
                                                        (size_t )(n_eig_vects * sizeof(float)),
                                                        cudaMemcpyDefault));
                        CHECK_CUDA(cudaMemcpy((float* )eig_vects,
                                                        eigVecs.raw(),
                                                        (size_t )(n_eig_vects * MCSRG->get_num_vertices()
                                                                * sizeof(float)),
                                                        cudaMemcpyDefault));
                    }

                    break;
                }
                case CUDA_R_64F:
                    {
                    nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim()
                            || n_clusters > static_cast<int>(MCSRG->get_num_vertices())) // base index is 0
                        return NVGRAPH_STATUS_INVALID_VALUE;
                    nvgraph::ValuedCsrGraph<int, double> network =
                            *MCSRG->get_valued_csr_graph(weight_index);
                    Vector<int> clust(MCSRG->get_num_vertices(), handle->stream);
                    Vector<double> eigVals(n_eig_vects, handle->stream);
                    Vector<double> eigVecs(MCSRG->get_num_vertices() * n_eig_vects, handle->stream);
                    if (evs_type == 0)
                            {
                        int restartIter_lanczos = 15 + n_eig_vects;
                        rc = partition<int, double>(network,
                                                    n_clusters,
                                                    n_eig_vects,
                                                    evs_max_it,
                                                    restartIter_lanczos,
                                                    evs_tol,
                                                    kmean_max_it,
                                                    kmean_tol,
                                                    clust.raw(),
                                                    eigVals,
                                                    eigVecs,
                                                    iters_lanczos,
                                                    iters_kmeans);
                    }
                    else
                    {
                        cusolverDnHandle_t cusolverHandle;
                        cusolverDnCreate(&cusolverHandle);
                        rc = partition_lobpcg<int, double>(network,
                                                           NULL, // preconditioner
                                                           cusolverHandle,
                                                           n_clusters,
                                                           n_eig_vects,
                                                           evs_max_it,
                                                           evs_tol,
                                                           kmean_max_it,
                                                           kmean_tol,
                                                           clust.raw(),
                                                           eigVals,
                                                           eigVecs,
                                                           iters_lanczos,
                                                           iters_kmeans);
                    }
                    // give a copy of results to the user
                    if (rc == NVGRAPH_OK)
                            {
                        CHECK_CUDA(cudaMemcpy((int* )clustering,
                                                        clust.raw(),
                                                        (size_t )(MCSRG->get_num_vertices() * sizeof(int)),
                                                        cudaMemcpyDefault));
                        CHECK_CUDA(cudaMemcpy((double* )eig_vals,
                                                        eigVals.raw(),
                                                        (size_t )(n_eig_vects * sizeof(double)),
                                                        cudaMemcpyDefault));
                        CHECK_CUDA(cudaMemcpy((double* )eig_vects,
                                                        eigVecs.raw(),
                                                        (size_t )(n_eig_vects * MCSRG->get_num_vertices()
                                                                * sizeof(double)),
                                                        cudaMemcpyDefault));
                    }
                    break;
                }
                default:
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
            }
        }
        NVGRAPH_CATCHES(rc)
        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphAnalyzeBalancedCut_impl(nvgraphHandle_t handle,
                                                               const nvgraphGraphDescr_t descrG,
                                                               const size_t weight_index,
                                                               const int n_clusters,
                                                               const int* clustering,
                                                               float * edgeCut,
                                                               float * ratioCut) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_size(weight_index))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (descrG->TT != NVGRAPH_CSR_32) // supported topologies
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (n_clusters < 2)
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (clustering == NULL || edgeCut == NULL || ratioCut == NULL)
                return NVGRAPH_STATUS_INVALID_VALUE;

            switch (descrG->T)
            {
                case CUDA_R_32F:
                    {
                    float edge_cut, ratio_cut;
                    nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim()
                            || n_clusters > static_cast<int>(MCSRG->get_num_vertices()))
                        return NVGRAPH_STATUS_INVALID_VALUE;
                    nvgraph::ValuedCsrGraph<int, float> network =
                            *MCSRG->get_valued_csr_graph(weight_index);
                    Vector<int> clust(MCSRG->get_num_vertices(), handle->stream);
                    CHECK_CUDA(cudaMemcpy(clust.raw(),
                                                    (int* )clustering,
                                                    (size_t )(MCSRG->get_num_vertices() * sizeof(int)),
                                                    cudaMemcpyDefault));
                    rc = analyzePartition<int, float>(network,
                                                      n_clusters,
                                                      clust.raw(),
                                                      edge_cut,
                                                      ratio_cut);
                    *edgeCut = edge_cut;
                    *ratioCut = ratio_cut;
                    break;
                }
                case CUDA_R_64F:
                    {
                    double edge_cut, ratio_cut;
                    nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim()
                            || n_clusters > static_cast<int>(MCSRG->get_num_vertices())) // base index is 0
                        return NVGRAPH_STATUS_INVALID_VALUE;
                    nvgraph::ValuedCsrGraph<int, double> network =
                            *MCSRG->get_valued_csr_graph(weight_index);
                    Vector<int> clust(MCSRG->get_num_vertices(), handle->stream);
                    CHECK_CUDA(cudaMemcpy(clust.raw(),
                                                    (int* )clustering,
                                                    (size_t )(MCSRG->get_num_vertices() * sizeof(int)),
                                                    cudaMemcpyDefault));
                    rc = analyzePartition<int, double>(network,
                                                       n_clusters,
                                                       clust.raw(),
                                                       edge_cut,
                                                       ratio_cut);
                    *edgeCut = static_cast<float>(edge_cut);
                    *ratioCut = static_cast<float>(ratio_cut);
                    break;
                }

                default:
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
            }
        }
        NVGRAPH_CATCHES(rc)
        return getCAPIStatusForError(rc);

    }

    nvgraphStatus_t NVGRAPH_API nvgraphHeavyEdgeMatching_impl(nvgraphHandle_t handle,
                                                              const nvgraphGraphDescr_t descrG,
                                                              const size_t weight_index,
                                                              const nvgraphEdgeWeightMatching_t similarity_metric,
                                                              int* aggregates,
                                                              size_t* num_aggregates) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_size(weight_index))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (descrG->TT != NVGRAPH_CSR_32) // supported topologies
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (aggregates == NULL)
                return NVGRAPH_STATUS_INVALID_VALUE;
            Matching_t sim_metric;
            switch (similarity_metric)
            {
                case NVGRAPH_UNSCALED: {
                    sim_metric = USER_PROVIDED;
                    break;
                }
                case NVGRAPH_SCALED_BY_ROW_SUM: {
                    sim_metric = SCALED_BY_ROW_SUM;
                    break;
                }
                case NVGRAPH_SCALED_BY_DIAGONAL: {
                    sim_metric = SCALED_BY_DIAGONAL;
                    break;
                }
                default:
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
            }

            switch (descrG->T)
            {
                case CUDA_R_32F:
                    {
                    nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim())
                        return NVGRAPH_STATUS_INVALID_VALUE;
                    nvgraph::ValuedCsrGraph<int, float> network =
                            *MCSRG->get_valued_csr_graph(weight_index);
                    Vector<int> agg(MCSRG->get_num_vertices(), handle->stream);
                    int num_agg = 0;
                    nvgraph::Size2Selector<int, float> one_phase_hand_checking(sim_metric);
                    rc = one_phase_hand_checking.setAggregates(network, agg, num_agg);
                    *num_aggregates = static_cast<size_t>(num_agg);
                    CHECK_CUDA(cudaMemcpy((int* )aggregates,
                                                    agg.raw(),
                                                    (size_t )(MCSRG->get_num_vertices() * sizeof(int)),
                                                    cudaMemcpyDefault));
                    break;
                }
                case CUDA_R_64F:
                    {
                    nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim())
                        return NVGRAPH_STATUS_INVALID_VALUE;
                    nvgraph::ValuedCsrGraph<int, double> network =
                            *MCSRG->get_valued_csr_graph(weight_index);
                    Vector<int> agg(MCSRG->get_num_vertices(), handle->stream);
                    Vector<int> agg_global(MCSRG->get_num_vertices(), handle->stream);
                    int num_agg = 0;
                    nvgraph::Size2Selector<int, double> one_phase_hand_checking(sim_metric);
                    rc = one_phase_hand_checking.setAggregates(network, agg, num_agg);
                    *num_aggregates = static_cast<size_t>(num_agg);
                    CHECK_CUDA(cudaMemcpy((int* )aggregates,
                                                    agg.raw(),
                                                    (size_t )(MCSRG->get_num_vertices() * sizeof(int)),
                                                    cudaMemcpyDefault));
                    break;
                }
                default:
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
            }
        }
        NVGRAPH_CATCHES(rc)
        return getCAPIStatusForError(rc);

    }

    nvgraphStatus_t NVGRAPH_API nvgraphSpectralModularityMaximization_impl(nvgraphHandle_t handle,
                                                                           const nvgraphGraphDescr_t descrG,
                                                                           const size_t weight_index,
                                                                           const int n_clusters,
                                                                           const int n_eig_vects,
                                                                           const float evs_tolerance,
                                                                           const int evs_max_iter,
                                                                           const float kmean_tolerance,
                                                                           const int kmean_max_iter,
                                                                           int* clustering,
                                                                           void* eig_vals,
                                                                           void* eig_vects) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_size(weight_index))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (descrG->TT != NVGRAPH_CSR_32) // supported topologies
                return NVGRAPH_STATUS_GRAPH_TYPE_NOT_SUPPORTED;

            int evs_max_it, kmean_max_it;
            int iters_lanczos, iters_kmeans;
            float evs_tol, kmean_tol;

            if (evs_max_iter > 0)
                evs_max_it = evs_max_iter;
            else
                evs_max_it = 4000;

            if (evs_tolerance == 0.0f)
                evs_tol = 1.0E-3f;
            else if (evs_tolerance < 1.0f && evs_tolerance > 0.0f)
                evs_tol = evs_tolerance;
            else
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (kmean_max_iter > 0)
                kmean_max_it = kmean_max_iter;
            else
                kmean_max_it = 200;

            if (kmean_tolerance == 0.0f)
                kmean_tol = 1.0E-2f;
            else if (kmean_tolerance < 1.0f && kmean_tolerance > 0.0f)
                kmean_tol = kmean_tolerance;
            else
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (n_clusters < 2)
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (n_eig_vects > n_clusters)
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (clustering == NULL || eig_vals == NULL || eig_vects == NULL)
                return NVGRAPH_STATUS_INVALID_VALUE;

            switch (descrG->T)
            {
                case CUDA_R_32F:
                    {
                    nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim()
                            || n_clusters > static_cast<int>(MCSRG->get_num_vertices())) // base index is 0
                        return NVGRAPH_STATUS_INVALID_VALUE;
                    nvgraph::ValuedCsrGraph<int, float> network =
                            *MCSRG->get_valued_csr_graph(weight_index);
                    Vector<int> clust(MCSRG->get_num_vertices(), handle->stream);
                    Vector<float> eigVals(n_eig_vects, handle->stream);
                    Vector<float> eigVecs(MCSRG->get_num_vertices() * n_eig_vects, handle->stream);
                    int restartIter_lanczos = 15 + n_eig_vects;
                    rc = modularity_maximization<int, float>(network,
                                                             n_clusters,
                                                             n_eig_vects,
                                                             evs_max_it,
                                                             restartIter_lanczos,
                                                             evs_tol,
                                                             kmean_max_it,
                                                             kmean_tol,
                                                             clust.raw(),
                                                             eigVals,
                                                             eigVecs,
                                                             iters_lanczos,
                                                             iters_kmeans);

                    // give a copy of results to the user
                    if (rc == NVGRAPH_OK)
                            {
                        CHECK_CUDA(cudaMemcpy((int* )clustering,
                                                        clust.raw(),
                                                        (size_t )(MCSRG->get_num_vertices() * sizeof(int)),
                                                        cudaMemcpyDefault));
                        CHECK_CUDA(cudaMemcpy((float* )eig_vals,
                                                        eigVals.raw(),
                                                        (size_t )(n_eig_vects * sizeof(float)),
                                                        cudaMemcpyDefault));
                        CHECK_CUDA(cudaMemcpy((float* )eig_vects,
                                                        eigVecs.raw(),
                                                        (size_t )(n_eig_vects * MCSRG->get_num_vertices()
                                                                * sizeof(float)),
                                                        cudaMemcpyDefault));
                    }

                    break;
                }
                case CUDA_R_64F:
                    {
                    nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim()
                            || n_clusters > static_cast<int>(MCSRG->get_num_vertices())) // base index is 0
                        return NVGRAPH_STATUS_INVALID_VALUE;
                    nvgraph::ValuedCsrGraph<int, double> network =
                            *MCSRG->get_valued_csr_graph(weight_index);
                    Vector<int> clust(MCSRG->get_num_vertices(), handle->stream);
                    Vector<double> eigVals(n_eig_vects, handle->stream);
                    Vector<double> eigVecs(MCSRG->get_num_vertices() * n_eig_vects, handle->stream);
                    int restartIter_lanczos = 15 + n_eig_vects;
                    rc = modularity_maximization<int, double>(network,
                                                              n_clusters,
                                                              n_eig_vects,
                                                              evs_max_it,
                                                              restartIter_lanczos,
                                                              evs_tol,
                                                              kmean_max_it,
                                                              kmean_tol,
                                                              clust.raw(),
                                                              eigVals,
                                                              eigVecs,
                                                              iters_lanczos,
                                                              iters_kmeans);
                    // give a copy of results to the user
                    if (rc == NVGRAPH_OK)
                            {
                        CHECK_CUDA(cudaMemcpy((int* )clustering,
                                                        clust.raw(),
                                                        (size_t )(MCSRG->get_num_vertices() * sizeof(int)),
                                                        cudaMemcpyDefault));
                        CHECK_CUDA(cudaMemcpy((double* )eig_vals,
                                                        eigVals.raw(),
                                                        (size_t )(n_eig_vects * sizeof(double)),
                                                        cudaMemcpyDefault));
                        CHECK_CUDA(cudaMemcpy((double* )eig_vects,
                                                        eigVecs.raw(),
                                                        (size_t )(n_eig_vects * MCSRG->get_num_vertices()
                                                                * sizeof(double)),
                                                        cudaMemcpyDefault));
                    }
                    break;
                }
                default:
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
            }
        }
        NVGRAPH_CATCHES(rc)
        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphAnalyzeModularityClustering_impl(nvgraphHandle_t handle,
                                                                        const nvgraphGraphDescr_t descrG,
                                                                        const size_t weight_index,
                                                                        const int n_clusters,
                                                                        const int* clustering,
                                                                        float * modularity) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_int_size(weight_index))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->graphStatus != HAS_VALUES) // need a MultiValuedCsrGraph
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (descrG->TT != NVGRAPH_CSR_32) // supported topologies
                return NVGRAPH_STATUS_GRAPH_TYPE_NOT_SUPPORTED;

            if (n_clusters < 2)
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (clustering == NULL || modularity == NULL)
                return NVGRAPH_STATUS_INVALID_VALUE;

            switch (descrG->T)
            {
                case CUDA_R_32F:
                    {
                    float mod;
                    nvgraph::MultiValuedCsrGraph<int, float> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, float>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim()
                            || n_clusters > static_cast<int>(MCSRG->get_num_vertices()))
                        return NVGRAPH_STATUS_INVALID_VALUE;
                    nvgraph::ValuedCsrGraph<int, float> network =
                            *MCSRG->get_valued_csr_graph(weight_index);
                    Vector<int> clust(MCSRG->get_num_vertices(), handle->stream);
                    CHECK_CUDA(cudaMemcpy(clust.raw(),
                                                    (int* )clustering,
                                                    (size_t )(MCSRG->get_num_vertices() * sizeof(int)),
                                                    cudaMemcpyDefault));
                    rc = analyzeModularity<int, float>(network,
                                                       n_clusters,
                                                       clust.raw(),
                                                       mod);
                    *modularity = mod;
                    break;
                }
                case CUDA_R_64F:
                    {
                    double mod;
                    nvgraph::MultiValuedCsrGraph<int, double> *MCSRG =
                            static_cast<nvgraph::MultiValuedCsrGraph<int, double>*>(descrG->graph_handle);
                    if (weight_index >= MCSRG->get_num_edge_dim()
                            || n_clusters > static_cast<int>(MCSRG->get_num_vertices())) // base index is 0
                        return NVGRAPH_STATUS_INVALID_VALUE;
                    Vector<int> clust(MCSRG->get_num_vertices(), handle->stream);
                    CHECK_CUDA(cudaMemcpy(clust.raw(),
                                                    (int* )clustering,
                                                    (size_t )(MCSRG->get_num_vertices() * sizeof(int)),
                                                    cudaMemcpyDefault));
                    nvgraph::ValuedCsrGraph<int, double> network =
                            *MCSRG->get_valued_csr_graph(weight_index);
                    rc = analyzeModularity<int, double>(network,
                                                        n_clusters,
                                                        clust.raw(),
                                                        mod);
                    *modularity = static_cast<float>(mod);
                    break;
                }

                default:
                    return NVGRAPH_STATUS_TYPE_NOT_SUPPORTED;
            }
        }
        NVGRAPH_CATCHES(rc)
        return getCAPIStatusForError(rc);
    }

    nvgraphStatus_t NVGRAPH_API nvgraphSpectralClustering_impl(nvgraphHandle_t handle, // nvGRAPH library handle.
                                                               const nvgraphGraphDescr_t descrG, // nvGRAPH graph descriptor, should contain the connectivity information in NVGRAPH_CSR_32 or NVGRAPH_CSR_32 at least 1 edge set (weights)
                                                               const size_t weight_index, // Index of the edge set for the weights.
                                                               const struct SpectralClusteringParameter *params, //parameters, see struct SpectralClusteringParameter
                                                               int* clustering, // (output) clustering
                                                               void* eig_vals, // (output) eigenvalues
                                                               void* eig_vects) {// (output) eigenvectors
        if (check_ptr(params) || check_ptr(clustering) || check_ptr(eig_vals) || check_ptr(eig_vects))
            FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
        if (params->algorithm == NVGRAPH_MODULARITY_MAXIMIZATION)
            return nvgraph::nvgraphSpectralModularityMaximization_impl(handle,
                                                                       descrG,
                                                                       weight_index,
                                                                       params->n_clusters,
                                                                       params->n_eig_vects,
                                                                       params->evs_tolerance,
                                                                       params->evs_max_iter,
                                                                       params->kmean_tolerance,
                                                                       params->kmean_max_iter,
                                                                       clustering,
                                                                       eig_vals,
                                                                       eig_vects);
        else if (params->algorithm == NVGRAPH_BALANCED_CUT_LANCZOS)
            return nvgraph::nvgraphBalancedCutClustering_impl(handle,
                                                              descrG,
                                                              weight_index,
                                                              params->n_clusters,
                                                              params->n_eig_vects,
                                                              0,
                                                              params->evs_tolerance,
                                                              params->evs_max_iter,
                                                              params->kmean_tolerance,
                                                              params->kmean_max_iter,
                                                              clustering,
                                                              eig_vals,
                                                              eig_vects);
        else if (params->algorithm == NVGRAPH_BALANCED_CUT_LOBPCG)
            return nvgraph::nvgraphBalancedCutClustering_impl(handle,
                                                              descrG,
                                                              weight_index,
                                                              params->n_clusters,
                                                              params->n_eig_vects,
                                                              1,
                                                              params->evs_tolerance,
                                                              params->evs_max_iter,
                                                              params->kmean_tolerance,
                                                              params->kmean_max_iter,
                                                              clustering,
                                                              eig_vals,
                                                              eig_vects);
        else
            return NVGRAPH_STATUS_INVALID_VALUE;
    }

    nvgraphStatus_t NVGRAPH_API nvgraphAnalyzeClustering_impl(nvgraphHandle_t handle, // nvGRAPH library handle.
                                                              const nvgraphGraphDescr_t descrG, // nvGRAPH graph descriptor, should contain the connectivity information in NVGRAPH_CSR_32 at least 1 edge set (weights)
                                                              const size_t weight_index, // Index of the edge set for the weights.
                                                              const int n_clusters, //number of clusters
                                                              const int* clustering, // clustering to analyse
                                                              nvgraphClusteringMetric_t metric, // metric to compute to measure the clustering quality
                                                              float * score) {// (output) clustering score telling how good the clustering is for the selected metric.
        if (check_ptr(clustering) || check_ptr(score))
            FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);
        if (metric == NVGRAPH_MODULARITY)
            return nvgraphAnalyzeModularityClustering_impl(handle,
                                                           descrG,
                                                           weight_index,
                                                           n_clusters,
                                                           clustering,
                                                           score);
        else if (metric == NVGRAPH_EDGE_CUT) {
            float dummy = 0;
            return nvgraph::nvgraphAnalyzeBalancedCut_impl(handle,
                                                           descrG,
                                                           weight_index,
                                                           n_clusters,
                                                           clustering,
                                                           score,
                                                           &dummy);
        }
        else if (metric == NVGRAPH_RATIO_CUT) {
            float dummy = 0;
            return nvgraph::nvgraphAnalyzeBalancedCut_impl(handle,
                                                           descrG,
                                                           weight_index,
                                                           n_clusters,
                                                           clustering,
                                                           &dummy,
                                                           score);
        }
        else
            return NVGRAPH_STATUS_INVALID_VALUE;
    }

    nvgraphStatus_t NVGRAPH_API nvgraphTriangleCount_impl(nvgraphHandle_t handle,
                                                          const nvgraphGraphDescr_t descrG,
                                                          uint64_t* result) {
        NVGRAPH_ERROR rc = NVGRAPH_OK;
        try
        {
            if (check_context(handle) || check_graph(descrG) || check_ptr(result))
                FatalError("Incorrect parameters.", NVGRAPH_ERR_BAD_PARAMETERS);

            if (descrG->TT != NVGRAPH_CSR_32 && descrG->TT != NVGRAPH_CSC_32) // supported topologies
                return NVGRAPH_STATUS_INVALID_VALUE;

            if (descrG->graphStatus != HAS_TOPOLOGY && descrG->graphStatus != HAS_VALUES)
            {
                return NVGRAPH_STATUS_INVALID_VALUE; // should have topology
            }

            nvgraph::CsrGraph<int> *CSRG = static_cast<nvgraph::CsrGraph<int>*>(descrG->graph_handle);
            if (CSRG == NULL)
                return NVGRAPH_STATUS_MAPPING_ERROR;
            nvgraph::triangles_counting::TrianglesCount<int> counter(*CSRG); /* stream, device */
            rc = counter.count();
            uint64_t s_res = counter.get_triangles_count();
            *result = static_cast<uint64_t>(s_res);

        }
        NVGRAPH_CATCHES(rc)
        return getCAPIStatusForError(rc);
    }

} /*namespace nvgraph*/

/*************************
 *        API
 *************************/

nvgraphStatus_t NVGRAPH_API nvgraphGetProperty(libraryPropertyType type, int *value) {
    switch (type) {
        case MAJOR_VERSION:
            *value = CUDART_VERSION / 1000;
            break;
        case MINOR_VERSION:
            *value = (CUDART_VERSION % 1000) / 10;
            break;
        case PATCH_LEVEL:
            *value = 0;
            break;
        default:
            return NVGRAPH_STATUS_INVALID_VALUE;
    }
    return NVGRAPH_STATUS_SUCCESS;
}

nvgraphStatus_t NVGRAPH_API nvgraphCreate(nvgraphHandle_t *handle) {
    return nvgraph::nvgraphCreate_impl(handle);
}

nvgraphStatus_t NVGRAPH_API nvgraphCreateMulti(nvgraphHandle_t *handle,
                                               int numDevices,
                                               int* devices) {
    return nvgraph::nvgraphCreateMulti_impl(handle, numDevices, devices);
}

nvgraphStatus_t NVGRAPH_API nvgraphDestroy(nvgraphHandle_t handle) {
    return nvgraph::nvgraphDestroy_impl(handle);
}

nvgraphStatus_t NVGRAPH_API nvgraphCreateGraphDescr(nvgraphHandle_t handle,
                                                    nvgraphGraphDescr_t *descrG) {
    return nvgraph::nvgraphCreateGraphDescr_impl(handle, descrG);
}

nvgraphStatus_t NVGRAPH_API nvgraphDestroyGraphDescr(nvgraphHandle_t handle,
                                                     nvgraphGraphDescr_t descrG) {
    return nvgraph::nvgraphDestroyGraphDescr_impl(handle, descrG);
}

nvgraphStatus_t NVGRAPH_API nvgraphSetStream(nvgraphHandle_t handle, cudaStream_t stream) {
    return nvgraph::nvgraphSetStream_impl(handle, stream);
}

nvgraphStatus_t NVGRAPH_API nvgraphSetGraphStructure(nvgraphHandle_t handle,
                                                     nvgraphGraphDescr_t descrG,
                                                     void* topologyData,
                                                     nvgraphTopologyType_t topologyType) {
    return nvgraph::nvgraphSetGraphStructure_impl(handle, descrG, topologyData, topologyType);
}

nvgraphStatus_t NVGRAPH_API nvgraphGetGraphStructure(nvgraphHandle_t handle,
                                                     nvgraphGraphDescr_t descrG,
                                                     void* topologyData,
                                                     nvgraphTopologyType_t* topologyType) {
    return nvgraph::nvgraphGetGraphStructure_impl(handle, descrG, topologyData, topologyType);
}
nvgraphStatus_t NVGRAPH_API nvgraphAllocateVertexData(nvgraphHandle_t handle,
                                                      nvgraphGraphDescr_t descrG,
                                                      size_t numsets,
                                                      cudaDataType_t *settypes) {
    return nvgraph::nvgraphAllocateVertexData_impl(handle, descrG, numsets, settypes);
}

nvgraphStatus_t NVGRAPH_API nvgraphAllocateEdgeData(nvgraphHandle_t handle,
                                                    nvgraphGraphDescr_t descrG,
                                                    size_t numsets,
                                                    cudaDataType_t *settypes) {
    return nvgraph::nvgraphAllocateEdgeData_impl(handle, descrG, numsets, settypes);
}

nvgraphStatus_t NVGRAPH_API nvgraphExtractSubgraphByVertex(nvgraphHandle_t handle,
                                                           nvgraphGraphDescr_t descrG,
                                                           nvgraphGraphDescr_t subdescrG,
                                                           int *subvertices,
                                                           size_t numvertices) {
    return nvgraph::nvgraphExtractSubgraphByVertex_impl(handle,
                                                        descrG,
                                                        subdescrG,
                                                        subvertices,
                                                        numvertices);
}

nvgraphStatus_t NVGRAPH_API nvgraphExtractSubgraphByEdge(nvgraphHandle_t handle,
                                                         nvgraphGraphDescr_t descrG,
                                                         nvgraphGraphDescr_t subdescrG,
                                                         int *subedges,
                                                         size_t numedges) {
    return nvgraph::nvgraphExtractSubgraphByEdge_impl(handle, descrG, subdescrG, subedges, numedges);
}

nvgraphStatus_t NVGRAPH_API nvgraphSetVertexData(nvgraphHandle_t handle,
                                                 nvgraphGraphDescr_t descrG,
                                                 void *vertexData,
                                                 size_t setnum) {
    return nvgraph::nvgraphSetVertexData_impl(handle, descrG, vertexData, setnum);
}

nvgraphStatus_t NVGRAPH_API nvgraphGetVertexData(nvgraphHandle_t handle,
                                                 nvgraphGraphDescr_t descrG,
                                                 void *vertexData,
                                                 size_t setnum) {
    return nvgraph::nvgraphGetVertexData_impl(handle, descrG, vertexData, setnum);
}

nvgraphStatus_t NVGRAPH_API nvgraphConvertTopology(nvgraphHandle_t handle,
                                                   nvgraphTopologyType_t srcTType,
                                                   void *srcTopology,
                                                   void *srcEdgeData,
                                                   cudaDataType_t *dataType,
                                                   nvgraphTopologyType_t dstTType,
                                                   void *dstTopology,
                                                   void *dstEdgeData) {
    return nvgraph::nvgraphConvertTopology_impl(handle,
                                                srcTType,
                                                srcTopology,
                                                srcEdgeData,
                                                dataType,
                                                dstTType,
                                                dstTopology,
                                                dstEdgeData);
}

nvgraphStatus_t NVGRAPH_API nvgraphConvertGraph(nvgraphHandle_t handle,
                                                nvgraphGraphDescr_t srcDescrG,
                                                nvgraphGraphDescr_t dstDescrG,
                                                nvgraphTopologyType_t dstTType) {
    return nvgraph::nvgraphConvertGraph_impl(handle, srcDescrG, dstDescrG, dstTType);
}

nvgraphStatus_t NVGRAPH_API nvgraphSetEdgeData(nvgraphHandle_t handle,
                                               nvgraphGraphDescr_t descrG,
                                               void *edgeData,
                                               size_t setnum) {
    return nvgraph::nvgraphSetEdgeData_impl(handle, descrG, edgeData, setnum);
}

nvgraphStatus_t NVGRAPH_API nvgraphGetEdgeData(nvgraphHandle_t handle,
                                               nvgraphGraphDescr_t descrG,
                                               void *edgeData,
                                               size_t setnum) {
    return nvgraph::nvgraphGetEdgeData_impl(handle, descrG, edgeData, setnum);
}

nvgraphStatus_t NVGRAPH_API nvgraphSrSpmv(nvgraphHandle_t handle,
                                          const nvgraphGraphDescr_t descrG,
                                          const size_t weight_index,
                                          const void *alpha,
                                          const size_t x,
                                          const void *beta,
                                          const size_t y,
                                          const nvgraphSemiring_t SR) {
    return nvgraph::nvgraphSrSpmv_impl_cub(handle, descrG, weight_index, alpha, x, beta, y, SR);
}

nvgraphStatus_t NVGRAPH_API nvgraphSssp(nvgraphHandle_t handle,
                                        const nvgraphGraphDescr_t descrG,
                                        const size_t weight_index,
                                        const int *source_vert,
                                        const size_t sssp) {
    return nvgraph::nvgraphSssp_impl(handle, descrG, weight_index, source_vert, sssp);
}

//nvgraphTraversal

typedef enum {
    NVGRAPH_TRAVERSAL_DISTANCES_INDEX = 0,
    NVGRAPH_TRAVERSAL_PREDECESSORS_INDEX = 1,
    NVGRAPH_TRAVERSAL_MASK_INDEX = 2,
    NVGRAPH_TRAVERSAL_UNDIRECTED_FLAG_INDEX = 3,
    NVGRAPH_TRAVERSAL_ALPHA = 4,
    NVGRAPH_TRAVERSAL_BETA = 5
} nvgraphTraversalParameterIndex_t;

nvgraphStatus_t NVGRAPH_API nvgraphTraversalParameterInit(nvgraphTraversalParameter_t *param) {
    if (check_ptr(param))
        return NVGRAPH_STATUS_INVALID_VALUE;

    param->pad[NVGRAPH_TRAVERSAL_DISTANCES_INDEX] = INT_MAX;
    param->pad[NVGRAPH_TRAVERSAL_PREDECESSORS_INDEX] = INT_MAX;
    param->pad[NVGRAPH_TRAVERSAL_MASK_INDEX] = INT_MAX;
    param->pad[NVGRAPH_TRAVERSAL_UNDIRECTED_FLAG_INDEX] = 0;
    param->pad[NVGRAPH_TRAVERSAL_ALPHA] = TRAVERSAL_DEFAULT_ALPHA;
    param->pad[NVGRAPH_TRAVERSAL_BETA] = TRAVERSAL_DEFAULT_BETA;

    return NVGRAPH_STATUS_SUCCESS;
}

nvgraphStatus_t NVGRAPH_API nvgraphTraversalSetDistancesIndex(nvgraphTraversalParameter_t *param,
                                                              const size_t value) {
    if (check_ptr(param))
        return NVGRAPH_STATUS_INVALID_VALUE;

    param->pad[NVGRAPH_TRAVERSAL_DISTANCES_INDEX] = value;

    return NVGRAPH_STATUS_SUCCESS;
}

nvgraphStatus_t NVGRAPH_API nvgraphTraversalGetDistancesIndex(const nvgraphTraversalParameter_t param,
                                                              size_t *value) {
    if (check_ptr(value))
        return NVGRAPH_STATUS_INVALID_VALUE;

    *value = param.pad[NVGRAPH_TRAVERSAL_DISTANCES_INDEX];

    return NVGRAPH_STATUS_SUCCESS;
}

nvgraphStatus_t NVGRAPH_API nvgraphTraversalSetPredecessorsIndex(nvgraphTraversalParameter_t *param,
                                                                 const size_t value) {
    if (check_ptr(param))
        return NVGRAPH_STATUS_INVALID_VALUE;

    param->pad[NVGRAPH_TRAVERSAL_PREDECESSORS_INDEX] = value;

    return NVGRAPH_STATUS_SUCCESS;
}

nvgraphStatus_t NVGRAPH_API nvgraphTraversalGetPredecessorsIndex(const nvgraphTraversalParameter_t param,
                                                                 size_t *value) {
    if (check_ptr(value))
        return NVGRAPH_STATUS_INVALID_VALUE;

    *value = param.pad[NVGRAPH_TRAVERSAL_PREDECESSORS_INDEX];

    return NVGRAPH_STATUS_SUCCESS;
}

nvgraphStatus_t NVGRAPH_API nvgraphTraversalSetEdgeMaskIndex(nvgraphTraversalParameter_t *param,
                                                             const size_t value) {
    if (check_ptr(param))
        return NVGRAPH_STATUS_INVALID_VALUE;

    param->pad[NVGRAPH_TRAVERSAL_MASK_INDEX] = value;

    return NVGRAPH_STATUS_SUCCESS;
}

nvgraphStatus_t NVGRAPH_API nvgraphTraversalGetEdgeMaskIndex(const nvgraphTraversalParameter_t param,
                                                             size_t *value) {
    if (check_ptr(value))
        return NVGRAPH_STATUS_INVALID_VALUE;

    *value = param.pad[NVGRAPH_TRAVERSAL_MASK_INDEX];

    return NVGRAPH_STATUS_SUCCESS;
}

nvgraphStatus_t NVGRAPH_API nvgraphTraversalSetUndirectedFlag(nvgraphTraversalParameter_t *param,
                                                              const size_t value) {
    if (check_ptr(param))
        return NVGRAPH_STATUS_INVALID_VALUE;

    param->pad[NVGRAPH_TRAVERSAL_UNDIRECTED_FLAG_INDEX] = value;

    return NVGRAPH_STATUS_SUCCESS;

}

nvgraphStatus_t NVGRAPH_API nvgraphTraversalGetUndirectedFlag(const nvgraphTraversalParameter_t param,
                                                              size_t *value) {
    if (check_ptr(value))
        return NVGRAPH_STATUS_INVALID_VALUE;

    *value = param.pad[NVGRAPH_TRAVERSAL_UNDIRECTED_FLAG_INDEX];

    return NVGRAPH_STATUS_SUCCESS;
}

nvgraphStatus_t NVGRAPH_API nvgraphTraversalSetAlpha(nvgraphTraversalParameter_t *param,
                                                     const size_t value) {
    if (check_ptr(param))
        return NVGRAPH_STATUS_INVALID_VALUE;

    param->pad[NVGRAPH_TRAVERSAL_ALPHA] = value;

    return NVGRAPH_STATUS_SUCCESS;

}

nvgraphStatus_t NVGRAPH_API nvgraphTraversalGetAlpha(const nvgraphTraversalParameter_t param,
                                                     size_t *value) {
    if (check_ptr(value))
        return NVGRAPH_STATUS_INVALID_VALUE;

    *value = param.pad[NVGRAPH_TRAVERSAL_ALPHA];

    return NVGRAPH_STATUS_SUCCESS;
}

nvgraphStatus_t NVGRAPH_API nvgraphTraversalSetBeta(nvgraphTraversalParameter_t *param,
                                                    const size_t value) {
    if (check_ptr(param))
        return NVGRAPH_STATUS_INVALID_VALUE;

    param->pad[NVGRAPH_TRAVERSAL_BETA] = value;

    return NVGRAPH_STATUS_SUCCESS;

}

nvgraphStatus_t NVGRAPH_API nvgraphTraversalGetBeta(const nvgraphTraversalParameter_t param,
                                                    size_t *value) {
    if (check_ptr(value))
        return NVGRAPH_STATUS_INVALID_VALUE;

    *value = param.pad[NVGRAPH_TRAVERSAL_BETA];

    return NVGRAPH_STATUS_SUCCESS;
}

nvgraphStatus_t NVGRAPH_API nvgraphTraversal(nvgraphHandle_t handle,
                                             const nvgraphGraphDescr_t descrG,
                                             const nvgraphTraversal_t traversalT,
                                             const int *source_vert,
                                             const nvgraphTraversalParameter_t params) {
    return nvgraph::nvgraphTraversal_impl(handle, descrG, traversalT, source_vert, params);
}

/**
 * CAPI Method for calling 2d BFS algorithm.
 * @param handle Nvgraph context handle.
 * @param descrG Graph handle (must be 2D partitioned)
 * @param source_vert The source vertex ID
 * @param distances Pointer to memory allocated to store the distances.
 * @param predecessors Pointer to memory allocated to store the predecessors
 * @return Status code.
 */
nvgraphStatus_t NVGRAPH_API nvgraph2dBfs(nvgraphHandle_t handle,
                                         const nvgraphGraphDescr_t descrG,
                                         const int32_t source_vert,
                                         int32_t* distances,
                                         int32_t* predecessors) {
    return nvgraph::nvgraph2dBfs_impl(handle, descrG, source_vert, distances, predecessors);
}

//nvgraphWidestPath

nvgraphStatus_t NVGRAPH_API nvgraphWidestPath(nvgraphHandle_t handle,
                                              const nvgraphGraphDescr_t descrG,
                                              const size_t weight_index,
                                              const int *source_vert,
                                              const size_t widest_path) {
    return nvgraph::nvgraphWidestPath_impl(handle, descrG, weight_index, source_vert, widest_path);
}

nvgraphStatus_t NVGRAPH_API nvgraphPagerank(nvgraphHandle_t handle,
                                            const nvgraphGraphDescr_t descrG,
                                            const size_t weight_index,
                                            const void *alpha,
                                            const size_t bookmark,
                                            const int has_guess,
                                            const size_t pagerank_index,
                                            const float tolerance,
                                            const int max_iter) {
    return nvgraph::nvgraphPagerank_impl(handle,
                                         descrG,
                                         weight_index,
                                         alpha,
                                         bookmark,
                                         has_guess,
                                         pagerank_index,
                                         tolerance,
                                         max_iter);
}

nvgraphStatus_t NVGRAPH_API nvgraphKrylovPagerank(nvgraphHandle_t handle,
                                                  const nvgraphGraphDescr_t descrG,
                                                  const size_t weight_index,
                                                  const void *alpha,
                                                  const size_t bookmark,
                                                  const float tolerance,
                                                  const int max_iter,
                                                  const int subspace_size,
                                                  const int has_guess,
                                                  const size_t rank) {
    return nvgraph::nvgraphKrylovPagerank_impl(handle,
                                               descrG,
                                               weight_index,
                                               alpha,
                                               bookmark,
                                               tolerance,
                                               max_iter,
                                               subspace_size,
                                               has_guess,
                                               rank);
}

nvgraphStatus_t NVGRAPH_API nvgraphBalancedCutClustering(nvgraphHandle_t handle,
                                                         const nvgraphGraphDescr_t descrG,
                                                         const size_t weight_index,
                                                         const int n_clusters,
                                                         const int n_eig_vects,
                                                         const int evs_type,
                                                         const float evs_tolerance,
                                                         const int evs_max_iter,
                                                         const float kmean_tolerance,
                                                         const int kmean_max_iter,
                                                         int* clustering,
                                                         void* eig_vals,
                                                         void* eig_vects) {
    return nvgraph::nvgraphBalancedCutClustering_impl(handle,
                                                      descrG,
                                                      weight_index,
                                                      n_clusters,
                                                      n_eig_vects,
                                                      evs_type,
                                                      evs_tolerance,
                                                      evs_max_iter,
                                                      kmean_tolerance,
                                                      kmean_max_iter,
                                                      clustering,
                                                      eig_vals,
                                                      eig_vects);
}

nvgraphStatus_t NVGRAPH_API nvgraphAnalyzeBalancedCut(nvgraphHandle_t handle,
                                                      const nvgraphGraphDescr_t descrG,
                                                      const size_t weight_index,
                                                      const int n_clusters,
                                                      const int* clustering,
                                                      float * edgeCut,
                                                      float * ratioCut) {
    return nvgraph::nvgraphAnalyzeBalancedCut_impl(handle,
                                                   descrG,
                                                   weight_index,
                                                   n_clusters,
                                                   clustering,
                                                   edgeCut,
                                                   ratioCut);
}

nvgraphStatus_t NVGRAPH_API nvgraphHeavyEdgeMatching(nvgraphHandle_t handle,
                                                     const nvgraphGraphDescr_t descrG,
                                                     const size_t weight_index,
                                                     const nvgraphEdgeWeightMatching_t similarity_metric,
                                                     int* aggregates,
                                                     size_t* num_aggregates) {
    return nvgraph::nvgraphHeavyEdgeMatching_impl(handle,
                                                  descrG,
                                                  weight_index,
                                                  similarity_metric,
                                                  aggregates,
                                                  num_aggregates);
}

nvgraphStatus_t NVGRAPH_API nvgraphSpectralModularityMaximization(nvgraphHandle_t handle,
                                                                  const nvgraphGraphDescr_t descrG,
                                                                  const size_t weight_index,
                                                                  const int n_clusters,
                                                                  const int n_eig_vects,
                                                                  const float evs_tolerance,
                                                                  const int evs_max_iter,
                                                                  const float kmean_tolerance,
                                                                  const int kmean_max_iter,
                                                                  int* clustering,
                                                                  void* eig_vals,
                                                                  void* eig_vects) {
    return nvgraph::nvgraphSpectralModularityMaximization_impl(handle,
                                                               descrG,
                                                               weight_index,
                                                               n_clusters,
                                                               n_eig_vects,
                                                               evs_tolerance,
                                                               evs_max_iter,
                                                               kmean_tolerance,
                                                               kmean_max_iter,
                                                               clustering,
                                                               eig_vals,
                                                               eig_vects);
}

nvgraphStatus_t NVGRAPH_API nvgraphAnalyzeModularityClustering(nvgraphHandle_t handle,
                                                               const nvgraphGraphDescr_t descrG,
                                                               const size_t weight_index,
                                                               const int n_clusters,
                                                               const int* clustering,
                                                               float * modularity) {
    return nvgraph::nvgraphAnalyzeModularityClustering_impl(handle,
                                                            descrG,
                                                            weight_index,
                                                            n_clusters,
                                                            clustering,
                                                            modularity);
}

nvgraphStatus_t NVGRAPH_API nvgraphSpectralClustering(nvgraphHandle_t handle, // nvGRAPH library handle.
                                                      const nvgraphGraphDescr_t descrG, // nvGRAPH graph descriptor, should contain the connectivity information in NVGRAPH_CSR_32 or NVGRAPH_CSR_32 at least 1 edge set (weights)
                                                      const size_t weight_index, // Index of the edge set for the weights.
                                                      const struct SpectralClusteringParameter *params, //parameters, see struct SpectralClusteringParameter
                                                      int* clustering, // (output) clustering
                                                      void* eig_vals,   // (output) eigenvalues
                                                      void* eig_vects)  // (output) eigenvectors
{
    return nvgraph::nvgraphSpectralClustering_impl(handle,
                                                   descrG,
                                                   weight_index,
                                                   params,
                                                   clustering,
                                                   eig_vals,
                                                   eig_vects);
}

nvgraphStatus_t NVGRAPH_API nvgraphAnalyzeClustering(nvgraphHandle_t handle, // nvGRAPH library handle.
                                                     const nvgraphGraphDescr_t descrG, // nvGRAPH graph descriptor, should contain the connectivity information in NVGRAPH_CSR_32 at least 1 edge set (weights)
                                                     const size_t weight_index, // Index of the edge set for the weights.
                                                     const int n_clusters, //number of clusters
                                                     const int* clustering, // clustering to analyse
                                                     nvgraphClusteringMetric_t metric, // metric to compute to measure the clustering quality
                                                     float * score) // (output) clustering score telling how good the clustering is for the selected metric.
{
    return nvgraph::nvgraphAnalyzeClustering_impl(handle,
                                                  descrG,
                                                  weight_index,
                                                  n_clusters,
                                                  clustering,
                                                  metric,
                                                  score);
}

nvgraphStatus_t NVGRAPH_API nvgraphTriangleCount(nvgraphHandle_t handle,
                                                 const nvgraphGraphDescr_t descrG,
                                                 uint64_t* result)
{
    return nvgraph::nvgraphTriangleCount_impl(handle, descrG, result);
}


nvgraphStatus_t NVGRAPH_API nvgraphLouvain (cudaDataType_t index_type, cudaDataType_t val_type, const size_t num_vertex, const size_t num_edges,
                            void* csr_ptr, void* csr_ind, void* csr_val, int weighted, int has_init_cluster, void* init_cluster,
                            void* final_modularity, void* best_cluster_vec, void* num_level)
{
    NVLOUVAIN_STATUS status = NVLOUVAIN_OK;
    if ((csr_ptr == NULL) || (csr_ind == NULL) || ((csr_val == NULL) && (weighted == 1)) ||
        ((init_cluster == NULL) && (has_init_cluster == 1)) || (final_modularity == NULL) || (best_cluster_vec == NULL) || (num_level == NULL))
       return NVGRAPH_STATUS_INVALID_VALUE;

    std::ostream log(0);
    bool weighted_b = weighted;
    bool has_init_cluster_b = has_init_cluster;
    if (val_type == CUDA_R_32F)
        status = nvlouvain::louvain ((int*)csr_ptr, (int*)csr_ind, (float*)csr_val, num_vertex, num_edges,
               weighted_b, has_init_cluster_b, (int*)init_cluster, *((float*)final_modularity),
              (int*)best_cluster_vec,*((int*)num_level), log);
    else
        status = nvlouvain::louvain ((int*)csr_ptr, (int*)csr_ind, (double*)csr_val, num_vertex, num_edges,
                weighted_b, has_init_cluster_b, (int*)init_cluster, *((double*)final_modularity),
                (int*)best_cluster_vec,*((int*)num_level), log);

    if (status != NVLOUVAIN_OK)
        return NVGRAPH_STATUS_INTERNAL_ERROR;

    return NVGRAPH_STATUS_SUCCESS;
}

nvgraphStatus_t NVGRAPH_API nvgraphJaccard (cudaDataType_t index_type, cudaDataType_t val_type, const size_t n,
                            const size_t e, void* csr_ptr, void* csr_ind, void* csr_val, int weighted, void* v, void* gamma, void* weight_j)
{
    int status = 0;

    if ((csr_ptr == NULL) || (csr_ind == NULL) || ((csr_val == NULL) && (weighted == 1)) || (gamma == NULL) || (weight_j == NULL))
        return NVGRAPH_STATUS_INVALID_VALUE;

    bool weighted_b = weighted;
    cudaStream_t stream{nullptr};

    if (val_type == CUDA_R_32F)
    {
        float* weight_i = NULL, *weight_s = NULL, *work = NULL;
        NVG_RMM_TRY(RMM_ALLOC((void**)&weight_i, sizeof(float) * e, stream));
        NVG_RMM_TRY(RMM_ALLOC((void**)&weight_s, sizeof(float) * e, stream));
        if (weighted_b == true)
        {
            NVG_RMM_TRY(RMM_ALLOC((void**)&work, sizeof(float) * n, stream));
            status = nvlouvain::jaccard <true> (n, e, (int*) csr_ptr, (int*) csr_ind, (float*) csr_val, (float*) v, work, *((float*) gamma), weight_i, weight_s, (float*)weight_j);
            NVG_RMM_TRY(RMM_FREE(work, stream));
        }
        else
        {
            NVG_RMM_TRY(RMM_ALLOC((void**)&work, sizeof(float) * n, stream));
            nvlouvain::fill(e, (float*)weight_j, (float)1.0);
            status = nvlouvain::jaccard <false> (n, e, (int*) csr_ptr, (int*) csr_ind, (float*) csr_val, (float*) v, work, *((float*) gamma), weight_i, weight_s, (float*)weight_j);
            NVG_RMM_TRY(RMM_FREE(work, stream));
        }
        NVG_RMM_TRY(RMM_FREE(weight_s, stream));
        NVG_RMM_TRY(RMM_FREE(weight_i, stream));
    }
    else
    {
        double* weight_i = NULL, *weight_s = NULL, *work = NULL;
        NVG_RMM_TRY(RMM_ALLOC((void**)&weight_i, sizeof(double) * e, stream));
        NVG_RMM_TRY(RMM_ALLOC((void**)&weight_s, sizeof(double) * e, stream));
        if (weighted_b == true)
        {
            NVG_RMM_TRY(RMM_ALLOC((void**)&work, sizeof(double) * n, stream));
            status = nvlouvain::jaccard <true> (n, e, (int*) csr_ptr, (int*) csr_ind, (double*) csr_val, (double*) v, work, *((double*) gamma), weight_i, weight_s, (double*)weight_j);
            NVG_RMM_TRY(RMM_FREE(work, stream));
        }
        else
        {
            NVG_RMM_TRY(RMM_ALLOC((void**)&work, sizeof(double) * n, stream));
            nvlouvain::fill(e, (double*)weight_j, (double)1.0);
            status = nvlouvain::jaccard <false> (n, e, (int*) csr_ptr, (int*) csr_ind, (double*) csr_val, (double*) v, work, *((double*) gamma), weight_i, weight_s, (double*)weight_j);
            NVG_RMM_TRY(RMM_FREE(work, stream));
        }
        NVG_RMM_TRY(RMM_FREE(weight_s, stream));
        NVG_RMM_TRY(RMM_FREE(weight_i, stream));
    }

    if (status != 0)
        return NVGRAPH_STATUS_INTERNAL_ERROR;

    return NVGRAPH_STATUS_SUCCESS;
}

nvgraphStatus_t NVGRAPH_API nvgraphAttachGraphStructure(nvgraphHandle_t handle,
                                                        nvgraphGraphDescr_t descrG,
                                                        void* topologyData,
                                                        nvgraphTopologyType_t TT) {
    return nvgraph::nvgraphAttachGraphStructure_impl( handle, descrG, topologyData, TT);
}

nvgraphStatus_t NVGRAPH_API nvgraphAttachVertexData(nvgraphHandle_t handle,
                                                     nvgraphGraphDescr_t descrG,
                                                     size_t setnum,
                                                     cudaDataType_t settype,
                                                     void *vertexData) {
    return nvgraph::nvgraphAttachVertexData_impl( handle, descrG, setnum, settype, vertexData);
}

nvgraphStatus_t NVGRAPH_API nvgraphAttachEdgeData(nvgraphHandle_t handle,
                                                  nvgraphGraphDescr_t descrG,
                                                  size_t setnum,
                                                  cudaDataType_t settype,
                                                  void *edgeData) {
    return nvgraph::nvgraphAttachEdgeData_impl( handle, descrG, setnum, settype, edgeData);
}

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
#include "include/nvgraph_error.hxx"
#include "include/rmm_shared_ptr.hxx"
#include "include/valued_csr_graph.hxx"
#include "include/multi_valued_csr_graph.hxx"
#include "include/nvgraph_vector.hxx"
#include "include/nvgraph_cusparse.hxx"
#include "include/nvgraph_cublas.hxx"
#include "include/nvgraph_csrmv.hxx"
#include "include/partition.hxx"
#include "include/size2_selector.hxx"
#include "include/modularity_maximization.hxx"
#include "include/csrmv_cub.h"
#include "include/nvgraphP.h"  // private header, contains structures, and potentially other things, used in the public C API that should never be exposed.
#include "include/nvgraph_experimental.h"  // experimental header, contains hidden API entries, can be shared only under special circumstances without reveling internal things
#include "include/debug_macros.h"

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

nvgraphStatus_t NVGRAPH_API nvgraphLouvain (cudaDataType_t index_type, cudaDataType_t val_type, const size_t num_vertex, const size_t num_edges,
                            void* csr_ptr, void* csr_ind, void* csr_val, int weighted, int has_init_cluster, void* init_cluster,
                            void* final_modularity, void* best_cluster_vec, void* num_level, int max_iter)
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
              (int*)best_cluster_vec,*((int*)num_level), max_iter, log);
    else
        status = nvlouvain::louvain ((int*)csr_ptr, (int*)csr_ind, (double*)csr_val, num_vertex, num_edges,
                weighted_b, has_init_cluster_b, (int*)init_cluster, *((double*)final_modularity),
                (int*)best_cluster_vec,*((int*)num_level), max_iter, log);

    if (status != NVLOUVAIN_OK)
        return NVGRAPH_STATUS_INTERNAL_ERROR;

    return NVGRAPH_STATUS_SUCCESS;
}

nvgraphStatus_t NVGRAPH_API nvgraphAttachGraphStructure(nvgraphHandle_t handle,
                                                        nvgraphGraphDescr_t descrG,
                                                        void* topologyData,
                                                        nvgraphTopologyType_t TT) {
    return nvgraph::nvgraphAttachGraphStructure_impl( handle, descrG, topologyData, TT);
}

nvgraphStatus_t NVGRAPH_API nvgraphAttachEdgeData(nvgraphHandle_t handle,
                                                  nvgraphGraphDescr_t descrG,
                                                  size_t setnum,
                                                  cudaDataType_t settype,
                                                  void *edgeData) {
    return nvgraph::nvgraphAttachEdgeData_impl( handle, descrG, setnum, settype, edgeData);
}

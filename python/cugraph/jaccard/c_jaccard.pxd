cdef extern from "library_types.h":

    ctypedef enum cudaDataType_t:
        CUDA_R_16F= 2,
        CUDA_C_16F= 6,
        CUDA_R_32F= 0,
        CUDA_C_32F= 4,
        CUDA_R_64F= 1,
        CUDA_C_64F= 5,
        CUDA_R_8I = 3,
        CUDA_C_8I = 7,
        CUDA_R_8U = 8,
        CUDA_C_8U = 9,
        CUDA_R_32I= 10,
        CUDA_C_32I= 11,
        CUDA_R_32U= 12,
        CUDA_C_32U= 13

cdef extern from "nvgraph.h":

    ctypedef enum nvgraphStatus_t:
        pass

    cdef nvgraphStatus_t nvgraphJaccard (cudaDataType_t index_type,
                                         cudaDataType_t val_type,
                                         const size_t n,
                                         const size_t e,
                                         void* csr_ptr,
                                         void *csr_ind,
                                         void* csr_val,
                                         int weighted,
                                         void* v,
                                         void* gamma,
                                         void* weight_j)

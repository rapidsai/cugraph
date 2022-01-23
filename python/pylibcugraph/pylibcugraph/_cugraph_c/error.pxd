# Copyright (c) 2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Have cython use python 3 syntax
# cython: language_level = 3

cdef extern from "cugraph_c/error.h":

    ctypedef enum cugraph_error_code_t:
        CUGRAPH_SUCCESS
        CUGRAPH_UNKNOWN_ERROR
        CUGRAPH_INVALID_HANDLE
        CUGRAPH_ALLOC_ERROR
        CUGRAPH_INVALID_INPUT
        CUGRAPH_NOT_IMPLEMENTED
        CUGRAPH_UNSUPPORTED_TYPE_COMBINATION

    ctypedef struct cugraph_error_t:
       pass

    const char* \
        cugraph_error_message(
            const cugraph_error_t* error
        )

    void \
        cugraph_error_free(
            cugraph_error_t* error
        )

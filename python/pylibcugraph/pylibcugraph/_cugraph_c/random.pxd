# Copyright (c) 2023, NVIDIA CORPORATION.
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

from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.resource_handle cimport (
    cugraph_resource_handle_t,
)

cdef extern from "cugraph_c/random.h":
    ctypedef struct cugraph_rng_state_t:
        pass

    cdef cugraph_error_code_t cugraph_rng_state_create(
        const cugraph_resource_handle_t* handle,
        size_t seed,
        cugraph_rng_state_t** state,
        cugraph_error_t** error,
    )

    cdef void cugraph_rng_state_free(cugraph_rng_state_t* p)

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
from pylibcugraph._cugraph_c.random cimport (
    cugraph_rng_state_create,
    cugraph_rng_state_free,
    cugraph_rng_state_t,
)
from pylibcugraph._cugraph_c.resource_handle cimport (
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph.utils cimport (
    assert_success,
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle
)

import time
import os
import socket

def generate_default_seed():
    h = hash(
            (
                socket.gethostname(),
                os.getpid(),
                time.perf_counter_ns()
            )
        )

    return h

cdef class CuGraphRandomState:
    """
        This class wraps a cugraph_rng_state_t instance, which represents a
        random state.
    """

    def __cinit__(self, ResourceHandle resource_handle, seed=None):
        """
        Constructs a new CuGraphRandomState instance.

        Parameters
        ----------
        resource_handle: pylibcugraph.ResourceHandle (Required)
            The cugraph resource handle for this process.
        seed: int (Optional)
            The random seed of this random state object.
            Defaults to the hash of the hostname, pid, and time.

        """

        cdef cugraph_error_code_t error_code
        cdef cugraph_error_t* error_ptr

        cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
            resource_handle.c_resource_handle_ptr
        
        cdef cugraph_rng_state_t* new_rng_state_ptr

        if seed is None:
            seed = generate_default_seed()
        
        # reinterpret as unsigned
        seed &= (2**64 - 1)

        error_code = cugraph_rng_state_create(
            c_resource_handle_ptr,
            <size_t>seed,
            &new_rng_state_ptr,
            &error_ptr    
        )
        assert_success(error_code, error_ptr, "cugraph_rng_state_create")
    
        self.rng_state_ptr = new_rng_state_ptr
    
    def __dealloc__(self):
        """
        Destroys this CuGraphRandomState instance.  Properly calls
        free to destroy the underlying C++ object.
        """
        cugraph_rng_state_free(self.rng_state_ptr)

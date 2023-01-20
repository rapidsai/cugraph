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
    
    # reinterpret as unsigned
    return h & (2**64 - 1)

class global_random_instance_wrapper:
    global_random_instance = None

cdef class CuGraphRandomState:
    """
        This class wraps a cugraph_rng_state_t instance, which represents a
        random state.  Users can opt to use the global random state by
        calling is_initialized(), initialize(), and get(), which are class
        methods, or they can create their own random state variable by
        calling this class's constructor.

        Pylibcugraph algorithms that require a random state will use the
        global random state if no random state is provided by the user.
        In this case, the pylibcugraph algorithm is responsible for
        initializing the global random state if necessary.
    """

    def __cinit__(self, ResourceHandle resource_handle, seed=generate_default_seed()):
        """
        Constructs a new CuGraphRandomState instance.

        Parameters
        ----------
        resource_handle: pylibcugraph.ResourceHandle (Required)
            The cugraph resource handle for this process.
        seed: unsigned int (Optional)
            The random seed of this random state object.
            Defaults to the hash of the hostname, pid, and time.

        """

        cdef cugraph_error_code_t error_code
        cdef cugraph_error_t* error_ptr

        cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
            resource_handle.c_resource_handle_ptr
        
        cdef cugraph_rng_state_t* new_rng_state_ptr

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

    @classmethod
    def get(cls):
        """
        Class method that returns the global random state instance.  If the
        global random state instance has not been initialized, an error
        is thrown instead.
        """
        if global_random_instance_wrapper.global_random_instance is None:
            raise ValueError('Global random state has not been initialized!')
        return global_random_instance_wrapper.global_random_instance
    
    @classmethod
    def initialize(cls, resource_handle, seed=generate_default_seed()):
        """
        Class method that initializes the global random state.  If the
        global random state was already initialized, the old instance
        is cleaned up and replaced with the new instance.

        Parameters
        ----------
        resource_handle: pylibcugraph.ResourceHandle (Required)
            The cugraph resource handle for this process.
        seed: unsigned int (Optional)
            The random seed of this random state object.
            Defaults to the hash of the hostname, pid, and time.
        """
        global_random_instance_wrapper.global_random_instance = CuGraphRandomState(resource_handle, seed)

    @classmethod
    def is_initialized(cls):
        """
        Class method that checks if the global random state was initialized or not.

        Returns
        -------
        True if the global random state instance has been initialized, False otherwise.
        """
        return global_random_instance_wrapper.global_random_instance is not None
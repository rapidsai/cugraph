# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

from libc.stdint cimport int8_t


cdef extern from "cugraph_c/types.h":

    ctypedef enum bool_t:
        FALSE
        TRUE

    ctypedef enum cugraph_data_type_id_t:
        INT32
        INT64
        FLOAT32
        FLOAT64
        SIZE_T

    ctypedef int8_t byte_t

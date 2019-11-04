# Copyright (c) 2019, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cugraph.community.c_triangle_count cimport *
from cugraph.structure.c_graph cimport *
from cugraph.utilities.column_utils cimport *
from libc.stdint cimport uintptr_t

import cudf
import cudf._lib as libcudf
import rmm


def triangles(graph_ptr):
    """
    Call gdf_triangle_count_nvgraph
    """
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph* g = <gdf_graph*> graph

    err = cugraph::add_adj_list(g)
    libcudf.cudf.check_gdf_error(err)

    cdef uint64_t result
    err = gdf_triangle_count_nvgraph(g, &result)
    libcudf.cudf.check_gdf_error(err)
    return result

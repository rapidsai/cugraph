# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import numpy as np
from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.utility cimport move
from libcpp.memory cimport unique_ptr
from cython.operator cimport dereference as deref

import cudf

from cugraph.structure.graph_utilities cimport (populate_graph_container,
                                                graph_container_t,
                                                numberTypeEnum,
                                                )
from cugraph.raft.common.handle cimport handle_t
from cugraph.structure import graph_primtypes_wrapper
from cugraph.sampling.random_walks cimport (call_random_walks,
                                            call_rw_paths,
                                            random_walk_ret_t,
                                            random_walk_path_t,
                                            )
from cugraph.structure.graph_primtypes cimport (move_device_buffer_to_column,
                                                move_device_buffer_to_series,
                                                )


def node2vec(input_graph, start_vertices, max_depth, p, q):
    """
    Call node2vec
    """
    # Step 1: Declare the different variables

    # Pointers required for CSR Graph

    # Pointers for node2vec

    # Step 2: Verify that input_graph has the expected format

    # Step 3: Extract CSR offsets, indices and weights

    # Step 4: Setup number of vertices and number of edges

    # Step 5: Check if source index is valid

    # Step 6: Generation of the result cudf.DataFrame

    # Step 7: Associate <uintptr_t> to cudf Series

    # Step 8: Call node2vec, note this is not correct

    """
    c_node2vec.call_node2vec[](handle_[0],
                               graph_container,
                               <int*> c_paths_ptr,
                               <float*> c_weights_ptr,
                               <int*> c_offsets_ptr,
                               <int*> sources,
                               <float> p,
                               <float> q)
    """

    # Stubbed out code - layer 2 being the wrapper in cugraph
    return 222
# Copyright (c) 2020, NVIDIA CORPORATION.
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

from cugraph.dask.community.louvain cimport louvain as c_louvain
from cugraph.structure.graph_primtypes cimport *
from cugraph.structure import graph_primtypes_wrapper

import cudf
import numpy as np


def louvain(input_graph, max_iter, resolution):
    """
    Call MG Louvain
    """
    # FIXME: view_adj_list() is not supported for a distributed graph but should
    # still be done?
    # if not input_graph.adjlist:
    #     input_graph.view_adj_list()

    weights = None
    final_modularity = None

    [offsets, indices] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    if input_graph.adjlist.weights is not None:
        [weights] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.weights], [np.float32, np.float64])
    else:
        weights = cudf.Series(np.full(num_edges, 1.0, dtype=np.float32))

    ####
    # FIXME: call louvain as declared in louvain.pxd here
    ####

    # Create the output dataframe
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['partition'] = cudf.Series(np.zeros(num_verts,dtype=np.int32))


    return df, final_modularity

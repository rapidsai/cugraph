# Copyright (c) 2021, NVIDIA CORPORATION.
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

import pytest
from pylibcugraph.tests import utils
import cupy
import cugraph
import pylibcugraph
import numpy as np


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_scc(graph_file):
    """
    FIXME: rewrite once SCC is implemented.
    """
    cu_M = utils.read_csv_file(graph_file)
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")

    offsets, indices, weights = G.view_adj_list()
    cupy_off = cupy.array(offsets)
    cudf_ind = indices

    cupy_labels = cupy.array(np.zeros(G.number_of_vertices()))
    pylibcugraph.strongly_connected_components(cupy_off.__cuda_array_interface__,
                                               cudf_ind.__cuda_array_interface__,
                                               None,
                                               G.number_of_vertices(),
                                               G.number_of_edges(directed_edges=True),
                                               cupy_labels.__cuda_array_interface__
                                               )
    print(cupy_labels)
    df = cugraph.strongly_connected_components(G)
    print(df)

@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_wcc(graph_file):
    """
    FIXME: rewrite once WCC is implemented.
    """
    cu_M = utils.read_csv_file(graph_file)
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")

    
    cupy_src = cupy.array(cu_M["0"])
    cudf_dst = cu_M["1"]

    cupy_labels = cupy.array(np.zeros(G.number_of_vertices()), dtype='int32')
    pylibcugraph.weakly_connected_components(cupy_src.__cuda_array_interface__,
                                             cudf_dst.__cuda_array_interface__,
                                             None,
                                             G.number_of_vertices(),
                                             G.number_of_edges(directed_edges=True),
                                             cupy_labels.__cuda_array_interface__
                                             )


    print(cupy_labels)
    df = cugraph.weakly_connected_components(G)
    print(df)

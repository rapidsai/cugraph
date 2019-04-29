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

import cugraph
import cudf
import pytest
import numpy as np
import networkx as nx
from scipy.io import mmread


def compareEdges(cg, nxg, verts):
    src, dest = cg.view_edge_list()
    if (len(src) != nxg.size()):
        assert False
    for i in range(len(src)):
        if not nxg.has_edge(verts[src[i]], verts[dest[i]]):
            assert False
    return True


def ReadMtxFile(mmFile):
    print('Reading ' + str(mmFile) + '...')
    return mmread(mmFile).asfptype()


def cugraph_call(M, verts):
    G = cugraph.Graph()
    rows = cudf.Series(M.row)
    cols = cudf.Series(M.col)
    G.add_edge_list(rows, cols, None)
    cu_verts = cudf.Series(verts)
    Sg = cugraph.subgraph(G, cu_verts)
    return Sg


def nx_call(M, verts):
    G = nx.DiGraph(M)
    Sg = nx.subgraph(G, verts)
    return Sg


datasets = ['/datasets/networks/karate.mtx',
            '/datasets/networks/dolphins.mtx',
            '/datasets/networks/netscience.mtx']


@pytest.mark.parametrize('graph_file', datasets)
def test_subgraph_extraction(graph_file):
    M = ReadMtxFile(graph_file)
    verts = np.zeros(3, dtype=np.int32)
    verts[0] = 0
    verts[1] = 1
    verts[2] = 17
    cu_Sg = cugraph_call(M, verts)
    nx_Sg = nx_call(M, verts)
    assert compareEdges(cu_Sg, nx_Sg, verts)

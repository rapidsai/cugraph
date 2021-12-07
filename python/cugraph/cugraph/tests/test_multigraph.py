# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

import gc

import pytest
import networkx as nx
import numpy as np

import cugraph
from cugraph.tests import utils


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_multigraph(graph_file):
    # FIXME: Migrate to new test fixtures for Graph setup once available
    cuM = utils.read_csv_file(graph_file)
    G = cugraph.MultiDiGraph()
    G.from_cudf_edgelist(cuM, source="0", destination="1", edge_attr="2")

    nxM = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)
    Gnx = nx.from_pandas_edgelist(
        nxM,
        source="0",
        target="1",
        edge_attr="weight",
        create_using=nx.MultiDiGraph(),
    )

    assert G.number_of_edges() == Gnx.number_of_edges()
    assert G.number_of_nodes() == Gnx.number_of_nodes()
    cuedges = cugraph.to_pandas_edgelist(G)
    cuedges.rename(columns={"src": "source", "dst": "target",
                   "weights": "weight"}, inplace=True)
    cuedges["weight"] = cuedges["weight"].round(decimals=3)
    nxedges = nx.to_pandas_edgelist(Gnx).astype(dtype={"source": "int32",
                                                       "target": "int32",
                                                       "weight": "float32"})
    cuedges = cuedges.sort_values(by=["source", "target"]).\
        reset_index(drop=True)
    nxedges = nxedges.sort_values(by=["source", "target"]).\
        reset_index(drop=True)
    nxedges["weight"] = nxedges["weight"].round(decimals=3)
    assert nxedges.equals(cuedges[["source", "target", "weight"]])


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_Graph_from_MultiGraph(graph_file):
    # FIXME: Migrate to new test fixtures for Graph setup once available
    cuM = utils.read_csv_file(graph_file)
    GM = cugraph.MultiGraph()
    GM.from_cudf_edgelist(cuM, source="0", destination="1", edge_attr="2")
    nxM = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)
    GnxM = nx.from_pandas_edgelist(
        nxM,
        source="0",
        target="1",
        edge_attr="weight",
        create_using=nx.MultiGraph(),
    )

    G = cugraph.Graph(GM)
    Gnx = nx.Graph(GnxM)
    assert Gnx.number_of_edges() == G.number_of_edges()

    GdM = cugraph.MultiDiGraph()
    GdM.from_cudf_edgelist(cuM, source="0", destination="1", edge_attr="2")
    GnxdM = nx.from_pandas_edgelist(
        nxM,
        source="0",
        target="1",
        edge_attr="weight",
        create_using=nx.MultiGraph(),
    )
    Gd = cugraph.DiGraph(GdM)
    Gnxd = nx.DiGraph(GnxdM)
    assert Gnxd.number_of_edges() == Gd.number_of_edges()


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_multigraph_sssp(graph_file):
    # FIXME: Migrate to new test fixtures for Graph setup once available
    cuM = utils.read_csv_file(graph_file)
    G = cugraph.MultiDiGraph()
    G.from_cudf_edgelist(cuM, source="0", destination="1", edge_attr="2")
    cu_paths = cugraph.sssp(G, 0)
    max_val = np.finfo(cu_paths["distance"].dtype).max
    cu_paths = cu_paths[cu_paths["distance"] != max_val]
    nxM = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)
    Gnx = nx.from_pandas_edgelist(
        nxM,
        source="0",
        target="1",
        edge_attr="weight",
        create_using=nx.MultiDiGraph(),
    )
    nx_paths = nx.single_source_dijkstra_path_length(Gnx, 0)

    cu_dist = cu_paths.sort_values(by='vertex')['distance'].to_numpy()
    nx_dist = [i[1] for i in sorted(nx_paths.items())]

    assert (cu_dist == nx_dist).all()

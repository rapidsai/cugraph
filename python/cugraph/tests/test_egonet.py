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

import gc

import pytest

import cugraph
from cugraph.tests import utils

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx

print("Networkx version : {} ".format(nx.__version__))

SEEDS = [0, 5, 13]
RADIUS = [1, 2, 3]


@pytest.mark.parametrize("graph_file", utils.DATASETS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("radius", RADIUS)
def test_ego_graph_nx(graph_file, seed, radius):
    gc.collect()

    # Nx
    df = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)
    Gnx = nx.from_pandas_edgelist(
        df, create_using=nx.Graph(), source="0", target="1", edge_attr="weight"
    )
    ego_nx = nx.ego_graph(Gnx, seed, radius=radius)

    # cugraph
    ego_cugraph = cugraph.ego_graph(Gnx, seed, radius=radius)

    assert nx.is_isomorphic(ego_nx, ego_cugraph)


@pytest.mark.parametrize("graph_file", utils.DATASETS)
@pytest.mark.parametrize("seeds", [[0, 5, 13]])
@pytest.mark.parametrize("radius", [1, 2, 3])
def test_batched_ego_graphs(graph_file, seeds, radius):
    """
    Compute the  induced subgraph of neighbors for each node in seeds
    within a given radius.
    Parameters
    ----------
    G : cugraph.Graph, networkx.Graph, CuPy or SciPy sparse matrix
        Graph or matrix object, which should contain the connectivity
        information. Edge weights, if present, should be single or double
        precision floating point values.
    seeds : cudf.Series
        Specifies the seeds of the induced egonet subgraphs
    radius: integer, optional
        Include all neighbors of distance<=radius from n.

    Returns
    -------
    ego_edge_lists : cudf.DataFrame
        GPU data frame containing all induced sources identifiers,
        destination identifiers, edge weights
    seeds_offsets: cudf.Series
        Series containing the starting offset in the returned edge list
        for each seed.
    """
    gc.collect()

    # Nx
    df = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)
    Gnx = nx.from_pandas_edgelist(
        df, create_using=nx.Graph(), source="0", target="1", edge_attr="weight"
    )

    # cugraph
    df, offsets = cugraph.batched_ego_graphs(Gnx, seeds, radius=radius)
    for i in range(len(seeds)):
        ego_nx = nx.ego_graph(Gnx, seeds[i], radius=radius)
        ego_df = df[offsets[i]:offsets[i+1]]
        ego_cugraph = nx.from_pandas_edgelist(ego_df,
                                              source="src",
                                              target="dst",
                                              edge_attr="weight")
    assert nx.is_isomorphic(ego_nx, ego_cugraph)

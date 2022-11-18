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

import gc

import pytest

import cudf
import cugraph
from cugraph.testing import utils
from cugraph.experimental.datasets import DATASETS

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


@pytest.mark.parametrize("graph_file", DATASETS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("radius", RADIUS)
def test_ego_graph_nx(graph_file, seed, radius):
    gc.collect()

    # Nx
    dataset_path = graph_file.get_path()
    df = utils.read_csv_for_nx(dataset_path, read_weights_in_sp=True)
    Gnx = nx.from_pandas_edgelist(
        df, create_using=nx.Graph(), source="0", target="1", edge_attr="weight"
    )
    ego_nx = nx.ego_graph(Gnx, seed, radius=radius)

    # cugraph
    ego_cugraph = cugraph.ego_graph(Gnx, seed, radius=radius)

    assert nx.is_isomorphic(ego_nx, ego_cugraph)


@pytest.mark.parametrize("graph_file", DATASETS)
@pytest.mark.parametrize("seeds", [[0, 5, 13]])
@pytest.mark.parametrize("radius", [1, 2, 3])
def test_batched_ego_graphs(graph_file, seeds, radius):
    gc.collect()

    # Nx
    dataset_path = graph_file.get_path()
    df = utils.read_csv_for_nx(dataset_path, read_weights_in_sp=True)
    Gnx = nx.from_pandas_edgelist(
        df, create_using=nx.Graph(), source="0", target="1", edge_attr="weight"
    )

    # cugraph
    df, offsets = cugraph.batched_ego_graphs(Gnx, seeds, radius=radius)
    for i in range(len(seeds)):
        ego_nx = nx.ego_graph(Gnx, seeds[i], radius=radius)
        ego_df = df[offsets[i] : offsets[i + 1]]
        ego_cugraph = nx.from_pandas_edgelist(
            ego_df, source="src", target="dst", edge_attr="weight"
        )
    assert nx.is_isomorphic(ego_nx, ego_cugraph)


@pytest.mark.parametrize("graph_file", DATASETS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("radius", RADIUS)
def test_multi_column_ego_graph(graph_file, seed, radius):
    gc.collect()

    dataset_path = graph_file.get_path()
    df = utils.read_csv_file(dataset_path, read_weights_in_sp=True)
    df.rename(columns={"0": "src_0", "1": "dst_0"}, inplace=True)
    df["src_1"] = df["src_0"] + 1000
    df["dst_1"] = df["dst_0"] + 1000

    G1 = cugraph.Graph()
    G1.from_cudf_edgelist(
        df, source=["src_0", "src_1"], destination=["dst_0", "dst_1"], edge_attr="2"
    )

    seed_df = cudf.DataFrame()
    seed_df["v_0"] = [seed]
    seed_df["v_1"] = [seed + 1000]

    ego_cugraph_res = cugraph.ego_graph(G1, seed_df, radius=radius)

    G2 = cugraph.Graph()
    G2.from_cudf_edgelist(df, source="src_0", destination="dst_0", edge_attr="2")
    ego_cugraph_exp = cugraph.ego_graph(G2, seed, radius=radius)

    # FIXME: Replace with multi-column view_edge_list()
    edgelist_df = ego_cugraph_res.edgelist.edgelist_df
    edgelist_df_res = ego_cugraph_res.unrenumber(edgelist_df, "src")
    edgelist_df_res = ego_cugraph_res.unrenumber(edgelist_df_res, "dst")
    for i in range(len(edgelist_df_res)):
        assert ego_cugraph_exp.has_edge(
            edgelist_df_res["0_src"].iloc[i], edgelist_df_res["0_dst"].iloc[i]
        )

# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
import random

import pytest

import cudf
import cugraph
from cugraph.testing import utils
from cugraph.utilities import ensure_cugraph_obj_for_nx
from cugraph.experimental.datasets import DATASETS

import networkx as nx


def cugraph_call(G, partitions):
    df = cugraph.spectralModularityMaximizationClustering(
        G, partitions, num_eigen_vects=(partitions - 1)
    )
    score = cugraph.analyzeClustering_modularity(G, partitions, df, "vertex", "cluster")
    return score


def random_call(G, partitions):
    random.seed(0)
    num_verts = G.number_of_vertices()
    assignment = []
    for i in range(num_verts):
        assignment.append(random.randint(0, partitions - 1))

    assignment_cu = cudf.DataFrame(assignment, columns=["cluster"])
    assignment_cu["vertex"] = assignment_cu.index

    score = cugraph.analyzeClustering_modularity(
        G, partitions, assignment_cu, "vertex", "cluster"
    )
    return score


PARTITIONS = [2, 4, 8]


@pytest.mark.parametrize("graph_file", DATASETS)
@pytest.mark.parametrize("partitions", PARTITIONS)
def test_modularity_clustering(graph_file, partitions):
    gc.collect()

    # Read in the graph and get a cugraph object
    G = graph_file.get_graph()
    # read_weights_in_sp=False => value column dtype is float64
    G.edgelist.edgelist_df["weights"] = G.edgelist.edgelist_df["weights"].astype(
        "float64"
    )

    # Get the modularity score for partitioning versus random assignment
    cu_score = cugraph_call(G, partitions)
    rand_score = random_call(G, partitions)

    # Assert that the partitioning has better modularity than the random
    # assignment
    assert cu_score > rand_score


@pytest.mark.parametrize("graph_file", DATASETS)
@pytest.mark.parametrize("partitions", PARTITIONS)
def test_modularity_clustering_nx(graph_file, partitions):
    # Read in the graph and get a cugraph object
    dataset_path = graph_file.get_path()
    csv_data = utils.read_csv_for_nx(dataset_path, read_weights_in_sp=True)

    nxG = nx.from_pandas_edgelist(
        csv_data,
        source="0",
        target="1",
        edge_attr="weight",
        create_using=nx.Graph(),
    )
    assert nx.is_directed(nxG) is False
    assert nx.is_weighted(nxG) is True

    cuG, isNx = ensure_cugraph_obj_for_nx(nxG)
    assert cugraph.is_directed(cuG) is False
    assert cugraph.is_weighted(cuG) is True

    # Get the modularity score for partitioning versus random assignment
    cu_score = cugraph_call(cuG, partitions)
    rand_score = random_call(cuG, partitions)

    # Assert that the partitioning has better modularity than the random
    # assignment
    assert cu_score > rand_score


@pytest.mark.parametrize("graph_file", DATASETS)
@pytest.mark.parametrize("partitions", PARTITIONS)
def test_modularity_clustering_multi_column(graph_file, partitions):
    # Read in the graph and get a cugraph object
    dataset_path = graph_file.get_path()
    cu_M = utils.read_csv_file(dataset_path, read_weights_in_sp=False)
    cu_M.rename(columns={"0": "src_0", "1": "dst_0"}, inplace=True)
    cu_M["src_1"] = cu_M["src_0"] + 1000
    cu_M["dst_1"] = cu_M["dst_0"] + 1000

    G1 = cugraph.Graph()
    G1.from_cudf_edgelist(
        cu_M, source=["src_0", "src_1"], destination=["dst_0", "dst_1"], edge_attr="2"
    )

    df1 = cugraph.spectralModularityMaximizationClustering(
        G1, partitions, num_eigen_vects=(partitions - 1)
    )

    cu_score = cugraph.analyzeClustering_modularity(
        G1, partitions, df1, ["0_vertex", "1_vertex"], "cluster"
    )

    G2 = cugraph.Graph()
    G2.from_cudf_edgelist(cu_M, source="src_0", destination="dst_0", edge_attr="2")

    rand_score = random_call(G2, partitions)
    # Assert that the partitioning has better modularity than the random
    # assignment
    assert cu_score > rand_score


# Test to ensure DiGraph objs are not accepted
# Test all combinations of default/managed and pooled/non-pooled allocation


def test_digraph_rejected():
    df = cudf.DataFrame()
    df["src"] = cudf.Series(range(10))
    df["dst"] = cudf.Series(range(10))
    df["val"] = cudf.Series(range(10))

    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(
        df, source="src", destination="dst", edge_attr="val", renumber=False
    )

    with pytest.raises(ValueError):
        cugraph_call(G, 2)

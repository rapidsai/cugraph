# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import gc

import random
import pytest

import cudf
import cugraph
from cugraph.testing import utils, DEFAULT_DATASETS


def cugraph_call(G, partitions):
    df = cugraph.spectralModularityMaximizationClustering(
        G, partitions, num_eigen_vects=(partitions - 1)
    )

    modularity_score = cugraph.analyzeClustering_modularity(
        G, partitions, df, "vertex", "cluster"
    )
    edge_cut_score = cugraph.analyzeClustering_edge_cut(
        G, partitions, df, "vertex", "cluster"
    )
    return modularity_score, edge_cut_score


def random_call(G, partitions):
    random.seed(0)
    num_verts = G.number_of_vertices()
    assignment = []
    for i in range(num_verts):
        assignment.append(random.randint(0, partitions - 1))

    assignment_cu = cudf.DataFrame(assignment, columns=["cluster"])
    assignment_cu["vertex"] = assignment_cu.index
    assignment_cu = assignment_cu.astype("int32")

    modularity_score = cugraph.analyzeClustering_modularity(
        G, partitions, assignment_cu, "vertex", "cluster"
    )
    edge_cut_score = cugraph.analyzeClustering_edge_cut(
        G, partitions, assignment_cu, "vertex", "cluster"
    )
    return modularity_score, edge_cut_score


PARTITIONS = [2, 4, 8]


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
@pytest.mark.parametrize("partitions", PARTITIONS)
def test_modularity_clustering(graph_file, partitions):
    gc.collect()

    # Read in the graph and get a cugraph object
    G = graph_file.get_graph()
    # read_weights_in_sp=False => value column dtype is float64
    G.edgelist.edgelist_df["weights"] = G.edgelist.edgelist_df["weights"].astype(
        "float64"
    )

    rand_modularity, rand_edge_cut = random_call(G, partitions)

    # Retry strategy: spectralModularityMaximizationClustering is a randomized
    # algorithm that may not converge or produce good results on every run.
    # Try up to 10 times, similar to the C API test strategy.
    cu_modularity = None
    cu_edge_cut = None
    for trial in range(10):
        cu_modularity, cu_edge_cut = cugraph_call(G, partitions)
        # Break early if we get a good result
        if cu_modularity > rand_modularity:
            break

    # Assert that the partitioning has better modularity than the random
    # assignment
    assert cu_modularity > rand_modularity
    # Assert that the partitioning has better edge cut than the random
    # assignment (lower is better for edge cut)
    assert cu_edge_cut < rand_edge_cut


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
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

    G2 = cugraph.Graph()
    G2.from_cudf_edgelist(cu_M, source="src_0", destination="dst_0", edge_attr="2")

    rand_modularity, rand_edge_cut = random_call(G2, partitions)

    # Retry strategy: spectralModularityMaximizationClustering is a randomized
    # algorithm that may not converge or produce good results on every run.
    # Try up to 10 times, similar to the C API test strategy.
    cu_modularity = None
    cu_edge_cut = None
    for trial in range(10):
        df1 = cugraph.spectralModularityMaximizationClustering(
            G1, partitions, num_eigen_vects=(partitions - 1)
        )

        cu_modularity = cugraph.analyzeClustering_modularity(
            G1, partitions, df1, ["0_vertex", "1_vertex"], "cluster"
        )
        cu_edge_cut = cugraph.analyzeClustering_edge_cut(
            G1, partitions, df1, ["0_vertex", "1_vertex"], "cluster"
        )

        # Break early if we get a good result
        if cu_modularity > rand_modularity:
            break

    # Assert that the partitioning has better modularity than the random
    # assignment
    assert cu_modularity > rand_modularity
    # Assert that the partitioning has better edge cut than the random
    # assignment (lower is better for edge cut)
    assert cu_edge_cut < rand_edge_cut


# Test to ensure DiGraph objs are not accepted
# Test all combinations of default/managed and pooled/non-pooled allocation


@pytest.mark.sg
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

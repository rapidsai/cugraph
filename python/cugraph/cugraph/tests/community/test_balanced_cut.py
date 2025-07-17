# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
from cugraph.testing import DEFAULT_DATASETS


def cugraph_call(G, partitions):
    df = cugraph.spectralBalancedCutClustering(
        G, partitions, num_eigen_vects=partitions
    )

    score = cugraph.analyzeClustering_edge_cut(G, partitions, df, "vertex", "cluster")
    return set(df["vertex"].to_numpy()), score


def random_call(G, partitions):
    random.seed(0)
    num_verts = G.number_of_vertices()

    score = 0.0
    for repeat in range(20):
        assignment = []
        for i in range(num_verts):
            assignment.append(random.randint(0, partitions - 1))

        assign_cu = cudf.DataFrame(assignment, columns=["cluster"])
        assign_cu["vertex"] = assign_cu.index
        assign_cu = assign_cu.astype("int32")

        score += cugraph.analyzeClustering_edge_cut(G, partitions, assign_cu)

    return set(range(num_verts)), (score / 10.0)


PARTITIONS = [2, 4, 8]


# Test all combinations of default/managed and pooled/non-pooled allocation


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
@pytest.mark.parametrize("partitions", PARTITIONS)
def test_edge_cut_clustering(graph_file, partitions):
    gc.collect()

    # read_weights_in_sp=True => value column dtype is float32
    G_edge = graph_file.get_graph(ignore_weights=True)

    # Get the edge_cut score for partitioning versus random assignment
    cu_vid, cu_score = cugraph_call(G_edge, partitions)
    rand_vid, rand_score = random_call(G_edge, partitions)

    # Assert that the partitioning has better edge_cut than the random
    # assignment
    dataset_name = graph_file.metadata["name"]
    print("graph_file = ", dataset_name, ", partitions = ", partitions)
    print(cu_score, rand_score)
    assert cu_score < rand_score


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DEFAULT_DATASETS)
@pytest.mark.parametrize("partitions", PARTITIONS)
def test_edge_cut_clustering_with_edgevals(graph_file, partitions):
    gc.collect()

    G_edge = graph_file.get_graph()

    # read_weights_in_sp=False => value column dtype is float64
    G_edge.edgelist.edgelist_df["weights"] = G_edge.edgelist.edgelist_df[
        "weights"
    ].astype("float64")

    # Get the edge_cut score for partitioning versus random assignment
    cu_vid, cu_score = cugraph_call(G_edge, partitions)
    rand_vid, rand_score = random_call(G_edge, partitions)

    # Assert that the partitioning has better edge_cut than the random
    # assignment
    print(cu_score, rand_score)
    assert cu_score < rand_score

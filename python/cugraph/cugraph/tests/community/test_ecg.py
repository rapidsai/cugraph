# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
import cugraph

from cugraph.testing import utils
from cugraph.experimental.datasets import karate, dolphins, netscience

from pathlib import PurePath


def cugraph_call(G, min_weight, ensemble_size):
    df = cugraph.ecg(G, min_weight, ensemble_size)
    num_parts = df["partition"].max() + 1
    score = cugraph.analyzeClustering_modularity(
        G, num_parts, df, "vertex", "partition"
    )

    return score, num_parts


def golden_call(graph_file):
    if graph_file == PurePath(utils.RAPIDS_DATASET_ROOT_DIR) / "dolphins.csv":
        return 0.4962422251701355
    if graph_file == PurePath(utils.RAPIDS_DATASET_ROOT_DIR) / "karate.csv":
        return 0.38428664207458496
    if graph_file == PurePath(utils.RAPIDS_DATASET_ROOT_DIR) / "netscience.csv":
        return 0.9279554486274719


DATASETS = [karate, dolphins, netscience]

MIN_WEIGHTS = [0.05, 0.10, 0.15]

ENSEMBLE_SIZES = [16, 32]


@pytest.mark.parametrize("graph_file", DATASETS)
@pytest.mark.parametrize("min_weight", MIN_WEIGHTS)
@pytest.mark.parametrize("ensemble_size", ENSEMBLE_SIZES)
def test_ecg_clustering(graph_file, min_weight, ensemble_size):
    gc.collect()

    # Read in the graph and get a cugraph object

    G = graph_file.get_graph()
    dataset_path = graph_file.get_path()
    # read_weights_in_sp=False => value column dtype is float64
    G.edgelist.edgelist_df["weights"] = G.edgelist.edgelist_df["weights"].astype(
        "float64"
    )

    # Get the modularity score for partitioning versus random assignment
    cu_score, num_parts = cugraph_call(G, min_weight, ensemble_size)
    golden_score = golden_call(dataset_path)

    # Assert that the partitioning has better modularity than the random
    # assignment
    assert cu_score > (0.95 * golden_score)


@pytest.mark.parametrize("graph_file", DATASETS)
@pytest.mark.parametrize("min_weight", MIN_WEIGHTS)
@pytest.mark.parametrize("ensemble_size", ENSEMBLE_SIZES)
def test_ecg_clustering_nx(graph_file, min_weight, ensemble_size):
    gc.collect()
    dataset_path = graph_file.get_path()
    # Read in the graph and get a NetworkX graph
    M = utils.read_csv_for_nx(dataset_path, read_weights_in_sp=True)
    G = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.Graph()
    )

    # Get the modularity score for partitioning versus random assignment
    df_dict = cugraph.ecg(G, min_weight, ensemble_size, "weight")

    assert isinstance(df_dict, dict)

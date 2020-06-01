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

import gc

import pytest

import cugraph
from cugraph.tests import utils


def cugraph_call(G, min_weight, ensemble_size):
    df = cugraph.ecg(G, min_weight, ensemble_size)
    num_parts = df['partition'].max() + 1
    score = cugraph.analyzeClustering_modularity(G, num_parts, df['partition'])
    return score, num_parts


def golden_call(graph_file):
    if graph_file == '../datasets/dolphins.csv':
        return 0.4962422251701355
    if graph_file == '../datasets/karate.csv':
        return 0.38428664207458496
    if graph_file == '../datasets/netscience.csv':
        return 0.9279554486274719


DATASETS = ['../datasets/karate.csv',
            '../datasets/dolphins.csv',
            '../datasets/netscience.csv']

MIN_WEIGHTS = [.05, .10, .15]

ENSEMBLE_SIZES = [16, 32]


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('min_weight', MIN_WEIGHTS)
@pytest.mark.parametrize('ensemble_size', ENSEMBLE_SIZES)
def test_ecg_clustering(graph_file,
                        min_weight,
                        ensemble_size):
    gc.collect()

    # Read in the graph and get a cugraph object
    cu_M = utils.read_csv_file(graph_file, read_weights_in_sp=False)
    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1', edge_attr='2')

    # Get the modularity score for partitioning versus random assignment
    cu_score, num_parts = cugraph_call(G, min_weight, ensemble_size)
    golden_score = golden_call(graph_file)

    # Assert that the partitioning has better modularity than the random
    # assignment
    assert cu_score > (.95 * golden_score)

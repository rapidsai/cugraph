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
import random

import pytest

import cudf
import cugraph
from cugraph.tests import utils


def cugraph_call(G, partitions):
    df = cugraph.spectralBalancedCutClustering(G, partitions,
                                               num_eigen_vects=partitions)

    df = df.sort_values('vertex')
    score = cugraph.analyzeClustering_edge_cut(G, partitions, df['cluster'])
    return set(df['vertex'].to_array()), score


def random_call(G, partitions):
    random.seed(0)
    num_verts = G.number_of_vertices()
    assignment = []
    for i in range(num_verts):
        assignment.append(random.randint(0, partitions-1))
    assignment_cu = cudf.Series(assignment)
    score = cugraph.analyzeClustering_edge_cut(G, partitions, assignment_cu)
    return set(range(num_verts)), score


PARTITIONS = [2, 4, 8]


# Test all combinations of default/managed and pooled/non-pooled allocation

@pytest.mark.parametrize('graph_file', utils.DATASETS)
@pytest.mark.parametrize('partitions', PARTITIONS)
def test_edge_cut_clustering(graph_file, partitions):
    gc.collect()

    # Read in the graph and get a cugraph object
    cu_M = utils.read_csv_file(graph_file, read_weights_in_sp=False)

    G_edge = cugraph.Graph()
    G_edge.from_cudf_edgelist(cu_M, source='0', destination='1')

    # Get the edge_cut score for partitioning versus random assignment
    '''cu_vid, cu_score = cugraph_call(G_adj, partitions)
    rand_vid, rand_score = random_call(G_adj, partitions)
    '''
    # Assert that the partitioning has better edge_cut than the random
    # assignment
    '''assert cu_score < rand_score'''

    # Get the edge_cut score for partitioning versus random assignment
    cu_vid, cu_score = cugraph_call(G_edge, partitions)
    rand_vid, rand_score = random_call(G_edge, partitions)

    # Assert that the partitioning has better edge_cut than the random
    # assignment
    print(cu_score, rand_score)
    assert cu_score < rand_score


@pytest.mark.parametrize('graph_file', utils.DATASETS)
@pytest.mark.parametrize('partitions', PARTITIONS)
def test_edge_cut_clustering_with_edgevals(graph_file, partitions):
    gc.collect()

    # Read in the graph and get a cugraph object
    cu_M = utils.read_csv_file(graph_file, read_weights_in_sp=False)

    G_edge = cugraph.Graph()
    G_edge.from_cudf_edgelist(cu_M, source='0', destination='1',
                              edge_attr='2')

    # Get the edge_cut score for partitioning versus random assignment
    '''cu_vid, cu_score = cugraph_call(G_adj, partitions)
    rand_vid, rand_score = random_call(G_adj, partitions)
    '''
    # Assert that the partitioning has better edge_cut than the random
    # assignment
    '''assert cu_score < rand_score'''

    # Get the edge_cut score for partitioning versus random assignment
    cu_vid, cu_score = cugraph_call(G_edge, partitions)
    rand_vid, rand_score = random_call(G_edge, partitions)

    # Assert that the partitioning has better edge_cut than the random
    # assignment
    print(cu_score, rand_score)
    assert cu_score < rand_score


# Test to ensure DiGraph objs are not accepted
# Test all combinations of default/managed and pooled/non-pooled allocation

def test_digraph_rejected():
    gc.collect()

    df = cudf.DataFrame()
    df['src'] = cudf.Series(range(10))
    df['dst'] = cudf.Series(range(10))
    df['val'] = cudf.Series(range(10))

    G = cugraph.DiGraph()
    G.from_cudf_edgelist(df, source="src",
                         destination="dst",
                         edge_attr="val",
                         renumber=False)

    with pytest.raises(Exception):
        cugraph_call(G, 2)

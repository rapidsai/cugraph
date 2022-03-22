# Copyright (c) 2022, NVIDIA CORPORATION.
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


# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import pytest
from cugraph.tests import utils
import cugraph.compat.nx as nx
import numpy as np
import importlib


MAX_ITERATIONS = [500]
TOLERANCE = [1.0e-06]
ALPHA = [0.85]
PERSONALIZATION_PERC = [10]
HAS_GUESS = [0]

# =============================================================================
# Pytest fixtures
# =============================================================================


@pytest.fixture(scope="module", params=['networkx', 'nxcompat'])
def which_import(request):
    if (request.param == 'networkx'):
        return importlib.import_module("networkx")
    if (request.param == 'nxcompat'):
        return importlib.import_module("cugraph.compat.nx")


# The function selects personalization_perc% of accessible vertices in graph M
# and randomly assigns them personalization values
def get_personalization(personalization_perc, nnz_vtx):

    personalization = None
    if personalization_perc != 0:
        personalization = {}
        personalization_count = int(
            (nnz_vtx.size * personalization_perc) / 100.0)
        nnz_vtx = np.random.choice(nnz_vtx,
                                   min(nnz_vtx.size,
                                       personalization_count),
                                   replace=False)

        nnz_val = np.random.random(nnz_vtx.size)
        nnz_val = nnz_val / sum(nnz_val)
        for vtx, val in zip(nnz_vtx, nnz_val):
            personalization[vtx] = val

    return personalization


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED_WEIGHTS)
def test_with_noparams(graph_file):

    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight",
        create_using=nx.DiGraph()
    )
    pr = nx.pagerank(Gnx)
    print(type(Gnx))
    assert type(pr) == dict

    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight",
        create_using=nx.DiGraph()
    )
    pr = nx.pagerank(Gnx)
    print(type(Gnx))
    assert type(pr) == dict


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED_WEIGHTS)
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
def test_with_max_iter(graph_file, max_iter):

    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight",
        create_using=nx.DiGraph()
    )

    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight",
        create_using=nx.DiGraph()
    )
    pr = nx.pagerank(Gnx, max_iter=max_iter)
    print(type(Gnx))
    assert type(pr) == dict


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
@pytest.mark.parametrize("personalization_perc", PERSONALIZATION_PERC)
def test_perc_spec(graph_file, max_iter, personalization_perc, which_import):

    test_import = which_import
    nx = test_import
    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight",
        create_using=nx.DiGraph()
    )

    # NetworkX PageRank
    M = utils.read_csv_for_nx(graph_file)
    nnz_vtx = np.unique(M[['0', '1']])
    Gnx = nx.from_pandas_edgelist(M,
                                  source="0",
                                  target="1",
                                  edge_attr="weight",
                                  create_using=nx.DiGraph())

    personalization_dict = get_personalization(personalization_perc, nnz_vtx)
    pr_cu = nx.pagerank(
        Gnx, max_iter=max_iter,
        personalization=personalization_dict
        )
    print(type(Gnx))
    print(pr_cu)

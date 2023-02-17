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
import gc
import importlib

import pytest
import numpy as np
from pylibcugraph.testing.utils import gen_fixture_params_product

from cugraph.testing import utils
from cugraph.experimental.datasets import karate


MAX_ITERATIONS = [100, 200]
TOLERANCE = [1.0e-06]
ALPHA = [0.85, 0.70]
PERS_PERCENT = [0, 15]
HAS_GUESS = [0, 1]

FILES_UNDIRECTED = [karate.get_path()]

# these are only used in the missing parameter tests.
KARATE_RANKING = [11, 9, 14, 15, 18, 20, 22, 17, 21, 12, 26, 16, 28, 19]

KARATE_PERS_RANKING = [11, 16, 17, 21, 4, 10, 5, 6, 12, 7, 9, 24, 19, 25]

KARATE_ITER_RANKINGS = [11, 9, 14, 15, 18, 20, 22, 17, 21, 12, 26, 16, 28, 19]

KARATE_NSTART_RANKINGS = [11, 9, 14, 15, 18, 20, 22, 17, 21, 12, 26, 16, 28, 19]


# =============================================================================
# Pytest fixtures
# =============================================================================
def setup_function():
    gc.collect()


datasets = FILES_UNDIRECTED
fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    (MAX_ITERATIONS, "max_iter"),
    (TOLERANCE, "tol"),
    (PERS_PERCENT, "pers_percent"),
    (HAS_GUESS, "has_guess"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(
        zip(
            ("graph_file", "max_iter", "tol", "pers_percent", "has_guess"),
            request.param,
        )
    )

    return parameters


@pytest.fixture(scope="module")
def input_expected_output(input_combo):
    """
    This fixture returns the expected results from the pagerank algorithm.
    """
    import networkx

    M = utils.read_csv_for_nx(input_combo["graph_file"])

    Gnx = networkx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=networkx.DiGraph()
    )
    nnz_vtx = np.unique(M[["0", "1"]])
    personalization = get_personalization(input_combo["pers_percent"], nnz_vtx)
    input_combo["nstart"] = None
    nstart = None
    if input_combo["has_guess"] == 1:
        z = {k: 1.0 / Gnx.number_of_nodes() for k in Gnx.nodes()}
        input_combo["nstart"] = z
        nstart = z

    pr = networkx.pagerank(
        Gnx,
        max_iter=input_combo["max_iter"],
        tol=input_combo["tol"],
        personalization=personalization,
        nstart=nstart,
    )
    input_combo["personalization"] = personalization
    input_combo["nx_pr_rankings"] = pr
    return input_combo


@pytest.fixture(scope="module", params=["networkx", "nxcompat"])
def which_import(request):
    if request.param == "networkx":
        return importlib.import_module("networkx")
    if request.param == "nxcompat":
        return importlib.import_module("cugraph.experimental.compat.nx")


# The function selects personalization_perc% of accessible vertices in graph M
# and randomly assigns them personalization values
def get_personalization(personalization_perc, nnz_vtx):
    personalization = None
    if personalization_perc != 0:
        personalization = {}
        personalization_count = int((nnz_vtx.size * personalization_perc) / 100.0)
        nnz_vtx = np.random.choice(
            nnz_vtx, min(nnz_vtx.size, personalization_count), replace=False
        )

        nnz_val = np.random.random(nnz_vtx.size)
        nnz_val = nnz_val / sum(nnz_val)
        for vtx, val in zip(nnz_vtx, nnz_val):
            personalization[vtx] = val
    return personalization


@pytest.mark.parametrize("graph_file", FILES_UNDIRECTED)
def test_with_noparams(graph_file, which_import):
    nx = which_import

    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.DiGraph()
    )
    pr = nx.pagerank(Gnx)

    # Rounding issues show up in runs but this tests that the
    # cugraph and networkx algrorithms are being correctly called.
    assert (sorted(pr, key=pr.get)[:14]) == KARATE_RANKING


@pytest.mark.parametrize("graph_file", FILES_UNDIRECTED)
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
def test_with_max_iter(graph_file, max_iter, which_import):
    nx = which_import
    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.DiGraph()
    )
    pr = nx.pagerank(Gnx, max_iter=max_iter)
    # Rounding issues show up in runs but this tests that the
    # cugraph and networkx algrorithms are being correctly called.
    assert (sorted(pr, key=pr.get)[:14]) == KARATE_ITER_RANKINGS


@pytest.mark.parametrize("graph_file", FILES_UNDIRECTED)
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
def test_perc_spec(graph_file, max_iter, which_import):
    nx = which_import

    # simple personalization to validate running
    personalization = {
        20: 0.7237260913723357,
        12: 0.03952608674390543,
        22: 0.2367478218837589,
    }

    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.DiGraph()
    )

    # NetworkX PageRank
    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.DiGraph()
    )
    # uses the same personalization for each imported package

    pr = nx.pagerank(Gnx, max_iter=max_iter, personalization=personalization)

    # Rounding issues show up in runs but this tests that the
    # cugraph and networkx algrorithms are being correctly called.
    assert (sorted(pr, key=pr.get)[:14]) == KARATE_PERS_RANKING


@pytest.mark.parametrize("graph_file", FILES_UNDIRECTED)
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
def test_with_nstart(graph_file, max_iter, which_import):
    nx = which_import

    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.DiGraph()
    )

    z = {k: 1.0 / Gnx.number_of_nodes() for k in Gnx.nodes()}

    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.DiGraph()
    )
    pr = nx.pagerank(Gnx, max_iter=max_iter, nstart=z)

    # Rounding issues show up in runs but this tests that the
    # cugraph and networkx algrorithms are being correctly called.
    assert (sorted(pr, key=pr.get)[:14]) == KARATE_NSTART_RANKINGS


def test_fixture_data(input_expected_output, which_import):
    nx = which_import
    M = utils.read_csv_for_nx(input_expected_output["graph_file"])
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.DiGraph()
    )
    pr = nx.pagerank(
        Gnx,
        max_iter=input_expected_output["max_iter"],
        tol=input_expected_output["tol"],
        personalization=input_expected_output["personalization"],
        nstart=input_expected_output["nstart"],
    )
    actual = sorted(pr.items())
    expected = sorted(input_expected_output["nx_pr_rankings"].items())
    assert all([a == pytest.approx(b, abs=1.0e-04) for a, b in zip(actual, expected)])

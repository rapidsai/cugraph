# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
import networkx as nx
import numpy as np
import pytest

import nx_cugraph as nxcg
from nx_cugraph import _nxver

from .testing_utils import assert_graphs_equal

if _nxver < (3, 2):
    pytest.skip("Need NetworkX >=3.2 to test generators", allow_module_level=True)


def compare(name, create_using, *args, is_vanilla=False):
    exc1 = exc2 = None
    func = getattr(nx, name)
    if isinstance(create_using, nxcg.CudaGraph):
        nx_create_using = nxcg.to_networkx(create_using)
    elif isinstance(create_using, type) and issubclass(
        create_using, (nxcg.Graph, nxcg.CudaGraph)
    ):
        nx_create_using = create_using.to_networkx_class()
    elif isinstance(create_using, nx.Graph):
        nx_create_using = create_using.copy()
    else:
        nx_create_using = create_using
    try:
        if is_vanilla:
            G = func(*args)
        else:
            G = func(*args, create_using=nx_create_using)
    except Exception as exc:
        exc1 = exc
    try:
        if is_vanilla:
            Gcg = func(*args, backend="cugraph")
        else:
            Gcg = func(*args, create_using=create_using, backend="cugraph")
    except ZeroDivisionError:
        raise
    except NotImplementedError as exc:
        if name in {"complete_multipartite_graph"}:  # nx.__version__[:3] <= "3.2"
            return
        exc2 = exc
    except Exception as exc:
        if exc1 is None:  # pragma: no cover (debug)
            raise
        exc2 = exc
    if exc1 is not None or exc2 is not None:
        assert type(exc1) is type(exc2)
        return
    if isinstance(Gcg, nxcg.Graph):
        # If the graph is empty, it may be on host, otherwise it should be on device
        if len(G):
            assert Gcg._is_on_gpu
            assert not Gcg._is_on_cpu
        assert_graphs_equal(G, Gcg._cudagraph)
    else:
        assert_graphs_equal(G, Gcg)
    # Ensure the output type is correct
    if is_vanilla:
        if _nxver < (3, 3) or nx.config.backends.cugraph.use_compat_graphs:
            assert isinstance(Gcg, nxcg.Graph)
        else:
            assert isinstance(Gcg, nxcg.CudaGraph)
    elif isinstance(create_using, type) and issubclass(
        create_using, (nxcg.Graph, nxcg.CudaGraph)
    ):
        assert type(Gcg) is create_using
    elif isinstance(create_using, (nxcg.Graph, nxcg.CudaGraph)):
        assert type(Gcg) is type(create_using)


N = list(range(-1, 5))
CREATE_USING = [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
COMPLETE_CREATE_USING = [
    nx.Graph,
    nx.DiGraph,
    nx.MultiGraph,
    nx.MultiDiGraph,
    nxcg.Graph,
    nxcg.DiGraph,
    nxcg.MultiGraph,
    nxcg.MultiDiGraph,
    nxcg.CudaGraph,
    nxcg.CudaDiGraph,
    nxcg.CudaMultiGraph,
    nxcg.CudaMultiDiGraph,
    # These raise NotImplementedError
    # nx.Graph(),
    # nx.DiGraph(),
    # nx.MultiGraph(),
    # nx.MultiDiGraph(),
    nxcg.Graph(),
    nxcg.DiGraph(),
    nxcg.MultiGraph(),
    nxcg.MultiDiGraph(),
    nxcg.CudaGraph(),
    nxcg.CudaDiGraph(),
    nxcg.CudaMultiGraph(),
    nxcg.CudaMultiDiGraph(),
    None,
    object,  # Bad input
    7,  # Bad input
]
GENERATORS_NOARG = [
    # classic
    "null_graph",
    "trivial_graph",
    # small
    "bull_graph",
    "chvatal_graph",
    "cubical_graph",
    "desargues_graph",
    "diamond_graph",
    "dodecahedral_graph",
    "frucht_graph",
    "heawood_graph",
    "house_graph",
    "house_x_graph",
    "icosahedral_graph",
    "krackhardt_kite_graph",
    "moebius_kantor_graph",
    "octahedral_graph",
    "petersen_graph",
    "sedgewick_maze_graph",
    "tetrahedral_graph",
    "truncated_cube_graph",
    "truncated_tetrahedron_graph",
    "tutte_graph",
]
GENERATORS_NOARG_VANILLA = [
    # classic
    "complete_multipartite_graph",
    # small
    "pappus_graph",
    # social
    "davis_southern_women_graph",
    "florentine_families_graph",
    "karate_club_graph",
    "les_miserables_graph",
]
GENERATORS_N = [
    # classic
    "circular_ladder_graph",
    "complete_graph",
    "cycle_graph",
    "empty_graph",
    "ladder_graph",
    "path_graph",
    "star_graph",
    "wheel_graph",
]
GENERATORS_M_N = [
    # classic
    "barbell_graph",
    "lollipop_graph",
    "tadpole_graph",
    # bipartite
    "complete_bipartite_graph",
]
GENERATORS_M_N_VANILLA = [
    # classic
    "complete_multipartite_graph",
    "turan_graph",
    # community
    "caveman_graph",
]


@pytest.mark.parametrize("name", GENERATORS_NOARG)
@pytest.mark.parametrize("create_using", COMPLETE_CREATE_USING)
def test_generator_noarg(name, create_using):
    print(name, create_using, type(create_using))
    if isinstance(create_using, nxcg.CudaGraph) and name in {
        # fmt: off
        "bull_graph", "chvatal_graph", "cubical_graph", "diamond_graph",
        "house_graph", "house_x_graph", "icosahedral_graph", "krackhardt_kite_graph",
        "octahedral_graph", "petersen_graph", "truncated_cube_graph", "tutte_graph",
        # fmt: on
    }:
        # The _raise_on_directed decorator used in networkx doesn't like our graphs.
        if create_using.is_directed():
            with pytest.raises(AssertionError):
                compare(name, create_using)
        else:
            with pytest.raises(TypeError):
                compare(name, create_using)
    else:
        compare(name, create_using)


@pytest.mark.parametrize("name", GENERATORS_NOARG_VANILLA)
def test_generator_noarg_vanilla(name):
    print(name)
    compare(name, None, is_vanilla=True)


@pytest.mark.parametrize("name", GENERATORS_N)
@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("create_using", CREATE_USING)
def test_generator_n(name, n, create_using):
    print(name, n, create_using)
    compare(name, create_using, n)


@pytest.mark.parametrize("name", GENERATORS_N)
@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("create_using", COMPLETE_CREATE_USING)
def test_generator_n_complete(name, n, create_using):
    print(name, n, create_using)
    compare(name, create_using, n)


@pytest.mark.parametrize("name", GENERATORS_M_N)
@pytest.mark.parametrize("create_using", CREATE_USING)
@pytest.mark.parametrize("m", N)
@pytest.mark.parametrize("n", N)
def test_generator_m_n(name, create_using, m, n):
    print(name, m, n, create_using)
    compare(name, create_using, m, n)


@pytest.mark.parametrize("name", GENERATORS_M_N_VANILLA)
@pytest.mark.parametrize("m", N)
@pytest.mark.parametrize("n", N)
def test_generator_m_n_vanilla(name, m, n):
    print(name, m, n)
    compare(name, None, m, n, is_vanilla=True)


@pytest.mark.parametrize("name", GENERATORS_M_N)
@pytest.mark.parametrize("create_using", COMPLETE_CREATE_USING)
@pytest.mark.parametrize("m", [4])
@pytest.mark.parametrize("n", [4])
def test_generator_m_n_complete(name, create_using, m, n):
    print(name, m, n, create_using)
    compare(name, create_using, m, n)


@pytest.mark.parametrize("name", GENERATORS_M_N_VANILLA)
@pytest.mark.parametrize("m", [4])
@pytest.mark.parametrize("n", [4])
def test_generator_m_n_complete_vanilla(name, m, n):
    print(name, m, n)
    compare(name, None, m, n, is_vanilla=True)


def test_bad_lollipop_graph():
    compare("lollipop_graph", None, [0, 1], [1, 2])


def test_can_convert_karate_club():
    # Karate club graph has string node values.
    # This really tests conversions, but it's here so we can use `assert_graphs_equal`.
    G = nx.karate_club_graph()
    G.add_node(0, foo="bar")  # string dtype with a mask
    G.add_node(1, object=object())  # haha
    Gcg = nxcg.from_networkx(G, preserve_all_attrs=True)
    assert_graphs_equal(G, Gcg)
    Gnx = nxcg.to_networkx(Gcg)
    assert nx.utils.graphs_equal(G, Gnx)
    assert isinstance(Gcg.node_values["club"], np.ndarray)
    assert Gcg.node_values["club"].dtype.kind == "U"
    assert isinstance(Gcg.node_values["foo"], np.ndarray)
    assert isinstance(Gcg.node_masks["foo"], np.ndarray)
    assert Gcg.node_values["foo"].dtype.kind == "U"
    assert isinstance(Gcg.node_values["object"], np.ndarray)
    assert Gcg.node_values["object"].dtype.kind == "O"
    assert isinstance(Gcg.node_masks["object"], np.ndarray)

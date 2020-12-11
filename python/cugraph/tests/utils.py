# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import os
from itertools import product

# Assume test environment has the following dependencies installed
import pytest
import pandas as pd
import networkx as nx
import numpy as np
import cupy as cp
from cupyx.scipy.sparse.coo import coo_matrix as cp_coo_matrix
from cupyx.scipy.sparse.csr import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse.csc import csc_matrix as cp_csc_matrix
from scipy.sparse.coo import coo_matrix as sp_coo_matrix
from scipy.sparse.csr import csr_matrix as sp_csr_matrix
from scipy.sparse.csc import csc_matrix as sp_csc_matrix

import cudf
import dask_cudf

import cugraph
from cugraph.dask.common.mg_utils import get_client


CUPY_MATRIX_TYPES = [cp_coo_matrix, cp_csr_matrix, cp_csc_matrix]
SCIPY_MATRIX_TYPES = [sp_coo_matrix, sp_csr_matrix, sp_csc_matrix]

#
# Datasets
#
DATASETS_UNDIRECTED = ["../datasets/karate.csv", "../datasets/dolphins.csv"]

DATASETS_UNDIRECTED_WEIGHTS = [
    "../datasets/netscience.csv",
]

DATASETS_UNRENUMBERED = ["../datasets/karate-disjoint.csv"]

DATASETS = [
    "../datasets/karate-disjoint.csv",
    "../datasets/dolphins.csv",
    "../datasets/netscience.csv",
]
#            '../datasets/email-Eu-core.csv']

STRONGDATASETS = [
    "../datasets/dolphins.csv",
    "../datasets/netscience.csv",
    "../datasets/email-Eu-core.csv",
]

DATASETS_KTRUSS = [
    ("../datasets/polbooks.csv", "../datasets/ref/ktruss/polbooks.csv")
]

DATASETS_SMALL = [
    "../datasets/karate.csv",
    "../datasets/dolphins.csv",
    "../datasets/polbooks.csv",
]

MATRIX_INPUT_TYPES = [
    pytest.param(
        cp_coo_matrix, marks=pytest.mark.matrix_types, id="CuPy.coo_matrix"
    ),
    pytest.param(
        cp_csr_matrix, marks=pytest.mark.matrix_types, id="CuPy.csr_matrix"
    ),
    pytest.param(
        cp_csc_matrix, marks=pytest.mark.matrix_types, id="CuPy.csc_matrix"
    ),
]

NX_INPUT_TYPES = [
    pytest.param(nx.Graph, marks=pytest.mark.nx_types, id="nx.Graph"),
]

NX_DIR_INPUT_TYPES = [
    pytest.param(nx.Graph, marks=pytest.mark.nx_types, id="nx.DiGraph"),
]

CUGRAPH_INPUT_TYPES = [
    pytest.param(
        cugraph.Graph, marks=pytest.mark.cugraph_types, id="cugraph.Graph"
    ),
]

CUGRAPH_DIR_INPUT_TYPES = [
    pytest.param(
        cugraph.DiGraph, marks=pytest.mark.cugraph_types, id="cugraph.DiGraph"
    ),
]


def read_csv_for_nx(csv_file, read_weights_in_sp=True, read_weights=True):
    print("Reading " + str(csv_file) + "...")
    if read_weights:
        if read_weights_in_sp is True:
            df = pd.read_csv(
                csv_file,
                delimiter=" ",
                header=None,
                names=["0", "1", "weight"],
                dtype={"0": "int32", "1": "int32", "weight": "float32"},
            )
        else:
            df = pd.read_csv(
                csv_file,
                delimiter=" ",
                header=None,
                names=["0", "1", "weight"],
                dtype={"0": "int32", "1": "int32", "weight": "float64"},
            )
    else:
        df = pd.read_csv(
            csv_file,
            delimiter=" ",
            header=None,
            names=["0", "1"],
            usecols=["0", "1"],
            dtype={"0": "int32", "1": "int32"},
        )
    return df


def create_obj_from_csv(
    csv_file_name, obj_type, csv_has_weights=True, edgevals=False
):
    """
    Return an object based on obj_type populated with the contents of
    csv_file_name
    """
    if obj_type in [cugraph.Graph, cugraph.DiGraph]:
        return generate_cugraph_graph_from_file(
            csv_file_name,
            directed=(obj_type is cugraph.DiGraph),
            edgevals=edgevals,
        )

    elif obj_type in SCIPY_MATRIX_TYPES + CUPY_MATRIX_TYPES:
        # FIXME: assuming float32
        if csv_has_weights:
            (rows, cols, weights) = np.genfromtxt(
                csv_file_name, delimiter=" ", dtype=np.float32, unpack=True
            )
        else:
            (rows, cols) = np.genfromtxt(
                csv_file_name, delimiter=" ", dtype=np.float32, unpack=True
            )

        if (csv_has_weights is False) or (edgevals is False):
            # COO matrices must have a value array. Also if edgevals are to be
            # ignored (False), reset all weights to 1.
            weights = np.array([1] * len(rows))

        if obj_type in CUPY_MATRIX_TYPES:
            coo = cp_coo_matrix(
                (cp.asarray(weights), (cp.asarray(rows), cp.asarray(cols))),
                dtype=np.float32,
            )
        else:
            coo = sp_coo_matrix(
                (weights, (np.array(rows, dtype=int),
                           np.array(cols, dtype=int))),
            )

        if obj_type in [cp_csr_matrix, sp_csr_matrix]:
            return coo.tocsr(copy=False)
        elif obj_type in [cp_csc_matrix, sp_csc_matrix]:
            return coo.tocsc(copy=False)
        else:
            return coo

    elif obj_type in [nx.Graph, nx.DiGraph]:
        return generate_nx_graph_from_file(
            csv_file_name, directed=(obj_type is nx.DiGraph), edgevals=edgevals
        )

    else:
        raise TypeError(f"unsupported type: {obj_type}")


def read_csv_file(csv_file, read_weights_in_sp=True):
    print("Reading " + str(csv_file) + "...")
    if read_weights_in_sp is True:
        return cudf.read_csv(
            csv_file,
            delimiter=" ",
            dtype=["int32", "int32", "float32"],
            header=None,
        )
    else:
        return cudf.read_csv(
            csv_file,
            delimiter=" ",
            dtype=["int32", "int32", "float64"],
            header=None,
        )


def read_dask_cudf_csv_file(
    csv_file, read_weights_in_sp=True, single_partition=True
):
    print("Reading " + str(csv_file) + "...")
    if read_weights_in_sp is True:
        if single_partition:
            chunksize = os.path.getsize(csv_file)
            return dask_cudf.read_csv(
                csv_file,
                chunksize=chunksize,
                delimiter=" ",
                names=["src", "dst", "weight"],
                dtype=["int32", "int32", "float32"],
                header=None,
            )
        else:
            return dask_cudf.read_csv(
                csv_file,
                delimiter=" ",
                names=["src", "dst", "weight"],
                dtype=["int32", "int32", "float32"],
                header=None,
            )
    else:
        if single_partition:
            chunksize = os.path.getsize(csv_file)
            return dask_cudf.read_csv(
                csv_file,
                chunksize=chunksize,
                delimiter=" ",
                names=["src", "dst", "weight"],
                dtype=["int32", "int32", "float32"],
                header=None,
            )
        else:
            return dask_cudf.read_csv(
                csv_file,
                delimiter=" ",
                names=["src", "dst", "weight"],
                dtype=["int32", "int32", "float64"],
                header=None,
            )


def generate_nx_graph_from_file(graph_file, directed=True, edgevals=False):
    M = read_csv_for_nx(graph_file, read_weights_in_sp=edgevals)
    edge_attr = "weight" if edgevals else None
    Gnx = nx.from_pandas_edgelist(
        M,
        create_using=(nx.DiGraph() if directed else nx.Graph()),
        source="0",
        target="1",
        edge_attr=edge_attr,
    )
    return Gnx


def generate_cugraph_graph_from_file(
    graph_file, directed=True, edgevals=False
):
    cu_M = read_csv_file(graph_file)
    G = cugraph.DiGraph() if directed else cugraph.Graph()

    if edgevals:
        G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")
    else:
        G.from_cudf_edgelist(cu_M, source="0", destination="1")
    return G


def generate_mg_batch_cugraph_graph_from_file(graph_file, directed=True):
    client = get_client()
    _ddf = read_dask_cudf_csv_file(graph_file)
    ddf = client.persist(_ddf)
    G = cugraph.DiGraph() if directed else cugraph.Graph()
    G.from_dask_cudf_edgelist(ddf)
    return G


def build_cu_and_nx_graphs(graph_file, directed=True, edgevals=False):
    G = generate_cugraph_graph_from_file(graph_file, directed=directed,
                                         edgevals=edgevals)
    Gnx = generate_nx_graph_from_file(graph_file, directed=directed,
                                      edgevals=edgevals)
    return G, Gnx


def build_mg_batch_cu_and_nx_graphs(graph_file, directed=True):
    G = generate_mg_batch_cugraph_graph_from_file(
        graph_file, directed=directed
    )
    Gnx = generate_nx_graph_from_file(graph_file, directed=directed)
    return G, Gnx


def random_edgelist(
    e=1024,
    ef=16,
    dtypes={"src": np.int32, "dst": np.int32, "val": float},
    drop_duplicates=True,
    seed=None,
):
    """Create a random edge list

    Parameters
    ----------
    e : int
        Number of edges
    ef : int
        Edge factor (average number of edges per vertex)
    dtypes : dict
        Mapping of column names to types.
        Supported type is {"src": int, "dst": int, "val": float}
    drop_duplicates
        Drop duplicates
    seed : int (optional)
        Randomstate seed

    Examples
    --------
    >>> from cugraph.tests import utils
    >>> # genrates 20 df with 100M edges each and write to disk
    >>> for x in range(20):
    >>>    df = utils.random_edgelist(e=100000000, ef=64,
    >>>                               dtypes={'src':np.int32, 'dst':np.int32},
    >>>                               seed=x)
    >>>    df.to_csv('df'+str(x), header=False, index=False)
    >>>    #df.to_parquet('files_parquet/df'+str(x), index=False)
    """
    state = np.random.RandomState(seed)
    columns = dict(
        (k, make[dt](e // ef, e, state)) for k, dt in dtypes.items()
    )

    df = pd.DataFrame(columns)
    if drop_duplicates:
        df = df.drop_duplicates(subset=["src", "dst"])
        print("Generated " + str(df.shape[0]) + " edges")
    return df


def make_int32(v, e, rstate):
    return rstate.randint(low=0, high=v, size=e, dtype=np.int32)


def make_int64(v, e, rstate):
    return rstate.randint(low=0, high=v, size=e, dtype=np.int64)


def make_float(v, e, rstate):
    return rstate.rand(e)


make = {float: make_float, np.int32: make_int32, np.int64: make_int64}


def genFixtureParamsProduct(*args):
    """
    Returns the cartesian product of the param lists passed in. The lists must
    be flat lists of pytest.param objects, and the result will be a flat list
    of pytest.param objects with values and meta-data combined accordingly. A
    flat list of pytest.param objects is required for pytest fixtures to
    properly recognize the params. The combinations also include ids generated
    from the param values and id names associated with each list. For example:

    genFixtureParamsProduct( ([pytest.param(True, marks=[pytest.mark.A_good]),
                               pytest.param(False, marks=[pytest.mark.A_bad])],
                              "A"),
                             ([pytest.param(True, marks=[pytest.mark.B_good]),
                               pytest.param(False, marks=[pytest.mark.B_bad])],
                              "B") )

    results in fixture param combinations:

    True, True   - marks=[A_good, B_good] - id="A=True,B=True"
    True, False  - marks=[A_good, B_bad]  - id="A=True,B=False"
    False, True  - marks=[A_bad, B_good]  - id="A=False,B=True"
    False, False - marks=[A_bad, B_bad]   - id="A=False,B=False"

    Simply using itertools.product on the lists would result in a list of
    sublists of individual param objects (ie. not "merged"), which would not be
    recognized properly as params for a fixture by pytest.

    NOTE: This function is only needed for parameterized fixtures.
    Tests/benchmarks will automatically get this behavior when specifying
    multiple @pytest.mark.parameterize(param_name, param_value_list)
    decorators.
    """
    # Enforce that each arg is a list of pytest.param objs and separate params
    # and IDs.
    paramLists = []
    ids = []
    paramType = pytest.param().__class__
    for (paramList, id) in args:
        for param in paramList:
            assert isinstance(param, paramType)
        paramLists.append(paramList)
        ids.append(id)

    retList = []
    for paramCombo in product(*paramLists):
        values = [p.values[0] for p in paramCombo]
        marks = [m for p in paramCombo for m in p.marks]
        comboid = ",".join(
            ["%s=%s" % (id, p.values[0]) for (p, id) in zip(paramCombo, ids)]
        )
        retList.append(pytest.param(values, marks=marks, id=comboid))
    return retList


# shared between min and max spanning tree tests
def compare_mst(mst_cugraph, mst_nx):
    mst_nx_df = nx.to_pandas_edgelist(mst_nx)
    edgelist_df = mst_cugraph.view_edge_list()
    assert len(mst_nx_df) == len(edgelist_df)

    # check cycles
    Gnx = nx.from_pandas_edgelist(
        edgelist_df.to_pandas(),
        create_using=nx.Graph(),
        source="src",
        target="dst",
    )
    try:
        lc = nx.find_cycle(Gnx, source=None, orientation="ignore")
        print(lc)
    except nx.NetworkXNoCycle:
        pass

    # check total weight
    cg_sum = edgelist_df["weights"].sum()
    nx_sum = mst_nx_df["weight"].sum()
    print(cg_sum)
    print(nx_sum)
    assert np.isclose(cg_sum, nx_sum)

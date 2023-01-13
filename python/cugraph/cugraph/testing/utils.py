# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

# Assume test environment has the following dependencies installed
import pytest
import pandas as pd
import networkx as nx
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import coo_matrix as cp_coo_matrix
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix
from scipy.sparse import coo_matrix as sp_coo_matrix
from scipy.sparse import csr_matrix as sp_csr_matrix
from scipy.sparse import csc_matrix as sp_csc_matrix
from pathlib import Path
import cudf
import dask_cudf

import cugraph
from cugraph.dask.common.mg_utils import get_client


CUPY_MATRIX_TYPES = [cp_coo_matrix, cp_csr_matrix, cp_csc_matrix]
SCIPY_MATRIX_TYPES = [sp_coo_matrix, sp_csr_matrix, sp_csc_matrix]

RAPIDS_DATASET_ROOT_DIR = os.getenv(
    "RAPIDS_DATASET_ROOT_DIR", os.path.join(os.path.dirname(__file__), "../datasets")
)
RAPIDS_DATASET_ROOT_DIR_PATH = Path(RAPIDS_DATASET_ROOT_DIR)

#
# Datasets
#
DATASETS_UNDIRECTED = [
    RAPIDS_DATASET_ROOT_DIR_PATH / f for f in ["karate.csv", "dolphins.csv"]
]

DATASETS_UNDIRECTED_WEIGHTS = [RAPIDS_DATASET_ROOT_DIR_PATH / "netscience.csv"]

DATASETS_UNRENUMBERED = [Path(RAPIDS_DATASET_ROOT_DIR) / "karate-disjoint.csv"]

DATASETS = [
    RAPIDS_DATASET_ROOT_DIR_PATH / f
    for f in ["karate-disjoint.csv", "dolphins.csv", "netscience.csv"]
]

DATASETS_MULTI_EDGES = [
    RAPIDS_DATASET_ROOT_DIR_PATH / f
    for f in ["karate_multi_edge.csv", "dolphins_multi_edge.csv"]
]

DATASETS_STR_ISLT_V = [
    RAPIDS_DATASET_ROOT_DIR_PATH / f for f in ["karate_mod.mtx", "karate_str.mtx"]
]

DATASETS_SELF_LOOPS = [
    RAPIDS_DATASET_ROOT_DIR_PATH / f
    for f in ["karate_s_loop.csv", "dolphins_s_loop.csv"]
]


#            '../datasets/email-Eu-core.csv']

STRONGDATASETS = [
    RAPIDS_DATASET_ROOT_DIR_PATH / f
    for f in ["dolphins.csv", "netscience.csv", "email-Eu-core.csv"]
]


DATASETS_KTRUSS = [
    (
        RAPIDS_DATASET_ROOT_DIR_PATH / "polbooks.csv",
        RAPIDS_DATASET_ROOT_DIR_PATH / "ref/ktruss/polbooks.csv",
    )
]

DATASETS_TSPLIB = [
    (RAPIDS_DATASET_ROOT_DIR_PATH / f,) + (d,)
    for (f, d) in [
        ("gil262.tsp", 2378),
        ("eil51.tsp", 426),
        ("kroA100.tsp", 21282),
        ("tsp225.tsp", 3916),
    ]
]

DATASETS_SMALL = [
    RAPIDS_DATASET_ROOT_DIR_PATH / f
    for f in ["karate.csv", "dolphins.csv", "polbooks.csv"]
]


MATRIX_INPUT_TYPES = [
    pytest.param(cp_coo_matrix, marks=pytest.mark.matrix_types, id="CuPy.coo_matrix"),
    pytest.param(cp_csr_matrix, marks=pytest.mark.matrix_types, id="CuPy.csr_matrix"),
    pytest.param(cp_csc_matrix, marks=pytest.mark.matrix_types, id="CuPy.csc_matrix"),
]

NX_INPUT_TYPES = [
    pytest.param(nx.Graph, marks=pytest.mark.nx_types, id="nx.Graph"),
]

NX_DIR_INPUT_TYPES = [
    pytest.param(nx.Graph, marks=pytest.mark.nx_types, id="nx.DiGraph"),
]

CUGRAPH_INPUT_TYPES = [
    pytest.param(cugraph.Graph(), marks=pytest.mark.cugraph_types, id="cugraph.Graph"),
]

CUGRAPH_DIR_INPUT_TYPES = [
    pytest.param(
        cugraph.Graph(directed=True),
        marks=pytest.mark.cugraph_types,
        id="cugraph.Graph(directed=True)",
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
    csv_file_name, obj_type, csv_has_weights=True, edgevals=False, directed=False
):
    """
    Return an object based on obj_type populated with the contents of
    csv_file_name
    """
    if obj_type in [cugraph.Graph]:
        return generate_cugraph_graph_from_file(
            csv_file_name,
            directed=directed,
            edgevals=edgevals,
        )
    elif isinstance(obj_type, cugraph.Graph):
        return generate_cugraph_graph_from_file(
            csv_file_name,
            directed=directed,
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
                (weights, (np.array(rows, dtype=int), np.array(cols, dtype=int))),
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


def read_dask_cudf_csv_file(csv_file, read_weights_in_sp=True, single_partition=True):
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


def generate_cugraph_graph_from_file(graph_file, directed=True, edgevals=False):
    cu_M = read_csv_file(graph_file)

    G = cugraph.Graph(directed=directed)

    if edgevals:
        G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")
    else:
        G.from_cudf_edgelist(cu_M, source="0", destination="1")
    return G


def generate_mg_batch_cugraph_graph_from_file(graph_file, directed=True):
    client = get_client()
    _ddf = read_dask_cudf_csv_file(graph_file)
    ddf = client.persist(_ddf)
    G = cugraph.Graph(directed=directed)
    G.from_dask_cudf_edgelist(ddf)
    return G


def build_cu_and_nx_graphs(graph_file, directed=True, edgevals=False):
    G = generate_cugraph_graph_from_file(
        graph_file, directed=directed, edgevals=edgevals
    )
    Gnx = generate_nx_graph_from_file(graph_file, directed=directed, edgevals=edgevals)
    return G, Gnx


def build_mg_batch_cu_and_nx_graphs(graph_file, directed=True):
    G = generate_mg_batch_cugraph_graph_from_file(graph_file, directed=directed)
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
    >>> from cugraph.testing import utils
    >>> # genrates 20 df with 100M edges each and write to disk
    >>> for x in range(20):
    >>>    df = utils.random_edgelist(e=100000000, ef=64,
    >>>                               dtypes={'src':np.int32, 'dst':np.int32},
    >>>                               seed=x)
    >>>    df.to_csv('df'+str(x), header=False, index=False)
    >>>    #df.to_parquet('files_parquet/df'+str(x), index=False)
    """
    state = np.random.RandomState(seed)
    columns = dict((k, make[dt](e // ef, e, state)) for k, dt in dtypes.items())

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

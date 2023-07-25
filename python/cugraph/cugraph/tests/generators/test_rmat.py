# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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


import pytest

import cudf

from cugraph.generators import rmat
import cugraph
from cupy.sparse import coo_matrix, triu, tril
import numpy as np
import cupy as cp


##############################################################################

_scale_values = [2, 4, 16]
_scale_test_ids = [f"scale={x}" for x in _scale_values]
_graph_types = [cugraph.Graph, None, int]
_graph_test_ids = [f"create_using={getattr(x,'__name__',str(x))}" for x in _graph_types]
_clip_and_flip = [False, True]
_clip_and_flip_test_ids = [f"clip_and_flip={x}" for x in _clip_and_flip]
_scramble_vertex_ids = [False, True]
_scramble_vertex_ids_test_ids = [
    f"scramble_vertex_ids={x}" for x in _scramble_vertex_ids
]
_include_edge_weights = [False, True]
_include_edge_weights_test_ids = [
    f"include_edge_weights={x}" for x in _include_edge_weights
]
_dtype = [np.float32, cp.float32, None, "FLOAT64", "float32"]
_dtype_test_ids = [f"_dtype={x}" for x in _dtype]
_min_max_weight_values = [[None, None], [0, 1], [2, 5]]
_min_max_weight_values_test_ids = [
    f"min_max_weight_values={x}" for x in _min_max_weight_values
]
_include_edge_ids = [False, True]
_include_edge_ids_test_ids = [f"include_edge_ids={x}" for x in _include_edge_ids]
_include_edge_types = [False, True]
_include_edge_types_test_ids = [f"include_edge_types={x}" for x in _include_edge_types]
_min_max_edge_type_values = [[None, None], [0, 1], [2, 5]]
_min_max_edge_type_values_test_ids = [
    f"min_max_edge_type_values={x}" for x in _min_max_edge_type_values
]


def _call_rmat(
    scale,
    num_edges,
    create_using,
    clip_and_flip=False,
    scramble_vertex_ids=False,
    include_edge_weights=False,
    dtype=None,
    minimum_weight=None,
    maximum_weight=None,
    include_edge_ids=False,
    include_edge_types=False,
    min_edge_type_value=None,
    max_edge_type_value=None,
    mg=False,
):
    """
    Simplifies calling RMAT by requiring only specific args that are varied by
    these tests and hard-coding all others.
    """
    return rmat(
        scale=scale,
        num_edges=num_edges,
        a=0.57,  # from Graph500
        b=0.19,  # from Graph500
        c=0.19,  # from Graph500
        seed=24,
        clip_and_flip=clip_and_flip,
        scramble_vertex_ids=scramble_vertex_ids,
        create_using=create_using,
        include_edge_weights=include_edge_weights,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
        dtype=dtype,
        include_edge_ids=include_edge_ids,
        include_edge_types=include_edge_types,
        min_edge_type_value=min_edge_type_value,
        max_edge_type_value=max_edge_type_value,
        mg=mg,
    )


###############################################################################


@pytest.mark.sg
@pytest.mark.parametrize(
    "include_edge_weights", _include_edge_weights, ids=_include_edge_weights_test_ids
)
@pytest.mark.parametrize("dtype", _dtype, ids=_dtype_test_ids)
@pytest.mark.parametrize(
    "min_max_weight", _min_max_weight_values, ids=_min_max_weight_values_test_ids
)
@pytest.mark.parametrize(
    "scramble_vertex_ids", _scramble_vertex_ids, ids=_scramble_vertex_ids_test_ids
)
def test_rmat_edge_weights(
    include_edge_weights, dtype, min_max_weight, scramble_vertex_ids
):
    """
    Verifies that the edge weights returned by rmat() are valid. Also verifies that
    valid values are passed to 'dtype', 'minimum_weight' and 'maximum_weight'.

    """
    scale = 2
    num_edges = (2**scale) * 4
    create_using = None  # Returns the edgelist from RMAT
    minimum_weight, maximum_weight = min_max_weight

    if include_edge_weights:
        if (
            minimum_weight is None
            or maximum_weight is None
            or dtype
            not in [
                np.float32,
                np.float64,
                cp.float32,
                cp.float64,
                "float32",
                "float64",
            ]
        ):
            with pytest.raises(ValueError):
                _call_rmat(
                    scale,
                    num_edges,
                    create_using,
                    scramble_vertex_ids=scramble_vertex_ids,
                    include_edge_weights=include_edge_weights,
                    dtype=dtype,
                    minimum_weight=minimum_weight,
                    maximum_weight=maximum_weight,
                )
        else:
            df = _call_rmat(
                scale,
                num_edges,
                create_using,
                scramble_vertex_ids=scramble_vertex_ids,
                include_edge_weights=include_edge_weights,
                dtype=dtype,
                minimum_weight=minimum_weight,
                maximum_weight=maximum_weight,
            )

            # Check that there is a 'weights' column
            assert "weights" in df.columns

            edge_weights_err1 = df.query("{} - weights < 0.0001".format(maximum_weight))
            edge_weights_err2 = df.query(
                "{} - weights > -0.0001".format(minimum_weight)
            )

            # Check that edge weights values are between 'minimum_weight'
            # and 'maximum_weight.
            assert len(edge_weights_err1) == 0
            assert len(edge_weights_err2) == 0
    else:
        df = _call_rmat(
            scale,
            num_edges,
            create_using,
            scramble_vertex_ids=scramble_vertex_ids,
            include_edge_weights=include_edge_weights,
            dtype=dtype,
            minimum_weight=minimum_weight,
            maximum_weight=maximum_weight,
        )
        assert len(df.columns) == 2


@pytest.mark.sg
@pytest.mark.parametrize("scale", _scale_values, ids=_scale_test_ids)
@pytest.mark.parametrize(
    "include_edge_ids", _include_edge_ids, ids=_include_edge_ids_test_ids
)
@pytest.mark.parametrize(
    "scramble_vertex_ids", _scramble_vertex_ids, ids=_scramble_vertex_ids_test_ids
)
def test_rmat_edge_ids(scale, include_edge_ids, scramble_vertex_ids):
    """
    Verifies that the edge ids returned by rmat() are valid.

    """
    num_edges = (2**scale) * 4
    create_using = None  # Returns the edgelist from RMAT
    df = _call_rmat(
        scale,
        num_edges,
        create_using,
        scramble_vertex_ids=scramble_vertex_ids,
        include_edge_ids=include_edge_ids,
    )

    if include_edge_ids:
        assert "edge_id" in df.columns
        df["index"] = df.index
        edge_id_err = df.query("index != edge_id")
        assert len(edge_id_err) == 0

    else:
        assert len(df.columns) == 2


@pytest.mark.sg
@pytest.mark.parametrize(
    "include_edge_types",
    _include_edge_types,
    ids=_include_edge_types_test_ids,
)
@pytest.mark.parametrize(
    "min_max_edge_type_value",
    _min_max_edge_type_values,
    ids=_min_max_edge_type_values_test_ids,
)
@pytest.mark.parametrize(
    "scramble_vertex_ids", _scramble_vertex_ids, ids=_scramble_vertex_ids_test_ids
)
def test_rmat_edge_types(
    include_edge_types, min_max_edge_type_value, scramble_vertex_ids
):
    """
    Verifies that the edge types returned by rmat() are valid and that valid values
    are passed for 'min_edge_type_value' and 'max_edge_type_value'.

    """
    scale = 2
    num_edges = (2**scale) * 4
    create_using = None  # Returns the edgelist from RMAT
    min_edge_type_value, max_edge_type_value = min_max_edge_type_value

    if include_edge_types:
        if min_edge_type_value is None or max_edge_type_value is None:
            with pytest.raises(ValueError):
                _call_rmat(
                    scale,
                    num_edges,
                    create_using,
                    scramble_vertex_ids=scramble_vertex_ids,
                    include_edge_types=include_edge_types,
                    min_edge_type_value=min_edge_type_value,
                    max_edge_type_value=max_edge_type_value,
                )
        else:
            df = _call_rmat(
                scale,
                num_edges,
                create_using,
                scramble_vertex_ids=scramble_vertex_ids,
                include_edge_types=include_edge_types,
                min_edge_type_value=min_edge_type_value,
                max_edge_type_value=max_edge_type_value,
            )

            # Check that there is an 'edge_type' column
            assert "edge_type" in df.columns
            edge_types_err1 = df.query("{} < edge_type".format(max_edge_type_value))
            edge_types_err2 = df.query("{} > edge_type".format(min_edge_type_value))

            # Check that edge weights values are between 'min_edge_type_value'
            # and 'max_edge_type_value'.
            assert len(edge_types_err1) == 0
            assert len(edge_types_err2) == 0
    else:
        df = _call_rmat(
            scale,
            num_edges,
            create_using,
            scramble_vertex_ids=scramble_vertex_ids,
            include_edge_types=include_edge_types,
            min_edge_type_value=min_edge_type_value,
            max_edge_type_value=max_edge_type_value,
        )
        assert len(df.columns) == 2


@pytest.mark.sg
@pytest.mark.parametrize("scale", [2, 4, 8], ids=_scale_test_ids)
@pytest.mark.parametrize(
    "include_edge_weights", _include_edge_weights, ids=_include_edge_weights_test_ids
)
@pytest.mark.parametrize("clip_and_flip", _clip_and_flip, ids=_clip_and_flip_test_ids)
def test_rmat_clip_and_flip(scale, include_edge_weights, clip_and_flip):
    """
    Verifies that there are edges only in the lower triangular part of
    the adjacency matrix when 'clip_and_flip' is set to 'true'.

    Note: 'scramble_vertex_ids' nullifies the effect of 'clip_and_flip' therefore
    both flags should not be set to 'True' in order to test the former

    """
    num_edges = (2**scale) * 4
    create_using = None  # Returns the edgelist from RMAT
    minimum_weight = 0
    maximum_weight = 1
    dtype = np.float32
    df = _call_rmat(
        scale,
        num_edges,
        create_using,
        clip_and_flip=clip_and_flip,
        scramble_vertex_ids=False,
        include_edge_weights=include_edge_weights,
        dtype=dtype,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
    )

    if not include_edge_weights:
        df["weights"] = 1
        # cupy coo_matrix only support 'float32', 'float64', 'complex64'
        # and 'complex128'.
        df["weights"] = df["weights"].astype("float32")

    dim = df[["src", "dst"]].max().max() + 1
    src = df["src"].to_cupy()
    dst = df["dst"].to_cupy()
    weights = df["weights"].to_cupy()
    adj_matrix = coo_matrix((weights, (src, dst)), shape=(dim, dim)).toarray()

    upper_coo = triu(adj_matrix)
    diag = tril(upper_coo)

    if clip_and_flip:
        # Except the diagonal, There should be no edge in the upper triangular part of
        # the graph adjacency matrix.
        assert diag.nnz == upper_coo.nnz


@pytest.mark.sg
@pytest.mark.parametrize("graph_type", _graph_types, ids=_graph_test_ids)
def test_rmat_return_type(graph_type):
    """
    Verifies that the return type returned by rmat() is valid (or the proper
    exception is raised) based on inputs.

    """
    scale = 2
    num_edges = (2**scale) * 4

    if graph_type not in [cugraph.Graph, None]:
        with pytest.raises(TypeError):
            _call_rmat(scale, num_edges, graph_type)

    else:
        G_or_df = _call_rmat(scale, num_edges, graph_type)

        if graph_type is None:
            assert type(G_or_df) is cudf.DataFrame
        else:
            assert type(G_or_df) is graph_type

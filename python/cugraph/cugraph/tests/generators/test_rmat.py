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
import dask_cudf

from cugraph.dask.common.mg_utils import (
    is_single_gpu,
    setup_local_dask_cluster,
    teardown_local_dask_cluster,
)
from cugraph.generators import rmat
import cugraph


##############################################################################
_cluster = None
_client = None
_is_single_gpu = is_single_gpu()
_visible_devices = None
_scale_values = [2, 4, 16]
_scale_test_ids = [f"scale={x}" for x in _scale_values]
_mg_values = [False, True]
_mg_test_ids = [f"mg={x}" for x in _mg_values]
_graph_types = [cugraph.Graph, None, int]
_graph_test_ids = [f"create_using={getattr(x,'__name__',str(x))}" for x in _graph_types]


def _call_rmat(scale, num_edges, create_using, mg):
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
        clip_and_flip=False,
        scramble_vertex_ids=True,
        create_using=create_using,
        mg=mg,
    )


###############################################################################
def setup_module():
    global _cluster
    global _client
    global _visible_devices
    if not _is_single_gpu:
        (_cluster, _client) = setup_local_dask_cluster(p2p=True)
        _visible_devices = _client.scheduler_info()["workers"]


def teardown_module():
    if not _is_single_gpu:
        teardown_local_dask_cluster(_cluster, _client)


###############################################################################
@pytest.mark.filterwarnings("ignore:make_current is deprecated:DeprecationWarning")
@pytest.mark.parametrize("scale", _scale_values, ids=_scale_test_ids)
@pytest.mark.parametrize("mg", _mg_values, ids=_mg_test_ids)
def test_rmat_edgelist(scale, mg):
    """
    Verifies that the edgelist returned by rmat() is valid based on inputs.
    """
    if mg and _is_single_gpu:
        pytest.skip("skipping MG testing on Single GPU system")

    num_edges = (2**scale) * 4
    create_using = None  # Returns the edgelist from RMAT

    df = _call_rmat(scale, num_edges, create_using, mg)

    if mg:
        assert df.npartitions == len(_visible_devices)
        df_to_check = df.compute()
    else:
        df_to_check = df

    assert len(df_to_check) == num_edges


@pytest.mark.filterwarnings("ignore:make_current is deprecated:DeprecationWarning")
@pytest.mark.parametrize("graph_type", _graph_types, ids=_graph_test_ids)
@pytest.mark.parametrize("mg", _mg_values, ids=_mg_test_ids)
def test_rmat_return_type(graph_type, mg):
    """
    Verifies that the return type returned by rmat() is valid (or the proper
    exception is raised) based on inputs.
    """
    if mg and _is_single_gpu:
        pytest.skip("skipping MG testing on Single GPU system")

    scale = 2
    num_edges = (2**scale) * 4

    if (mg and (graph_type is not None)) or (graph_type not in [cugraph.Graph, None]):
        with pytest.raises(TypeError):
            _call_rmat(scale, num_edges, graph_type, mg)

    else:
        G_or_df = _call_rmat(scale, num_edges, graph_type, mg)

        if graph_type is None:
            assert type(G_or_df) is dask_cudf.DataFrame if mg else cudf.DataFrame
        else:
            assert type(G_or_df) is graph_type

# Copyright (c) 2020, NVIDIA CORPORATION.
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
import gc

import cudf

import cugraph
from cugraph.tests.dask.mg_context import MGContext, skip_if_not_enough_devices
import cugraph.dask.structure.replication as replication
from cugraph.dask.common.mg_utils import is_single_gpu
import cugraph.tests.utils as utils

DATASETS_OPTIONS = utils.DATASETS_SMALL
DIRECTED_GRAPH_OPTIONS = [False, True]
# MG_DEVICE_COUNT_OPTIONS = [1, 2, 3, 4]
MG_DEVICE_COUNT_OPTIONS = [1]


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("input_data_path", DATASETS_OPTIONS)
@pytest.mark.parametrize("mg_device_count", MG_DEVICE_COUNT_OPTIONS)
def test_replicate_cudf_dataframe_with_weights(
    input_data_path, mg_device_count
):
    gc.collect()
    skip_if_not_enough_devices(mg_device_count)
    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )
    with MGContext(number_of_devices=mg_device_count,
                   p2p=True):
        worker_to_futures = replication.replicate_cudf_dataframe(df)
        for worker in worker_to_futures:
            replicated_df = worker_to_futures[worker].result()
            assert df.equals(replicated_df), (
                "There is a mismatch in one " "of the replications"
            )


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("input_data_path", DATASETS_OPTIONS)
@pytest.mark.parametrize("mg_device_count", MG_DEVICE_COUNT_OPTIONS)
def test_replicate_cudf_dataframe_no_weights(input_data_path, mg_device_count):
    gc.collect()
    skip_if_not_enough_devices(mg_device_count)
    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst"],
        dtype=["int32", "int32"],
    )
    with MGContext(number_of_devices=mg_device_count,
                   p2p=True):
        worker_to_futures = replication.replicate_cudf_dataframe(df)
        for worker in worker_to_futures:
            replicated_df = worker_to_futures[worker].result()
            assert df.equals(replicated_df), (
                "There is a mismatch in one " "of the replications"
            )


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("input_data_path", DATASETS_OPTIONS)
@pytest.mark.parametrize("mg_device_count", MG_DEVICE_COUNT_OPTIONS)
def test_replicate_cudf_series(input_data_path, mg_device_count):
    gc.collect()
    skip_if_not_enough_devices(mg_device_count)
    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )
    with MGContext(number_of_devices=mg_device_count,
                   p2p=True):
        for column in df.columns.values:
            series = df[column]
            worker_to_futures = replication.replicate_cudf_series(series)
            for worker in worker_to_futures:
                replicated_series = worker_to_futures[worker].result()
                assert series.equals(replicated_series), (
                    "There is a " "mismatch in one of the replications"
                )
            # FIXME: If we do not clear this dictionary, when comparing
            # results for the 2nd column, one of the workers still
            # has a value from the 1st column
            worker_to_futures = {}


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("graph_file", DATASETS_OPTIONS)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("mg_device_count", MG_DEVICE_COUNT_OPTIONS)
def test_enable_batch_no_context(graph_file, directed, mg_device_count):
    gc.collect()
    skip_if_not_enough_devices(mg_device_count)
    G = utils.generate_cugraph_graph_from_file(graph_file, directed)
    assert G.batch_enabled is False, "Internal property should be False"
    with pytest.raises(Exception):
        G.enable_batch()


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("graph_file", DATASETS_OPTIONS)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("mg_device_count", MG_DEVICE_COUNT_OPTIONS)
def test_enable_batch_no_context_view_adj(
    graph_file, directed, mg_device_count
):
    gc.collect()
    skip_if_not_enough_devices(mg_device_count)
    G = utils.generate_cugraph_graph_from_file(graph_file, directed)
    assert G.batch_enabled is False, "Internal property should be False"
    G.view_adj_list()


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("graph_file", DATASETS_OPTIONS)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("mg_device_count", MG_DEVICE_COUNT_OPTIONS)
def test_enable_batch_context_then_views(
    graph_file, directed, mg_device_count
):
    gc.collect()
    skip_if_not_enough_devices(mg_device_count)
    G = utils.generate_cugraph_graph_from_file(graph_file, directed)
    with MGContext(number_of_devices=mg_device_count,
                   p2p=True):
        assert G.batch_enabled is False, "Internal property should be False"
        G.enable_batch()
        assert G.batch_enabled is True, "Internal property should be True"
        assert G.batch_edgelists is not None, (
            "The graph should have " "been created with an " "edgelist"
        )
        assert G.batch_adjlists is None
        G.view_adj_list()
        assert G.batch_adjlists is not None

        assert G.batch_transposed_adjlists is None
        G.view_transposed_adj_list()
        assert G.batch_transposed_adjlists is not None


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("graph_file", DATASETS_OPTIONS)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("mg_device_count", MG_DEVICE_COUNT_OPTIONS)
def test_enable_batch_view_then_context(graph_file, directed, mg_device_count):
    gc.collect()
    skip_if_not_enough_devices(mg_device_count)
    G = utils.generate_cugraph_graph_from_file(graph_file, directed)

    assert G.batch_adjlists is None
    G.view_adj_list()
    assert G.batch_adjlists is None

    assert G.batch_transposed_adjlists is None
    G.view_transposed_adj_list()
    assert G.batch_transposed_adjlists is None

    with MGContext(number_of_devices=mg_device_count,
                   p2p=True):
        assert G.batch_enabled is False, "Internal property should be False"
        G.enable_batch()
        assert G.batch_enabled is True, "Internal property should be True"
        assert G.batch_edgelists is not None, (
            "The graph should have " "been created with an " "edgelist"
        )
        assert G.batch_adjlists is not None
        assert G.batch_transposed_adjlists is not None


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("graph_file", DATASETS_OPTIONS)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("mg_device_count", MG_DEVICE_COUNT_OPTIONS)
def test_enable_batch_context_no_context_views(
    graph_file, directed, mg_device_count
):
    gc.collect()
    skip_if_not_enough_devices(mg_device_count)
    G = utils.generate_cugraph_graph_from_file(graph_file, directed)
    with MGContext(number_of_devices=mg_device_count,
                   p2p=True):
        assert G.batch_enabled is False, "Internal property should be False"
        G.enable_batch()
        assert G.batch_enabled is True, "Internal property should be True"
        assert G.batch_edgelists is not None, (
            "The graph should have " "been created with an " "edgelist"
        )
    G.view_edge_list()
    G.view_adj_list()
    G.view_transposed_adj_list()


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("graph_file", DATASETS_OPTIONS)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("mg_device_count", MG_DEVICE_COUNT_OPTIONS)
def test_enable_batch_edgelist_replication(
    graph_file, directed, mg_device_count
):
    gc.collect()
    skip_if_not_enough_devices(mg_device_count)
    G = utils.generate_cugraph_graph_from_file(graph_file, directed)
    with MGContext(number_of_devices=mg_device_count,
                   p2p=True):
        G.enable_batch()
        df = G.edgelist.edgelist_df
        for worker in G.batch_edgelists:
            replicated_df = G.batch_edgelists[worker].result()
            assert df.equals(replicated_df), "Replication of edgelist failed"


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("graph_file", DATASETS_OPTIONS)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("mg_device_count", MG_DEVICE_COUNT_OPTIONS)
def test_enable_batch_adjlist_replication_weights(
    graph_file, directed, mg_device_count
):
    gc.collect()
    skip_if_not_enough_devices(mg_device_count)
    df = cudf.read_csv(
        graph_file,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )
    G = cugraph.DiGraph() if directed else cugraph.Graph()
    G.from_cudf_edgelist(
        df, source="src", destination="dst", edge_attr="value"
    )
    with MGContext(number_of_devices=mg_device_count,
                   p2p=True):
        G.enable_batch()
        G.view_adj_list()
        adjlist = G.adjlist
        offsets = adjlist.offsets
        indices = adjlist.indices
        weights = adjlist.weights
        for worker in G.batch_adjlists:
            (rep_offsets, rep_indices, rep_weights) = G.batch_adjlists[worker]
            assert offsets.equals(rep_offsets.result()), (
                "Replication of " "adjlist offsets failed"
            )
            assert indices.equals(rep_indices.result()), (
                "Replication of " "adjlist indices failed"
            )
            assert weights.equals(rep_weights.result()), (
                "Replication of " "adjlist weights failed"
            )


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("graph_file", DATASETS_OPTIONS)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("mg_device_count", MG_DEVICE_COUNT_OPTIONS)
def test_enable_batch_adjlist_replication_no_weights(
    graph_file, directed, mg_device_count
):
    gc.collect()
    skip_if_not_enough_devices(mg_device_count)
    df = cudf.read_csv(
        graph_file,
        delimiter=" ",
        names=["src", "dst"],
        dtype=["int32", "int32"],
    )
    G = cugraph.DiGraph() if directed else cugraph.Graph()
    G.from_cudf_edgelist(df, source="src", destination="dst")
    with MGContext(number_of_devices=mg_device_count,
                   p2p=True):
        G.enable_batch()
        G.view_adj_list()
        adjlist = G.adjlist
        offsets = adjlist.offsets
        indices = adjlist.indices
        weights = adjlist.weights
        for worker in G.batch_adjlists:
            (rep_offsets, rep_indices, rep_weights) = G.batch_adjlists[worker]
            assert offsets.equals(rep_offsets.result()), (
                "Replication of " "adjlist offsets failed"
            )
            assert indices.equals(rep_indices.result()), (
                "Replication of " "adjlist indices failed"
            )
            assert weights is None and rep_weights is None

# Copyright (c) 2021-2023, NVIDIA CORPORATION.
#
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
#

from dask.distributed import wait, default_client
import cugraph.dask.comms.comms as Comms
import dask_cudf
import cudf

from pylibcugraph import ResourceHandle
from pylibcugraph import weakly_connected_components as pylibcugraph_wcc


def convert_to_cudf(cp_arrays):
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """
    cupy_vertex, cupy_labels = cp_arrays
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertex
    df["labels"] = cupy_labels

    return df


def _call_plc_wcc(sID, mg_graph_x, do_expensive_check):
    return pylibcugraph_wcc(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        offsets=None,
        indices=None,
        weights=None,
        labels=None,
        do_expensive_check=do_expensive_check,
    )


def weakly_connected_components(input_graph):
    """
    Generate the Weakly Connected Components and attach a component label to
    each vertex.

    Parameters
    ----------
    input_graph : cugraph.Graph
        The graph descriptor should contain the connectivity information
        and weights. The adjacency list will be computed if not already
        present.
        The current implementation only supports undirected graphs.

    Returns
    -------
    result : dask_cudf.DataFrame
        GPU distributed data frame containing 2 dask_cudf.Series

    ddf['vertex']: dask_cudf.Series
        Contains the vertex identifiers
    ddf['labels']: dask_cudf.Series
        Contains the wcc labels

    Examples
    --------
    >>> import cugraph.dask as dcg
    >>> import dask_cudf
    >>> # ... Init a DASK Cluster
    >>> #    see https://docs.rapids.ai/api/cugraph/stable/dask-cugraph.html
    >>> # Download dataset from https://github.com/rapidsai/cugraph/datasets/..
    >>> chunksize = dcg.get_chunksize(datasets_path / "karate.csv")
    >>> ddf = dask_cudf.read_csv(datasets_path / "karate.csv",
    ...                          chunksize=chunksize, delimiter=" ",
    ...                          names=["src", "dst", "value"],
    ...                          dtype=["int32", "int32", "float32"])
    >>> dg = cugraph.Graph(directed=False)
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst',
    ...                            edge_attr='value')
    >>> result = dcg.weakly_connected_components(dg)

    """

    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")

    # Initialize dask client
    client = default_client()

    do_expensive_check = False

    result = [
        client.submit(
            _call_plc_wcc,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            do_expensive_check,
            workers=[w],
            allow_other_workers=False,
        )
        for w in Comms.get_workers()
    ]

    wait(result)

    cudf_result = [client.submit(convert_to_cudf, cp_arrays) for cp_arrays in result]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result).persist()
    wait(ddf)
    # Wait until the inactive futures are released
    wait([(r.release(), c_r.release()) for r, c_r in zip(result, cudf_result)])

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "vertex")

    return ddf

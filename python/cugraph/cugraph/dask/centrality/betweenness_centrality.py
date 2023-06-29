# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

from dask.distributed import wait, get_client
from pylibcugraph import (
    ResourceHandle,
    betweenness_centrality as pylibcugraph_betweenness_centrality,
    edge_betweenness_centrality as pylibcugraph_edge_betweenness_centrality,
)
import cugraph.dask.comms.comms as Comms
from cugraph.dask.common.input_utils import get_distributed_data
import dask_cudf
import cudf
import cupy as cp
import warnings
import dask
from typing import Union


def convert_to_cudf(cp_arrays: cp.ndarray, edge_bc: bool) -> cudf.DataFrame:
    """
    create a cudf DataFrame from cupy arrays
    """
    df = cudf.DataFrame()
    if edge_bc:
        cupy_src_vertices, cupy_dst_vertices, cupy_values, cupy_edge_ids = \
            cp_arrays
        df["src"] = cupy_src_vertices
        df["dst"] = cupy_dst_vertices
        df["betweenness_centrality"] = cupy_values
        if cupy_edge_ids is not None:
            df["edge_ids"] = cupy_edge_ids
    
    else:
        cupy_vertices, cupy_values = cp_arrays
        df["vertex"] = cupy_vertices
        df["betweenness_centrality"] = cupy_values
    return df


def _call_plc_betweenness_centrality(
    mg_graph_x,
    sID: bytes,
    k: Union[int, cudf.Series],
    random_state: int,
    normalized: bool,
    endpoints: bool,
    do_expensive_check: bool,
    edge_bc: bool,
) -> cudf.DataFrame:

    if edge_bc:
        cp_arrays = pylibcugraph_edge_betweenness_centrality(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        k=k,
        random_state=random_state,
        normalized=normalized,
        do_expensive_check=do_expensive_check,
    )
    else:
        cp_arrays = pylibcugraph_betweenness_centrality(
            resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
            graph=mg_graph_x,
            k=k,
            random_state=random_state,
            normalized=normalized,
            include_endpoints=endpoints,
            do_expensive_check=do_expensive_check,
        )
    return convert_to_cudf(cp_arrays, edge_bc)


def _mg_call_plc_betweenness_centrality(
    input_graph,
    client: dask.distributed.client.Client,
    sID: bytes,
    k: dict,
    random_state: int,
    normalized: bool,
    do_expensive_check: bool,
    endpoints: bool = False,
    edge_bc: bool = False,
) -> dask_cudf.DataFrame:

    result = [
        client.submit(
            _call_plc_betweenness_centrality,
            input_graph._plc_graph[w],
            sID,
            k if isinstance(k, (int, type(None))) else k[w][0],
            hash((random_state, i)),
            normalized,
            endpoints,
            do_expensive_check,
            edge_bc,
            workers=[w],
            allow_other_workers=False,
            pure=False,
        )
        for i, w in enumerate(Comms.get_workers())
    ]

    ddf = dask_cudf.from_delayed(result, verify_meta=False).persist()
    wait(ddf)
    wait([r.release() for r in result])
    return ddf


def betweenness_centrality(
    input_graph,
    k: Union[
        int, list, cudf.Series, cudf.DataFrame, dask_cudf.Series, dask_cudf.DataFrame
    ] = None,
    normalized: bool = True,
    endpoints: bool = False,
    random_state: int = None,
) -> dask_cudf.DataFrame:
    """
    Compute the betweenness centrality for all vertices of the graph G.
    Betweenness centrality is a measure of the number of shortest paths that
    pass through a vertex.  A vertex with a high betweenness centrality score
    has more paths passing through it and is therefore believed to be more
    important.

    To improve performance. rather than doing an all-pair shortest path,
    a sample of k starting vertices can be used.

    CuGraph does not currently support 'weight' parameters as seen in the
    corresponding networkX call.

    Parameters
    ----------
    input_graph: cuGraph.Graph
        The graph can be either directed (Graph(directed=True)) or undirected.
        Weights in the graph are ignored, the current implementation uses a parallel
        variation of the Brandes Algorithm (2001) to compute exact or approximate
        betweenness. If weights are provided in the edgelist, they will not be
        used.

    k : int, list or (dask)cudf object or None, optional (default=None)
        If k is not None, use k node samples to estimate betweenness.  Higher
        values give better approximation.  If k is either a list or a (dask)cudf,
        use its content for estimation: it contain vertex identifiers. If k is None
        (the default), all the vertices are used to estimate betweenness.  Vertices
        obtained through sampling or defined as a list will be used as sources for
        traversals inside the algorithm.

    normalized : bool, optional (default=True)
        If True normalize the resulting betweenness centrality values

    endpoints : bool, optional (default=False)
        If true, include the endpoints in the shortest path counts.

    random_state : int, optional (default=None)
        if k is specified and k is an integer, use random_state to initialize the
        random number generator.
        Using None defaults to a hash of process id, time, and hostname
        If k is either None or list or cudf objects: random_state parameter is
        ignored.

    Returns
    -------
    betweenness_centrality : dask_cudf.DataFrame
        GPU distributed data frame containing two dask_cudf.Series of size V:
        the vertex identifiers and the corresponding betweenness centrality values.

        ddf['vertex'] : dask_cudf.Series
            Contains the vertex identifiers
        ddf['betweenness_centrality'] : dask_cudf.Series
            Contains the betweenness centrality of vertices

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
    >>> dg = cugraph.Graph(directed=True)
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst')
    >>> pr = dcg.betweenness_centrality(dg)

    """

    if input_graph.store_transposed is True:
        warning_msg = (
            "Betweenness centrality expects the 'store_transposed' flag "
            "to be set to 'False' for optimal performance during "
            "the graph creation"
        )
        warnings.warn(warning_msg, UserWarning)

    if not isinstance(k, (dask_cudf.DataFrame, dask_cudf.Series)):
        if isinstance(k, (cudf.DataFrame, cudf.Series, list)):
            if isinstance(k, list):
                k_dtype = input_graph.nodes().dtype
                k = cudf.Series(k, dtype=k_dtype)

        if isinstance(k, (cudf.Series, cudf.DataFrame)):
            splits = cp.array_split(cp.arange(len(k)), len(Comms.get_workers()))
            k = {w: [k.iloc[splits[i]]] for i, w in enumerate(Comms.get_workers())}

    else:
        if k is not None:
            k = get_distributed_data(k)
            wait(k)
            k = k.worker_to_parts

    if input_graph.renumbered:
        if isinstance(k, dask_cudf.DataFrame):
            tmp_col_names = k.columns

        elif isinstance(k, dask_cudf.Series):
            tmp_col_names = None

        if isinstance(k, (dask_cudf.DataFrame, dask_cudf.Series)):
            k = input_graph.lookup_internal_vertex_id(k, tmp_col_names)

    # FIXME: should we add this parameter as an option?
    do_expensive_check = False

    client = get_client()

    ddf = _mg_call_plc_betweenness_centrality(
        input_graph=input_graph,
        client=client,
        sID=Comms.get_session_id(),
        k=k,
        random_state=random_state,
        normalized=normalized,
        endpoints=endpoints,
        do_expensive_check=do_expensive_check,
    )

    if input_graph.renumbered:
        return input_graph.unrenumber(ddf, "vertex")

    return ddf


def edge_betweenness_centrality(
    input_graph,
    k: Union[
        int, list, cudf.Series, cudf.DataFrame, dask_cudf.Series, dask_cudf.DataFrame
    ] = None,
    normalized: bool = True,
    random_state: int = None,
) -> dask_cudf.DataFrame:
    """
    Compute the edge betweenness centrality for all edges of the graph G.
    Betweenness centrality is a measure of the number of shortest paths
    that pass over an edge.  An edge with a high betweenness centrality
    score has more paths passing over it and is therefore believed to be
    more important.

    To improve performance. rather than doing an all-pair shortest path,
    a sample of k starting vertices can be used.

    CuGraph does not currently support 'weight' parameters as seen in the
    corresponding networkX call.

    Parameters
    ----------
    input_graph: cuGraph.Graph
        The graph can be either directed (Graph(directed=True)) or undirected.
        Weights in the graph are ignored, the current implementation uses a parallel
        variation of the Brandes Algorithm (2001) to compute exact or approximate
        betweenness. If weights are provided in the edgelist, they will not be
        used.

    k : int, list or (dask)cudf object or None, optional (default=None)
        If k is not None, use k node samples to estimate betweenness.  Higher
        values give better approximation.  If k is either a list or a (dask)cudf,
        use its content for estimation: it contain vertex identifiers. If k is None
        (the default), all the vertices are used to estimate betweenness.  Vertices
        obtained through sampling or defined as a list will be used as sources for
        traversals inside the algorithm.

    normalized : bool, optional (default=True)
        If True normalize the resulting betweenness centrality values

    random_state : int, optional (default=None)
        if k is specified and k is an integer, use random_state to initialize the
        random number generator.
        Using None defaults to a hash of process id, time, and hostname
        If k is either None or list or cudf objects: random_state parameter is
        ignored.

    Returns
    -------
    betweenness_centrality : dask_cudf.DataFrame
        GPU distributed data frame containing two dask_cudf.Series of size V:
        the vertex identifiers and the corresponding betweenness centrality values.

        ddf['src'] : dask_cudf.Series
            Contains the vertex identifiers of the source of each edge

        ddf['dst'] : dask_cudf.Series
            Contains the vertex identifiers of the destination of each edge

        ddf['betweenness_centrality'] : dask_cudf.Series
            Contains the betweenness centrality of edges
        
        ddf["edge_id"] : dask_cudf.Series
            Contains the edge ids of edges if there are.

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
    >>> dg = cugraph.Graph(directed=True)
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst')
    >>> pr = dcg.edge_betweenness_centrality(dg)

    """

    if input_graph.store_transposed is True:
        warning_msg = (
            "Betweenness centrality expects the 'store_transposed' flag "
            "to be set to 'False' for optimal performance during "
            "the graph creation"
        )
        warnings.warn(warning_msg, UserWarning)

    if not isinstance(k, (dask_cudf.DataFrame, dask_cudf.Series)):
        if isinstance(k, (cudf.DataFrame, cudf.Series, list)):
            if isinstance(k, list):
                k_dtype = input_graph.nodes().dtype
                k = cudf.Series(k, dtype=k_dtype)

        if isinstance(k, (cudf.Series, cudf.DataFrame)):
            splits = cp.array_split(cp.arange(len(k)), len(Comms.get_workers()))
            k = {w: [k.iloc[splits[i]]] for i, w in enumerate(Comms.get_workers())}

    else:
        if k is not None:
            k = get_distributed_data(k)
            wait(k)
            k = k.worker_to_parts

    if input_graph.renumbered:
        if isinstance(k, dask_cudf.DataFrame):
            tmp_col_names = k.columns

        elif isinstance(k, dask_cudf.Series):
            tmp_col_names = None

        if isinstance(k, (dask_cudf.DataFrame, dask_cudf.Series)):
            k = input_graph.lookup_internal_vertex_id(k, tmp_col_names)

    # FIXME: should we add this parameter as an option?
    do_expensive_check = False

    client = get_client()

    ddf = _mg_call_plc_betweenness_centrality(
        input_graph=input_graph,
        client=client,
        sID=Comms.get_session_id(),
        k=k,
        random_state=random_state,
        normalized=normalized,
        do_expensive_check=do_expensive_check,
        edge_bc=True,
    )

    if input_graph.renumbered:
        return input_graph.unrenumber(ddf, "vertex")

    return ddf

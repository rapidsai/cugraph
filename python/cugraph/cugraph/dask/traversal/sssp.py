# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
from cugraph.dask.common.input_utils import get_distributed_data
import cugraph.dask.comms.comms as Comms
import cupy
import cudf
import dask_cudf
from pylibcugraph import sssp as pylibcugraph_sssp
from pylibcugraph import (ResourceHandle,
                          GraphProperties,
                          MGGraph)


def _call_plc_sssp(
                  sID,
                  data,
                  src_col_name,
                  dst_col_name,
                  num_edges,
                  source,
                  cutoff,
                  compute_predecessors=True,
                  do_expensive_check=False):

    comms_handle = Comms.get_handle(sID)
    resource_handle = ResourceHandle(comms_handle.getHandle())

    srcs = data[0][src_col_name]
    dsts = data[0][dst_col_name]
    weights = data[0]['value'] \
        if 'value' in data[0].columns \
        else cudf.Series((srcs + 1) / (srcs + 1), dtype='float32')
    if weights.dtype not in ('float32', 'double'):
        weights = weights.astype('double')

    mg = MGGraph(
        resource_handle=resource_handle,
        graph_properties=GraphProperties(is_multigraph=False),
        src_array=srcs,
        dst_array=dsts,
        weight_array=weights,
        store_transposed=False,
        num_edges=num_edges,
        do_expensive_check=do_expensive_check
    )

    vertices, distances, predecessors = pylibcugraph_sssp(
        resource_handle=resource_handle,
        graph=mg,
        source=source,
        cutoff=cutoff,
        compute_predecessors=compute_predecessors,
        do_expensive_check=do_expensive_check
    )

    return cudf.DataFrame({
        'distance': cudf.Series(distances),
        'vertex': cudf.Series(vertices),
        'predecessor': cudf.Series(predecessors),
    })


def sssp(input_graph, source, cutoff=None, check_source=True):
    """
    Compute the distance and predecessors for shortest paths from the specified
    source to all the vertices in the input_graph. The distances column will
    store the distance from the source to each vertex. The predecessors column
    will store each vertex's predecessor in the shortest path. Vertices that
    are unreachable will have a distance of infinity denoted by the maximum
    value of the data type and the predecessor set as -1. The source vertex's
    predecessor is also set to -1.  The input graph must contain edge list as
    dask-cudf dataframe with one partition per GPU.

    Parameters
    ----------
    input_graph : cugraph.Graph
        cuGraph graph descriptor, should contain the connectivity information
        as dask cudf edge list dataframe.

    source : Integer
        Specify source vertex

    cutoff : double, optional (default = None)
        Maximum edge weight sum considered by the algorithm

    check_source : bool, optional (default=True)
        If True, performs more extensive tests on the start vertices
        to ensure validitity, at the expense of increased run time.

    Returns
    -------
    df : dask_cudf.DataFrame
        df['vertex'] gives the vertex id

        df['distance'] gives the path distance from the
        starting vertex

        df['predecessor'] gives the vertex id it was
        reached from in the traversal

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
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst',
    ...                            edge_attr='value')
    >>> df = dcg.sssp(dg, 0)

    """

    client = default_client()
    # FIXME: 'legacy_renum_only' will not trigger the C++ renumbering
    # In the future, once all the algos follow the C/Pylibcugraph path,
    # compute_renumber_edge_list will only be used for multicolumn and
    # string vertices since the renumbering will be done in pylibcugraph
    input_graph.compute_renumber_edge_list(
        transposed=False, legacy_renum_only=True)
    ddf = input_graph.edgelist.edgelist_df
    num_edges = len(ddf)
    data = get_distributed_data(ddf)

    src_col_name = input_graph.renumber_map.renumbered_src_col_name
    dst_col_name = input_graph.renumber_map.renumbered_dst_col_name

    def check_valid_vertex(G, source):
        is_valid_vertex = G.has_node(source)
        if not is_valid_vertex:
            raise ValueError('Invalid source vertex')

    if check_source:
        check_valid_vertex(input_graph, source)

    if cutoff is None:
        cutoff = cupy.inf

    if input_graph.renumbered:
        source = input_graph.lookup_internal_vertex_id(
            cudf.Series([source])).fillna(-1).compute()
        source = source.iloc[0]

    result = [client.submit(
            _call_plc_sssp,
            Comms.get_session_id(),
            wf[1],
            src_col_name,
            dst_col_name,
            num_edges,
            source,
            cutoff,
            True,
            False,
            workers=[wf[0]])
            for idx, wf in enumerate(data.worker_to_parts.items())]
    wait(result)
    ddf = dask_cudf.from_delayed(result)

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, 'vertex')
        ddf = input_graph.unrenumber(ddf, 'predecessor')
        ddf["predecessor"] = ddf["predecessor"].fillna(-1)

    return ddf

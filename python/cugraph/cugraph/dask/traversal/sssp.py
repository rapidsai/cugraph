# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
import cupy
import cudf
import dask_cudf
from pylibcugraph import sssp as pylibcugraph_sssp, ResourceHandle
import warnings


def _call_plc_sssp(
    sID, mg_graph_x, source, cutoff, compute_predecessors, do_expensive_check
):
    vertices, distances, predecessors = pylibcugraph_sssp(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        source=source,
        cutoff=cutoff,
        compute_predecessors=compute_predecessors,
        do_expensive_check=do_expensive_check,
    )
    return cudf.DataFrame(
        {
            "distance": cudf.Series(distances),
            "vertex": cudf.Series(vertices),
            "predecessor": cudf.Series(predecessors),
        }
    )


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

    # FIXME: Implement a better way to check if the graph is weighted similar
    # to 'simpleGraph'
    if len(input_graph.edgelist.edgelist_df.columns) != 3:
        warning_msg = (
            "'SSSP' requires the input graph to be weighted: Unweighted "
            "graphs will not be supported in the next release."
        )
        warnings.warn(warning_msg, PendingDeprecationWarning)

    client = default_client()

    def check_valid_vertex(G, source):
        is_valid_vertex = G.has_node(source)
        if not is_valid_vertex:
            raise ValueError("Invalid source vertex")

    if check_source:
        check_valid_vertex(input_graph, source)

    if cutoff is None:
        cutoff = cupy.inf

    if input_graph.renumbered:
        source = (
            input_graph.lookup_internal_vertex_id(cudf.Series([source]))
            .fillna(-1)
            .compute()
        )
        source = source.iloc[0]

    do_expensive_check = False
    compute_predecessors = True
    result = [
        client.submit(
            _call_plc_sssp,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            source,
            cutoff,
            compute_predecessors,
            do_expensive_check,
            workers=[w],
            allow_other_workers=False,
        )
        for w in Comms.get_workers()
    ]

    wait(result)
    ddf = dask_cudf.from_delayed(result).persist()
    wait(ddf)

    # Wait until the inactive futures are released
    wait([r.release() for r in result])

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "vertex")
        ddf = input_graph.unrenumber(ddf, "predecessor")
        ddf["predecessor"] = ddf["predecessor"].fillna(-1)

    return ddf

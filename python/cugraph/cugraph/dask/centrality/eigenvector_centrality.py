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

from dask.distributed import wait, default_client

from pylibcugraph import (
    eigenvector_centrality as pylib_eigen,
    ResourceHandle,
)
import cugraph.dask.comms.comms as Comms
import dask_cudf
import cudf
import warnings


def _call_plc_eigenvector_centrality(
    sID,
    mg_graph_x,
    max_iterations,
    epsilon,
    do_expensive_check,
):

    return pylib_eigen(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        epsilon=epsilon,
        max_iterations=max_iterations,
        do_expensive_check=do_expensive_check,
    )


def convert_to_cudf(cp_arrays):
    """
    create a cudf DataFrame from cupy arrays
    """
    cupy_vertices, cupy_values = cp_arrays
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertices
    df["eigenvector_centrality"] = cupy_values
    return df


def eigenvector_centrality(input_graph, max_iter=100, tol=1.0e-6):
    """
    Compute the eigenvector centrality for a graph G.

    Eigenvector centrality computes the centrality for a node based on the
    centrality of its neighbors. The eigenvector centrality for node i is the
    i-th element of the vector x defined by the eigenvector equation.

    Parameters
    ----------
    input_graph : cuGraph.Graph or networkx.Graph
        cuGraph graph descriptor with connectivity information. The graph can
        contain either directed or undirected edges.

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned. This can
        be used to limit the execution time and do an early exit before the
        solver reaches the convergence tolerance.

    tol : float, optional (default=1e-6)
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.
        The lower the tolerance the better the approximation. If this value is
        0.0f, cuGraph will use the default value which is 1.0e-6.
        Setting too small a tolerance can lead to non-convergence due to
        numerical roundoff. Usually values between 1e-2 and 1e-6 are
        acceptable.

    normalized : not supported
        If True normalize the resulting eigenvector centrality values

    Returns
    -------
    df : dask_cudf.DataFrame
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding eigenvector centrality values.
        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['eigenvector_centrality'] : cudf.Series
            Contains the eigenvector centrality of vertices

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
    >>> dg = cugraph.Graph()
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst',
    ...                            edge_attr='value')
    >>> ec = dcg.eigenvector_centrality(dg)

    """
    client = default_client()

    if input_graph.store_transposed is False:
        warning_msg = (
            "Eigenvector centrality expects the 'store_transposed' "
            "flag to be set to 'True' for optimal performance "
            "during the graph creation"
        )
        warnings.warn(warning_msg, UserWarning)

    # FIXME: should we add this parameter as an option?
    do_expensive_check = False

    cupy_result = [
        client.submit(
            _call_plc_eigenvector_centrality,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            max_iter,
            tol,
            do_expensive_check,
            workers=[w],
            allow_other_workers=False,
        )
        for w in Comms.get_workers()
    ]

    wait(cupy_result)

    cudf_result = [
        client.submit(
            convert_to_cudf, cp_arrays, workers=client.who_has(cp_arrays)[cp_arrays.key]
        )
        for cp_arrays in cupy_result
    ]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result).persist()
    wait(ddf)

    # Wait until the inactive futures are released
    wait([(r.release(), c_r.release()) for r, c_r in zip(cupy_result, cudf_result)])

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "vertex")

    return ddf

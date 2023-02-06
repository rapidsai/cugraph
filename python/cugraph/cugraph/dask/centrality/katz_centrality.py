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
from pylibcugraph import ResourceHandle, katz_centrality as pylibcugraph_katz
import cugraph.dask.comms.comms as Comms
import dask_cudf
import cudf
import warnings


def _call_plc_katz_centrality(
    sID, mg_graph_x, betas, alpha, beta, epsilon, max_iterations, do_expensive_check
):

    return pylibcugraph_katz(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        betas=betas,
        alpha=alpha,
        beta=beta,
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
    df["katz_centrality"] = cupy_values
    return df


def katz_centrality(
    input_graph,
    alpha=None,
    beta=1.0,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    normalized=True,
):
    """
    Compute the Katz centrality for the nodes of the graph G.

    Parameters
    ----------
    input_graph : cuGraph.Graph
        cuGraph graph descriptor with connectivity information. The graph can
        contain either directed or undirected edges.

    alpha : float, optional (default=None)
        Attenuation factor. If alpha is not specified then
        it is internally calculated as 1/(degree_max) where degree_max is the
        maximum out degree.

        NOTE
            The maximum acceptable value of alpha for convergence
            alpha_max = 1/(lambda_max) where lambda_max is the largest
            eigenvalue of the graph.
            Since lambda_max is always lesser than or equal to degree_max for a
            graph, alpha_max will always be greater than or equal to
            (1/degree_max). Therefore, setting alpha to (1/degree_max) will
            guarantee that it will never exceed alpha_max thus in turn
            fulfilling the requirement for convergence.

    beta : float, optional (default=None)
        Weight scalar added to each vertex's new Katz Centrality score in every
        iteration. If beta is not specified then it is set as 1.0.

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned. This can
        be used to limit the execution time and do an early exit before the
        solver reaches the convergence tolerance.
        If this value is lower or equal to 0 cuGraph will use the default
        value, which is 100.

    tol : float, optional (default=1.0e-5)
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.
        The lower the tolerance the better the approximation. If this value is
        0.0f, cuGraph will use the default value which is 1.0e-6.
        Setting too small a tolerance can lead to non-convergence due to
        numerical roundoff. Usually values between 1e-2 and 1e-6 are
        acceptable.

    nstart : dask_cudf.Dataframe, optional (default=None)
        Distributed GPU Dataframe containing the initial guess for katz
        centrality.

        nstart['vertex'] : dask_cudf.Series
            Contains the vertex identifiers
        nstart['values'] : dask_cudf.Series
            Contains the katz centrality values of vertices

    normalized : not supported
        If True normalize the resulting katz centrality values

    Returns
    -------
    katz_centrality : dask_cudf.DataFrame
        GPU distributed data frame containing two dask_cudf.Series of size V:
        the vertex identifiers and the corresponding katz centrality values.

        ddf['vertex'] : dask_cudf.Series
            Contains the vertex identifiers
        ddf['katz_centrality'] : dask_cudf.Series
            Contains the katz centrality of vertices

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
    >>> pr = dcg.katz_centrality(dg)

    """
    client = default_client()

    if input_graph.store_transposed is False:
        warning_msg = (
            "Katz centrality expects the 'store_transposed' flag "
            "to be set to 'True' for optimal performance during "
            "the graph creation"
        )
        warnings.warn(warning_msg, UserWarning)

    if alpha is None:
        degree_max = input_graph.degree()["degree"].max().compute()
        alpha = 1 / (degree_max)

    if (alpha is not None) and (alpha <= 0.0):
        raise ValueError(f"'alpha' must be a positive float or None, " f"got: {alpha}")

    # FIXME: should we add this parameter as an option?
    do_expensive_check = False

    initial_hubs_guess_values = None
    if nstart:
        if input_graph.renumbered:
            if len(input_graph.renumber_map.implementation.col_names) > 1:
                cols = nstart.columns[:-1].to_list()
            else:
                cols = "vertex"
            nstart = input_graph.add_internal_vertex_id(nstart, "vertex", cols)
            initial_hubs_guess_values = nstart[nstart.columns[0]].compute()
        else:
            initial_hubs_guess_values = nstart["values"]
            if isinstance(nstart, dask_cudf.DataFrame):
                initial_hubs_guess_values = initial_hubs_guess_values.compute()

    cupy_result = [
        client.submit(
            _call_plc_katz_centrality,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            initial_hubs_guess_values,
            alpha,
            beta,
            tol,
            max_iter,
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
        return input_graph.unrenumber(ddf, "vertex")

    return ddf

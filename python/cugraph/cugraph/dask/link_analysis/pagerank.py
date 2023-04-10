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
import dask_cudf
import cudf
import numpy as np
import warnings
from cugraph.dask.common.input_utils import get_distributed_data

from pylibcugraph import (
    ResourceHandle,
    pagerank as pylibcugraph_pagerank,
    personalized_pagerank as pylibcugraph_p_pagerank,
)


def convert_to_cudf(cp_arrays):
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """
    cupy_vertices, cupy_pagerank = cp_arrays
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertices
    df["pagerank"] = cupy_pagerank

    return df


# FIXME: Move this function to the utility module so that it can be
# shared by other algos
def ensure_valid_dtype(input_graph, input_df, input_df_name):
    if input_graph.properties.weighted is False:
        edge_attr_dtype = np.float64
    else:
        edge_attr_dtype = input_graph.input_df["value"].dtype

    input_df_dtype = input_df["values"].dtype
    if input_df_dtype != edge_attr_dtype:
        warning_msg = (
            f"PageRank requires '{input_df_name}' values "
            "to match the graph's 'edge_attr' type. "
            f"edge_attr type is: {edge_attr_dtype} and got "
            f"'{input_df_name}' values of type: "
            f"{input_df_dtype}."
        )
        warnings.warn(warning_msg, UserWarning)
        input_df = input_df.astype({"values": edge_attr_dtype})

    return input_df


def renumber_vertices(input_graph, input_df):
    input_df = input_graph.add_internal_vertex_id(
        input_df, "vertex", "vertex"
    ).compute()

    return input_df


def _call_plc_pagerank(
    sID,
    mg_graph_x,
    pre_vtx_o_wgt_vertices,
    pre_vtx_o_wgt_sums,
    initial_guess_vertices,
    initial_guess_values,
    alpha,
    epsilon,
    max_iterations,
    do_expensive_check,
):

    return pylibcugraph_pagerank(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        precomputed_vertex_out_weight_vertices=pre_vtx_o_wgt_vertices,
        precomputed_vertex_out_weight_sums=pre_vtx_o_wgt_sums,
        initial_guess_vertices=initial_guess_vertices,
        initial_guess_values=initial_guess_values,
        alpha=alpha,
        epsilon=epsilon,
        max_iterations=max_iterations,
        do_expensive_check=do_expensive_check,
    )


def _call_plc_personalized_pagerank(
    sID,
    mg_graph_x,
    pre_vtx_o_wgt_vertices,
    pre_vtx_o_wgt_sums,
    data_personalization,
    initial_guess_vertices,
    initial_guess_values,
    alpha,
    epsilon,
    max_iterations,
    do_expensive_check,
):
    personalization_vertices = data_personalization["vertex"]
    personalization_values = data_personalization["values"]
    return pylibcugraph_p_pagerank(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        precomputed_vertex_out_weight_vertices=pre_vtx_o_wgt_vertices,
        precomputed_vertex_out_weight_sums=pre_vtx_o_wgt_sums,
        personalization_vertices=personalization_vertices,
        personalization_values=personalization_values,
        initial_guess_vertices=initial_guess_vertices,
        initial_guess_values=initial_guess_values,
        alpha=alpha,
        epsilon=epsilon,
        max_iterations=max_iterations,
        do_expensive_check=do_expensive_check,
    )


def pagerank(
    input_graph,
    alpha=0.85,
    personalization=None,
    precomputed_vertex_out_weight=None,
    max_iter=100,
    tol=1.0e-5,
    nstart=None,
):
    """
    Find the PageRank values for each vertex in a graph using multiple GPUs.
    cuGraph computes an approximation of the Pagerank using the power method.
    The input graph must contain edge list as  dask-cudf dataframe with
    one partition per GPU.
    All edges will have an edge_attr value of 1.0 if not provided.

    Parameters
    ----------
    input_graph : cugraph.Graph
        cuGraph graph descriptor, should contain the connectivity information
        as dask cudf edge list dataframe(edge weights are not used for this
        algorithm).

    alpha : float, optional (default=0.85)
        The damping factor alpha represents the probability to follow an
        outgoing edge, standard value is 0.85.
        Thus, 1.0-alpha is the probability to “teleport” to a random vertex.
        Alpha should be greater than 0.0 and strictly lower than 1.0.

    personalization : cudf.Dataframe, optional (default=None)
        GPU Dataframe containing the personalization information.
        (a performance optimization)
        personalization['vertex'] : cudf.Series
            Subset of vertices of graph for personalization
        personalization['values'] : cudf.Series
            Personalization values for vertices

    precomputed_vertex_out_weight : cudf.Dataframe, optional (default=None)
        GPU Dataframe containing the precomputed vertex out weight
        (a performance optimization)
        information.
        precomputed_vertex_out_weight['vertex'] : cudf.Series
            Subset of vertices of graph for precomputed_vertex_out_weight
        precomputed_vertex_out_weight['sums'] : cudf.Series
            Corresponding precomputed sum of outgoing vertices weight

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned. This can
        be used to limit the execution time and do an early exit before the
        solver reaches the convergence tolerance.
        If this value is lower or equal to 0 cuGraph will use the default
        value, which is 100.

    tol : float, optional (default=1e-05)
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.
        The lower the tolerance the better the approximation. If this value is
        0.0f, cuGraph will use the default value which is 1.0E-5.
        Setting too small a tolerance can lead to non-convergence due to
        numerical roundoff. Usually values between 0.01 and 0.00001 are
        acceptable.

    nstart : cudf.Dataframe, optional (default=None)
        GPU Dataframe containing the initial guess for pagerank.
        (a performance optimization)
        nstart['vertex'] : cudf.Series
            Subset of vertices of graph for initial guess for pagerank values
        nstart['values'] : cudf.Series
            Pagerank values for vertices

    Returns
    -------
    PageRank : dask_cudf.DataFrame
        GPU data frame containing two dask_cudf.Series of size V: the
        vertex identifiers and the corresponding PageRank values.

        NOTE: if the input cugraph.Graph was created using the renumber=False
        option of any of the from_*_edgelist() methods, pagerank assumes that
        the vertices in the edgelist are contiguous and start from 0.
        If the actual set of vertices in the edgelist is not
        contiguous (has gaps) or does not start from zero, pagerank will assume
        the "missing" vertices are isolated vertices in the graph, and will
        compute and return pagerank values for each. If this is not the desired
        behavior, ensure the input cugraph.Graph is created from the
        from_*_edgelist() functions with the renumber=True option (the default)

        ddf['vertex'] : dask_cudf.Series
            Contains the vertex identifiers
        ddf['pagerank'] : dask_cudf.Series
            Contains the PageRank score

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
    >>> pr = dcg.pagerank(dg)

    """

    # Initialize dask client
    client = default_client()

    if input_graph.store_transposed is False:
        warning_msg = (
            "Pagerank expects the 'store_transposed' flag "
            "to be set to 'True' for optimal performance during "
            "the graph creation"
        )
        warnings.warn(warning_msg, UserWarning)

    initial_guess_vertices = None
    initial_guess_values = None
    precomputed_vertex_out_weight_vertices = None
    precomputed_vertex_out_weight_sums = None

    do_expensive_check = False

    # FIXME: Distribute the 'precomputed_vertex_out_weight'
    # across GPUs for performance optimization
    if precomputed_vertex_out_weight is not None:
        if input_graph.renumbered is True:
            precomputed_vertex_out_weight = renumber_vertices(
                input_graph, precomputed_vertex_out_weight
            )
        precomputed_vertex_out_weight_vertices = precomputed_vertex_out_weight["vertex"]
        precomputed_vertex_out_weight_sums = precomputed_vertex_out_weight["sums"]

    # FIXME: Distribute the 'nstart' across GPUs for performance optimization
    if nstart is not None:
        if input_graph.renumbered is True:
            nstart = renumber_vertices(input_graph, nstart)
        nstart = ensure_valid_dtype(input_graph, nstart, "nstart")
        initial_guess_vertices = nstart["vertex"]
        initial_guess_values = nstart["values"]

    if personalization is not None:
        if input_graph.renumbered is True:
            personalization = renumber_vertices(input_graph, personalization)
        personalization = ensure_valid_dtype(
            input_graph, personalization, "personalization"
        )

        personalization_ddf = dask_cudf.from_cudf(
            personalization, npartitions=len(Comms.get_workers())
        )

        data_prsztn = get_distributed_data(personalization_ddf)

        result = [
            client.submit(
                _call_plc_personalized_pagerank,
                Comms.get_session_id(),
                input_graph._plc_graph[w],
                precomputed_vertex_out_weight_vertices,
                precomputed_vertex_out_weight_sums,
                data_personalization[0],
                initial_guess_vertices,
                initial_guess_values,
                alpha,
                tol,
                max_iter,
                do_expensive_check,
                workers=[w],
                allow_other_workers=False,
            )
            for w, data_personalization in data_prsztn.worker_to_parts.items()
        ]
    else:
        result = [
            client.submit(
                _call_plc_pagerank,
                Comms.get_session_id(),
                input_graph._plc_graph[w],
                precomputed_vertex_out_weight_vertices,
                precomputed_vertex_out_weight_sums,
                initial_guess_vertices,
                initial_guess_values,
                alpha,
                tol,
                max_iter,
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

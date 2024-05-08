# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

import warnings

import dask
from dask.distributed import wait, default_client
import dask_cudf
import cudf
import numpy as np
from pylibcugraph import (
    pagerank as plc_pagerank,
    personalized_pagerank as plc_p_pagerank,
    exceptions as plc_exceptions,
    ResourceHandle,
)

import cugraph.dask.comms.comms as Comms
from cugraph.dask.common.part_utils import (
    persist_dask_df_equal_parts_per_worker,
)
from cugraph.exceptions import FailedToConvergeError


def convert_to_return_tuple(plc_pr_retval):
    """
    Using the PLC pagerank return tuple, creates a cudf DataFrame from the cupy
    arrays and extracts the (optional) bool.
    """
    if len(plc_pr_retval) == 3:
        cupy_vertices, cupy_pagerank, converged = plc_pr_retval
    else:
        cupy_vertices, cupy_pagerank = plc_pr_retval
        converged = True

    df = cudf.DataFrame()
    df["vertex"] = cupy_vertices
    df["pagerank"] = cupy_pagerank

    return (df, converged)


# FIXME: Move this function to the utility module so that it can be
# shared by other algos
def ensure_valid_dtype(input_graph, input_df, input_df_name):
    if input_graph.properties.weighted is False:
        # If the graph is not weighted, an artificial weight column
        # of type 'float32' is added and it must match the user
        # personalization/nstart values.
        edge_attr_dtype = np.float32
    else:
        edge_attr_dtype = input_graph.input_df["value"].dtype

    if "values" in input_df.columns:
        input_df_values_dtype = input_df["values"].dtype
        if input_df_values_dtype != edge_attr_dtype:
            warning_msg = (
                f"PageRank requires '{input_df_name}' values "
                "to match the graph's 'edge_attr' type. "
                f"edge_attr type is: {edge_attr_dtype} and got "
                f"'{input_df_name}' values of type: "
                f"{input_df_values_dtype}."
            )
            warnings.warn(warning_msg, UserWarning)
            input_df = input_df.astype({"values": edge_attr_dtype})

    vertex_dtype = input_graph.edgelist.edgelist_df.dtypes.iloc[0]
    input_df_vertex_dtype = input_df["vertex"].dtype
    if input_df_vertex_dtype != vertex_dtype:
        warning_msg = (
            f"PageRank requires '{input_df_name}' vertex "
            "to match the graph's 'vertex' type. "
            f"input graph's vertex type is: {vertex_dtype} and got "
            f"'{input_df_name}' vertex of type: "
            f"{input_df_vertex_dtype}."
        )
        warnings.warn(warning_msg, UserWarning)
        input_df = input_df.astype({"vertex": vertex_dtype})

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
    fail_on_nonconvergence,
):
    try:
        return plc_pagerank(
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
            fail_on_nonconvergence=fail_on_nonconvergence,
        )
    # Re-raise this as a cugraph exception so users trying to catch this do not
    # have to know to import another package.
    except plc_exceptions.FailedToConvergeError as exc:
        raise FailedToConvergeError from exc


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
    fail_on_nonconvergence,
):
    personalization_vertices = data_personalization["vertex"]
    personalization_values = data_personalization["values"]
    try:
        return plc_p_pagerank(
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
            fail_on_nonconvergence=fail_on_nonconvergence,
        )
    # Re-raise this as a cugraph exception so users trying to catch this do not
    # have to know to import another package.
    except plc_exceptions.FailedToConvergeError as exc:
        raise FailedToConvergeError from exc


def pagerank(
    input_graph,
    alpha=0.85,
    personalization=None,
    precomputed_vertex_out_weight=None,
    max_iter=100,
    tol=1.0e-5,
    nstart=None,
    fail_on_nonconvergence=True,
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

    fail_on_nonconvergence : bool (default=True)
        If the solver does not reach convergence, raise an exception if
        fail_on_nonconvergence is True. If fail_on_nonconvergence is False,
        the return value is a tuple of (pagerank, converged) where pagerank is
        a cudf.DataFrame as described below, and converged is a boolean
        indicating if the solver converged (True) or not (False).

    Returns
    -------
    The return value varies based on the value of the fail_on_nonconvergence
    paramter.  If fail_on_nonconvergence is True:

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

    If fail_on_nonconvergence is False:

    (PageRank, converged) : tuple of (dask_cudf.DataFrame, bool)
       PageRank is the GPU dataframe described above, converged is a bool
       indicating if the solver converged (True) or not (False).

    Examples
    --------
    >>> import cugraph.dask as dcg
    >>> import dask_cudf
    >>> # ... Init a DASK Cluster
    >>> #    see https://docs.rapids.ai/api/cugraph/stable/dask-cugraph.html
    >>> # Download dataset from https://github.com/rapidsai/cugraph/datasets/..
    >>> chunksize = dcg.get_chunksize(datasets_path / "karate.csv")
    >>> ddf = dask_cudf.read_csv(datasets_path / "karate.csv",
    ...                          blocksize=chunksize, delimiter=" ",
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
        precomputed_vertex_out_weight = ensure_valid_dtype(
            input_graph, precomputed_vertex_out_weight, "precomputed_vertex_out_weight"
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

        data_prsztn = persist_dask_df_equal_parts_per_worker(
            personalization_ddf, client, return_type="dict"
        )

        empty_df = cudf.DataFrame(columns=list(personalization_ddf.columns))
        empty_df = empty_df.astype(
            dict(zip(personalization_ddf.columns, personalization_ddf.dtypes))
        )

        result = [
            client.submit(
                _call_plc_personalized_pagerank,
                Comms.get_session_id(),
                input_graph._plc_graph[w],
                precomputed_vertex_out_weight_vertices,
                precomputed_vertex_out_weight_sums,
                data_personalization[0] if data_personalization else empty_df,
                initial_guess_vertices,
                initial_guess_values,
                alpha,
                tol,
                max_iter,
                do_expensive_check,
                fail_on_nonconvergence,
                workers=[w],
                allow_other_workers=False,
            )
            for w, data_personalization in data_prsztn.items()
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
                fail_on_nonconvergence,
                workers=[w],
                allow_other_workers=False,
            )
            for w in Comms.get_workers()
        ]

    wait(result)

    vertex_dtype = input_graph.edgelist.edgelist_df.dtypes.iloc[0]

    # Have each worker convert tuple of arrays and bool from PLC to cudf
    # DataFrames and bools. This will be a list of futures.
    result_tuples = [
        client.submit(convert_to_return_tuple, cp_arrays) for cp_arrays in result
    ]

    # Convert the futures to dask delayed objects so the tuples can be
    # split. nout=2 is passed since each tuple/iterable is a fixed length of 2.
    result_tuples = [dask.delayed(r, nout=2) for r in result_tuples]

    # Create the ddf and get the converged bool from the delayed objs.  Use a
    # meta DataFrame to pass the expected dtypes for the DataFrame to prevent
    # another compute to determine them automatically.
    meta = cudf.DataFrame(columns=["vertex", "pagerank"])
    meta = meta.astype({"pagerank": "float64", "vertex": vertex_dtype})
    ddf = dask_cudf.from_delayed([t[0] for t in result_tuples], meta=meta).persist()
    converged = all(dask.compute(*[t[1] for t in result_tuples]))

    wait(ddf)

    # Wait until the inactive futures are released
    wait([(r.release(), c_r.release()) for r, c_r in zip(result, result_tuples)])

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "vertex")

    if fail_on_nonconvergence:
        return ddf
    else:
        return (ddf, converged)

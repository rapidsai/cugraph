# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

import warnings

import cudf
import numpy as np

from pylibcugraph import (
    pagerank as plc_pagerank,
    personalized_pagerank as plc_p_pagerank,
    exceptions as plc_exceptions,
    ResourceHandle,
)

from cugraph.utilities import (
    ensure_cugraph_obj_for_nx,
    df_score_to_dictionary,
)
from cugraph.exceptions import FailedToConvergeError


def renumber_vertices(input_graph, input_df):
    if len(input_graph.renumber_map.implementation.col_names) > 1:
        cols = input_df.columns[:-1].to_list()
    else:
        cols = "vertex"
    input_df = input_graph.add_internal_vertex_id(input_df, "vertex", cols)

    return input_df


# FIXME: Move this function to the utility module so that it can be
# shared by other algos
def ensure_valid_dtype(input_graph, input_df, input_df_name):
    if input_graph.edgelist.weights is False:
        # If the graph is not weighted, an artificial weight column
        # of type 'float32' is added and it must match the user
        # personalization/nstart values.
        edge_attr_dtype = np.float32
    else:
        edge_attr_dtype = input_graph.edgelist.edgelist_df["weights"].dtype

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

    vertex_dtype = input_graph.edgelist.edgelist_df.dtypes[0]
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


def pagerank(
    G,
    alpha=0.85,
    personalization=None,
    precomputed_vertex_out_weight=None,
    max_iter=100,
    tol=1.0e-5,
    nstart=None,
    weight=None,
    dangling=None,
    fail_on_nonconvergence=True,
):
    """Find the PageRank score for every vertex in a graph. cuGraph computes an
    approximation of the Pagerank eigenvector using the power method. The
    number of iterations depends on the properties of the network itself; it
    increases when the tolerance descreases and/or alpha increases toward the
    limiting value of 1. The user is free to use default values or to provide
    inputs for the initial guess, tolerance and maximum number of iterations.
    All edges will have an edge_attr value of 1.0 if not provided.

    Parameters
    ----------
    G : cugraph.Graph or networkx.Graph
        cuGraph graph descriptor, should contain the connectivity information
        as an edge list.
        The transposed adjacency list will be computed if not already present.

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
        information(a performance optimization).

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
        (a performance optimization).

        nstart['vertex'] : cudf.Series
            Subset of vertices of graph for initial guess for pagerank values

        nstart['values'] : cudf.Series
            Pagerank values for vertices

    weight: str, optional (default=None)
        The attribute column to be used as edge weights if Graph is a NetworkX
        Graph. This parameter is here for NetworkX compatibility and is ignored
        in case of a cugraph.Graph

    dangling : dict, optional (default=None)
        This parameter is here for NetworkX compatibility and ignored

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

    PageRank : cudf.DataFrame
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding PageRank values.

        NOTE: if the input cugraph.Graph was created using the renumber=False
        option of any of the from_*_edgelist() methods, pagerank assumes that
        the vertices in the edgelist are contiguous and start from 0.
        If the actual set of vertices in the edgelist is not
        contiguous (has gaps) or does not start from zero, pagerank will assume
        the "missing" vertices are isolated vertices in the graph, and will
        compute and return pagerank values for each. If this is not the desired
        behavior, ensure the input cugraph.Graph is created from the
        from_*_edgelist() functions with the renumber=True option (the default)

        df['vertex'] : cudf.Series
            Contains the vertex identifiers

        df['pagerank'] : cudf.Series
            Contains the PageRank score

    If fail_on_nonconvergence is False:

    (PageRank, converged) : tuple of (cudf.DataFrame, bool)
       PageRank is the GPU dataframe described above, converged is a bool
       indicating if the solver converged (True) or not (False).

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> pr = cugraph.pagerank(G, alpha = 0.85, max_iter = 500, tol = 1.0e-05)
    """

    initial_guess_vertices = None
    initial_guess_values = None
    pre_vtx_o_wgt_vertices = None
    pre_vtx_o_wgt_sums = None

    G, isNx = ensure_cugraph_obj_for_nx(G, weight, store_transposed=True)
    if G.store_transposed is False:
        warning_msg = (
            "Pagerank expects the 'store_transposed' flag "
            "to be set to 'True' for optimal performance during "
            "the graph creation"
        )
        warnings.warn(warning_msg, UserWarning)

    do_expensive_check = False

    if nstart is not None:
        if G.renumbered is True:
            nstart = renumber_vertices(G, nstart)
        nstart = ensure_valid_dtype(G, nstart, "nstart")
        initial_guess_vertices = nstart["vertex"]
        initial_guess_values = nstart["values"]

    if precomputed_vertex_out_weight is not None:
        if G.renumbered is True:
            precomputed_vertex_out_weight = renumber_vertices(
                G, precomputed_vertex_out_weight
            )
        precomputed_vertex_out_weight = ensure_valid_dtype(
            G, precomputed_vertex_out_weight, "precomputed_vertex_out_weight"
        )
        pre_vtx_o_wgt_vertices = precomputed_vertex_out_weight["vertex"]
        pre_vtx_o_wgt_sums = precomputed_vertex_out_weight["sums"]

    try:
        if personalization is not None:
            if not isinstance(personalization, cudf.DataFrame):
                raise NotImplementedError(
                    "personalization other than a cudf dataframe currently not "
                    "supported"
                )
            if G.renumbered is True:
                personalization = renumber_vertices(G, personalization)

            personalization = ensure_valid_dtype(G, personalization, "personalization")

            result_tuple = plc_p_pagerank(
                resource_handle=ResourceHandle(),
                graph=G._plc_graph,
                precomputed_vertex_out_weight_vertices=pre_vtx_o_wgt_vertices,
                precomputed_vertex_out_weight_sums=pre_vtx_o_wgt_sums,
                personalization_vertices=personalization["vertex"],
                personalization_values=personalization["values"],
                initial_guess_vertices=initial_guess_vertices,
                initial_guess_values=initial_guess_values,
                alpha=alpha,
                epsilon=tol,
                max_iterations=max_iter,
                do_expensive_check=do_expensive_check,
                fail_on_nonconvergence=fail_on_nonconvergence,
            )
        else:
            result_tuple = plc_pagerank(
                resource_handle=ResourceHandle(),
                graph=G._plc_graph,
                precomputed_vertex_out_weight_vertices=pre_vtx_o_wgt_vertices,
                precomputed_vertex_out_weight_sums=pre_vtx_o_wgt_sums,
                initial_guess_vertices=initial_guess_vertices,
                initial_guess_values=initial_guess_values,
                alpha=alpha,
                epsilon=tol,
                max_iterations=max_iter,
                do_expensive_check=do_expensive_check,
                fail_on_nonconvergence=fail_on_nonconvergence,
            )
    # Re-raise this as a cugraph exception so users trying to catch this do not
    # have to know to import another package.
    except plc_exceptions.FailedToConvergeError as exc:
        raise FailedToConvergeError from exc

    df = cudf.DataFrame()
    df["vertex"] = result_tuple[0]
    df["pagerank"] = result_tuple[1]

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        df = df_score_to_dictionary(df, "pagerank")

    if fail_on_nonconvergence:
        return df
    else:
        return (df, result_tuple[2])

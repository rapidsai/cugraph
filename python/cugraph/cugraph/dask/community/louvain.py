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

from __future__ import annotations

from dask.distributed import wait, default_client
import cugraph.dask.comms.comms as Comms
import dask_cudf
import dask
from dask import delayed
import cudf
import cupy as cp
import numpy

from pylibcugraph import ResourceHandle
from pylibcugraph import louvain as pylibcugraph_louvain
from typing import Tuple, TYPE_CHECKING

import warnings

if TYPE_CHECKING:
    from cugraph import Graph


def convert_to_cudf(result: cp.ndarray) -> Tuple[cudf.DataFrame, float]:
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """
    cupy_vertex, cupy_partition, modularity = result
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertex
    df["partition"] = cupy_partition

    return df, modularity


def _call_plc_louvain(
    sID: bytes,
    mg_graph_x,
    max_level: int,
    threshold: float,
    resolution: float,
    do_expensive_check: bool,
) -> Tuple[cp.ndarray, cp.ndarray, float]:
    return pylibcugraph_louvain(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        max_level=max_level,
        threshold=threshold,
        resolution=resolution,
        do_expensive_check=do_expensive_check,
    )


# FIXME: max_level should default to 100 once max_iter is removed
def louvain(
    input_graph: Graph,
    max_level: int = None,
    max_iter: int = None,
    resolution: float = 1.0,
    threshold: float = 1e-7,
) -> Tuple[dask_cudf.DataFrame, float]:
    """
    Compute the modularity optimizing partition of the input graph using the
    Louvain method

    It uses the Louvain method described in:

    VD Blondel, J-L Guillaume, R Lambiotte and E Lefebvre: Fast unfolding of
    community hierarchies in large networks, J Stat Mech P10008 (2008),
    http://arxiv.org/abs/0803.0476

    Parameters
    ----------
    G : cugraph.Graph
        The graph descriptor should contain the connectivity information
        and weights. The adjacency list will be computed if not already
        present.
        The current implementation only supports undirected graphs.

    max_level : integer, optional (default=100)
        This controls the maximum number of levels of the Louvain
        algorithm. When specified the algorithm will terminate after no more
        than the specified number of levels. No error occurs when the
        algorithm terminates early in this manner.

    max_iter : integer, optional (default=None)
        This parameter is deprecated in favor of max_level.  Previously
        it was used to control the maximum number of levels of the Louvain
        algorithm.

    resolution: float, optional (default=1.0)
        Called gamma in the modularity formula, this changes the size
        of the communities.  Higher resolutions lead to more smaller
        communities, lower resolutions lead to fewer larger communities.

    threshold: float, optional (default=1e-7)
        Modularity gain threshold for each level. If the gain of
        modularity between 2 levels of the algorithm is less than the
        given threshold then the algorithm stops and returns the
        resulting communities.

    Returns
    -------
    parts : dask_cudf.DataFrame
        GPU data frame of size V containing two columns the vertex id and the
        partition id it is assigned to.

        ddf['vertex'] : cudf.Series
            Contains the vertex identifiers
        ddf['partition'] : cudf.Series
            Contains the partition assigned to the vertices

    modularity_score : float
        a floating point number containing the global modularity score of the
        partitioning.

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
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst')
    >>> parts, modularity_score = dcg.louvain(dg)

    """

    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")

    # FIXME: This max_iter logic and the max_level defaulting can be deleted
    #        in favor of defaulting max_level in call once max_iter is deleted
    if max_iter:
        if max_level:
            raise ValueError(
                "max_iter is deprecated.  Cannot specify both max_iter and max_level"
            )

        warning_msg = (
            "max_iter has been renamed max_level.  Use of max_iter is "
            "deprecated and will no longer be supported in the next releases. "
        )
        warnings.warn(warning_msg, FutureWarning)
        max_level = max_iter

    if max_level is None:
        max_level = 100

    # Initialize dask client
    client = default_client()

    do_expensive_check = False

    result = [
        client.submit(
            _call_plc_louvain,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            max_level,
            threshold,
            resolution,
            do_expensive_check,
            workers=[w],
            allow_other_workers=False,
        )
        for w in Comms.get_workers()
    ]

    wait(result)

    part_mod_score = [client.submit(convert_to_cudf, r) for r in result]
    wait(part_mod_score)

    vertex_dtype = input_graph.edgelist.edgelist_df.dtypes.iloc[0]
    empty_df = cudf.DataFrame(
        {
            "vertex": numpy.empty(shape=0, dtype=vertex_dtype),
            "partition": numpy.empty(shape=0, dtype="int32"),
        }
    )

    part_mod_score = [delayed(lambda x: x, nout=2)(r) for r in part_mod_score]

    ddf = dask_cudf.from_delayed(
        [r[0] for r in part_mod_score], meta=empty_df, verify_meta=False
    ).persist()

    mod_score = dask.array.from_delayed(
        part_mod_score[0][1], shape=(1,), dtype=float
    ).compute()

    wait(ddf)
    wait(mod_score)

    wait([r.release() for r in part_mod_score])

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "vertex")

    return ddf, mod_score

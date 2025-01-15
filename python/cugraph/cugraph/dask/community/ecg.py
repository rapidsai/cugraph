# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

from pylibcugraph import ResourceHandle
from pylibcugraph import ecg as pylibcugraph_ecg
import numpy
import cupy as cp
from typing import Tuple, TYPE_CHECKING

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


def _call_plc_ecg(
    sID: bytes,
    mg_graph_x,
    max_iter: int,
    resolution: int,
    random_state: int,
    theta: int,
    do_expensive_check: bool,
) -> Tuple[cp.ndarray, cp.ndarray, float]:
    return pylibcugraph_ecg(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        random_state=random_state,
        graph=mg_graph_x,
        max_level=max_iter,
        resolution=resolution,
        theta=theta,
        do_expensive_check=do_expensive_check,
    )


def ecg(
    input_graph,
    min_weight: float = 0.0001,
    ensemble_size: int = 100,
    max_level: int = 10,
    threshold: float = 1e-7,
    resolution: float = 1.0,
    random_state: int = None,
    weight=None,
) -> Tuple[dask_cudf.DataFrame, float]:
    """
    Compute the Ensemble Clustering for Graphs (ECG) partition of the input
    graph. ECG runs truncated Louvain on an ensemble of permutations of the
    input graph, then uses the ensemble partitions to determine weights for
    the input graph. The final result is found by running full Louvain on
    the input graph using the determined weights.

    See https://arxiv.org/abs/1809.05578 for further information.

    Parameters
    ----------
    input_graph : cugraph.Graph or NetworkX Graph
        The graph descriptor should contain the connectivity information
        and weights. The adjacency list will be computed if not already
        present.

    min_weight : float, optional (default=0.5)
        The minimum value to assign as an edgeweight in the ECG algorithm.
        It should be a value in the range [0,1] usually left as the default
        value of .05

    ensemble_size : integer, optional (default=16)
        The number of graph permutations to use for the ensemble.
        The default value is 16, larger values may produce higher quality
        partitions for some graphs.

    max_level : integer, optional (default=100)
        This controls the maximum number of levels/iterations of the ECG
        algorithm. When specified the algorithm will terminate after no more
        than the specified number of iterations. No error occurs when the
        algorithm terminates early in this manner.

    threshold: float
        Modularity gain threshold for each level. If the gain of
        modularity between 2 levels of the algorithm is less than the
        given threshold then the algorithm stops and returns the
        resulting communities.
        Defaults to 1e-7.

    resolution: float, optional (default=1.0)
        Called gamma in the modularity formula, this changes the size
        of the communities.  Higher resolutions lead to more smaller
        communities, lower resolutions lead to fewer larger communities.
        Defaults to 1.

    random_state: int, optional(default=None)
        Random state to use when generating samples.  Optional argument,
        defaults to a hash of process id, time, and hostname.

    weight : str, optional (default=None)
        Deprecated.
        This parameter is here for NetworkX compatibility and
        represents which NetworkX data column represents Edge weights.

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
    ...                          blocksize=chunksize, delimiter=" ",
    ...                          names=["src", "dst", "value"],
    ...                          dtype=["int32", "int32", "float32"])
    >>> dg = cugraph.Graph()
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst')
    >>> parts, modularity_score = dcg.ecg(dg)

    """

    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")

    # Return a client if one has started
    client = default_client()

    do_expensive_check = False

    result = [
        client.submit(
            _call_plc_ecg,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            max_iter,
            resolution,
            random_state,
            theta,
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

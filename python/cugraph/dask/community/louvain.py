# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import operator as op

from dask.distributed import wait, default_client

import cugraph.comms.comms as Comms
from cugraph.dask.common.input_utils import get_distributed_data
from cugraph.structure.shuffle import shuffle
from cugraph.dask.community import louvain_wrapper as c_mg_louvain
from cugraph.utilities.utils import is_cuda_version_less_than

import dask_cudf


def call_louvain(sID,
                 data,
                 num_verts,
                 num_edges,
                 vertex_partition_offsets,
                 sorted_by_degree,
                 max_level,
                 resolution):

    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)

    return c_mg_louvain.louvain(data[0],
                                num_verts,
                                num_edges,
                                vertex_partition_offsets,
                                wid,
                                handle,
                                sorted_by_degree,
                                max_level,
                                resolution)


def louvain(input_graph, max_iter=100, resolution=1.0):
    """
    Compute the modularity optimizing partition of the input graph using the
    Louvain method on multiple GPUs

    Examples
    --------
    >>> import cugraph.dask as dcg
    >>> Comms.initialize(p2p=True)
    >>> chunksize = dcg.get_chunksize(input_data_path)
    >>> ddf = dask_cudf.read_csv('datasets/karate.csv', chunksize=chunksize,
                                 delimiter=' ',
                                 names=['src', 'dst', 'value'],
                                 dtype=['int32', 'int32', 'float32'])
    >>> dg = cugraph.Graph()
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst',
                                   edge_attr='value')
    >>> parts, modularity_score = dcg.louvain(dg)
    """
    # FIXME: finish docstring: describe parameters, etc.

    # MG Louvain currently requires CUDA 10.2 or higher.
    # FIXME: remove this check once RAPIDS drops support for CUDA < 10.2
    if is_cuda_version_less_than((10, 2)):
        raise NotImplementedError("Multi-GPU Louvain is not implemented for "
                                  "this version of CUDA. Ensure CUDA version "
                                  "10.2 or higher is installed.")

    # FIXME: dask methods to populate graphs from edgelists are only present on
    # DiGraph classes. Disable the Graph check for now and assume inputs are
    # symmetric DiGraphs.
    # if type(graph) is not Graph:
    #     raise Exception("input graph must be undirected")
    client = default_client()
    # Calling renumbering results in data that is sorted by degree
    input_graph.compute_renumber_edge_list(transposed=False)
    sorted_by_degree = True

    (ddf,
     num_verts,
     partition_row_size,
     partition_col_size,
     vertex_partition_offsets) = shuffle(input_graph, transposed=False)

    num_edges = len(ddf)
    data = get_distributed_data(ddf)

    futures = [client.submit(call_louvain,
                             Comms.get_session_id(),
                             wf[1],
                             num_verts,
                             num_edges,
                             vertex_partition_offsets,
                             sorted_by_degree,
                             max_iter,
                             resolution,
                             workers=[wf[0]])
               for idx, wf in enumerate(data.worker_to_parts.items())]

    wait(futures)

    # futures is a list of Futures containing tuples of (DataFrame, mod_score),
    # unpack using separate calls to client.submit with a callable to get
    # individual items.
    # FIXME: look into an alternate way (not returning a tuples, accessing
    # tuples differently, etc.) since multiple client.submit() calls may not be
    # optimal.
    df_futures = [client.submit(op.getitem, f, 0) for f in futures]
    mod_score_futures = [client.submit(op.getitem, f, 1) for f in futures]

    ddf = dask_cudf.from_delayed(df_futures)
    # Each worker should have computed the same mod_score
    mod_score = mod_score_futures[0].result()

    if input_graph.renumbered:
        # MG renumbering is lazy, but it's safe to assume it's been called at
        # this point if renumbered=True
        ddf = input_graph.unrenumber(ddf, "vertex")

    return (ddf, mod_score)

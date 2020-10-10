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

from dask.distributed import wait, default_client

import cugraph.comms.comms as Comms
from cugraph.dask.common.input_utils import get_distributed_data
from cugraph.structure.shuffle import shuffle
from cugraph.dask.community import louvain_wrapper as c_mg_louvain

def call_louvain(sID,
                 data,
                 num_verts,
                 num_edges,
                 partition_row_size,
                 partition_col_size,
                 vertex_partition_offsets,
                 sorted_by_degree,
                 max_level,
                 resolution):

    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)

    return c_mg_louvain.louvain(data[0],
                                num_verts,
                                num_edges,
                                partition_row_size,
                                partition_col_size,
                                vertex_partition_offsets,
                                wid,
                                handle,
                                sorted_by_degree,
                                max_level,
                                resolution)


def louvain(input_graph, max_iter=100, resolution=1.0, load_balance=True, prows=None, pcols=None):
    """
    Compute the modularity optimizing partition of the input graph using the
    Louvain method on multiple GPUs

    Examples
    --------
    >>> import cugraph.dask as dcg
    >>> Comms.initialize()
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

    # FIXME: import here to prevent circular import: cugraph->louvain
    # wrapper->cugraph/structure->cugraph/dask->dask/louvain->cugraph/structure
    # from cugraph.structure.graph import Graph

    # FIXME: dask methods to populate graphs from edgelists are only present on
    # DiGraph classes. Disable the Graph check for now and assume inputs are
    # symmetric DiGraphs.
    # if type(graph) is not Graph:
    #     raise Exception("input graph must be undirected")
    client = default_client()
    # Calling renumbering results in data that is sorted by degree
    input_graph.compute_renumber_edge_list(transposed=False)
    sorted_by_degree = True

    if prows is None:
        (ddf,
         num_verts,
         partition_row_size,
         partition_col_size,
         vertex_partition_offsets) = shuffle(input_graph, transposed=False)
    else:
        (ddf,
         num_verts,
         partition_row_size,
         partition_col_size,
         vertex_partition_offsets) = shuffle(input_graph, prows=prows, pcols=pcols, transposed=False)

    num_edges = len(ddf)
    data = get_distributed_data(ddf)

    print('number_map = ', input_graph.renumber_map.implementation.ddf.compute())

    result = dict([(data.worker_info[wf[0]]["rank"],
                    client.submit(
                        call_louvain,
                        Comms.get_session_id(),
                        wf[1],
                        num_verts,
                        num_edges,
                        partition_row_size,
                        partition_col_size,
                        vertex_partition_offsets,
                        sorted_by_degree,
                        max_iter,
                        resolution,
                        workers=[wf[0]]))
                   for idx, wf in enumerate(data.worker_to_parts.items())])

    wait(result)

    (parts, modularity_score) = result[0].result()

    print('result = ', result)

    if input_graph.renumbered:
        # MG renumbering is lazy, but it's safe to assume it's been called at
        # this point if renumbered=True
        parts = input_graph.unrenumber(parts, "vertex")

    return parts, modularity_score

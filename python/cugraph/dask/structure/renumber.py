# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
from cugraph.dask.common.input_utils import get_distributed_data
from cugraph.structure.shuffle import shuffle
from cugraph.dask.structure import renumber_wrapper as renumber
import cugraph.comms.comms as Comms
import dask_cudf

def call_renumber(sID,
                  data,
                  num_verts,
                  num_edges.
                  is_mnmg):
    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)
    return renumber.mg_renumber(data[0],
                                num_verts,
                                num_edges,
                                wid,
                                handle,
                                is_mnmg)

def renumber(input_graph):
    from cugraph.structure.graph import null_check

    client = default_client()

    input_graph.compute_renumber_edge_list(transposed=False)
    (ddf,
     num_verts,
     partition_row_size,
     partition_col_size,
     vertex_partition_offsets) = shuffle(input_graph, transposed=False)
    num_edges = len(ddf)

    if isinstance(ddf, dask_cudf.DataFrame)::
        is_mnmg = True
    else:
        is_mnmg = False

    if is_mnmg:
        result = [client.submit(call_renumber,
                                Comms.get_session_id(),
                                wf[1],
                                num_verts,
                                num_edges,
                                is_mnmg,
                                workers=[wf[0]])
                  for idx, wf in enumerate(data.worker_to_parts.items())]
        wait(result)
        ddf = dask_cudf.from_delayed(result)
        #
        # FIXME: logic below...?
        # if input_graph.renumbered:
        #    return input_graph.unrenumber(ddf, 'vertex')

        return ddf
    else:
        df = input_graph.edgelist.edgelist_df
        call_renumber(Comms.get_session_id(),
                      df,
                      num_verts,
                      num_edges,
                      is_mnmg)

# Copyright (c) 2021, NVIDIA CORPORATION.
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
from cugraph.dask.structure import renumber_wrapper as renumber_w
import cugraph.comms.comms as Comms
import dask_cudf


def call_renumber(sID,
                  data,
                  num_verts,
                  num_edges,
                  is_mnmg):
    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)
    return renumber_w.mg_renumber(data[0],
                                  num_verts,
                                  num_edges,
                                  wid,
                                  handle,
                                  is_mnmg)


def renumber(input_graph):

    client = default_client()

    ddf = input_graph.edgelist.edgelist_df

    num_edges = len(ddf)

    if isinstance(ddf, dask_cudf.DataFrame):
        is_mnmg = True
    else:
        is_mnmg = False

    num_verts = input_graph.number_of_vertices()

    if is_mnmg:
        data = get_distributed_data(ddf)
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
    else:
        call_renumber(Comms.get_session_id(),
                      ddf,
                      num_verts,
                      num_edges,
                      is_mnmg)
    return ddf

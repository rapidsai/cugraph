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

from cugraph.raft.dask.common.comms import Comms
from cugraph.dask.common.input_utils import DistributedDataHandler
from dask.distributed import wait, default_client
from cugraph.raft.dask.common.comms import worker_state
from cugraph.opg.link_analysis import mg_pagerank_wrapper as mg_pagerank
from dask.delayed import delayed
from cugraph.dask.common.part_utils import load_balance_func
import warnings

def call_pagerank(sID, data, local_data, alpha, max_iter, tol, personalization, nstart):
    sessionstate = worker_state(sID)
    return mg_pagerank.mg_pagerank(data[0],
                                   local_data,
                                   sessionstate['handle'],
                                   alpha,
                                   max_iter,
                                   tol,
                                   personalization,
                                   nstart)


def pagerank(input_graph,
             alpha=0.85,
             personalization=None,
             max_iter=100,
             tol=1.0e-5,
             nstart=None,
             load_balance=True):

    if tol != 1.0e-5:
        warnings.warn("Tolerance is currently not supported. Setting it to default 1.0e-5")
    tol = 1.0e-5
    if personalization is not None or nstart is not None:
        warnings.warn("personalization and nstart currently not supported. Setting them to None")
    personalization = None
    nstart = None

    client = default_client()
    _ddf = input_graph.edgelist.edgelist_df
    ddf = _ddf.sort_values(by='dst', ignore_index=True)

    if load_balance:
        ddf = load_balance_func(ddf, by='dst')

    data = DistributedDataHandler.create(data=ddf)
    comms = Comms(comms_p2p=False)
    comms.init(data.workers)
    local_data = data.calculate_local_data(comms)
    result = dict([(data.worker_info[wf[0]]["rank"],
                    client.submit(
            call_pagerank,
            comms.sessionId,
            wf[1],
            local_data,
            alpha,
            max_iter,
            tol,
            personalization,
            nstart,
            workers=[wf[0]]))
            for idx, wf in enumerate(data.worker_to_parts.items())])
    wait(result)

    return result[0].result()


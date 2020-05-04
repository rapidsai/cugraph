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

from cugraph.dask.common.comms import CommsContext
from cugraph.dask.common.input_utils import DistributedDataHandler
from dask.distributed import wait, default_client
from cugraph.dask.common.comms import worker_state

def common_func(sID, data):
    print(data)
    sessionstate = worker_state(sID)
    print("nworkers: ", sessionstate['nworkers'],"  id: ", sessionstate['wid'])
    print("INSIDE common_func: ", sID)
    return 1

def pagerank(input_graph):
    print("INSIDE DASK PAGERANK")
    client = default_client()
    ddf = input_graph.edgelist

    data = DistributedDataHandler.create(data=ddf)

    comms = CommsContext(comms_p2p=False)
    comms.init(workers=data.workers)

    data.calculate_parts_to_sizes(comms)
    #self.ranks = data.ranks

    result = dict([(data.worker_info[wf[0]]["rank"],
                            client.submit(
            common_func,
            comms.sessionId,
            wf[1],
            workers=[wf[0]]))
            for idx, wf in enumerate(data.worker_to_parts.items())])
    wait(result)
    print(result)

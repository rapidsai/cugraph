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
 from cugraph.opg.link_analysis import mg_pagerank_wrapper as mg_pagerank


 def common_func(sID, data):
     print(data)
     sessionstate = worker_state(sID)
     print("nworkers: ", sessionstate['nworkers'],"  id: ", sessionstate['wid'])
     mg_pagerank.mg_pagerank(data[0], sessionstate['comm'])
     return 1

 def pagerank(input_graph):
     print("INSIDE DASK PAGERANK")
     client = default_client()
     _ddf = input_graph.edgelist.edgelist_df
     ddf = _ddf.sort_values(by='dst', ignore_index=True)
     data = DistributedDataHandler.create(data=ddf)

     comms = CommsContext(comms_p2p=False)
     comms.init(data.gpu_futures)
     data.calculate_parts_to_sizes(comms)
     #self.ranks = data.ranks

     print("Calling function")
     result = dict([(data.worker_info[wf[0]]["rank"],
                             client.submit(
             common_func,
             comms.sessionId,
             wf[1],
             workers=[wf[0]]))
             for idx, wf in enumerate(data.worker_to_parts.items())])
     wait(result)

 def get_n_gpus():
     import os
     try:
         return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
     except KeyError:
         return len(os.popen("nvidia-smi -L").read().strip().split("\n"))


 def get_chunksize(input_path):
     """
     Calculate the appropriate chunksize for dask_cudf.read_csv
     to get a number of partitions equal to the number of GPUs

     Examples
     --------
     >>> import dask_cugraph.pagerank as dcg
     >>> chunksize = dcg.get_chunksize(edge_list.csv)
     """

     import os
     from glob import glob
     import math

     input_files = sorted(glob(str(input_path)))
     if len(input_files) == 1:
         size = os.path.getsize(input_files[0])
         chunksize = math.ceil(size/get_n_gpus())
     else:
         size = [os.path.getsize(_file) for _file in input_files]
         chunksize = max(size)
     return chunksize

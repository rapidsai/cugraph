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

'''
    def sort_values_binned(self, by):
        """Sorty by the given column and ensure that the same key
        doesn't spread across multiple partitions.
        """
        # Get sorted partitions
        parts = self.sort_values(by=by).to_delayed()

        # Get unique keys in each partition
        @delayed
        def get_unique(p):
            return set(p[by].unique())

        uniques = list(compute(*map(get_unique, parts)))

        joiner = {}
        for i in range(len(uniques)):
            joiner[i] = to_join = {}
            for j in range(i + 1, len(uniques)):
                intersect = uniques[i] & uniques[j]
                # If the keys intersect
                if intersect:
                    # Remove keys
                    uniques[j] -= intersect
                    to_join[j] = frozenset(intersect)
                else:
                    break

        @delayed
        def join(df, other, keys):
            others = [
                other.query("{by}==@k".format(by=by)) for k in sorted(keys)
            ]
            return cudf.concat([df] + others)

        @delayed
        def drop(df, keep_keys):
            locvars = locals()
            for i, k in enumerate(keep_keys):
                locvars["k{}".format(i)] = k

            conds = [
                "{by}==@k{i}".format(by=by, i=i) for i in range(len(keep_keys))
            ]
            expr = " or ".join(conds)
            return df.query(expr)

        for i in range(len(parts)):
            if uniques[i]:
                parts[i] = drop(parts[i], uniques[i])
                for joinee, intersect in joiner[i].items():
                    parts[i] = join(parts[i], parts[joinee], intersect)

        results = [p for i, p in enumerate(parts) if uniques[i]]
        return from_delayed(results, meta=self._meta).reset_index()
'''

def pagerank(input_graph):
    print("INSIDE DASK PAGERANK")
    client = default_client()
    _ddf = input_graph.edgelist.edgelist_df
    ddf = _ddf.sort_values(by='dst')
    data = DistributedDataHandler.create(data=ddf)

    comms = CommsContext(comms_p2p=False)
    comms.init(workers=data.workers)

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
    print(result)


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

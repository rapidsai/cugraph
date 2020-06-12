#
# Copyright (c) 2020, NVIDIA CORPORATION.
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


from collections.abc import Sequence

from collections import OrderedDict
from dask_cudf.core import DataFrame as dcDataFrame
from dask_cudf.core import Series as daskSeries

from cugraph.raft.dask.common.utils import get_client
from cugraph.dask.common.part_utils import _extract_partitions
from dask.distributed import default_client
from toolz import first

from functools import reduce


class DistributedDataHandler:
    """
    Class to centralize distributed data management. Functionalities include:
    - Data colocation
    - Worker information extraction
    - GPU futures extraction,

    Additional functionality can be added as needed. This class **does not**
    contain the actual data, just the metadata necessary to handle it,
    including common pieces of code that need to be performed to call
    Dask functions.

    The constructor is not meant to be used directly, but through the factory
    method DistributedDataHandler.create

    """

    def __init__(self, gpu_futures=None, workers=None,
                 datatype=None, multiple=False, client=None):
        self.client = get_client(client)
        self.gpu_futures = gpu_futures
        self.worker_to_parts = _workers_to_parts(gpu_futures)
        self.workers = workers
        self.datatype = datatype
        self.multiple = multiple
        self.worker_info = None
        self.total_rows = None
        self.ranks = None
        self.parts_to_sizes = None

    @classmethod
    def get_client(cls, client=None):
        return default_client() if client is None else client

    """ Class methods for initalization """

    @classmethod
    def create(cls, data, client=None):
        """
        Creates a distributed data handler instance with the given
        distributed data set(s).

        Parameters
        ----------

        data : dask.array, dask.dataframe, or unbounded Sequence of
               dask.array or dask.dataframe.

        client : dask.distributedClient
        """

        client = cls.get_client(client)

        multiple = isinstance(data, Sequence)

        if isinstance(first(data) if multiple else data,
                      (dcDataFrame, daskSeries)):
            datatype = 'cudf'
        else:
            raise Exception("Graph data must be dask-cudf dataframe")

        gpu_futures = client.sync(_extract_partitions, data, client)
        workers = tuple(set(map(lambda x: x[0], gpu_futures)))

        return DistributedDataHandler(gpu_futures=gpu_futures, workers=workers,
                                      datatype=datatype, multiple=multiple,
                                      client=client)

    """ Methods to calculate further attributes """

    def calculate_worker_and_rank_info(self, comms):

        self.worker_info = comms.worker_info(comms.worker_addresses)
        self.ranks = dict()

        for w, futures in self.worker_to_parts.items():
            self.ranks[w] = self.worker_info[w]["rank"]

    def calculate_parts_to_sizes(self, comms=None, ranks=None):

        if self.worker_info is None and comms is not None:
            self.calculate_worker_and_rank_info(comms)

        self.total_rows = 0

        self.parts_to_sizes = dict()

        parts = [(wf[0], self.client.submit(
            _get_rows,
            wf[1],
            self.multiple,
            workers=[wf[0]],
            pure=False))
            for idx, wf in enumerate(self.worker_to_parts.items())]

        sizes = self.client.compute(parts, sync=True)

        for w, sizes_parts in sizes:
            sizes, total = sizes_parts
            self.parts_to_sizes[self.worker_info[w]["rank"]] = \
                sizes

            self.total_rows += total


""" Internal methods, API subject to change """


def _workers_to_parts(futures):
    """
    Builds an ordered dict mapping each worker to their list
    of parts
    :param futures: list of (worker, part) tuples
    :return:
    """
    w_to_p_map = OrderedDict()
    for w, p in futures:
        if w not in w_to_p_map:
            w_to_p_map[w] = []
        w_to_p_map[w].append(p)
    return w_to_p_map


def _get_rows(objs, multiple):
    def get_obj(x): return x[0] if multiple else x
    total = list(map(lambda x: get_obj(x).shape[0], objs))
    return total, reduce(lambda a, b: a + b, total)

# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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


from collections.abc import Sequence

from collections import OrderedDict
from dask_cudf.core import DataFrame as dcDataFrame
from dask_cudf.core import Series as daskSeries

import cugraph.dask.comms.comms as Comms

# FIXME: this raft import breaks the library if ucx-py is
# not available. They are necessary only when doing MG work.
from cugraph.dask.common.read_utils import MissingUCXPy

try:
    from raft_dask.common.utils import get_client
except ImportError as err:
    # FIXME: Generalize since err.name is arr when
    # libnuma.so.1 is not available
    if err.name == "ucp" or err.name == "arr":
        get_client = MissingUCXPy()
    else:
        raise
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

    def __init__(
        self, gpu_futures=None, workers=None, datatype=None, multiple=False, client=None
    ):
        self.client = get_client(client)
        self.gpu_futures = gpu_futures
        self.worker_to_parts = _workers_to_parts(gpu_futures)
        self.workers = workers
        self.datatype = datatype
        self.multiple = multiple
        self.worker_info = None
        self.total_rows = None
        self.max_vertex_id = None
        self.ranks = None
        self.parts_to_sizes = None
        self.local_data = None

    @classmethod
    def get_client(cls, client=None):
        return default_client() if client is None else client

    """ Class methods for initalization """

    @classmethod
    def create(cls, data, client=None, batch_enabled=False):
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

        if isinstance(first(data) if multiple else data, (dcDataFrame, daskSeries)):
            datatype = "cudf"
        else:
            raise Exception("Graph data must be dask-cudf dataframe")

        gpu_futures = client.sync(
            _extract_partitions, data, client, batch_enabled=batch_enabled
        )
        workers = tuple(OrderedDict.fromkeys(map(lambda x: x[0], gpu_futures)))
        return DistributedDataHandler(
            gpu_futures=gpu_futures,
            workers=workers,
            datatype=datatype,
            multiple=multiple,
            client=client,
        )

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

        parts = [
            (
                wf[0],
                self.client.submit(
                    _get_rows, wf[1], self.multiple, workers=[wf[0]], pure=False
                ),
            )
            for idx, wf in enumerate(self.worker_to_parts.items())
        ]

        sizes = self.client.compute(parts, sync=True)

        for w, sizes_parts in sizes:
            sizes, total = sizes_parts
            self.parts_to_sizes[self.worker_info[w]["rank"]] = sizes

            self.total_rows += total

    def calculate_local_data(self, comms, by):

        if self.worker_info is None and comms is not None:
            self.calculate_worker_and_rank_info(comms)

        local_data = dict(
            [
                (
                    self.worker_info[wf[0]]["rank"],
                    self.client.submit(_get_local_data, wf[1], by, workers=[wf[0]]),
                )
                for idx, wf in enumerate(self.worker_to_parts.items())
            ]
        )

        _local_data_dict = self.client.compute(local_data, sync=True)
        local_data_dict = {"edges": [], "offsets": [], "verts": []}
        max_vid = 0
        for rank in range(len(_local_data_dict)):
            data = _local_data_dict[rank]
            local_data_dict["edges"].append(data[0])
            if rank == 0:
                local_offset = 0
            else:
                prev_data = _local_data_dict[rank - 1]
                local_offset = prev_data[1] + 1
            local_data_dict["offsets"].append(local_offset)
            local_data_dict["verts"].append(data[1] - local_offset + 1)
            if data[2] > max_vid:
                max_vid = data[2]

        import numpy as np

        local_data_dict["edges"] = np.array(local_data_dict["edges"], dtype=np.int32)
        local_data_dict["offsets"] = np.array(
            local_data_dict["offsets"], dtype=np.int32
        )
        local_data_dict["verts"] = np.array(local_data_dict["verts"], dtype=np.int32)
        self.local_data = local_data_dict
        self.max_vertex_id = max_vid


def _get_local_data(df, by):
    df = df[0]
    num_local_edges = len(df)
    local_by_max = df[by].iloc[-1]
    local_max = df[["src", "dst"]].max().max()
    return num_local_edges, local_by_max, local_max


""" Internal methods, API subject to change """


def _workers_to_parts(futures):
    """
    Builds an ordered dict mapping each worker to their list
    of parts
    :param futures: list of (worker, part) tuples
    :return:
    """
    w_to_p_map = OrderedDict.fromkeys(Comms.get_workers())
    for w, p in futures:
        if w_to_p_map[w] is None:
            w_to_p_map[w] = []
        w_to_p_map[w].append(p)
    keys_to_delete = [w for (w, p) in w_to_p_map.items() if p is None]
    for k in keys_to_delete:
        del w_to_p_map[k]
    return w_to_p_map


def _get_rows(objs, multiple):
    def get_obj(x):
        return x[0] if multiple else x

    total = list(map(lambda x: get_obj(x).shape[0], objs))
    return total, reduce(lambda a, b: a + b, total)


def get_mg_batch_data(dask_cudf_data, batch_enabled=False):
    data = DistributedDataHandler.create(
        data=dask_cudf_data, batch_enabled=batch_enabled
    )
    return data


def get_distributed_data(input_ddf):
    ddf = input_ddf
    comms = Comms.get_comms()
    data = DistributedDataHandler.create(data=ddf)
    if data.worker_info is None and comms is not None:
        data.calculate_worker_and_rank_info(comms)
    return data


def get_vertex_partition_offsets(input_graph):
    import cudf

    renumber_vertex_count = input_graph.renumber_map.implementation.ddf.map_partitions(
        len
    ).compute()
    renumber_vertex_cumsum = renumber_vertex_count.cumsum()
    # Assume the input_graph edgelist was renumbered
    src_col_name = input_graph.renumber_map.renumbered_src_col_name
    vertex_dtype = input_graph.edgelist.edgelist_df[src_col_name].dtype
    vertex_partition_offsets = cudf.Series([0], dtype=vertex_dtype)
    vertex_partition_offsets = vertex_partition_offsets.append(
        cudf.Series(renumber_vertex_cumsum, dtype=vertex_dtype)
    )
    return vertex_partition_offsets

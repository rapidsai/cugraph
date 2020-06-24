# Copyright (c) 2019, NVIDIA CORPORATION.
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

import numpy as np
from tornado import gen
from collections import Sequence
from dask.distributed import futures_of, default_client, wait
from toolz import first
from collections import OrderedDict
import dask_cudf as dc
from dask.array.core import Array as daskArray
from dask_cudf.core import DataFrame as daskDataFrame
from dask_cudf.core import Series as daskSeries


'''
def hosts_to_parts(futures):
    """
    Builds an ordered dict mapping each host to their list
    of parts
    :param futures: list of (worker, part) tuples
    :return:
    """
    w_to_p_map = OrderedDict()
    for w, p in futures:
        host, port = parse_host_port(w)
        host_key = (host, port)
        if host_key not in w_to_p_map:
            w_to_p_map[host_key] = []
        w_to_p_map[host_key].append(p)
    return w_to_p_map


def workers_to_parts(futures):
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


def _func_get_rows(df):
    return df.shape[0]


def parts_to_ranks(client, worker_info, part_futures):
    """
    Builds a list of (rank, size) tuples of partitions
    :param worker_info: dict of {worker, {"rank": rank }}. Note: \
        This usually comes from the underlying communicator
    :param part_futures: list of (worker, future) tuples
    :return: [(part, size)] in the same order of part_futures
    """
    futures = [(worker_info[wf[0]]["rank"],
                client.submit(_func_get_rows,
                              wf[1],
                              workers=[wf[0]],
                              pure=False))
               for idx, wf in enumerate(part_futures)]

    sizes = client.compute(list(map(lambda x: x[1], futures)), sync=True)
    total = reduce(lambda a, b: a + b, sizes)

    return [(futures[idx][0], size) for idx, size in enumerate(sizes)], total


def _default_part_getter(f, idx): return f[idx]


def flatten_grouped_results(client, gpu_futures,
                            worker_results_map,
                            getter_func=_default_part_getter):
    """
    This function is useful when a series of partitions have been grouped by
    the worker responsible for the data and the resulting partitions are
    stored on each worker as a list. This happens when a communications
    implementation is used which does not allow multiple ranks per device, so
    the partitions need to be grouped on the ranks to be processed concurrently
    using different streams.

    :param client: Dask client
    :param gpu_futures: [(future, part)] worker to part list of tuples
    :param worker_results_map: { rank: future } where future is a list
           of data partitions on a Dask worker
    :param getter_func: a function that takes a future and partition index
           as arguments and returns the data for a specific partitions
    :return: the ordered list of futures holding each partition on the workers
    """
    futures = []
    completed_part_map = {}
    for rank, part in gpu_futures:
        if rank not in completed_part_map:
            completed_part_map[rank] = 0

        f = worker_results_map[rank]

        futures.append(client.submit(
            getter_func, f, completed_part_map[rank]))

        completed_part_map[rank] += 1

    return futures
'''


@gen.coroutine
def _extract_partitions(dask_obj, client=None):

    client = default_client() if client is None else client

    # dask.dataframe or dask.array
    if isinstance(dask_obj, (daskDataFrame, daskArray, daskSeries)):
        persisted = client.persist(dask_obj)
        parts = futures_of(persisted)

    # iterable of dask collections (need to colocate them)
    elif isinstance(dask_obj, Sequence):
        # NOTE: We colocate (X, y) here by zipping delayed
        # n partitions of them as (X1, y1), (X2, y2)...
        # and asking client to compute a single future for
        # each tuple in the list
        dela = [np.asarray(d.to_delayed()) for d in dask_obj]

        # TODO: ravel() is causing strange behavior w/ delayed Arrays which are
        # not yet backed by futures. Need to investigate this behavior.
        # ref: https://github.com/rapidsai/cuml/issues/2045
        raveled = [d.flatten() for d in dela]
        parts = client.compute([p for p in zip(*raveled)])

    yield wait(parts)
    key_to_part = [(str(part.key), part) for part in parts]
    who_has = yield client.who_has(parts)
    raise gen.Return([(first(who_has[key]), part)
                      for key, part in key_to_part])


def create_dict(futures):
    w_to_p_map = OrderedDict()
    for w, k, p in futures:
        if w not in w_to_p_map:
            w_to_p_map[w] = []
        w_to_p_map[w].append([p, k])
    return w_to_p_map


def set_global_index(df, cumsum):
    df.index = df.index + cumsum
    return df


def get_cumsum(df, by):
    return df[by].value_counts(sort=False).cumsum()


def repartition(ddf, cumsum):
    import math
    npartitions = ddf.npartitions
    count = math.ceil(len(ddf)/npartitions)
    new_divisions = [0]
    move_count = 0
    for i in range(npartitions-1):
        index = -1
        while(cumsum[i].iloc[index] > count - move_count > cumsum[i].iloc[0]):
            index = index - 1
        new_divisions.append(new_divisions[i] +
                             cumsum[i].iloc[index] +
                             move_count)
        move_count = cumsum[i].iloc[-1] - cumsum[i].iloc[index]
    new_divisions.append(new_divisions[i+1] +
                         cumsum[-1].iloc[-1] +
                         move_count - 1)
    return ddf.repartition(divisions=tuple(new_divisions))


def load_balance_func(ddf_, by, client=None):

    client = default_client() if client is None else client

    persisted = client.persist(ddf_)
    parts = futures_of(persisted)
    wait(parts)

    who_has = client.who_has(parts)
    key_to_part = [(str(part.key), part) for part in parts]
    gpu_fututres = [(first(who_has[key]),
                     part.key[1], part) for key, part in key_to_part]
    worker_to_data = create_dict(gpu_fututres)

    cumsum_parts = [client.submit(get_cumsum,
                    wf[1][0][0],
                    by,
                    workers=[wf[0]]).result()
                    for idx, wf in enumerate(worker_to_data.items())]

    num_rows = []
    for cumsum in cumsum_parts:
        num_rows.append(cumsum.iloc[-1])

    divisions = [sum(num_rows[0:x:1]) for x in range(0, len(num_rows) + 1)]
    divisions[-1] = divisions[-1] - 1
    divisions = tuple(divisions)

    futures = [client.submit(set_global_index,
               wf[1][0][0],
               divisions[wf[1][0][1]],
               workers=[wf[0]])
               for idx, wf in enumerate(worker_to_data.items())]
    wait(futures)

    ddf = dc.from_delayed(futures)
    ddf.divisions = divisions
    ddf = repartition(ddf, cumsum_parts)
    return ddf

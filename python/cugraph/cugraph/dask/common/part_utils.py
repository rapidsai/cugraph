# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

import numpy as np
from dask.distributed import futures_of, default_client, wait
from toolz import first
import collections
import dask_cudf
from dask.array.core import Array as daskArray
from dask_cudf.core import DataFrame as daskDataFrame
from dask_cudf.core import Series as daskSeries
from functools import reduce
import cugraph.dask.comms.comms as Comms
from dask.delayed import delayed
import cudf


def workers_to_parts(futures):
    """
    Builds an ordered dict mapping each worker to their list
    of parts
    :param futures: list of (worker, part) tuples
    :return:
    """
    w_to_p_map = collections.OrderedDict()
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
    futures = [
        (
            worker_info[wf[0]]["rank"],
            client.submit(_func_get_rows, wf[1], workers=[wf[0]], pure=False),
        )
        for idx, wf in enumerate(part_futures)
    ]

    sizes = client.compute(list(map(lambda x: x[1], futures)), sync=True)
    total = reduce(lambda a, b: a + b, sizes)

    return [(futures[idx][0], size) for idx, size in enumerate(sizes)], total


def persist_distributed_data(dask_df, client):
    client = default_client() if client is None else client
    worker_addresses = Comms.get_workers()
    _keys = dask_df.__dask_keys__()
    worker_dict = {}
    for i, key in enumerate(_keys):
        worker_dict[str(key)] = tuple([worker_addresses[i]])
    persisted = client.persist(dask_df, workers=worker_dict)
    parts = futures_of(persisted)
    return parts


async def _extract_partitions(dask_obj, client=None, batch_enabled=False):
    client = default_client() if client is None else client
    worker_list = Comms.get_workers()

    # dask.dataframe or dask.array
    if isinstance(dask_obj, (daskDataFrame, daskArray, daskSeries)):
        # parts = persist_distributed_data(dask_obj, client)
        # FIXME: persist data to the same worker when batch_enabled=True
        if batch_enabled:
            persisted = client.persist(dask_obj, workers=worker_list[0])
        else:
            # repartition the 'dask_obj' to get as many partitions as there
            # are workers
            dask_obj = dask_obj.repartition(npartitions=len(worker_list))
            # Have the first n workers persisting the n partitions
            # Ideally, there would be as many partitions as there are workers
            persisted = [
                client.persist(dask_obj.get_partition(p), workers=w)
                for p, w in enumerate(worker_list[: dask_obj.npartitions])
            ]
            # Persist empty dataframe/series with the remaining workers if
            # there are less partitions than workers
            if dask_obj.npartitions < len(worker_list):
                # The empty df should have the same column names and dtypes as
                # dask_obj
                if isinstance(dask_obj, dask_cudf.DataFrame):
                    empty_df = cudf.DataFrame(columns=list(dask_obj.columns))
                    empty_df = empty_df.astype(
                        dict(zip(dask_obj.columns, dask_obj.dtypes))
                    )
                else:
                    empty_df = cudf.Series(dtype=dask_obj.dtype)

                for p, w in enumerate(worker_list[dask_obj.npartitions :]):
                    empty_ddf = dask_cudf.from_cudf(empty_df, npartitions=1)
                    persisted.append(client.persist(empty_ddf, workers=w))

        parts = futures_of(persisted)
    # iterable of dask collections (need to colocate them)
    elif isinstance(dask_obj, collections.abc.Sequence):
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

    await wait(parts)
    key_to_part = [(str(part.key), part) for part in parts]
    who_has = await client.who_has(parts)
    return [(first(who_has[key]), part) for key, part in key_to_part]


def create_dict(futures):
    w_to_p_map = collections.OrderedDict()
    for w, k, p in futures:
        if w not in w_to_p_map:
            w_to_p_map[w] = []
        w_to_p_map[w].append([p, k])
    return w_to_p_map


def set_global_index(df, cumsum):
    df.index = df.index + cumsum
    df.index = df.index.astype("int64")
    return df


def get_cumsum(df, by):
    return df[by].value_counts(sort=False).cumsum()


def repartition(ddf, cumsum):
    # Calculate new optimal divisions and repartition the data
    # for load balancing.

    import math

    npartitions = ddf.npartitions
    count = math.ceil(len(ddf) / npartitions)
    new_divisions = [0]
    move_count = 0
    i = npartitions - 2
    for i in range(npartitions - 1):
        search_val = count - move_count
        index = cumsum[i].searchsorted(search_val)
        if index == len(cumsum[i]):
            index = -1
        elif index > 0:
            left = cumsum[i].iloc[index - 1]
            right = cumsum[i].iloc[index]
            index -= search_val - left < right - search_val
        new_divisions.append(new_divisions[i] + cumsum[i].iloc[index] + move_count)
        move_count = cumsum[i].iloc[-1] - cumsum[i].iloc[index]
    new_divisions.append(new_divisions[i + 1] + cumsum[-1].iloc[-1] + move_count - 1)

    return ddf.repartition(divisions=tuple(new_divisions))


def load_balance_func(ddf_, by, client=None):
    # Load balances the sorted dask_cudf DataFrame.
    # Input is a dask_cudf dataframe ddf_ which is sorted by
    # the column name passed as the 'by' argument.

    client = default_client() if client is None else client

    parts = persist_distributed_data(ddf_, client)
    wait(parts)

    who_has = client.who_has(parts)
    key_to_part = [(str(part.key), part) for part in parts]
    gpu_fututres = [
        (first(who_has[key]), part.key[1], part) for key, part in key_to_part
    ]
    worker_to_data = create_dict(gpu_fututres)

    # Calculate cumulative sum in each dataframe partition
    cumsum_parts = [
        client.submit(get_cumsum, wf[1][0][0], by, workers=[wf[0]]).result()
        for idx, wf in enumerate(worker_to_data.items())
    ]

    num_rows = []
    for cumsum in cumsum_parts:
        num_rows.append(cumsum.iloc[-1])

    # Calculate current partition divisions
    divisions = [sum(num_rows[0:x:1]) for x in range(0, len(num_rows) + 1)]
    divisions[-1] = divisions[-1] - 1
    divisions = tuple(divisions)

    # Set global index from 0 to len(dask_cudf_dataframe) so that global
    # indexing of divisions can be used for repartitioning.
    futures = [
        client.submit(
            set_global_index, wf[1][0][0], divisions[wf[1][0][1]], workers=[wf[0]]
        )
        for idx, wf in enumerate(worker_to_data.items())
    ]
    wait(futures)

    ddf = dask_cudf.from_delayed(futures)
    ddf.divisions = divisions

    # Repartition the data
    ddf = repartition(ddf, cumsum_parts)

    return ddf


def concat_dfs(df_list):
    """
    Concat a list of cudf dataframes
    """
    return cudf.concat(df_list)


def get_delayed_dict(ddf):
    """
    Returns a dicitionary with the dataframe tasks as keys and
    the dataframe delayed objects as values
    """
    df_delayed = {}
    for delayed_obj in ddf.to_delayed():
        df_delayed[str(delayed_obj.key)] = delayed_obj
    return df_delayed


def concat_within_workers(client, ddf):
    """
    Concats all partitions within workers without transfers
    """
    df_delayed = get_delayed_dict(ddf)

    result = []
    for worker, tasks in client.has_what().items():
        worker_task_list = []

        for task in list(tasks):
            if task in df_delayed:
                worker_task_list.append(df_delayed[task])
        concat_tasks = delayed(concat_dfs)(worker_task_list)
        result.append(client.persist(collections=concat_tasks, workers=worker))

    return dask_cudf.from_delayed(result)

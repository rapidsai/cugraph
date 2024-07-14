# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
from dask_cudf import DataFrame as daskDataFrame
from dask_cudf import Series as daskSeries
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
        worker_dict[key] = tuple([worker_addresses[i]])
    persisted = client.persist(dask_df, workers=worker_dict)
    parts = futures_of(persisted)
    return parts


def _create_empty_dask_df_future(meta_df, client, worker):
    df_future = client.scatter(meta_df.head(0), workers=[worker])
    wait(df_future)
    return [df_future]


def get_persisted_df_worker_map(dask_df, client):
    ddf_keys = futures_of(dask_df)
    output_map = {}
    for w, w_keys in client.has_what().items():
        output_map[w] = [ddf_k for ddf_k in ddf_keys if ddf_k.key in w_keys]
        if len(output_map[w]) == 0:
            output_map[w] = _create_empty_dask_df_future(dask_df._meta, client, w)
    return output_map


def _chunk_lst(ls, num_parts):
    return [ls[i::num_parts] for i in range(num_parts)]


def persist_dask_df_equal_parts_per_worker(
    dask_df, client, return_type="dask_cudf.DataFrame"
):
    """
    Persist dask_df with equal parts per worker
    Args:
        dask_df: dask_cudf.DataFrame
        client: dask.distributed.Client
        return_type: str, "dask_cudf.DataFrame" or "dict"
    Returns:
        persisted_keys: dict of {worker: [persisted_keys]}
    """
    if return_type not in ["dask_cudf.DataFrame", "dict"]:
        raise ValueError("return_type must be either 'dask_cudf.DataFrame' or 'dict'")

    ddf_keys = dask_df.to_delayed()
    rank_to_worker = Comms.rank_to_worker(client)
    # rank-worker mappings are in ascending order
    workers = dict(sorted(rank_to_worker.items())).values()

    ddf_keys_ls = _chunk_lst(ddf_keys, len(workers))
    persisted_keys_d = {}
    for w, ddf_k in zip(workers, ddf_keys_ls):
        persisted_keys_d[w] = client.compute(
            ddf_k, workers=w, allow_other_workers=False, pure=False
        )

    persisted_keys_ls = [
        item for sublist in persisted_keys_d.values() for item in sublist
    ]
    wait(persisted_keys_ls)
    if return_type == "dask_cudf.DataFrame":
        dask_df = dask_cudf.from_delayed(
            persisted_keys_ls, meta=dask_df._meta
        ).persist()
        wait(dask_df)
        return dask_df

    return persisted_keys_d


def get_length_of_parts(persisted_keys_d, client):
    """
    Get the length of each partition
    Args:
        persisted_keys_d: dict of {worker: [persisted_keys]}
        client: dask.distributed.Client
    Returns:
        length_of_parts: dict of {worker: [length_of_parts]}
    """
    length_of_parts = {}
    for w, p_keys in persisted_keys_d.items():
        length_of_parts[w] = [
            client.submit(
                len, p_key, pure=False, workers=[w], allow_other_workers=False
            )
            for p_key in p_keys
        ]

    for w, len_futures in length_of_parts.items():
        length_of_parts[w] = client.gather(len_futures)
    return length_of_parts


async def _extract_partitions(
    dask_obj, client=None, batch_enabled=False, broadcast_worker=None
):
    client = default_client() if client is None else client
    worker_list = Comms.get_workers()

    # dask.dataframe or dask.array
    if isinstance(dask_obj, (daskDataFrame, daskArray, daskSeries)):
        if batch_enabled:
            persisted = client.persist(dask_obj, workers=broadcast_worker)
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
        # each tuple in the list.
        dela = [np.asarray(d.to_delayed()) for d in dask_obj]

        # TODO: ravel() is causing strange behavior w/ delayed Arrays which are
        # not yet backed by futures. Need to investigate this behavior.
        # ref: https://github.com/rapidsai/cuml/issues/2045
        raveled = [d.flatten() for d in dela]
        parts = client.compute([p for p in zip(*raveled)])

    await wait(parts)
    key_to_part = [(part.key, part) for part in parts]
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
    key_to_part = [(part.key, part) for part in parts]
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

    # Calculate current partition divisions.
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
    Concat a list of cudf dataframes.
    """
    return cudf.concat(df_list)


def get_delayed_dict(ddf):
    """
    Returns a dicitionary with the dataframe tasks as keys and
    the dataframe delayed objects as values.
    """
    df_delayed = {}
    for delayed_obj in ddf.to_delayed():
        df_delayed[delayed_obj.key] = delayed_obj
    return df_delayed


def concat_within_workers(client, ddf):
    """
    Concats all partitions within workers without transfers.
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

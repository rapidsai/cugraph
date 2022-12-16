# Copyright (c) 2022, NVIDIA CORPORATION.
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

from __future__ import annotations
from cugraph.utilities.utils import import_optional
import cupy as cp
import cudf
import numpy as np
import dask.array as da
from dask.distributed import get_client

torch = import_optional("torch")
F = import_optional("dgl.backend")


# Feature Tensor to DataFrame Utils
def convert_to_column_major(t: torch.Tensor):
    return t.t().contiguous().t()


def create_ar_from_tensor(t: torch.Tensor):
    t = convert_to_column_major(t)
    if t.device.type == "cuda":
        ar = cp.from_dlpack(F.zerocopy_to_dlpack(t))
    else:
        ar = t.numpy()
    return ar


def create_df_from_ar(ar, feat_key="default", single_gpu=True):
    if not single_gpu:
        n_workers = len(get_client().scheduler_info()["workers"])
        n_partitions = n_workers * 2

    if ar.ndim == 2:
        n_rows, n_cols = ar.shape
    else:
        n_rows = ar.shape[0]
        n_cols = 1
    feat_columns = [f"{feat_key}_{i}" for i in range(n_cols)]

    if single_gpu:
        ar = cp.asarray(ar)
        if n_cols == 1:
            df = cudf.Series(data=ar, name=feat_columns[0]).to_frame()
        else:
            df = cudf.DataFrame(data=ar, columns=feat_columns)
    else:
        if n_cols == 1:
            ar = ar.reshape(-1, 1)
        chunksize = (n_rows + n_partitions - 1) // n_partitions
        # converting with meta because
        # dd.from_dask_array results in host frames
        # https://github.com/rapidsai/cudf/issues/9029
        meta_df = cudf.DataFrame(
            data=np.ones(shape=(1, n_cols), dtype=ar.dtype), columns=feat_columns
        )
        ar = da.from_array(ar, chunks=(chunksize, -1)).map_blocks(cp.asarray)
        df = ar.to_dask_dataframe(meta=meta_df, columns=feat_columns)

        def set_columns(df, feat_columns):
            df.columns = feat_columns
            return df

        df = df.map_partitions(set_columns, feat_columns)

    df = df.reset_index(drop=True)
    return df, feat_columns

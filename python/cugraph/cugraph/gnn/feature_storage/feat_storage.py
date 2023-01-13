# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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


from collections import defaultdict
import cudf
import cupy as cp
import numpy as np
import dask_cudf
from dask import dataframe as dd
import dask
import pandas as pd


class FeatureStore:
    def __init__(self, backend="numpy", client=None):
        self.fd = defaultdict(dict)
        if backend in ["dask_numpy", "dask_cupy"] and client is None:
            raise ValueError(f"Please provide dask client for backend={type(backend)}")
        self.backend = backend

        self._client = client

    def add_feat_from_df(self, feat_obj, type_name, feat_name):
        self.fd[type_name][feat_name] = self.cast_feat_obj_to_backend(
            feat_obj, self.backend
        )

    @staticmethod
    def cast_feat_obj_to_backend(feat_obj, backend):
        if backend == "cupy":
            if isinstance(feat_obj, (cudf.DataFrame, pd.DataFrame)):
                return _cast_to_cupy_ar(feat_obj.values)
            elif isinstance(feat_obj, (dask_cudf.DataFrame, dd.DataFrame)):
                return _cast_to_cupy_ar(feat_obj.values.compute())
            elif isinstance(feat_obj, (np.ndarray, cp.ndarray)):
                return _cast_to_cupy_ar(feat_obj)
            else:
                raise ValueError(f"feat_obj of {type(feat_obj)} is not supported")
        elif backend == "numpy":
            if isinstance(feat_obj, (cudf.DataFrame, pd.DataFrame)):
                return _cast_to_numpy_ar(feat_obj.values)
            elif isinstance(feat_obj, (dask_cudf.DataFrame, dd.DataFrame)):
                return feat_obj.values.map_blocks(_cast_to_numpy_ar).compute()
            elif isinstance(feat_obj, (np.ndarray, cp.ndarray)):
                return _cast_to_numpy_ar(feat_obj)
            else:
                raise ValueError(f"feat_obj of {type(feat_obj)} is not supported")
        elif backend == "dask_numpy":
            if isinstance(feat_obj, (cudf.DataFrame, pd.DataFrame)):
                return _create_dask_ar_from_ar(feat_obj.values, array_type="numpy")
            elif isinstance(feat_obj, (dask_cudf.DataFrame, dd.DataFrame)):
                return feat_obj.values.map_blocks(_cast_to_numpy_ar)
            else:
                raise ValueError(f"feat_obj of {type(feat_obj)} is not supported")

        elif backend == "dask_cupy":
            if isinstance(feat_obj, (cudf.DataFrame, pd.DataFrame)):
                return _create_dask_ar_from_ar(feat_obj.values, array_type="cupy")
            elif isinstance(feat_obj, (dask_cudf.DataFrame, dd.DataFrame)):
                return feat_obj.values.map_blocks(_cast_to_cupy_ar)
            else:
                raise ValueError(f"feat_obj of {type(feat_obj)} is not supported")
        else:
            raise ValueError(f"backend {backend} is not supported")

    def get_data(self, indices, type_name, feat_name, compute_results=True):
        ar = self.fd[type_name][feat_name]
        if isinstance(ar, dask.array.core.Array):
            # sort indices first to prevent shuffling at dask layer
            # 70x speedup
            indices_args = indices.argsort()
            # sort indices
            indices = indices.take(indices_args, axis=0)
            ar = ar[indices].compute()
            # unsort result to orignal requested order
            ar = ar.take(indices_args.argsort(), axis=0)
            return ar
        else:
            return ar[indices]

    def persist_feat_data(self):
        if self._client:
            n_workers = len(self._client.scheduler_info()["workers"])
            nparts = n_workers * 2
        else:
            nparts = 1
        self.fd = {
            type_n: {fname: _persist_ar(ar, nparts) for fname, ar in type_d.items()}
            for type_n, type_d in self.fd.items()
        }


def get_evenly_divided_values(value_to_be_distributed, times):
    return [
        value_to_be_distributed // times + int(x < value_to_be_distributed % times)
        for x in range(times)
    ]


def _repartition_dask_ar(ar, nparts):
    chunk1, chunk2 = ar.chunks
    new_chunk1 = get_evenly_divided_values(sum(chunk1), nparts)
    new_chunks = tuple(new_chunk1), chunk2
    ar = ar.rechunk(new_chunks).persist()
    ar = ar.compute_chunk_sizes()
    return ar


def _create_dask_ar_from_ar(ar, array_type="numpy"):
    raise NotImplementedError


def _persist_ar(ar, nparts):
    if hasattr(ar, "persist"):
        ar = ar.persist()
        ar = ar.compute_chunk_sizes()
        ar = _repartition_dask_ar(ar, nparts)
    return ar


def _cast_to_numpy_ar(ar):
    if isinstance(ar, cp.ndarray):
        ar = ar.get()
    else:
        ar = np.asarray(ar)
    return ar


def _cast_to_cupy_ar(ar):
    return cp.asarray(ar)

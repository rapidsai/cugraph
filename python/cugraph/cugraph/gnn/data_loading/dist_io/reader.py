# Copyright (c) 2024, NVIDIA CORPORATION.
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


import os
import re

import cudf

from typing import Callable, Iterator, Tuple, Dict, Optional

from cugraph.utilities.utils import import_optional, MissingModule

# Prevent PyTorch from being imported and causing an OOM error
torch = MissingModule("torch")


class DistSampleReader:
    def __init__(
        self,
        directory: str,
        *,
        format: str = "parquet",
        rank: Optional[int] = None,
        filelist=None,
    ):
        torch = import_optional("torch")

        self.__format = format
        self.__directory = directory

        if format != "parquet":
            raise ValueError("Invalid format (currently supported: 'parquet')")

        if filelist is None:
            files = os.listdir(directory)
            ex = re.compile(r"batch\=([0-9]+)\.([0-9]+)\-([0-9]+)\.([0-9]+)\.parquet")
            filematch = [ex.match(f) for f in files]
            filematch = [f for f in filematch if f]

            if rank is not None:
                filematch = [f for f in filematch if int(f[1]) == rank]

            batch_count = sum([int(f[4]) - int(f[2]) + 1 for f in filematch])
            filematch = sorted(filematch, key=lambda f: int(f[2]), reverse=True)

            self.__files = filematch
        else:
            self.__files = list(filelist)

        if rank is None:
            self.__batch_count = batch_count
        else:
            # TODO maybe remove this in favor of warning users that they are
            # probably going to cause a hang, instead of attempting to resolve
            # the hang for them by dropping batches.
            batch_count = torch.tensor([batch_count], device="cuda")
            torch.distributed.all_reduce(batch_count, torch.distributed.ReduceOp.MIN)
            self.__batch_count = int(batch_count)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[Dict[str, "torch.Tensor"], int, int]:
        torch = import_optional("torch")

        if len(self.__files) > 0:
            f = self.__files.pop()
            fname = f[0]
            start_inclusive = int(f[2])
            end_inclusive = int(f[4])

            if (end_inclusive - start_inclusive + 1) > self.__batch_count:
                end_inclusive = start_inclusive + self.__batch_count - 1
                self.__batch_count = 0
            else:
                self.__batch_count -= end_inclusive - start_inclusive + 1

            df = cudf.read_parquet(os.path.join(self.__directory, fname))
            tensors = {}
            for col in list(df.columns):
                s = df[col].dropna()
                if len(s) > 0:
                    tensors[col] = torch.as_tensor(s, device="cuda")
                df.drop(col, axis=1, inplace=True)

            return tensors, start_inclusive, end_inclusive

        raise StopIteration


class BufferedSampleReader:
    def __init__(
        self,
        nodes_call_groups: list["torch.Tensor"],
        sample_fn: Callable[..., Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]],
        *args,
        **kwargs,
    ):
        self.__sample_args = args
        self.__sample_kwargs = kwargs

        self.__nodes_call_groups = iter(nodes_call_groups)
        self.__sample_fn = sample_fn
        self.__current_call_id = 0
        self.__current_reader = None

    def __next__(self) -> Tuple[Dict[str, "torch.Tensor"], int, int]:
        new_reader = False

        if self.__current_reader is None:
            new_reader = True
        else:
            try:
                out = next(self.__current_reader)
            except StopIteration:
                new_reader = True

        if new_reader:
            # Will trigger StopIteration if there are no more call groups
            self.__current_reader = self.__sample_fn(
                self.__current_call_id,
                next(self.__nodes_call_groups),
                *self.__sample_args,
                **self.__sample_kwargs,
            )

            self.__current_call_id += 1
            out = next(self.__current_reader)

        return out

    def __iter__(self) -> Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]:
        return self

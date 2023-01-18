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


def get_n_workers():
    from dask.distributed import default_client

    client = default_client()
    return len(client.scheduler_info()["workers"])


def get_chunksize(input_path):
    """
    Calculate the appropriate chunksize for dask_cudf.read_csv
    to get a number of partitions equal to the number of GPUs.

    Examples
    --------
    >>> import cugraph.dask as dcg
    >>> chunksize = dcg.get_chunksize(datasets_path / 'netscience.csv')

    """

    import os
    from glob import glob
    import math

    input_files = sorted(glob(str(input_path)))
    if len(input_files) == 1:
        size = os.path.getsize(input_files[0])
        chunksize = math.ceil(size / get_n_workers())
    else:
        size = [os.path.getsize(_file) for _file in input_files]
        chunksize = max(size)
    return chunksize


class MissingUCXPy:
    def __call__(self, *args, **kwargs):
        raise ModuleNotFoundError(
            "ucx-py could not be imported but is required for MG operations"
        )

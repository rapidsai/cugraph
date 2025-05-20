# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

import cudf
import cupy


from typing import List, Dict


def create_df_from_disjoint_series(series_list: List[cudf.Series]):
    series_list.sort(key=lambda s: len(s), reverse=True)

    df = cudf.DataFrame()
    for s in series_list:
        df[s.name] = s

    return df


def create_df_from_disjoint_arrays(array_dict: Dict[str, cupy.array]):
    series_dict = {}
    for k in list(array_dict.keys()):
        if array_dict[k] is not None:
            series_dict[k] = cudf.Series(array_dict[k], name=k)

    return create_df_from_disjoint_series(list(series_dict.values()))

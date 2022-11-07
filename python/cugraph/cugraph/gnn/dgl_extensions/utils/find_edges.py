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

import cudf
import dask_cudf
from cugraph.experimental import PropertyGraph

src_n = PropertyGraph.src_col_name
dst_n = PropertyGraph.dst_col_name
type_n = PropertyGraph.type_col_name


def find_edges(pg, edge_ids_cap, etype):
    edge_ids = cudf.from_dlpack(edge_ids_cap)
    subset_df = pg.get_edge_data(edge_ids=edge_ids, columns=type_n, types=[etype])
    if isinstance(subset_df, dask_cudf.DataFrame):
        subset_df = subset_df.compute()
    return subset_df[src_n].to_dlpack(), subset_df[dst_n].to_dlpack()

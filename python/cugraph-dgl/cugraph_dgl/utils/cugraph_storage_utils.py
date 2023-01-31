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
import numpy as np
from cugraph.gnn.dgl_extensions.utils.sampling import eid_n, src_n, dst_n
from cugraph.utilities.utils import import_optional, MissingModule

dgl = import_optional("dgl")
F = import_optional("dgl.backend")


def _assert_valid_canonical_etype(canonical_etype):
    if not _is_valid_canonical_etype:
        error_message = (
            f"Invalid canonical_etype {canonical_etype} "
            + "canonical etype should be is a string triplet (str, str, str)"
            + "for source node type, edge type and destination node type"
        )
        raise dgl.DGLError(error_message)


def _is_valid_canonical_etype(canonical_etype):
    if not isinstance(canonical_etype, tuple):
        return False

    if len(canonical_etype) != 3:
        return False

    for t in canonical_etype:
        if not isinstance(t, str):
            return False
    return True


def add_edge_ids_to_edges_dict(edge_data_dict, edge_id_offset_d, id_dtype):
    eids_data_dict = {}
    for etype, df in edge_data_dict.items():
        # Do not modify input by user
        if len(df.columns) != 2:
            raise ValueError(
                "Provided dataframe in edge_dict contains more than 2 columns",
                "DataFrame with only 2 columns is supported",
                "Where first is treated as src and second as dst",
            )
        df = df.copy(deep=False)
        df = df.rename(columns={df.columns[0]: src_n, df.columns[1]: dst_n})
        df[eid_n] = id_dtype(1)
        df[eid_n] = df[eid_n].cumsum()
        df[eid_n] = df[eid_n] + edge_id_offset_d[etype] - 1
        df[eid_n] = df[eid_n].astype(id_dtype)
        eids_data_dict[etype] = df
    return eids_data_dict


def add_node_offset_to_edges_dict(edge_data_dict, node_id_offset_d):
    for etype, df in edge_data_dict.items():
        src_type, _, dst_type = etype
        df[src_n] = df[src_n] + node_id_offset_d[src_type]
        df[dst_n] = df[dst_n] + node_id_offset_d[dst_type]
    return edge_data_dict


if isinstance(F, MissingModule):
    backend_dtype_to_np_dtype_dict = MissingModule("dgl")
else:
    backend_dtype_to_np_dtype_dict = {
        F.bool: bool,
        F.uint8: np.uint8,
        F.int8: np.int8,
        F.int16: np.int16,
        F.int32: np.int32,
        F.int64: np.int64,
        F.float16: np.float16,
        F.float32: np.float32,
        F.float64: np.float64,
    }

# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

# from cugraph.utilities.grmat import grmat_gen
# from cugraph.utilities.pointer_utils import device_of_gpu_pointer
from cugraph.utilities.nx_factory import convert_from_nx
from cugraph.utilities.nx_factory import check_nx_graph
from cugraph.utilities.nx_factory import df_score_to_dictionary
from cugraph.utilities.nx_factory import df_edge_score_to_dictionary
from cugraph.utilities.nx_factory import cugraph_to_nx
from cugraph.utilities.nx_factory import is_networkx_graph
from cugraph.utilities.utils import (import_optional,
                                     ensure_cugraph_obj,
                                     is_matrix_type,
                                     is_cp_matrix_type,
                                     is_sp_matrix_type,
                                     )

# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

from cugraph.community.louvain import louvain
from cugraph.community.leiden import leiden
from cugraph.community.ecg import ecg
from cugraph.community.spectral_clustering import (
    spectralBalancedCutClustering,
    spectralModularityMaximizationClustering,
    analyzeClustering_modularity,
    analyzeClustering_edge_cut,
    analyzeClustering_ratio_cut,
)
from cugraph.community.subgraph_extraction import subgraph
from cugraph.community.triangle_count import triangles
from cugraph.community.egonet import ego_graph
from cugraph.community.egonet import batched_ego_graphs

# FIXME: special case for ktruss on CUDA 11.4: an 11.4 bug causes ktruss to
# crash in that environment. Allow ktruss to import on non-11.4 systems, but
# replace ktruss with a __UnsupportedModule instance, which lazily raises an
# exception when referenced.
from numba import cuda
__cuda_version = cuda.runtime.get_version()
__ktruss_unsupported_cuda_version = (11, 4)

class __UnsupportedModule:
    def __init__(self, exception):
        self.__excexption = exception

    def __getattr__(self, attr):
        raise self.__excexption

    def __call__(self, *args, **kwargs):
        raise self.__excexption


if __cuda_version != __ktruss_unsupported_cuda_version:
    from cugraph.community.ktruss_subgraph import ktruss_subgraph
    from cugraph.community.ktruss_subgraph import k_truss
else:
    __kuvs = ".".join([str(n) for n in __ktruss_unsupported_cuda_version])
    k_truss = __UnsupportedModule(
        NotImplementedError("k_truss is not currently supported in CUDA"
                            f" {__kuvs} environments.")
        )
    ktruss_subgraph = __UnsupportedModule(
        NotImplementedError("ktruss_subgraph is not currently supported in CUDA"
                            f" {__kuvs} environments.")
        )

# Copyright (c) 2023, NVIDIA CORPORATION.
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
import inspect

import networkx as nx

import cugraph_nx as cnx
from cugraph_nx.utils import networkx_algorithm


def test_match_signature_and_names():
    """Simple test to ensure our signatures and basic module layout match networkx."""
    for name, func in vars(cnx.interface.BackendInterface).items():
        if not isinstance(func, networkx_algorithm):
            continue

        # nx version >=3.2 uses utils.backends, version >=3.0,<3.2 uses classes.backends
        nx_backends = getattr(
            nx.utils, "backends", getattr(nx.classes, "backends", None)
        )
        if nx_backends is None:
            raise AttributeError(
                f"imported networkx version {nx.__version__} is not "
                "supported, must be >= 3.0"
            )

        dispatchable_func = nx_backends._registered_algorithms[name]
        # nx version >=3.2 uses orig_func, version >=3.0,<3.2 uses _orig_func
        orig_func = getattr(
            dispatchable_func, "orig_func", getattr(dispatchable_func, "_orig_func")
        )

        # Matching signatures?
        sig = inspect.signature(orig_func)
        assert sig == inspect.signature(func)

        # Matching function names?
        assert func.__name__ == dispatchable_func.__name__ == orig_func.__name__

        # Matching dispatch names?
        # nx version >=3.2 uses name, version >=3.0,<3.2 uses dispatchname
        assert func.name == getattr(
            dispatchable_func, "name", getattr(dispatchable_func, "dispatchname")
        )

        # Matching modules (i.e., where function defined)?
        assert (
            "networkx." + func.__module__.split(".", 1)[1]
            == dispatchable_func.__module__
            == orig_func.__module__
        )

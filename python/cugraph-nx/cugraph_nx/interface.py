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
from __future__ import annotations

import cugraph_nx as cnx

from . import algorithms


class Dispatcher:
    is_strongly_connected = algorithms.is_strongly_connected

    # Required conversions
    @staticmethod
    def convert_from_nx(*args, edge_attrs=None, weight=None, **kwargs):
        if weight is not None:
            # For networkx 3.0 and 3.1 compatibility
            if edge_attrs is not None:
                raise TypeError(
                    "edge_attrs and weight arguments should not both be given"
                )
            edge_attrs = {weight: 1}
        return cnx.from_networkx(*args, edge_attrs=edge_attrs, **kwargs)

    @staticmethod
    def convert_to_nx(obj, *, name: str | None = None):
        if isinstance(obj, cnx.Graph):
            return cnx.to_networkx(obj)
        return obj

    @staticmethod
    def on_start_tests(items):
        pass

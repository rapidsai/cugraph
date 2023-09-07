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
"""Tell NetworkX about the cugraph backend. This file can update itself:

$ make plugin-info  # Recommended method for development

or

$ python _nx_cugraph/__init__.py
"""


def get_info():
    """Target of ``networkx.plugin_info`` entry point.

    This tells NetworkX about the cugraph backend without importing nx_cugraph.
    """
    # Entries between BEGIN and END are automatically generated
    return {
        "backend_name": "cugraph",
        "project": "nx-cugraph",
        "package": "nx_cugraph",
        "functions": {
            # BEGIN: functions
            "betweenness_centrality",
            "edge_betweenness_centrality",
            "louvain_communities",
            # END: functions
        },
        "extra_docstrings": {
            # BEGIN: extra_docstrings
            "betweenness_centrality": "`weight` parameter is not yet supported.",
            "edge_betweenness_centrality": "`weight` parameter is not yet supported.",
            "louvain_communities": "`threshold` and `seed` parameters are currently ignored.",
            # END: extra_docstrings
        },
        "extra_parameters": {
            # BEGIN: extra_parameters
            "louvain_communities": {
                "max_level": "Upper limit of the number of macro-iterations.",
            },
            # END: extra_parameters
        },
    }


__version__ = "23.10.00"

if __name__ == "__main__":
    from pathlib import Path

    from _nx_cugraph.core import main

    filepath = Path(__file__)
    text = main(filepath)
    with filepath.open("w") as f:
        f.write(text)

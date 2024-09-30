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
import nx_cugraph as nxcg


def test_class_to_class():
    """Basic sanity checks to ensure metadata relating graph classes are accurate."""
    for prefix in ["", "Cuda"]:
        for suffix in ["Graph", "DiGraph", "MultiGraph", "MultiDiGraph"]:
            cls_name = f"{prefix}{suffix}"
            cls = getattr(nxcg, cls_name)
            assert cls.__name__ == cls_name
            G = cls()
            assert cls is G.__class__
            # cudagraph
            val = cls.to_cudagraph_class()
            val2 = G.to_cudagraph_class()
            assert val is val2
            assert val.__name__ == f"Cuda{suffix}"
            assert val.__module__.startswith("nx_cugraph")
            assert cls.is_directed() == G.is_directed() == val.is_directed()
            assert cls.is_multigraph() == G.is_multigraph() == val.is_multigraph()
            # networkx
            val = cls.to_networkx_class()
            val2 = G.to_networkx_class()
            assert val is val2
            assert val.__name__ == suffix
            assert val.__module__.startswith("networkx")
            val = val()
            assert cls.is_directed() == G.is_directed() == val.is_directed()
            assert cls.is_multigraph() == G.is_multigraph() == val.is_multigraph()
            # directed
            val = cls.to_directed_class()
            val2 = G.to_directed_class()
            assert val is val2
            assert val.__module__.startswith("nx_cugraph")
            assert val.is_directed()
            assert cls.is_multigraph() == G.is_multigraph() == val.is_multigraph()
            if "Di" in suffix:
                assert val is cls
            else:
                assert "Di" in val.__name__
                assert prefix in val.__name__
                assert cls.to_undirected_class() is cls
            # undirected
            val = cls.to_undirected_class()
            val2 = G.to_undirected_class()
            assert val is val2
            assert val.__module__.startswith("nx_cugraph")
            assert not val.is_directed()
            assert cls.is_multigraph() == G.is_multigraph() == val.is_multigraph()
            if "Di" not in suffix:
                assert val is cls
            else:
                assert "Di" not in val.__name__
                assert prefix in val.__name__
                assert cls.to_directed_class() is cls
            # "zero"
            if prefix == "Cuda":
                val = cls._to_compat_graph_class()
                val2 = G._to_compat_graph_class()
                assert val is val2
                assert val.__name__ == suffix
                assert val.__module__.startswith("nx_cugraph")
                assert val.to_cudagraph_class() is cls
                assert cls.is_directed() == G.is_directed() == val.is_directed()
                assert cls.is_multigraph() == G.is_multigraph() == val.is_multigraph()

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


from collections import defaultdict
from collections.abc import MutableMapping
from typing import Union, Dict, List, Tuple

from cugraph.utilities.utils import import_optional

import cugraph_dgl
from cugraph_dgl.typing import TensorType

torch = import_optional("torch")
dgl = import_optional("dgl")


class HeteroEdgeDataView(MutableMapping):
    """
    Duck-typed version of DGL's HeteroEdgeDataView.
    Used for accessing and modifying edge features.
    """

    def __init__(
        self,
        graph: "cugraph_dgl.Graph",
        etype: Union[Tuple[str, str, str], List[Tuple[str, str, str]]],
        edges: TensorType,
    ):
        self.__graph = graph
        self.__etype = etype
        self.__edges = edges

    @property
    def _etype(self) -> Tuple[str, str, str]:
        return self.__etype

    @property
    def _graph(self) -> "cugraph_dgl.Graph":
        return self.__graph

    @property
    def _edges(self) -> TensorType:
        return self.__edges

    def __getitem__(self, key: str):
        if isinstance(self._etype, list):
            return {t: self._graph._get_e_emb(t, key, self._edges) for t in self._etype}

        return self._graph._get_e_emb(self._etype, key, self._edges)

    def __setitem__(self, key: str, val: Union[TensorType, Dict[str, TensorType]]):
        if isinstance(self._etype, list):
            if not isinstance(val, dict):
                raise ValueError(
                    "There are multiple edge types in this view. "
                    "Expected a dictionary of values."
                )
            for t, v in val.items():
                if t not in self._etype:
                    raise ValueError("Attempted to modify a type out of view.")
                self._graph.set_e_emb(t, self._edges, {key: v})
        else:
            if isinstance(val, dict):
                raise ValueError(
                    "There is only one edge type in this view. "
                    "Expected a single tensor."
                )
            self._graph.set_e_emb(self._etype, self._edges, {key: v})

    def __delitem__(self, key: str):
        if isinstance(self._etype, list):
            for t in self._etype:
                self._graph.pop_e_emb(t, key)
        else:
            self._graph.pop_e_emb(self._etype, key)

    def _transpose(self, fetch_vals=True):
        if isinstance(self._etype, list):
            tr = defaultdict(dict)
            for etype in self._etype:
                for key in self._graph._get_e_emb_keys(etype):
                    tr[key][etype] = (
                        self._graph._get_e_emb(etype, key, self._edges)
                        if fetch_vals
                        else []
                    )
        else:
            tr = {}
            for key in self._graph._get_e_emb_keys(self._etype):
                tr[key] = (
                    self._graph._get_e_emb(self._etype, key, self._edges)
                    if fetch_vals
                    else []
                )

        return tr

    def __len__(self):
        return len(self._transpose(fetch_vals=False))

    def __iter__(self):
        return iter(self._transpose())

    def keys(self):
        return self._transpose(fetch_vals=False).keys()

    def values(self):
        return self._transpose().values()

    def __repr__(self):
        return repr(self._transpose(fetch_vals=False))


class HeteroNodeDataView(MutableMapping):
    """
    Duck-typed version of DGL's HeteroNodeDataView.
    Used for accessing and modifying node features.
    """

    def __init__(
        self,
        graph: "cugraph_dgl.Graph",
        ntype: Union[str, List[str]],
        nodes: TensorType,
    ):
        self.__graph = graph
        self.__ntype = ntype
        self.__nodes = nodes

    @property
    def _ntype(self) -> str:
        return self.__ntype

    @property
    def _graph(self) -> "cugraph_dgl.Graph":
        return self.__graph

    @property
    def _nodes(self) -> TensorType:
        return self.__nodes

    def __getitem__(self, key: str):
        if isinstance(self._ntype, list):
            return {t: self._graph._get_n_emb(t, key, self._nodes) for t in self._ntype}
        else:
            return self._graph._get_n_emb(self._ntype, key, self._nodes)

    def __setitem__(self, key: str, val: Union[TensorType, Dict[str, TensorType]]):
        if isinstance(self._ntype, list):
            if not isinstance(val, dict):
                raise ValueError(
                    "There are multiple node types in this view. "
                    "Expected a dictionary of values."
                )
            for t, v in val.items():
                if t not in self._ntype:
                    raise ValueError("Attempted to modify a type out of view.")
                self._graph._set_n_emb(t, self._nodes, {key: v})
        else:
            if isinstance(val, dict):
                raise ValueError(
                    "There is only one node type in this view. "
                    "Expected a single value tensor."
                )
            self._graph._set_n_emb(self._ntype, self._nodes, {key: val})

    def __delitem__(self, key: str):
        if isinstance(self._ntype, list):
            for t in self._ntype:
                self._graph._pop_n_emb(t, key)
        else:
            self._graph.pop_n_emb(self._ntype, key)

    def _transpose(self, fetch_vals=True):
        if isinstance(self._ntype, list):
            tr = defaultdict(dict)
            for ntype in self._ntype:
                for key in self._graph._get_n_emb_keys(ntype):
                    tr[key][ntype] = (
                        self._graph._get_n_emb(ntype, key, self._nodes)
                        if fetch_vals
                        else []
                    )
        else:
            tr = {}
            for key in self._graph._get_n_emb_keys(self._ntype):
                tr[key] = (
                    self._graph._get_n_emb(self._ntype, key, self._nodes)
                    if fetch_vals
                    else []
                )

        return tr

    def __len__(self):
        return len(self._transpose(fetch_vals=False))

    def __iter__(self):
        return iter(self._transpose())

    def keys(self):
        return self._transpose(fetch_vals=False).keys()

    def values(self):
        return self._transpose().values()

    def __repr__(self):
        return repr(self._transpose(fetch_vals=False))


class HeteroEdgeView:
    """
    Duck-typed version of DGL's HeteroEdgeView.
    """

    def __init__(self, graph):
        self.__graph = graph

    @property
    def _graph(self) -> "cugraph_dgl.Graph":
        return self.__graph

    def __getitem__(self, key):
        if isinstance(key, slice):
            if not (key.start is None and key.stop is None and key.stop is None):
                raise ValueError("Only full slices are supported in DGL.")
            edges = dgl.base.ALL
            etype = None
        elif key is None:
            edges = dgl.base.ALL
            etype = None
        elif isinstance(key, tuple):
            if len(key) == 3:
                edges = dgl.base.ALL
                etype = key
            else:
                edges = key
                etype = None
        elif isinstance(key, str):
            edges = dgl.base.ALL
            etype = key
        else:
            edges = key
            etype = None

        return HeteroEdgeDataView(
            graph=self.__graph,
            etype=etype,
            edges=edges,
        )

    def __call__(self, *args, **kwargs):
        if "device" in kwargs:
            return self.__graph.all_edges(*args, **kwargs)

        return self.__graph.all_edges(*args, **kwargs, device="cuda")


class HeteroNodeView:
    """
    Duck-typed version of DGL's HeteroNodeView.
    """

    def __init__(self, graph: "cugraph_dgl.Graph"):
        self.__graph = graph

    @property
    def _graph(self) -> "cugraph_dgl.Graph":
        return self.__graph

    def __getitem__(self, key):
        if isinstance(key, slice):
            if not (key.start is None and key.stop is None and key.stop is None):
                raise ValueError("Only full slices are supported in DGL.")
            nodes = dgl.base.ALL
            ntype = None
        elif isinstance(key, tuple):
            nodes, ntype = key
        elif key is None or isinstance(key, str):
            nodes = dgl.base.ALL
            ntype = key
        else:
            nodes = key
            ntype = None

        return HeteroNodeDataView(graph=self.__graph, ntype=ntype, nodes=nodes)

    def __call__(self, ntype=None):
        return torch.arange(
            0, self.__graph.num_nodes(ntype), dtype=self.__graph.idtype, device="cuda"
        )

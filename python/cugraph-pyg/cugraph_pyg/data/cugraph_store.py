# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

from typing import Optional, Tuple, Any, Union, List, Dict

from enum import Enum, auto

from dataclasses import dataclass
from collections import defaultdict
from itertools import chain
from functools import cached_property

import numpy as np
import cupy
import pandas
import cudf
import cugraph

from cugraph.utilities.utils import import_optional, MissingModule

dd = import_optional("dask.dataframe")
distributed = import_optional("dask.distributed")
dask_cudf = import_optional("dask_cudf")

torch = import_optional("torch")
torch_geometric = import_optional("torch_geometric")

Tensor = None if isinstance(torch, MissingModule) else torch.Tensor
NdArray = None if isinstance(cupy, MissingModule) else cupy.ndarray
DaskCudfSeries = None if isinstance(dask_cudf, MissingModule) else dask_cudf.Series

TensorType = Union[Tensor, NdArray, cudf.Series, DaskCudfSeries]
NodeType = (
    None
    if isinstance(torch_geometric, MissingModule)
    else torch_geometric.typing.NodeType
)
EdgeType = (
    None
    if isinstance(torch_geometric, MissingModule)
    else torch_geometric.typing.EdgeType
)


class EdgeLayout(Enum):
    COO = "coo"
    CSC = "csc"
    CSR = "csr"


@dataclass
class CuGraphEdgeAttr:
    """
    Defines the attributes of an :obj:`GraphStore` edge.
    """

    # The type of the edge
    edge_type: Optional[Any]

    # The layout of the edge representation
    layout: EdgeLayout

    # Whether the edge index is sorted, by destination node. Useful for
    # avoiding sorting costs when performing neighbor sampling, and only
    # meaningful for COO (CSC and CSR are sorted by definition)
    is_sorted: bool = False

    # The number of nodes in this edge type. If set to None, will attempt to
    # infer with the simple heuristic int(self.edge_index.max()) + 1
    size: Optional[Tuple[int, int]] = None

    # NOTE we define __post_init__ to force-cast layout
    def __post_init__(self):
        self.layout = EdgeLayout(self.layout)

    @classmethod
    def cast(cls, *args, **kwargs):
        """
        Cast to a CuGraphTensorAttr from a tuple, list, or dict.

        Returns
        -------
        CuGraphTensorAttr
            contains the data of the tuple, list, or dict passed in
        """
        if len(args) == 1 and len(kwargs) == 0:
            elem = args[0]
            if elem is None:
                return None
            if isinstance(elem, CuGraphEdgeAttr):
                return elem
            if isinstance(elem, (tuple, list)):
                return cls(*elem)
            if isinstance(elem, dict):
                return cls(**elem)
        return cls(*args, **kwargs)


class _field_status(Enum):
    UNSET = auto()


@dataclass
class CuGraphTensorAttr:
    """
    Defines the attributes of a class:`FeatureStore` tensor; in particular,
    all the parameters necessary to uniquely identify a tensor from the feature
    store.

    Note that the order of the attributes is important; this is the order in
    which attributes must be provided for indexing calls. Feature store
    implementor classes can define a different ordering by overriding
    :meth:`TensorAttr.__init__`.
    """

    # The group name that the tensor corresponds to. Defaults to UNSET.
    group_name: Optional[str] = _field_status.UNSET

    # The name of the tensor within its group. Defaults to UNSET.
    attr_name: Optional[str] = _field_status.UNSET

    # The node indices the rows of the tensor correspond to. Defaults to UNSET.
    index: Optional[Any] = _field_status.UNSET

    # The properties in the FeatureStore the rows of the tensor correspond to.
    # Defaults to UNSET.
    properties: Optional[Any] = _field_status.UNSET

    # The datatype of the tensor.  Defaults to UNSET.
    dtype: Optional[Any] = _field_status.UNSET

    # Convenience methods

    def is_set(self, key):
        """
        Whether an attribute is set in :obj:`TensorAttr`.
        """
        if key not in self.__dataclass_fields__:
            raise KeyError(key)
        attr = getattr(self, key)
        return type(attr) != _field_status or attr != _field_status.UNSET

    def is_fully_specified(self):
        """
        Whether the :obj:`TensorAttr` has no unset fields.
        """
        return all([self.is_set(key) for key in self.__dataclass_fields__])

    def fully_specify(self):
        """
        Sets all :obj:`UNSET` fields to :obj:`None`.
        """
        for key in self.__dataclass_fields__:
            if not self.is_set(key):
                setattr(self, key, None)
        return self

    def update(self, attr):
        """
        Updates an :class:`TensorAttr` with set attributes from another
        :class:`TensorAttr`.
        """
        for key in self.__dataclass_fields__:
            if attr.is_set(key):
                setattr(self, key, getattr(attr, key))

    @classmethod
    def cast(cls, *args, **kwargs):
        """
        Casts to a CuGraphTensorAttr from a tuple, list, or dict

        Returns
        -------
        CuGraphTensorAttr
            contains the data of the tuple, list, or dict passed in
        """
        if len(args) == 1 and len(kwargs) == 0:
            elem = args[0]
            if elem is None:
                return None
            if isinstance(elem, CuGraphTensorAttr):
                return elem
            if isinstance(elem, (tuple, list)):
                return cls(*elem)
            if isinstance(elem, dict):
                return cls(**elem)
        return cls(*args, **kwargs)


class EXPERIMENTAL__CuGraphStore:
    """
    Duck-typed version of PyG's GraphStore and FeatureStore.
    """

    # TODO allow (and possibly require) separate stores for node, edge attrs
    # For now edge attrs are entirely unsupported.
    # TODO add an "expensive check" argument that ensures the graph store
    # and feature store are valid and compatible with PyG.
    def __init__(
        self,
        F: cugraph.gnn.FeatureStore,
        G: Union[Dict[str, Tuple[TensorType]], Dict[str, int]],
        num_nodes_dict: Dict[str, int],
        multi_gpu: bool = False,
    ):
        """
        Constructs a new CuGraphStore from the provided
        arguments.

        Parameters
        ----------
        F: cugraph.gnn.FeatureStore (Required)
            The feature store containing this graph's features.
            Typed lexicographic-ordered numbering convention
            should match that of the graph.

        G: dict[str, tuple[TensorType]] or dict[str, int] (Required)
            Dictionary of edge indices.
            Option 1 (graph in memory):

                Pass the edge indices: i.e.
                {
                ('author', 'writes', 'paper'): [[0,1,2],[2,0,1]],
                ('author', 'affiliated', 'institution'): [[0,1],[0,1]]
                }


            Option 2 (graph not in memory):

                Pass the number of edges: i.e.
                {
                ('author', 'writes', 'paper'): 2,
                ('author', 'affiliated', 'institution'): 2
                }
                If the graph is not in memory, manipulating the edge indices
                or calling sampling is not possible.  This is for cases where
                sampling has already been done and samples were written to disk.

            Note: the internal cugraph representation will use
            offsetted vertex and edge ids.

        num_nodes_dict: dict (Required)
            A dictionary mapping each node type to the count of nodes
            of that type in the graph.

        multi_gpu: bool (Optional, default = False)
            Whether the store should be backed by a multi-GPU graph.
            Requires dask to have been set up.
        """

        if None in G:
            raise ValueError("Unspecified edge types not allowed in PyG")

        self.__vertex_dtype = torch.int64

        self._tensor_attr_cls = CuGraphTensorAttr
        self._tensor_attr_dict = defaultdict(list)

        construct_graph = True
        if isinstance(next(iter(G.values())), int):
            # User has passed in the number of edges
            # (not the actual edge index), so the number of edges
            # does not need to be counted.
            num_edges_dict = dict(G)  # make sure the cugraph store owns this dict
            construct_graph = False
        else:
            # User has passed in the actual edge index, so the
            # number of edges needs to be counted.
            num_edges_dict = {
                pyg_can_edge_type: len(ei[0]) for pyg_can_edge_type, ei in G.items()
            }

        self.__infer_offsets(num_nodes_dict, num_edges_dict)
        self.__infer_existing_tensors(F)
        self.__infer_edge_types(num_nodes_dict, num_edges_dict)

        self._edge_attr_cls = CuGraphEdgeAttr

        self.__features = F
        self.__graph = None
        self.__is_graph_owner = False

        if construct_graph:
            if multi_gpu:
                self.__graph = distributed.get_client().get_dataset(
                    "cugraph_graph", default=None
                )

            if self.__graph is None:
                self.__graph = self.__construct_graph(G, multi_gpu=multi_gpu)
                self.__is_graph_owner = True

        self.__subgraphs = {}

    def __del__(self):
        if self.__is_graph_owner:
            if isinstance(self.__graph._plc_graph, dict):
                distributed.get_client().unpublish_dataset("cugraph_graph")
            del self.__graph

    def __make_offsets(self, input_dict):
        offsets = {}
        offsets["stop"] = [input_dict[v] for v in sorted(input_dict.keys())]
        offsets["stop"] = torch.tensor(offsets["stop"])
        if torch.has_cuda:
            offsets["stop"] = offsets["stop"].cuda()

        cumsum = offsets["stop"].cumsum(0)
        offsets["start"] = cumsum - offsets["stop"]
        offsets["stop"] = cumsum - 1

        offsets["type"] = np.array(sorted(input_dict.keys()))

        return offsets

    def __infer_offsets(
        self,
        num_nodes_dict: Dict[str, int],
        num_edges_dict: Dict[Tuple[str, str, str], int],
    ) -> None:
        """
        Sets the vertex offsets for this store.
        """
        self.__vertex_type_offsets = self.__make_offsets(num_nodes_dict)

        # Need to convert tuples to string in order to use searchsorted
        # Can convert back using x.split('__')
        # Lexicographic ordering is unchanged.
        self.__edge_type_offsets = self.__make_offsets(
            {
                "__".join(pyg_can_edge_type): n
                for pyg_can_edge_type, n in num_edges_dict.items()
            }
        )

    def __construct_graph(
        self,
        edge_info: Dict[Tuple[str, str, str], List[TensorType]],
        multi_gpu: bool = False,
    ) -> cugraph.MultiGraph:
        """
        This function takes edge information and uses it to construct
        a cugraph Graph.  It determines the numerical edge type by
        sorting the keys of the input dictionary
        (the canonical edge types).

        Parameters
        ----------
        edge_info: Dict[Tuple[str, str, str], List[TensorType]] (Required)
            Input edge info dictionary, where keys are the canonical
            edge type and values are the edge index (src/dst).

        multi_gpu: bool (Optional, default=False)
            Whether to construct a single-GPU or multi-GPU cugraph Graph.
            Defaults to a single-GPU graph.
        Returns
        -------
        A newly-constructed directed cugraph.MultiGraph object.
        """
        # Ensure the original dict is not modified.
        edge_info_cg = {}

        # Iterate over the keys in sorted order so that the created
        # numerical types correspond to the lexicographic order
        # of the keys, which is critical to converting the numeric
        # keys back to canonical edge types later.
        # FIXME don't always convert to host arrays (#3383)
        for pyg_can_edge_type in sorted(edge_info.keys()):
            src_type, _, dst_type = pyg_can_edge_type
            srcs, dsts = edge_info[pyg_can_edge_type]

            src_offset = np.searchsorted(self.__vertex_type_offsets["type"], src_type)
            srcs_t = srcs + int(self.__vertex_type_offsets["start"][src_offset])
            if isinstance(srcs_t, torch.Tensor):
                srcs_t = srcs_t.cpu()
            else:
                if isinstance(srcs_t, dask_cudf.Series):
                    srcs_t = srcs_t.compute()
                if isinstance(srcs_t, cudf.Series):
                    srcs_t = srcs_t.values_host

            dst_offset = np.searchsorted(self.__vertex_type_offsets["type"], dst_type)
            dsts_t = dsts + int(self.__vertex_type_offsets["start"][dst_offset])
            if isinstance(dsts_t, torch.Tensor):
                dsts_t = dsts_t.cpu()
            else:
                if isinstance(dsts_t, dask_cudf.Series):
                    dsts_t = dsts_t.compute()
                if isinstance(dsts_t, cudf.Series):
                    dsts_t = dsts_t.values_host

            edge_info_cg[pyg_can_edge_type] = (srcs_t, dsts_t)

        na_src = np.concatenate(
            [
                edge_info_cg[pyg_can_edge_type][0]
                for pyg_can_edge_type in sorted(edge_info_cg.keys())
            ]
        )

        na_dst = np.concatenate(
            [
                edge_info_cg[pyg_can_edge_type][1]
                for pyg_can_edge_type in sorted(edge_info_cg.keys())
            ]
        )

        et_offsets = self.__edge_type_offsets
        na_etp = np.concatenate(
            [
                np.full(
                    int(et_offsets["stop"][i] - et_offsets["start"][i] + 1),
                    i,
                    dtype="int32",
                )
                for i in range(len(self.__edge_type_offsets["start"]))
            ]
        )

        df = pandas.DataFrame(
            {
                "src": pandas.Series(na_src),
                "dst": pandas.Series(na_dst),
                "etp": pandas.Series(na_etp),
            }
        )

        if multi_gpu:
            nworkers = len(distributed.get_client().scheduler_info()["workers"])
            df = dd.from_pandas(df, npartitions=nworkers).persist()
            df = df.map_partitions(cudf.DataFrame.from_pandas)
        else:
            df = cudf.from_pandas(df)

        df = df.reset_index(drop=True)

        graph = cugraph.MultiGraph(directed=True)
        if multi_gpu:
            graph.from_dask_cudf_edgelist(
                df,
                source="src",
                destination="dst",
                edge_type="etp",
            )
            distributed.get_client().publish_dataset(cugraph_graph=graph)
        else:
            graph.from_cudf_edgelist(
                df,
                source="src",
                destination="dst",
                edge_type="etp",
            )

        return graph

    @property
    def _edge_types_to_attrs(self) -> dict:
        return dict(self.__edge_types_to_attrs)

    @property
    def node_types(self) -> List[NodeType]:
        return list(self.__vertex_type_offsets["type"])

    @property
    def edge_types(self) -> List[EdgeType]:
        return list(self.__edge_types_to_attrs.keys())

    def canonical_edge_type_to_numeric(self, etype: EdgeType) -> int:
        return np.searchsorted(self.__edge_type_offsets["type"], "__".join(etype))

    def numeric_edge_type_to_canonical(self, etype: int) -> EdgeType:
        return tuple(self.__edge_type_offsets["type"][etype].split("__"))

    @cached_property
    def _is_delayed(self):
        if self.__graph is None:
            return False
        return self.__graph.is_multi_gpu()

    def _numeric_vertex_type_from_name(self, vertex_type_name: str) -> int:
        return np.searchsorted(self.__vertex_type_offsets["type"], vertex_type_name)

    def get_vertex_index(self, vtypes) -> TensorType:
        if isinstance(vtypes, str):
            vtypes = [vtypes]

        ix = torch.tensor([], dtype=torch.int64)

        if isinstance(self.__vertex_type_offsets, dict):
            vtypes = np.searchsorted(self.__vertex_type_offsets["type"], vtypes)
        for vtype in vtypes:
            start = int(self.__vertex_type_offsets["start"][vtype])
            stop = int(self.__vertex_type_offsets["stop"][vtype])
            ix = torch.concatenate(
                [
                    ix,
                    torch.arange(
                        start, stop + 1, 1, dtype=self.__vertex_dtype, device="cuda"
                    ),
                ]
            )

        return ix

    def put_edge_index(self, edge_index, edge_attr):
        """
        Adds additional edges to the graph.
        Not yet implemented.
        """
        raise NotImplementedError("Adding indices not supported.")

    def get_all_edge_attrs(self):
        """
        Gets a list of all edge types and indices in this store.

        Returns
        -------
        list[str]
            All edge types and indices in this store.
        """
        return self.__edge_types_to_attrs.values()

    def _get_edge_index(self, attr: CuGraphEdgeAttr) -> Tuple[TensorType, TensorType]:
        """
        Returns the edge index in the requested format
        (as defined by attr).  Currently, only unsorted
        COO is supported, which is returned as a (src,dst)
        tuple as expected by the PyG API.

        Parameters
        ----------
        attr: CuGraphEdgeAttr
            The CuGraphEdgeAttr specifying the
            desired edge type, layout (i.e. CSR, COO, CSC), and
            whether the returned index should be sorted (if COO).
            Currently, only unsorted COO is supported.

        Returns
        -------
        (src, dst) : Tuple[tensor type]
            Tuple of the requested edge index in COO form.
            Currently, only COO form is supported.
        """

        if self.__graph is None:
            raise ValueError("Graph is not in memory, cannot access edge index!")

        if attr.layout != EdgeLayout.COO:
            raise TypeError("Only COO direct access is supported!")

        # Currently, graph creation enforces that input vertex ids are always of
        # integer type.  Therefore, it is currently safe to assume that for MG
        # graphs, the src/dst col names are renumbered_src/dst
        # and for SG graphs, the src/dst col names are src/dst.
        # This may change in the future if/when renumbering or the graph
        # creation process is refactored.
        # See Issue #3201 for more details.
        if self._is_delayed:
            src_col_name = self.__graph.renumber_map.renumbered_src_col_name
            dst_col_name = self.__graph.renumber_map.renumbered_dst_col_name
        else:
            src_col_name = self.__graph.srcCol
            dst_col_name = self.__graph.dstCol

        # If there is only one edge type (homogeneous graph) then
        # bypass the edge filters for a significant speed improvement.
        if len(self.__edge_types_to_attrs) == 1:
            if attr.edge_type not in self.__edge_types_to_attrs:
                raise ValueError(
                    f"Requested edge type {attr.edge_type}" "is not present in graph."
                )

            df = self.__graph.edgelist.edgelist_df[[src_col_name, dst_col_name]]
            src_offset = 0
            dst_offset = 0
        else:
            src_type, _, dst_type = attr.edge_type
            src_offset = int(
                self.__vertex_type_offsets["start"][
                    self._numeric_vertex_type_from_name(src_type)
                ]
            )
            dst_offset = int(
                self.__vertex_type_offsets["start"][
                    self._numeric_vertex_type_from_name(dst_type)
                ]
            )
            coli = np.searchsorted(
                self.__edge_type_offsets["type"], "__".join(attr.edge_type)
            )

            df = self.__graph.edgelist.edgelist_df[
                [src_col_name, dst_col_name, self.__graph.edgeTypeCol]
            ]
            df = df[df[self.__graph.edgeTypeCol] == coli]
            df = df[[src_col_name, dst_col_name]]

        if self._is_delayed:
            df = df.compute()

        src = torch.as_tensor(df[src_col_name], device="cuda") - src_offset
        dst = torch.as_tensor(df[dst_col_name], device="cuda") - dst_offset

        src = src.to(self.__vertex_dtype)
        dst = dst.to(self.__vertex_dtype)

        if src.shape[0] != dst.shape[0]:
            raise IndexError("src and dst shape do not match!")

        return (src, dst)

    def get_edge_index(self, *args, **kwargs) -> Tuple[TensorType, TensorType]:
        """
        Synchronously gets an edge_index tensor from the materialized
        graph.

        Args:
            **attr(EdgeAttr): the edge attributes.

        Returns:
            EdgeTensorType: an edge_index tensor corresonding to the provided
            attributes, or None if there is no such tensor.

        Raises:
            KeyError: if the edge index corresponding to attr was not found.
        """

        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        edge_attr.layout = EdgeLayout(edge_attr.layout)
        # Override is_sorted for CSC and CSR:
        # TODO treat is_sorted specially in this function, where is_sorted=True
        # returns an edge index sorted by column.
        edge_attr.is_sorted = edge_attr.is_sorted or (
            edge_attr.layout in [EdgeLayout.CSC, EdgeLayout.CSR]
        )
        edge_index = self._get_edge_index(edge_attr)
        if edge_index is None:
            raise KeyError(f"An edge corresponding to '{edge_attr}' was not " f"found")
        return edge_index

    def _subgraph(self, edge_types: List[tuple] = None) -> cugraph.MultiGraph:
        """
        Returns a subgraph with edges limited to those of a given type

        Parameters
        ----------
        edge_types : list of pyg canonical edge types
            Directly references the graph's internal edge types.  Does
            not accept PyG edge type tuples.

        Returns
        -------
        The appropriate extracted subgraph.  Will extract the subgraph
        if it has not already been extracted.

        """
        if self.__graph is None:
            raise ValueError("Graph is not in memory, cannot get subgraph")

        if edge_types is not None and set(edge_types) != set(
            self.__edge_types_to_attrs.keys()
        ):
            raise ValueError(
                "Subgraphing is currently unsupported, please"
                " specify all edge types in the graph or leave"
                " this argument empty."
            )

        return self.__graph

    def _get_vertex_groups_from_sample(
        self, nodes_of_interest: TensorType, is_sorted: bool = False
    ) -> dict:
        """
        Given a tensor of nodes of interest, this
        method a single dictionary, noi_index.

        noi_index is the original vertex ids grouped by vertex type.

        Example Input: [5, 2, 1, 10, 11, 8]
        Output: {'red_vertex': [5, 1, 8], 'blue_vertex': [2], 'green_vertex': [10, 11]}

        """

        noi_index = {}

        vtypes = cudf.Series(self.__vertex_type_offsets["type"])
        if len(vtypes) == 1:
            noi_index[vtypes.iloc[0]] = nodes_of_interest
        else:
            noi_type_indices = torch.searchsorted(
                torch.as_tensor(self.__vertex_type_offsets["stop"], device="cuda"),
                nodes_of_interest,
            )

            noi_types = vtypes.iloc[cupy.asarray(noi_type_indices)].reset_index(
                drop=True
            )
            noi_starts = self.__vertex_type_offsets["start"][noi_type_indices]

            noi_types = cudf.Series(noi_types, name="t").groupby("t").groups

            for type_name, ix in noi_types.items():
                # store the renumbering for this vertex type
                # renumbered vertex id is the index of the old id
                ix = torch.as_tensor(ix, device="cuda")
                # subtract off the offsets
                noi_index[type_name] = nodes_of_interest[ix] - noi_starts[ix]

        return noi_index

    def _get_sample_from_vertex_groups(
        self, vertex_groups: Dict[str, TensorType]
    ) -> TensorType:
        """
        Inverse of _get_vertex_groups_from_sample() (although with de-offsetted ids).
        Given a dictionary of node types and de-offsetted node ids, return
        the global (non-renumbered) vertex ids.

        Example Input: {'horse': [1, 3, 5], 'duck': [1, 2]}
        Output: [1, 3, 5, 14, 15]
        """
        t = torch.tensor([], dtype=torch.int64, device="cuda")

        for group_name, ix in vertex_groups.items():
            type_id = self._numeric_vertex_type_from_name(group_name)
            if not ix.is_cuda:
                ix = ix.cuda()
            offset = self.__vertex_type_offsets["start"][type_id]
            u = ix + offset
            t = torch.concatenate([t, u])

        return t

    def _get_renumbered_edge_groups_from_sample(
        self, sampling_results: cudf.DataFrame, noi_index: dict
    ) -> Tuple[dict, dict]:
        """
        Given a cudf (NOT dask_cudf) DataFrame of sampling results and a dictionary
        of non-renumbered vertex ids grouped by vertex type, this method
        outputs two dictionaries:
            1. row_dict
            2. col_dict
        (1) row_dict corresponds to the renumbered source vertex ids grouped
            by PyG edge type - (src, type, dst) tuple.
        (2) col_dict corresponds to the renumbered destination vertex ids grouped
            by PyG edge type (src, type, dst) tuple.
        * The two outputs combined make a PyG "edge index".
        * The ith element of each array corresponds to the same edge.
        * The _get_vertex_groups_from_sample() method is usually called
          before this one to get the noi_index.

        Example Input: Series({
                'sources': [0, 5, 11, 3],
                'destinations': [8, 2, 3, 5]},
                'edge_type': [1, 3, 5, 14]
            }),
            {
                'blue_vertex': [0, 5],
                'red_vertex': [3, 11],
                'green_vertex': [2, 8]
            }
        Output: {
                ('blue', 'etype1', 'green'): [0, 1],
                ('red', 'etype2', 'red'): [1],
                ('red', 'etype3', 'blue'): [0]
            },
            {
                ('blue', 'etype1', 'green'): [1, 0],
                ('red', 'etype2', 'red'): [0],
                ('red', 'etype3', 'blue'): [1]
            }

        """
        row_dict = {}
        col_dict = {}
        if len(self.__edge_types_to_attrs) == 1:
            t_pyg_type = list(self.__edge_types_to_attrs.values())[0].edge_type
            src_type, _, dst_type = t_pyg_type

            dst_id_table = noi_index[dst_type]
            dst_id_map = (
                cudf.Series(cupy.asarray(dst_id_table), name="dst")
                .reset_index()
                .rename(columns={"index": "new_id"})
                .set_index("dst")
            )
            dst = dst_id_map["new_id"].loc[sampling_results.destinations]
            col_dict[t_pyg_type] = torch.as_tensor(dst.values, device="cuda")

            src_id_table = noi_index[src_type]
            src_id_map = (
                cudf.Series(cupy.asarray(src_id_table), name="src")
                .reset_index()
                .rename(columns={"index": "new_id"})
                .set_index("src")
            )
            src = src_id_map["new_id"].loc[sampling_results.sources]
            row_dict[t_pyg_type] = torch.as_tensor(src.values, device="cuda")

        else:
            # This will retrieve the single string representation.
            # It needs to be converted to a tuple in the for loop below.
            eoi_types = (
                cudf.Series(self.__edge_type_offsets["type"])
                .iloc[sampling_results.edge_type.astype("int32")]
                .reset_index(drop=True)
            )

            eoi_types = cudf.Series(eoi_types, name="t").groupby("t").groups

            for pyg_can_edge_type_str, ix in eoi_types.items():
                pyg_can_edge_type = tuple(pyg_can_edge_type_str.split("__"))
                src_type, _, dst_type = pyg_can_edge_type

                # Get the de-offsetted sources
                sources = torch.as_tensor(
                    sampling_results.sources.iloc[ix].values, device="cuda"
                )
                sources_ix = torch.searchsorted(
                    self.__vertex_type_offsets["stop"], sources
                )
                sources -= self.__vertex_type_offsets["start"][sources_ix]

                # Create the row entry for this type
                src_id_table = noi_index[src_type]
                src_id_map = (
                    cudf.Series(cupy.asarray(src_id_table), name="src")
                    .reset_index()
                    .rename(columns={"index": "new_id"})
                    .set_index("src")
                )
                src = src_id_map["new_id"].loc[cupy.asarray(sources)]
                row_dict[pyg_can_edge_type] = torch.as_tensor(src.values, device="cuda")

                # Get the de-offsetted destinations
                destinations = torch.as_tensor(
                    sampling_results.destinations.iloc[ix].values, device="cuda"
                )
                destinations_ix = torch.searchsorted(
                    self.__vertex_type_offsets["stop"], destinations
                )
                destinations -= self.__vertex_type_offsets["start"][destinations_ix]

                # Create the col entry for this type
                dst_id_table = noi_index[dst_type]
                dst_id_map = (
                    cudf.Series(cupy.asarray(dst_id_table), name="dst")
                    .reset_index()
                    .rename(columns={"index": "new_id"})
                    .set_index("dst")
                )
                dst = dst_id_map["new_id"].loc[cupy.asarray(destinations)]
                col_dict[pyg_can_edge_type] = torch.as_tensor(dst.values, device="cuda")

        return row_dict, col_dict

    def put_tensor(self, tensor, attr) -> None:
        raise NotImplementedError("Adding properties not supported.")

    def create_named_tensor(
        self, attr_name: str, properties: List[str], vertex_type: str, dtype: str
    ) -> None:
        """
        Create a named tensor that contains a subset of
        properties in the graph.

        Parameters
        ----------
        attr_name : str
            The name of the tensor within its group.
        properties : list[str]
            The properties the rows
            of the tensor correspond to.
        vertex_type : str
            The vertex type associated with this new tensor property.
        dtype : numpy/cupy dtype (i.e. 'int32') or torch dtype (i.e. torch.float)
            The datatype of the tensor.  Usually float32/float64.
        """
        self._tensor_attr_dict[vertex_type].append(
            CuGraphTensorAttr(
                vertex_type, attr_name, properties=properties, dtype=dtype
            )
        )

    def __infer_edge_types(
        self,
        num_nodes_dict: Dict[str, int],
        num_edges_dict: Dict[Tuple[str, str, str], int],
    ) -> None:
        self.__edge_types_to_attrs = {}

        for pyg_can_edge_type in sorted(num_edges_dict.keys()):
            sz_src = num_nodes_dict[pyg_can_edge_type[0]]
            sz_dst = num_nodes_dict[pyg_can_edge_type[-1]]
            self.__edge_types_to_attrs[pyg_can_edge_type] = CuGraphEdgeAttr(
                edge_type=pyg_can_edge_type,
                layout=EdgeLayout.COO,
                is_sorted=False,
                size=(sz_src, sz_dst),
            )

    def __infer_existing_tensors(self, F) -> None:
        """
        Infers the tensor attributes/features.
        """
        for attr_name, types_with_attr in F.get_feature_list().items():
            for vt in types_with_attr:
                attr_dtype = F.get_data(np.array([0]), vt, attr_name).dtype
                self.create_named_tensor(
                    attr_name=attr_name,
                    properties=None,
                    vertex_type=vt,
                    dtype=attr_dtype,
                )

    def get_all_tensor_attrs(self) -> List[CuGraphTensorAttr]:
        """
        Obtains all tensor attributes stored in this feature store.
        """
        # unpack and return the list of lists
        it = chain.from_iterable(self._tensor_attr_dict.values())
        return [CuGraphTensorAttr.cast(c) for c in it]

    def _get_tensor(self, attr: CuGraphTensorAttr) -> TensorType:
        feature_backend = self.__features.backend
        cols = attr.properties

        idx = attr.index
        if idx is not None:
            if feature_backend == "torch":
                if not isinstance(idx, torch.Tensor):
                    raise TypeError(
                        f"Type {type(idx)} invalid"
                        f" for feature store backend {feature_backend}"
                    )
                idx = idx.cpu()
            elif feature_backend == "numpy":
                # allow feature indexing through cupy arrays
                if isinstance(idx, cupy.ndarray):
                    idx = idx.get()
                elif isinstance(idx, torch.Tensor):
                    idx = np.asarray(idx.cpu())

        if cols is None:
            t = self.__features.get_data(idx, attr.group_name, attr.attr_name)
            if idx is None:
                t = t[-1]

            if isinstance(t, np.ndarray):
                t = torch.as_tensor(t, device="cuda")
            else:
                t = t.cuda()

            return t

        else:
            t = self.__features.get_data(idx, attr.group_name, cols[0])

            if len(t.shape) == 1:
                t = torch.tensor([t])

            for col in cols[1:]:
                u = self.__features.get_data(idx, attr.group_name, col)

                if len(u.shape) == 1:
                    u = torch.tensor([u])

                t = torch.concatenate([t, u])

            t = t.cuda()
            return t

    def _multi_get_tensor(self, attrs: List[CuGraphTensorAttr]) -> List[TensorType]:
        return [self._get_tensor(attr) for attr in attrs]

    def multi_get_tensor(self, attrs: List[CuGraphTensorAttr]) -> List[TensorType]:
        """
        Synchronously obtains a :class:`FeatureTensorType` object from the
        feature store for each tensor associated with the attributes in
        `attrs`.

        Parameters
        ----------
        attrs (List[TensorAttr]): a list of :class:`TensorAttr` attributes
        that identify the tensors to get.

        Returns
        -------
        List[FeatureTensorType]: a Tensor of the same type as the index for
        each attribute.

        Raises
        ------
            KeyError: if a tensor corresponding to an attr was not found.
            ValueError: if any input `TensorAttr` is not fully specified.
        """
        attrs = [
            self._infer_unspecified_attr(self._tensor_attr_cls.cast(attr))
            for attr in attrs
        ]
        bad_attrs = [attr for attr in attrs if not attr.is_fully_specified()]
        if len(bad_attrs) > 0:
            raise ValueError(
                f"The input TensorAttr(s) '{bad_attrs}' are not fully "
                f"specified. Please fully specify them by specifying all "
                f"'UNSET' fields"
            )

        tensors = self._multi_get_tensor(attrs)

        bad_attrs = [attrs[i] for i, v in enumerate(tensors) if v is None]
        if len(bad_attrs) > 0:
            raise KeyError(
                f"Tensors corresponding to attributes " f"'{bad_attrs}' were not found"
            )

        return [tensor for attr, tensor in zip(attrs, tensors)]

    def get_tensor(self, *args, **kwargs) -> TensorType:
        """
        Synchronously obtains a :class:`FeatureTensorType` object from the
        feature store. Feature store implementors guarantee that the call
        :obj:`get_tensor(put_tensor(tensor, attr), attr) = tensor` holds.

        Parameters
        ----------
        **attr (TensorAttr): Any relevant tensor attributes that correspond
            to the feature tensor. See the :class:`TensorAttr`
            documentation for required and optional attributes. It is the
            job of implementations of a :class:`FeatureStore` to store this
            metadata in a meaningful way that allows for tensor retrieval
            from a :class:`TensorAttr` object.

        Returns
        -------
        FeatureTensorType: a Tensor of the same type as the index.

        Raises
        ------
        KeyError: if the tensor corresponding to attr was not found.
        ValueError: if the input `TensorAttr` is not fully specified.
        """

        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        attr = self._infer_unspecified_attr(attr)

        if not attr.is_fully_specified():
            raise ValueError(
                f"The input TensorAttr '{attr}' is not fully "
                f"specified. Please fully specify the input by "
                f"specifying all 'UNSET' fields."
            )

        tensor = self._get_tensor(attr)
        if tensor is None:
            raise KeyError(f"A tensor corresponding to '{attr}' was not found")
        return tensor

    def _get_tensor_size(self, attr: CuGraphTensorAttr) -> Union[List, int]:
        return self._get_tensor(attr).size()

    def get_tensor_size(self, *args, **kwargs) -> Union[List, int]:
        """
        Obtains the size of a tensor given its attributes, or :obj:`None`
        if the tensor does not exist.
        """
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        if not attr.is_set("index"):
            attr.index = None
        return self._get_tensor_size(attr)

    def _remove_tensor(self, attr):
        raise NotImplementedError("Removing features not supported")

    def _infer_unspecified_attr(self, attr: CuGraphTensorAttr) -> CuGraphTensorAttr:
        if attr.properties == _field_status.UNSET:
            # attempt to infer property names
            if attr.group_name in self._tensor_attr_dict:
                for n in self._tensor_attr_dict[attr.group_name]:
                    if attr.attr_name == n.attr_name:
                        attr.properties = n.properties
            else:
                raise KeyError(f"Invalid group name {attr.group_name}")

        if attr.dtype == _field_status.UNSET:
            # attempt to infer dtype
            if attr.group_name in self._tensor_attr_dict:
                for n in self._tensor_attr_dict[attr.group_name]:
                    if attr.attr_name == n.attr_name:
                        attr.dtype = n.dtype

        return attr

    def __len__(self):
        return len(self.get_all_tensor_attrs())

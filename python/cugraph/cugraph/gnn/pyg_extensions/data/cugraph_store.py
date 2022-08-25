# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

from cugraph.experimental import MGPropertyGraph

from typing import Optional, Tuple, Any
from enum import Enum

import cupy
import cudf
import dask_cudf
import cugraph

from dataclasses import dataclass


class EdgeLayout(Enum):
    COO = 'coo'
    CSC = 'csc'
    CSR = 'csr'


@dataclass
class CuGraphEdgeAttr:
    r"""Defines the attributes of an :obj:`GraphStore` edge."""

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


def EXPERIMENTAL__to_pyg(G, backend='torch'):
    """
        Returns the PyG wrappers for the provided PropertyGraph or
        MGPropertyGraph.

    Parameters
    ----------
    G : PropertyGraph or MGPropertyGraph
        The graph to produce PyG wrappers for.

    Returns
    -------
    Tuple (CuGraphFeatureStore, CuGraphStore)
    Wrappers for the provided property graph.
    """
    return (
        EXPERIMENTAL__CuGraphFeatureStore(G, backend=backend),
        EXPERIMENTAL__CuGraphStore(G, backend=backend)
    )


_field_status = Enum("FieldStatus", "UNSET")


@dataclass
class CuGraphTensorAttr:
    r"""Defines the attributes of a class:`FeatureStore` tensor; in particular,
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

    # Convenience methods #####################################################

    def is_set(self, key):
        r"""Whether an attribute is set in :obj:`TensorAttr`."""
        if key not in self.__dataclass_fields__:
            raise KeyError(key)
        attr = getattr(self, key)
        return type(attr) != _field_status or attr != _field_status.UNSET

    def is_fully_specified(self):
        r"""Whether the :obj:`TensorAttr` has no unset fields."""
        return all([self.is_set(key) for key in self.__dataclass_fields__])

    def fully_specify(self):
        r"""Sets all :obj:`UNSET` fields to :obj:`None`."""
        for key in self.__dataclass_fields__:
            if not self.is_set(key):
                setattr(self, key, None)
        return self

    def update(self, attr):
        r"""Updates an :class:`TensorAttr` with set attributes from another
        :class:`TensorAttr`."""
        for key in self.__dataclass_fields__:
            if attr.is_set(key):
                setattr(self, key, getattr(attr, key))

    @classmethod
    def cast(cls, *args, **kwargs):
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
    Duck-typed version of PyG's GraphStore.
    """
    def __init__(self, G, backend='torch'):
        """
            G : PropertyGraph or MGPropertyGraph
                The cuGraph property graph where the
                data is being stored.
            backend : The backend that manages tensors (default = 'torch')
                Should usually be 'torch' ('torch', 'cupy' supported).
        """

        if backend == 'torch':
            from torch.utils.dlpack import from_dlpack
            from torch import int64 as vertex_type
            from torch import float32 as property_type
        elif backend == 'cupy':
            from cupy import from_dlpack
            from cupy import int64 as vertex_type
            from cupy import float32 as property_type
        else:
            raise ValueError(f'Invalid backend {backend}.')
        self.__backend = backend
        self.from_dlpack = from_dlpack
        self.vertex_type = vertex_type
        self.property_type = property_type

        self.__graph = G
        self.__subgraphs = {}

        self.__edge_type_lookup_table = G.get_edge_data(
            columns=[
                G.src_col_name,
                G.dst_col_name,
                G.type_col_name
            ]
        )

        self.__edge_types_to_attrs = {}
        for edge_type in self.__graph.edge_types:
            edges = self.__graph.get_edge_data(types=[edge_type])
            dsts = edges[self.__graph.dst_col_name].unique()
            srcs = edges[self.__graph.src_col_name].unique()

            if self.is_mg:
                dsts = dsts.compute()
                srcs = srcs.compute()

            dst_types = self.__graph.get_vertex_data(
                vertex_ids=dsts,
                columns=[self.__graph.type_col_name]
            )[self.__graph.type_col_name].unique()

            src_types = self.__graph.get_vertex_data(
                vertex_ids=srcs,
                columns=['_TYPE_']
            )._TYPE_.unique()

            if self.is_mg:
                dst_types = dst_types.compute()
                src_types = src_types.compute()

            err_string = (
                f'Edge type {edge_type} associated'
                'with multiple src/dst type pairs'
            )
            if len(dst_types) > 1 or len(src_types) > 1:
                raise TypeError(err_string)

            pyg_edge_type = (src_types[0], edge_type, dst_types[0])

            self.__edge_types_to_attrs[edge_type] = CuGraphEdgeAttr(
                edge_type=pyg_edge_type,
                layout=EdgeLayout.COO,
                is_sorted=False,
                size=len(edges)
            )

            self._edge_attr_cls = CuGraphEdgeAttr

    @property
    def _edge_types_to_attrs(self):
        return dict(self.__edge_types_to_attrs)

    @property
    def is_mg(self):
        return isinstance(self.__graph, MGPropertyGraph)

    def put_edge_index(self, edge_index, edge_attr):
        raise NotImplementedError('Adding indices not supported.')

    def get_all_edge_attrs(self):
        """
            Returns all edge types and indices in this store.
        """
        return self.__edge_types_to_attrs.values()

    def _get_edge_index(self, attr):
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

        if attr.layout != EdgeLayout.COO:
            raise TypeError('Only COO direct access is supported!')

        if isinstance(attr.edge_type, str):
            edge_type = attr.edge_type
        else:
            edge_type = attr.edge_type[1]

        # If there is only one edge type (homogeneous graph) then
        # bypass the edge filters for a significant speed improvement.
        if len(self.__graph.edge_types) == 1:
            if list(self.__graph.edge_types)[0] != edge_type:
                raise ValueError(
                    f'Requested edge type {edge_type}'
                    'is not present in graph.'
                )

            df = self.__graph.get_edge_data(
                edge_ids=None,
                types=None,
                columns=[
                    self.__graph.src_col_name,
                    self.__graph.dst_col_name
                ]
            )
        else:
            # FIXME unrestricted edge type names
            df = self.__graph.get_edge_data(
                edge_ids=None,
                types=[edge_type],
                columns=[
                    self.__graph.src_col_name,
                    self.__graph.dst_col_name
                ]
            )

        if self.is_mg:
            df = df.compute()

        src = self.from_dlpack(df[self.__graph.src_col_name].to_dlpack())
        dst = self.from_dlpack(df[self.__graph.dst_col_name].to_dlpack())
        if self.__backend == 'torch':
            src = src.to(self.vertex_type)
            dst = dst.to(self.vertex_type)
        else:
            # self.__backend == 'cupy'
            src = src.astype(self.vertex_type)
            dst = dst.astype(self.vertex_type)

        if src.shape[0] != dst.shape[0]:
            raise IndexError('src and dst shape do not match!')

        return (src, dst)

    def get_edge_index(self, *args, **kwargs):
        r"""Synchronously gets an edge_index tensor from the materialized
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
        edge_attr.is_sorted = edge_attr.is_sorted or (edge_attr.layout in [
            EdgeLayout.CSC, EdgeLayout.CSR
        ])
        edge_index = self._get_edge_index(edge_attr)
        if edge_index is None:
            raise KeyError(f"An edge corresponding to '{edge_attr}' was not "
                           f"found")
        return edge_index

    def _subgraph(self, edge_types):
        """
        Returns a subgraph with edges limited to those of a given type

        Parameters
        ----------
        edge_types : list of edge types
            Directly references the graph's internal edge types.  Does
            not accept PyG edge type tuples.

        Returns
        -------
        The appropriate extracted subgraph.  Will extract the subgraph
        if it has not already been extracted.

        """
        edge_types = tuple(sorted(edge_types))

        if edge_types not in self.__subgraphs:
            query = f'(_TYPE_=="{edge_types[0]}")'
            for t in edge_types[1:]:
                query += f' | (_TYPE_=="{t}")'
            selection = self.__graph.select_edges(query)

            # FIXME enforce int type
            sg = self.__graph.extract_subgraph(
                selection=selection,
                default_edge_weight=1.0,
                allow_multi_edges=True,
                renumber_graph=True,
                add_edge_data=False
            )
            self.__subgraphs[edge_types] = sg

        return self.__subgraphs[edge_types]

    def neighbor_sample(
            self,
            index,
            num_neighbors,
            replace,
            directed,
            edge_types):

        if isinstance(num_neighbors, dict):
            # FIXME support variable num neighbors per edge type
            num_neighbors = list(num_neighbors.values())[0]

        # FIXME eventually get uniform neighbor sample to accept longs
        if self.__backend == 'torch' and not index.is_cuda:
            index = index.cuda()
        index = cupy.from_dlpack(index.__dlpack__())

        # FIXME resolve the directed/undirected issue
        G = self._subgraph([et[1] for et in edge_types])

        index = cudf.Series(index)
        if self.is_mg:
            uniform_neighbor_sample = cugraph.dask.uniform_neighbor_sample
        else:
            uniform_neighbor_sample = cugraph.uniform_neighbor_sample
        sampling_results = uniform_neighbor_sample(
                G,
                index,
                # conversion required by cugraph api
                list(num_neighbors),
                replace
            )

        concat_fn = dask_cudf.concat if self.is_mg else cudf.concat

        nodes_of_interest = concat_fn(
            [sampling_results.destinations, sampling_results.sources]
        ).unique()

        noi = self.__graph.get_vertex_data(
            nodes_of_interest.compute() if self.is_mg else nodes_of_interest,
            columns=[self.__graph.vertex_col_name, self.__graph.type_col_name]
        )

        noi_types = noi[self.__graph.type_col_name].unique()
        noi = noi.groupby(self.__graph.type_col_name)

        if self.is_mg:
            noi_types = noi_types.compute()
        noi_types = noi_types.to_pandas()

        # these should contain the original ids, they will be auto-renumbered
        noi_groups = {}
        for t in noi_types:
            v = noi.get_group(t)
            if self.is_mg:
                v = v.compute()

            noi_groups[t] = self.from_dlpack(
                v[self.__graph.vertex_col_name].to_dlpack()
            )

        eoi = cudf.merge(
            sampling_results,
            self.__edge_type_lookup_table,
            left_on=[
                'sources',
                'destinations'
            ],
            right_on=[
                self.__graph.src_col_name,
                self.__graph.dst_col_name
            ]
        )
        eoi_types = eoi[self.__graph.type_col_name].unique()
        eoi = eoi.groupby(self.__graph.type_col_name)

        if self.is_mg:
            eoi_types = eoi_types.compute()
        eoi_types = eoi_types.to_pandas()

        #    to be pre-renumbered;
        # the pre-renumbering must match
        # the auto-renumbering
        row_dict = {}
        col_dict = {}
        for t in eoi_types:
            t_pyg_type = self.__edge_types_to_attrs[t].edge_type
            t_pyg_c_type = edge_type_to_str(t_pyg_type)
            gr = eoi.get_group(t)
            if self.is_mg:
                gr = gr.compute()

            sources = gr.sources
            src_id_table = cudf.DataFrame(
                {'id': range(len(noi_groups[t_pyg_type[0]]))},
                index=cudf.from_dlpack(noi_groups[t_pyg_type[0]].__dlpack__())
            )

            src = self.from_dlpack(
                src_id_table.loc[sources].to_dlpack()
            )
            row_dict[t_pyg_c_type] = src

            destinations = gr.destinations
            dst_id_table = cudf.DataFrame(
                {'id': cupy.arange(len(noi_groups[t_pyg_type[2]]))},
                index=cudf.from_dlpack(noi_groups[t_pyg_type[2]].__dlpack__())
            )
            dst = self.from_dlpack(
                dst_id_table.loc[destinations].to_dlpack()
            )
            col_dict[t_pyg_c_type] = dst

        # FIXME handle edge ids
        return (noi_groups, row_dict, col_dict, None)


class EXPERIMENTAL__CuGraphFeatureStore:
    """
        Duck-typed version of PyG's FeatureStore.
    """
    def __init__(self, G, reserved_keys=[], backend='torch'):
        """
        G : PropertyGraph or MGPropertyGraph where the graph is stored.
        reserved_keys : Properties in the graph that are not used for
            training (the 'x' attribute will ignore these properties).
        backend : The tensor backend (default = 'torch')
            Should usually be 'torch' ('torch', 'cupy' supported).
        """

        if backend == 'torch':
            from torch.utils.dlpack import from_dlpack
            from torch import int64 as vertex_type
            from torch import float32 as property_type
        elif backend == 'cupy':
            from cupy import from_dlpack
            from cupy import int64 as vertex_type
            from cupy import float32 as property_type
        else:
            raise ValueError(f'Invalid backend {backend}.')

        self.__backend = backend
        self.from_dlpack = from_dlpack
        self.vertex_type = vertex_type
        self.property_type = property_type

        self.__graph = G
        self.__reserved_keys = list(reserved_keys)
        self.__dict__['_tensor_attr_cls'] = CuGraphTensorAttr

        # TODO ensure all x properties are float32 type
        # TODO ensure y is of long type

    @property
    def is_mg(self):
        return isinstance(self.__graph, MGPropertyGraph)

    def put_tensor(self, tensor, attr):
        raise NotImplementedError('Adding properties not supported.')

    def create_named_tensor(self, attr, properties):
        """
            Create a named tensor that contains a subset of
            properties in the graph.
        """
        # FIXME implement this to allow props other than x and y
        raise NotImplementedError('Not yet supported')

    def get_all_tensor_attrs(self):
        r"""Obtains all tensor attributes stored in this feature store."""
        attrs = []
        for vertex_type in self.__graph.vertex_types:
            # FIXME handle differing properties by type
            # once property graph supports it

            # FIXME allow props other than x and y
            attrs.append(
                CuGraphTensorAttr(vertex_type, 'x')
            )
            if 'y' in self.__graph.vertex_property_names:
                attrs.append(
                    CuGraphTensorAttr(vertex_type, 'y')
                )

        return attrs

    def _get_tensor(self, attr):
        if attr.attr_name == 'x':
            cols = None
        else:
            cols = [attr.attr_name]

        idx = attr.index
        if self.__backend == 'torch' and not idx.is_cuda:
            idx = idx.cuda()
        idx = cupy.from_dlpack(idx.__dlpack__())

        if len(self.__graph.vertex_types) == 1:
            # make sure we don't waste computation if there's only 1 type
            df = self.__graph.get_vertex_data(
                vertex_ids=idx,
                types=None,
                columns=cols
            )
        else:
            df = self.__graph.get_vertex_data(
                vertex_ids=idx,
                types=[attr.group_name],
                columns=cols
            )

        # FIXME allow properties other than x and y
        if attr.attr_name == 'x':
            if 'y' in df.columns:
                df = df.drop('y', axis=1)

        idx_cols = [
            self.__graph.type_col_name,
            self.__graph.vertex_col_name
        ]

        for dropcol in self.__reserved_keys + idx_cols:
            df = df.drop(dropcol, axis=1)

        if self.is_mg:
            df = df.compute()

        # FIXME handle vertices without properties
        output = self.from_dlpack(
            df.fillna(0).to_dlpack()
        )

        # FIXME look up the dtypes for x and other properties
        if attr.attr_name == 'x' and output.dtype != self.property_type:
            if self.__backend == 'torch':
                output = output.to(self.property_type)
            else:
                # self.__backend == 'cupy'
                output = output.astype(self.property_type)

        return output

    def _multi_get_tensor(self, attrs):
        return [self._get_tensor(attr) for attr in attrs]

    def multi_get_tensor(self, attrs):
        r"""Synchronously obtains a :class:`FeatureTensorType` object from the
        feature store for each tensor associated with the attributes in
        `attrs`.

        Args:
            attrs (List[TensorAttr]): a list of :class:`TensorAttr` attributes
                that identify the tensors to get.

        Returns:
            List[FeatureTensorType]: a Tensor of the same type as the index for
                each attribute.

        Raises:
            KeyError: if a tensor corresponding to an attr was not found.
            ValueError: if any input `TensorAttr` is not fully specified.
        """
        attrs = [self._tensor_attr_cls.cast(attr) for attr in attrs]
        bad_attrs = [attr for attr in attrs if not attr.is_fully_specified()]
        if len(bad_attrs) > 0:
            raise ValueError(
                f"The input TensorAttr(s) '{bad_attrs}' are not fully "
                f"specified. Please fully specify them by specifying all "
                f"'UNSET' fields")

        tensors = self._multi_get_tensor(attrs)

        bad_attrs = [attrs[i] for i, v in enumerate(tensors) if v is None]
        if len(bad_attrs) > 0:
            raise KeyError(f"Tensors corresponding to attributes "
                           f"'{bad_attrs}' were not found")

        return [
            tensor
            for attr, tensor in zip(attrs, tensors)
        ]

    def get_tensor(self, *args, **kwargs):
        r"""Synchronously obtains a :class:`FeatureTensorType` object from the
        feature store. Feature store implementors guarantee that the call
        :obj:`get_tensor(put_tensor(tensor, attr), attr) = tensor` holds.

        Args:
            **attr (TensorAttr): Any relevant tensor attributes that correspond
                to the feature tensor. See the :class:`TensorAttr`
                documentation for required and optional attributes. It is the
                job of implementations of a :class:`FeatureStore` to store this
                metadata in a meaningful way that allows for tensor retrieval
                from a :class:`TensorAttr` object.

        Returns:
            FeatureTensorType: a Tensor of the same type as the index.

        Raises:
            KeyError: if the tensor corresponding to attr was not found.
            ValueError: if the input `TensorAttr` is not fully specified.
        """

        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        if not attr.is_fully_specified():
            raise ValueError(f"The input TensorAttr '{attr}' is not fully "
                             f"specified. Please fully specify the input by "
                             f"specifying all 'UNSET' fields.")

        tensor = self._get_tensor(attr)
        if tensor is None:
            raise KeyError(f"A tensor corresponding to '{attr}' was not found")
        return tensor

    def _get_tensor_size(self, attr):
        return self._get_tensor(attr).size

    def get_tensor_size(self, *args, **kwargs):
        r"""Obtains the size of a tensor given its attributes, or :obj:`None`
        if the tensor does not exist."""
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        if not attr.is_set('index'):
            attr.index = None
        return self._get_tensor_size(attr)

    def _remove_tensor(self, attr):
        raise NotImplementedError('Removing features not supported')

    def __len__(self):
        return len(self.get_all_tensor_attrs())


def edge_type_to_str(edge_type):
    """
    Converts the PyG (src, type, dst) edge representation into
    the equivalent C++ representation.

    edge_type : The PyG (src, type, dst) tuple edge representation
        to convert to the C++ representation.
    """
    # Since C++ cannot take dictionaries with tuples as key as input, edge type
    # triplets need to be converted into single strings.
    return edge_type if isinstance(edge_type, str) else '__'.join(edge_type)

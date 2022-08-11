from re import I
from cugraph.experimental import PropertyGraph
from cugraph.experimental import MGPropertyGraph

from torch_geometric.data.feature_store import (
    FeatureStore,
    TensorAttr
)
from torch_geometric.data.graph_store import (
    GraphStore,
    EdgeAttr,
    EdgeLayout
)

from typing import List, Optional, Tuple, Any
from torch_geometric.typing import (
    FeatureTensorType,
    EdgeTensorType
)

from torch_geometric.loader.utils import edge_type_to_str

import torch
import cupy
import cudf
import dask_cudf
import cugraph
from datetime import datetime

def to_pyg(G: PropertyGraph):
    return CuGraphFeatureStore(G), CuGraphStore(G)

class CuGraphStore(GraphStore):
    def __init__(self, G):
        """
            G : PropertyGraph or MGPropertyGraph
                The cuGraph property graph where the 
                data is being stored.
        """
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

            dst_types = self.__graph.get_vertex_data(vertex_ids=dsts, columns=['_TYPE_'])._TYPE_.unique()
            src_types = self.__graph.get_vertex_data(vertex_ids=srcs, columns=['_TYPE_'])._TYPE_.unique()

            if self.is_mg:
                dst_types = dst_types.compute()
                src_types = src_types.compute()
            
            if len(dst_types) > 1 or len(src_types) > 1:
                raise TypeError(f'Edge type {edge_type} has multiple src/dst type pairs associated with it')
            
            pyg_edge_type = (src_types[0], edge_type, dst_types[0])

            self.__edge_types_to_attrs[edge_type] = EdgeAttr(
                edge_type=pyg_edge_type,
                layout=EdgeLayout.COO,
                is_sorted=False,
                size=len(edges)
            )

        super().__init__()
    
    @property
    def is_mg(self):
        return isinstance(self.__graph, MGPropertyGraph)
    
    def _put_edge_index(self, edge_index: EdgeTensorType,
                        edge_attr: EdgeAttr) -> bool:
        raise NotImplementedError('Adding indices not supported.')

    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        """
            Returns all edge types and indices in this store.
        """
        return self.__edge_types_to_attrs.values()
    
    def _get_edge_index(self, attr: EdgeAttr) -> Optional[EdgeTensorType]:
        # returns the edge index for particular edge type

        if attr.layout != EdgeLayout.COO:
            raise TypeError('Only COO direct access is supported!')

        if len(self.__graph.edge_types) == 1:
            df = self.__graph.get_edge_data(
                edge_ids=None,
                types=None,
                columns=[
                    self.__graph.src_col_name,
                    self.__graph.dst_col_name
                ]
            )
        else:
            if isinstance(attr.edge_type, str):
                edge_type = attr.edge_type
            else:
                edge_type = attr.edge_type[1]

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
        
        src = torch.from_dlpack(df[self.__graph.src_col_name].to_dlpack()).to(torch.long)
        dst = torch.from_dlpack(df[self.__graph.dst_col_name].to_dlpack()).to(torch.long)
        assert src.shape[0] == dst.shape[0]

        # dst/src are flipped in PyG
        return (src, dst)
    
    def _subgraph(self, edge_types:Tuple[str]):
        """
        Returns a subgraph with edges limited to those of a given type
        """
        edge_types = tuple(edge_types)
        
        if edge_types not in self.__subgraphs:
            query = f'(_TYPE_=="{edge_types[0]}")'
            for t in edge_types[1:]:
                query += f' | (_TYPE_=="{t}")'
            selection = self.__graph.select_edges(query)

            sg = self.__graph.extract_subgraph(
                selection=selection,
                default_edge_weight=1.0,
                allow_multi_edges=True,
                renumber_graph=False, # FIXME enforce int type
                add_edge_data=False
            )
            self.__subgraphs[edge_types] = sg

        return self.__subgraphs[edge_types]

    @torch.no_grad()
    def neighbor_sample(
            self,
            index: torch.Tensor,
            num_neighbors: torch.Tensor,
            replace: bool,
            directed: bool,
            edge_types: List[Tuple[str]]) -> Any:
        
        start_time = datetime.now()
        
        if isinstance(num_neighbors, dict):
            # FIXME support variable num neighbors per edge type
            num_neighbors = list(num_neighbors.values())[0]

        if not isinstance(index, torch.Tensor):
            index = torch.Tensor(index)
        if not isinstance(num_neighbors, torch.Tensor):
            num_neighbors = torch.Tensor(num_neighbors).to(torch.long)
        
        if not index.is_cuda:
            index = index.cuda()
        if num_neighbors.is_cuda:
            num_neighbors = num_neighbors.cpu()
        
        if not index.dtype == torch.int32:
            index = index.to(torch.int32)

        if not num_neighbors.dtype == torch.int32:
            num_neighbors = num_neighbors.to(torch.int32)

        index = cupy.from_dlpack(index.__dlpack__())

        # FIXME resolve the directed/undirected issue
        #G = self.graph.extract_subgraph(add_edge_data=False, default_edge_weight=1.0, allow_multi_edges=True)
        G = self._subgraph([et[1] for et in edge_types])
        
        sampling_start = datetime.now()
        index = cudf.Series(index)
        if self.is_mg:
            sampling_results = cugraph.dask.uniform_neighbor_sample(
                G,
                index,
                list(num_neighbors), # conversion required by cugraph api
                replace
            )
        else:
            sampling_results = cugraph.uniform_neighbor_sample(
                G,
                index,
                list(num_neighbors), # conversion required by cugraph api
                replace
            )

        VERBOSE = True

        concat_fn = dask_cudf.concat if self.is_mg else cudf.concat

        end_time = datetime.now()
        td = end_time - start_time
        if VERBOSE:
            print('first half', td.total_seconds())
            print('sampling', (end_time - sampling_start).total_seconds())

        start_time = datetime.now()

        noi_start = datetime.now()
        nodes_of_interest = concat_fn(
            [sampling_results.destinations, sampling_results.sources]
        ).unique()
        noi_end = datetime.now()
        print('noi time:', (noi_end - noi_start).total_seconds())

        gd_start = datetime.now()
        noi = self.__graph.get_vertex_data(
            nodes_of_interest.compute() if self.is_mg else nodes_of_interest,
            columns=[self.__graph.vertex_col_name, self.__graph.type_col_name]
        )

        gd_end = datetime.now()
        print('get_vertex_data call:', (gd_end - gd_start).total_seconds())

        noi_group_start = datetime.now()
        noi_types = noi[self.__graph.type_col_name].unique()
        if len(noi_types) > 1:
            noi = noi.groupby(self.__graph.type_col_name)
        else:
            class FakeGroupBy:
                def __init__(self, df):
                    self.df = df

                def get_group(self, g):
                    return self.df
            noi = FakeGroupBy(noi)

        if self.is_mg:
            noi_types = noi_types.compute()
        noi_types = noi_types.to_pandas()

        noi_group_end = datetime.now()
        print('noi group time:', (noi_group_end - noi_group_start).total_seconds())

        # these should contain the original ids, they will be auto-renumbered
        noi_groups = {}
        for t in noi_types:
            v = noi.get_group(t)
            if self.is_mg:
                v = v.compute()
            noi_groups[t] = torch.from_dlpack(v[self.__graph.vertex_col_name].to_dlpack())

        eoi_group_start = datetime.now()        
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

        eoi_group_end = datetime.now()
        print('eoi_group_time:', (eoi_group_end - eoi_group_start).total_seconds())

        # PyG expects these to be pre-renumbered; the pre-renumbering must match
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
                {'id':range(len(noi_groups[t_pyg_type[0]]))},
                index=cudf.from_dlpack(noi_groups[t_pyg_type[0]].__dlpack__())
            )

            src = torch.from_dlpack(
                src_id_table.loc[sources].to_dlpack()
            )
            row_dict[t_pyg_c_type] = src
            
            destinations = gr.destinations
            dst_id_table = cudf.DataFrame(
                {'id':range(len(noi_groups[t_pyg_type[2]]))},
                index=cudf.from_dlpack(noi_groups[t_pyg_type[2]].__dlpack__())
            )
            dst = torch.from_dlpack(
                dst_id_table.loc[destinations].to_dlpack()
            )
            col_dict[t_pyg_c_type] = dst
        
        end_time = datetime.now()
        print('second half:', (end_time - start_time).total_seconds())

        #FIXME handle edge ids
        return (noi_groups, row_dict, col_dict, None)
        

class CuGraphFeatureStore(FeatureStore):
    def __init__(self, G:PropertyGraph, reserved_keys=[]):
        self.__graph = G
        self.__reserved_keys = list(reserved_keys)
        super().__init__()

        #TODO ensure all x properties are float32 type
        #TODO ensure y is of long type
    
    @property
    def is_mg(self):
        return isinstance(self.__graph, MGPropertyGraph)
    
    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        raise NotImplementedError('Adding properties not supported.')
    
    def create_named_tensor(self, attr:TensorAttr, properties:list) -> None:
        """
            Create a named tensor that contains a subset of
            properties in the graph.
        """
        #FIXME implement this to allow props other than x and y
        raise NotImplementedError('Not yet supported')
    
    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        r"""Obtains all tensor attributes stored in this feature store."""
        attrs = []
        for vertex_type in self.__graph.vertex_types:
             #FIXME handle differing properties by type 
             # once property graph supports it

            #FIXME allow props other than x and y
             attrs.append(
                TensorAttr(vertex_type, 'x')
             )
             if 'y' in self.__graph.vertex_property_names:
                attrs.append(
                    TensorAttr(vertex_type, 'y')
                )
            
        return attrs
    
    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        if attr.attr_name == 'x':
            cols = None
        else:
            cols = [attr.attr_name]
        
        idx = attr.index
        if not idx.is_cuda:
            idx = idx.cuda()
        idx = cupy.fromDlpack(idx.__dlpack__())

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

        #FIXME allow properties other than x and y
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

        #FIXME handle vertices without properties
        output = torch.from_dlpack(
            df.fillna(0).to_dlpack()
        )

        # FIXME look up the dtypes for x and other properties
        if attr.attr_name == 'x' and output.dtype != torch.float:
            output = output.to(torch.float)

        return output

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple:
        return self._get_tensor(attr).size()

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        raise NotImplementedError('Removing features not supported')
    
    def __len__(self):
        return len(self.get_all_tensor_attrs())
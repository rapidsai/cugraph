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

import re
from collections import defaultdict
from collections.abc import Callable, Collection, Iterator, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

Comparable = Any
DtypeLike = Any
DataFrame = Any
Series = Any
Self = Any

DEFAULT_IDS_DTYPE = np.int64


class SingleOrCollection:
    def __class_getitem__(cls, key: type) -> Any:
        return key | Collection[key]


class EdgeKey(NamedTuple):
    src_type: str  # Source
    edge_type: str  # Relation
    dst_type: str  # Object


class Dtype:
    """Datatype abstraction for PropertyGraph properties.

    This typically wraps a numpy dtype such as ``np.int32``, but can also support
    pandas dtypes and custom dtypes such as vectors.
    """

    # Immutable (for now)

    _dtype: DtypeLike

    def __init__(self, dtype: DtypeLike):
        self._dtype = dtype

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Dtype):
            try:
                other = Dtype(other)
            except Exception as exc:
                raise TypeError from exc
        return type(self) is type(other) and self._dtype == other._dtype

    def __hash__(self) -> int:
        return hash(self._dtype)

    @property
    def is_nullable(self) -> bool:
        raise NotImplementedError

    # numpy_dtype, cupy_dtype, pandas_dtype, cudf_dtype, etc.
    # Maybe: itemsize, is_signed_integer, name, kind, fill_value, etc.


class VectorDtype(Dtype):
    """Dtype representing a property that is a 1-d array."""

    _size: int
    _name: str
    _suffix_pattern: str
    _pattern: re.Pattern
    _sort_key: Callable[[re.Match], Comparable]

    _default_suffix_pattern: str = r"([\s_-]*)(\d+)(\s*)"

    @staticmethod
    def _default_sort_key(match: re.Match) -> tuple[int, str]:
        # Assume last group is the integer we want to sort by
        return (int(match.groups[-1]), match.group(0))

    def __init__(
        self,
        dtype: DtypeLike,
        size: int,
        name: str,
        *,
        suffix_pattern: str | None = None,
        sort_key: Callable[[re.Match], Comparable] | None = None,
    ):
        # Prepare regex for matching column names that comprise vector data
        super().__init__(dtype)
        self._size = size
        self._name = name
        if suffix_pattern is None:
            suffix_pattern = self._default_suffix_pattern
        self._suffix_pattern = suffix_pattern
        self._pattern: re.Pattern = re.compile(f"^({re.escape(name)}){suffix_pattern}$")
        if sort_key is None:
            sort_key = self._default_sort_key
        self._sort_key = sort_key

    @property
    def name(self) -> str:
        return self._name

    @property
    def pattern(self) -> re.Pattern:
        return self._pattern

    @property
    def size(self) -> int:
        return self._size

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Dtype):
            raise TypeError
        return (
            type(self) is type(other)
            and self._size == other._size
            and self._pattern == other._pattern
            and self._sort_key == other._sort_key
        )

    def __hash__(self) -> int:
        return hash((self._size, self._pattern, self._sort_key))


class TypeBase(Mapping):
    """Base class for type metadata for edge and vertex properties.

    An edge or vertex type is typically associated with a single dataframe
    of ids to properties, and the type is named upon construction.

    As the type metadata, TypeBase is a mapping from property name to dtype.
    """

    # Mutable (for now)
    # TODO: specify where/how data is stored?
    # For properties to be split, we may want e.g. `CompositeType`.

    _name: str
    _dtypes: dict[str, Dtype]
    _ids_dtype: Dtype

    def __init__(
        self,
        name: str,
        dtypes: Mapping[str, DtypeLike],
        ids_dtype: DtypeLike = DEFAULT_IDS_DTYPE,
    ):
        self._name = name
        self._dtypes = dtypes  # {property_name: Dtype}
        self._ids_dtype = ids_dtype

    @property
    def name(self) -> str:
        return self._name

    def _rename(self, name: str) -> Self:
        rv = object.__new__(type(self))
        for key, val in self.__dict__.items():
            if isinstance(val, dict | list):
                val = val.copy()  # Shallow copy
            rv.__dict__[key] = val
        rv._name = name
        return rv

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TypeBase):
            raise TypeError
        return (
            type(self) is type(other)
            and self._name == other._name
            and self._dtypes == other._dtypes
            and self._ids_dtype == other._ids_dtype
        )

    # Required Mapping methods
    # Implied methods: __contains__, keys, items, values, get, __eq__, __ne__
    def __getitem__(self, key: str) -> Dtype:
        return self._dtypes[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._dtypes)

    def __len__(self) -> int:
        return len(self._dtypes)


class VertexType(TypeBase):
    """Metadata for a vertex type that maps property name to dtype."""


class EdgeType(TypeBase):
    """Metadata for an edge type that maps property name to dtype."""

    src_types: dict[str, VertexType]
    dst_types: dict[str, VertexType]
    _ids_autogenerated: bool | None

    def __init__(
        self,
        name: str,
        dtypes: Mapping[str, DtypeLike],
        ids_dtype: DtypeLike = DEFAULT_IDS_DTYPE,
        src_types: SingleOrCollection[VertexType] | None = None,
        dst_types: SingleOrCollection[VertexType] | None = None,
        *,
        autogenerate_ids: bool | None = None,
    ):
        super().__init__(name, dtypes, ids_dtype)
        if src_types is None:
            src_types = ()
        elif isinstance(src_types, VertexType):
            src_types = [src_types]
        if dst_types is None:
            dst_types = ()
        elif isinstance(dst_types, VertexType):
            dst_types = [dst_types]
        self.src_types = {val.name: val for val in src_types}
        self.dst_types = {val.name: val for val in dst_types}
        self._ids_autogenerated = autogenerate_ids

    def __eq__(self, other: Any) -> bool:
        return (
            super().__eq__(other)
            and self.src_types == other.src_types
            and self.dst_types == other.dst_types
        )

    @property
    def ids_autogenerated(self) -> bool | None:
        # True, False, or None
        # None means no edge data has been added and auto-generation is undecided
        return self._ids_autogenerated


class TypesBase(Mapping):
    """Base class for metadata for all vertex types or edge types."""

    _kind: str = ""
    _type: type[TypeBase] = TypeBase

    def __init__(self, parent: PropertyGraph):
        self._parent = parent
        self._types = {}  # {type_name: type}

    @property
    def parent(self) -> PropertyGraph:
        return self._parent

    @property
    def properties(self) -> dict[str, set[str]]:
        # {property_name: {type_names}}
        # Could also be `{property_name: [type]}`, but type_name is probably nicer
        rv = defaultdict(set)
        for type_name, type_ in self._types.items():
            for property_name in type_:
                rv[property_name].add(type_name)
        return dict(rv)

    # Required Mapping methods (and __getitem__)
    # Implied methods: __contains__, keys, items, values, get, __eq__, __ne__
    def __iter__(self) -> Iterator[str]:
        return iter(self._types)

    def __len__(self) -> int:
        return len(self._types)


class VertexTypes(TypesBase):
    """Metadata for vertex types that maps vertex type names to vertex types."""

    # TODO: indicate whether some vertex ids are allowed to only appear in edge data.

    _kind = "vertex"
    _type = VertexType

    def _add(
        self,
        name: str,
        dtypes: Mapping[str, DtypeLike],
        *,
        ids_dtype: DtypeLike = DEFAULT_IDS_DTYPE,
    ) -> None:
        if name in self._types:
            raise ValueError(f"{name} {self._kind} type already exists")
        if not isinstance(dtypes, self._type):
            dtypes = self._type(name, dtypes, ids_dtype)
        elif isinstance(dtypes, TypeBase):
            raise TypeError
        else:  # Always make a copy
            # elif dtypes.name != name:  # Only copy when necessary
            dtypes = dtypes._rename(name)
        self._types[name] = dtypes

    def __getitem__(self, key: str) -> VertexType:
        return self._types[key]


class EdgeTypes(TypesBase):
    """Metadata for edge types that maps edge type names to edge types."""

    _kind = "edge"
    _type = EdgeType

    def _add(
        self,
        name: str,
        dtypes: Mapping[str, DtypeLike],
        *,
        ids_dtype: DtypeLike = DEFAULT_IDS_DTYPE,
        src_types: SingleOrCollection[VertexTypeLike] | None = None,
        dst_types: SingleOrCollection[VertexTypeLike] | None = None,
    ) -> None:
        if name in self._types:
            raise ValueError(f"{name} {self._kind} type already exists")

        # Convert vertex type strings to VertexType
        if isinstance(src_types, str):
            src_types = self._parent.vertex_types[src_types]
        elif src_types is not None:
            src_types = [
                self._parent.vertex_types[x] if isinstance(x, str) else x
                for x in src_types
            ]
        if isinstance(dst_types, str):
            dst_types = self._parent.vertex_types[dst_types]
        elif dst_types is None:
            dst_types = [
                self._parent.vertex_types[x] if isinstance(x, str) else x
                for x in dst_types
            ]

        if not isinstance(dtypes, self._type):
            dtypes = self._type(
                name, dtypes, ids_dtype, src_types=src_types, dst_types=dst_types
            )
        elif isinstance(dtypes, TypeBase):
            raise TypeError
        else:  # Always make a copy
            dtypes = dtypes._rename(name)
            if src_types is not None:
                dtypes.src_types = src_types
            if dst_types is not None:
                dtypes.dst_types = dst_types
        self._types[name] = dtypes

    def __getitem__(self, key: str) -> EdgeType:
        return self._types[key]


class DataBase:
    """Base class for implementations that hold and operate on edge and vertex data."""

    _parent: PropertyGraph
    _vertex_dfs: dict[str, DataFrame]
    _edge_dfs: dict[EdgeKey, DataFrame]

    def __init__(self, parent: PropertyGraph):
        self._parent = parent
        self._vertex_dfs = {}  # {type_name: df}
        self._edge_dfs = {}  # {(edgetype_name, srctype_name, dsttype_name): df}

    @property
    def parent(self) -> PropertyGraph:
        return self._parent


class PandasData(DataBase):
    """Pandas backend."""

    _vertex_dfs: dict[str, pd.DataFrame]
    _edge_dfs: dict[EdgeKey, pd.DataFrame]


class TypedVertexData:
    """TBD: result of accessor for vertex data.

    Backend-specific functionality can be added here, which can allow
    implementations to be "leaky" in a clean manner.
    """


class TypedEdgeData:
    """TBD: result of accessor for edge data.

    Backend-specific functionality can be added here, which can allow
    implementations to be "leaky" in a clean manner.
    """


class VertexData(Mapping):
    """An accessor for conveniently accessing vertex data."""

    _parent: PropertyGraph
    _data: dict[str, TypedVertexData]

    def __init__(self, parent: PropertyGraph, type_name: str):
        self._parent = parent
        self._type_name = type_name
        self._data = {}

    @property
    def type(self) -> VertexType:
        return self._parent.vertex_types[self._type_name]

    @property
    def ids(self) -> Series:  # Or -> Ids or VertexIds
        return self._parent.vertex_ids[self._type_name]

    # Required Mapping methods
    # Implied methods: __contains__, keys, items, values, get, __eq__, __ne__
    def __getitem__(self, key: str) -> Dtype:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class EdgeData(Mapping):
    """An accessor for conveniently accessing edge data."""

    _parent: PropertyGraph
    _data: dict[str, TypedEdgeData]

    def __init__(self, parent: PropertyGraph, type_name: str):
        self._parent = parent
        self._type_name = type_name
        self._data = {}

    @property
    def type(self) -> EdgeType:
        return self._parent.edge_types[self._type_name]

    @property
    def ids(self) -> Series:  # Or -> Ids or EdgeIds
        return self._parent.edge_ids[self._type_name]

    # Required Mapping methods
    # Implied methods: __contains__, keys, items, values, get, __eq__, __ne__
    def __getitem__(self, key: str) -> Dtype:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class PropertyGraph:
    _data: dict[str, DataBase]
    vertex_types: VertexTypes  # Like Mapping[str, VertexType]
    edge_types: EdgeTypes  # Like Mapping[str, EdgeType]

    vertex_ids: Mapping[str, Series]  # Or specific type: VertexIds
    edge_ids: Mapping[str, Series]  # Or specific type: EdgeIds

    vertex_data: VertexData  # Mapping[str, TypedVertexData]  # An accessor
    edge_data: EdgeData  # Mapping[str, TypedEdgeData]  # An accessor

    # Examples:
    # vertex_data[type_name].type <--> vertex_types[type_name]
    # vertex_data[type_name].ids <--> vertex_ids[type_name]
    # vertex_data[type_name].fillna()
    # vertex_data[type_name].issorted
    # vertex_data[type_name].persist()  # Or .dask.persist()
    # vertex_data[type_name].dask.rechunk("100 MiB")

    def __init__(self) -> None:
        self._data = {"pandas": PandasData()}
        self.vertex_types = VertexTypes(self)
        self.edge_types = EdgeTypes(self)

    def add_vertex_type(
        self,
        type_or_name: VertexTypeLike,
        dtypes: Mapping[str, DtypeLike] | None = None,
        *,
        ids_dtype: DtypeLike = DEFAULT_IDS_DTYPE,
    ) -> None:
        """Add new type for vertex data.

        Examples
        --------
        >>> pg.add_vertex_type("books", {"pages": int, "author": str})

        Or

        >>> vertex_type = VertexType("books", {"pages": int, "author": str})
        >>> pg.add_vertex_type(vertex_type)
        """
        if isinstance(type_or_name, TypeBase):
            if dtypes is not None:
                raise TypeError
            dtypes = type_or_name
            name = type_or_name.name
        else:
            name = type_or_name
        self.vertex_types._add(name, dtypes, ids_dtype=ids_dtype)

    def add_vertex_data(
        self,
        df: DataFrame,
        vertex_type: VertexTypeLike,
        ids_col: str = "ids",
        *,
        infer_type: bool = False,
        drop_extra: bool = False,
        where: str = "auto",
    ) -> None:
        """Add vertex data for a given type.

        Examples
        --------
        Here are three ways to add data for a new type.

        Simplest:

        >>> pg.add_vertex_data(df, "books", infer_type=True)

        Simple and explicit:

        >>> pg.add_vertex_type("books", {"pages": int, "author": str})
        >>> pg.add_vertex_data(df, "books")

        Most explicit:

        >>> vertex_type = VertexType("books", {"pages": int, "author": str})
        >>> pg.add_vertex_type(vertex_type)
        >>> pg.add_vertex_data(df, vertex_type)
        """
        # Adds new rows for the given vertex type
        # TODO: can/should we detect if ids already exist?
        ...

    def add_vertex_properties(
        self,
        df: DataFrame,
        vertex_type: VertexTypeLike,
        extra_vertex_type: VertexTypeLike | Mapping[str, DtypeLike],
        ids_col: str = "ids",
        *,
        infer_type: bool = False,
        drop_extra: bool = False,
    ) -> None:
        """Add new vertex properties (no new ids)"""
        # Adds new columns for the given vertex type
        ...

    def update_vertex_data(
        self,
        df: DataFrame,
        vertex_type: VertexTypeLike,
        extra_vertex_type: VertexTypeLike | Mapping[str, DtypeLike] | None = None,
        ids_col: str = "ids",
        *,
        may_add_ids: bool = False,
        may_add_properties: bool = False,
        infer_type: bool = False,
        drop_extra: bool = False,
    ) -> None:
        # Set ``may_add_ids`` to True to allow rows for new ids to be added
        # Set ``may_add_properties`` to True to allow new properties to be added
        ...

    def remove_vertex_data(
        self,
        types: SingleOrCollection[VertexTypeLike] | None = None,
        ids: SingleOrCollection[int] | None = None,
    ) -> None:
        # W/o specifying ids, remove all data for vertex type, but edges
        # may still use this type as src or dst type.
        # Should this return the removed data?
        ...

    def remove_vertex_properties(
        self,
        vertex_type: VertexTypeLike,
        properties: SingleOrCollection[str],
    ) -> None:
        # This changes the VertexType object.
        # Should this return the removed data?
        ...

    def get_vertex_data(
        self,
        types: SingleOrCollection[VertexTypeLike] | None = None,
        ids: SingleOrCollection[int] | None = None,
        properties: SingleOrCollection[str] | None = None,
        ids_col: str | None = None,
    ) -> dict[str, DataFrame]:
        # If ids_col is None, ids are the index w/ default name,
        # else the ids are moved to a column
        ...

    def get_vertex_dataframe(
        self,
        type_col: str = "type",
        types: SingleOrCollection[VertexTypeLike] | None = None,
        ids: SingleOrCollection[int] | None = None,
        properties: SingleOrCollection[str] | None = None,
        ids_col: str | None = None,
        where: str = "auto",
    ) -> DataFrame:
        ...

    def add_edge_type(
        self,
        type_or_name: EdgeTypeLike,
        dtypes: Mapping[str, DtypeLike] | None = None,
        *,
        ids_dtype: DtypeLike = DEFAULT_IDS_DTYPE,
        src_types: SingleOrCollection[VertexTypeLike] | None = None,
        dst_types: SingleOrCollection[VertexTypeLike] | None = None,
    ) -> None:
        if isinstance(type_or_name, TypeBase):
            if dtypes is not None:
                raise TypeError
            dtypes = type_or_name
            name = type_or_name.name
        else:
            name = type_or_name
        self.vertex_types._add(
            name, dtypes, ids_dtype=ids_dtype, src_types=src_types, dst_types=dst_types
        )

    def add_edge_data(
        self,
        df: DataFrame,
        edge_type: EdgeTypeLike,  # EdgeType or name
        ids_col: str | None = None,
        src_col: str = "src",
        dst_col: str = "dst",
        *,
        src_type: VertexTypeLike | None = None,
        dst_type: VertexTypeLike | None = None,
        infer_type: bool = False,
        drop_extra: bool = False,
        where: str = "auto",
    ) -> None:
        ...

    def add_edge_properties(
        self,
        df: DataFrame,
        edge_type: EdgeTypeLike,
        extra_edge_type: EdgeTypeLike | Mapping[str, DtypeLike],
        ids_col: str = "ids",
        src_col: str = "src",
        dst_col: str = "dst",
        *,
        src_type: VertexTypeLike | None = None,
        dst_type: VertexTypeLike | None = None,
        infer_type: bool = False,
        drop_extra: bool = False,
    ) -> None:
        ...

    def update_edge_data(
        self,
        df: DataFrame,
        edge_type: EdgeTypeLike,
        extra_edge_type: EdgeTypeLike | Mapping[str, DtypeLike],
        ids_col: str = "ids",
        src_col: str = "src",
        dst_col: str = "dst",
        *,
        src_type: VertexTypeLike | None = None,
        dst_type: VertexTypeLike | None = None,
        may_add_ids: bool = False,
        may_add_properties: bool = False,
        infer_type: bool = False,
        drop_extra: bool = False,
    ) -> None:
        ...

    def remove_edge_data(
        self,
        src_types: SingleOrCollection[VertexTypeLike] | None = None,  # Source
        edge_types: SingleOrCollection[EdgeTypeLike] | None = None,  # Relation
        dst_types: SingleOrCollection[VertexTypeLike] | None = None,  # Object
        ids: SingleOrCollection[int] | None = None,
    ) -> None:
        # Should this return the removed data?
        ...

    def remove_edge_properties(
        self,
        edge_type: EdgeTypeLike,
        properties: SingleOrCollection[str],
    ) -> None:
        # Should this return the removed data?
        ...

    def get_edge_data(
        self,
        src_types: SingleOrCollection[VertexTypeLike] | None = None,  # Source
        edge_types: SingleOrCollection[EdgeTypeLike] | None = None,  # Relation
        dst_types: SingleOrCollection[VertexTypeLike] | None = None,  # Object
        ids: SingleOrCollection[int] | None = None,
        properties: SingleOrCollection[str] | None = None,
        ids_col: str | None = None,
    ) -> dict[EdgeKey, DataFrame]:
        # Is this return type too specific to our expected implementation?
        ...

    def get_edge_dataframe(
        self,
        src_type_col: str = "src_type",  # Source
        edge_type_col: str = "edge_type",  # Relation
        dst_type_col: str = "dst_type",  # Object
        src_types: SingleOrCollection[VertexTypeLike] | None = None,  # Source
        edge_types: SingleOrCollection[EdgeTypeLike] | None = None,  # Relation
        dst_types: SingleOrCollection[VertexTypeLike] | None = None,  # Object
        ids: SingleOrCollection[int] | None = None,
        properties: SingleOrCollection[str] | None = None,
        ids_col: str | None = None,
        where: str = "auto",
    ) -> DataFrame:
        ...

    # Misc...

    def __len__(self) -> int:
        """The number of vertices in the graph"""
        ...

    def number_of_vertices(
        self,
        types: SingleOrCollection[VertexTypeLike] | None = None,
        include_edge_data: bool = True,
        # selection=None  # From old PG
    ) -> int:
        ...

    def number_of_edges(
        self,
        types: SingleOrCollection[EdgeTypeLike] | None = None,
        # src_types: Optional[SingleOrCollection[VertexTypeLike]] = None,  # Source
        # edge_types: Optional[SingleOrCollection[EdgeTypeLike]] = None,  # Relation
        # dst_types: Optional[SingleOrCollection[VertexTypeLike]] = None,  # Object
    ) -> int:
        ...

    def renumber_vertices_by_type(self, prev_id_col: str | None = None) -> DataFrame:
        # Can we know whether (and which) vertices exist only in edges?
        ...

    def renumber_edges_by_type(self, prev_id_col: str | None = None) -> DataFrame:
        ...

    # def fillna_vertices(self, ...):
    # def fillna_edges(self, ...):

    # TODO: filter queries!


class PandasPropertyGraph(PropertyGraph):
    _data: dict[str, PandasData]

    def __init__(self) -> None:
        # if pd is None: "needs pandas installed"
        super().__init__()
        self._data = {"pandas": PandasData()}

    # We could redefine methods w/o `where=`, but base PropertyGraph
    # should handle checks for us


# CudfPropertyGraph, DaskPropertyGraph, DaskCudfPropertyGraph

VertexTypeLike = str | VertexType
EdgeTypeLike = str | EdgeType

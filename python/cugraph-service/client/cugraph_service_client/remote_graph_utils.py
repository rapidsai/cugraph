# Copyright (c) 2022-2023, NVIDIA CORPORATION.
#
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

import importlib
import numpy as np


class MissingModule:
    """
    Raises RuntimeError when any attribute is accessed on instances of this
    class.

    Instances of this class are returned by import_optional() when a module
    cannot be found, which allows for code to import optional dependencies, and
    have only the code paths that use the module affected.
    """

    def __init__(self, mod_name):
        self.name = mod_name

    def __getattr__(self, attr):
        raise RuntimeError(f"This feature requires the {self.name} " "package/module")


def import_optional(mod, default_mod_class=MissingModule):
    """
    import the "optional" module 'mod' and return the module object or object.
    If the import raises ModuleNotFoundError, returns an instance of
    default_mod_class.

    This method was written to support importing "optional" dependencies so
    code can be written to run even if the dependency is not installed.

    Example
    -------
    >> from cugraph.utils import import_optional
    >> nx = import_optional("networkx")  # networkx is not installed
    >> G = nx.Graph()
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      ...
    RuntimeError: This feature requires the networkx package/module

    Example
    -------
    >> class CuDFFallback:
    ..   def __init__(self, mod_name):
    ..     assert mod_name == "cudf"
    ..     warnings.warn("cudf could not be imported, using pandas instead!")
    ..   def __getattr__(self, attr):
    ..     import pandas
    ..     return getattr(pandas, attr)
    ...
    >> from cugraph.utils import import_optional
    >> df_mod = import_optional("cudf", default_mod_class=CuDFFallback)
    <stdin>:4: UserWarning: cudf could not be imported, using pandas instead!
    >> df = df_mod.DataFrame()
    >> df
    Empty DataFrame
    Columns: []
    Index: []
    >> type(df)
    <class 'pandas.core.frame.DataFrame'>
    >>
    """
    try:
        return importlib.import_module(mod)
    except ModuleNotFoundError:
        return default_mod_class(mod_name=mod)


cudf = import_optional("cudf")
cupy = import_optional("cupy")
pandas = import_optional("pandas")
torch = import_optional("torch")


def _transform_to_backend_dtype_1d(data, series_name=None, backend="numpy", dtype=None):
    """
    Supports method-by-method selection of backend type (cupy, cudf, etc.)
    to avoid costly conversion such as row-major to column-major transformation.
    This method is used for 1-dimensional data, and does not perform unncessary
    transpositions or copies.

    Note: If using inferred dtypes, the returned series, array, or tensor may
    infer a different dtype than what was originally on the server (i.e promotion
    of int32 to int64).  In the future, the server may also return dtype to prevent
    this from occurring.

    data : np.ndarray
        The raw ndarray that will be transformed to the backend dtype.
    series_name : string
        The name of the series (only used for dataframe backends).
    backend : ('numpy', 'pandas', 'cupy', 'cudf', 'torch', 'torch:<device>')
              [default = 'numpy']
    dtype : ('int32', 'int64', 'float32', etc.)
        Optional. The data type to use when storing data in a series or array.
        If not set, it will be inferred for dataframe backends, and assumed as float64
        for array and tensor backends.

    """

    if backend == "numpy":
        if dtype == data.dtype:
            return data
        else:
            return np.array(data, dtype=dtype or "float64")
    elif backend == "cupy":
        return cupy.array(data, dtype=dtype or "float64")
    elif backend == "pandas":
        return pandas.Series(data, name=series_name, dtype=dtype)
    elif backend == "cudf":
        return cudf.Series(data, name=series_name, dtype=dtype)
    elif backend == "torch":
        return torch.tensor(data.astype(dtype=dtype or "float64"))

    backend = backend.split(":")
    if backend[0] == "torch":
        try:
            device = int(backend[1])
        except ValueError:
            device = backend[1]
        return torch.tensor(data.astype(dtype=dtype or "float64"), device=device)

    raise ValueError(f"invalid backend {backend[0]}")


def _transform_to_backend_dtype(data, column_names, backend="numpy", dtypes=None):
    """
    Supports method-by-method selection of backend type (cupy, cudf, etc.)
    to avoid costly conversion such as row-major to column-major transformation.
    If using an array or tensor backend, this method will likely be followed with
    one or more stack() operations to create a matrix or matrices.

    Note: If using inferred dtypes, the returned dataframes, arrays, or tensors may
    infer a different dtype than what was originally on the server (i.e promotion
    of int32 to int64).  In the future, the server may also return dtype to prevent
    this from occurring.

    data : numpy.ndarray
        The raw ndarray that will be transformed to the backend type.
    column_names : list[string]
        The names of the columns, if creating a dataframe.
    backend : ('numpy', 'pandas', 'cupy', 'cudf', 'torch', 'torch:<device>')
              [default = 'numpy']
        The data backend to convert the provided data to.
    dtypes : ('int32', 'int64', 'float32', etc.)
        Optional.  The data type to use when storing data in a dataframe or array.
        If not set, it will be inferred for dataframe backends, and assumed as float64
        for array and tensor backends.
        May be a list, or dictionary corresponding to column names.  Unspecified
        columns in the dictionary will have their dtype inferred.  Note: for array
        and tensor backends, the inferred type is always 'float64' which will result
        in a error for non-numeric inputs.
        i.e. ['int32', 'int64', 'int32', 'float64']
        i.e. {'col1':'int32', 'col2': 'int64', 'col3': 'float64'}
    """

    default_dtype = None if backend in ["cudf", "pandas"] else "float64"

    if dtypes is None:
        dtypes = [default_dtype] * data.shape[1]
    elif isinstance(dtypes, (list, tuple)):
        if len(dtypes) != data.shape[1]:
            raise ValueError("Datatype array length must match number of columns!")
    elif isinstance(dtypes, dict):
        dtypes = [
            dtypes[name] if name in dtypes else default_dtype for name in column_names
        ]
    else:
        raise ValueError("dtypes must be None, a list/tuple, or a dict")

    if not isinstance(data, np.ndarray):
        raise TypeError("Numpy ndarray expected")

    if backend == "cupy":
        return [cupy.array(data[:, c], dtype=dtypes[c]) for c in range(data.shape[1])]
    elif backend == "numpy":
        return [np.array(data[:, c], dtype=dtypes[c]) for c in range(data.shape[1])]

    elif backend == "pandas" or backend == "cudf":
        from_records = (
            pandas.DataFrame.from_records
            if backend == "pandas"
            else cudf.DataFrame.from_records
        )
        df = from_records(data, columns=column_names)
        for i, t in enumerate(dtypes):
            if t is not None:
                df[column_names[i]] = df[column_names[i]].astype(t)
        return df
    elif backend == "torch":
        return [
            torch.tensor(data[:, c].astype(dtypes[c])) for c in range(data.shape[1])
        ]

    backend = backend.split(":")
    if backend[0] == "torch":
        try:
            device = int(backend[1])
        except ValueError:
            device = backend[1]
        return [
            torch.tensor(data[:, c].astype(dtypes[c]), device=device)
            for c in range(data.shape[1])
        ]

    raise ValueError(f"invalid backend {backend[0]}")


def _offsets_to_backend_dtype(offsets, backend):
    """
    Transforms the offsets object into an appropriate object for the given backend.

    Parameters
    ----------
    offsets : cugraph_service_client.types.Offsets
        The offsets object to transform.
    backend : ('numpy', 'pandas', 'cupy', 'cudf', 'torch', 'torch:<device>')
              [default = 'numpy']
        The backend the offsets will be transformed to a type of.

    Returns
    -------
    An object of the desired backend.
    For cudf: A cudf DataFrame with index=type, start, stop columns
    For pandas: A pandas DataFrame with index=type, start, stop columns
    For cupy: A dict of {'type': np.ndarray, 'start': cp.ndarray, 'stop': cp.ndarray}
    For numpy: A dict of {'type': np.ndarray, 'start': np.ndarray, 'stop': np.ndarray}
    For torch: A dict of {'type': np.ndarray, 'start': Tensor, 'stop': Tensor}
    """
    if backend == "cudf" or backend == "pandas":
        df_clx = cudf.DataFrame if backend == "cudf" else pandas.DataFrame
        df = df_clx(
            {
                "start": offsets.start,
                "stop": offsets.stop,
            },
            index=offsets.type,
        )
        df.index.name = "type"
        return df

    if backend == "cupy":
        tn_clx = cupy.array
        device = None
    elif backend == "numpy":
        tn_clx = np.array
        device = None
    elif backend == "torch":
        tn_clx = torch.tensor
        device = "cpu"
    else:
        if "torch" not in backend:
            raise ValueError(f"Invalid backend {backend}")
        tn_clx = torch.tensor
        backend = backend.split(":")
        try:
            device = int(backend[1])
        except ValueError:
            device = backend[1]

    return_dict = {
        "type": np.array(offsets.type),
        "start": tn_clx(offsets.start),
        "stop": tn_clx(offsets.stop),
    }

    if device is not None:
        return_dict["start"] = return_dict["start"].to(device)
        return_dict["stop"] = return_dict["stop"].to(device)

    return return_dict

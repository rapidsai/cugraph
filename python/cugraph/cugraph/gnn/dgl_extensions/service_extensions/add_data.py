import cudf
import dask_cudf
import cupy as cp
from cugraph.experimental import MGPropertyGraph


def add_node_data_from_parquet(
    file_path, node_col_name, node_offset, ntype, gid, server
):
    pG = server.get_graph(gid)
    if isinstance(pG, MGPropertyGraph):
        df = dask_cudf.read_parquet(file_path)
    else:
        df = cudf.read_parquet(file_path)

    df[node_col_name] = df[node_col_name] + node_offset
    pG.add_vertex_data(df, vertex_col_name=node_col_name, type_name=ntype)

    columns_list = list(df.columns)

    return serialize_strings_to_array(columns_list)


def add_edge_data_from_parquet(
    file_path, node_col_names, canonical_etype, src_offset, dst_offset, gid, server
):
    pG = server.get_graph(gid)
    if isinstance(pG, MGPropertyGraph):
        df = dask_cudf.read_parquet(file_path)
    else:
        df = cudf.read_parquet(file_path)

    df[node_col_names[0]] = df[node_col_names] + src_offset
    df[node_col_names[1]] = df[node_col_names] + dst_offset
    pG.add_edge_data(df, vertex_col_names=node_col_names, type_name=canonical_etype)

    columns_list = list(df.columns)

    return serialize_strings_to_array(columns_list)


def convert_to_string_ar(string):
    return cp.asarray([ord(c) for c in string], cp.int32), len(string)


def serialize_strings_to_array(strings_list):
    ar_ls = []
    len_ls = []
    for s in strings_list:
        ar, s_len = convert_to_string_ar(s)
        ar_ls.append(ar)
        len_ls.append(s_len)
    return cp.concatenate(ar_ls), cp.asarray(len_ls, dtype=cp.int32)

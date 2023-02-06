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

"""
Create a synthetic graph that is similar to a citation network
and in OGB format.

Since this matches OGB MAG/MAD240 - the node types and edge types are set
Node Types:  Paper, Author, Institution
Edge Types:  cites, writes, affiliated with

Also, since the format is defined, so are subdirectories:

gived a "base_dir", the following 4 subdirectories are created
* paper
* paper__cites__paper
* author__affiliated_with__institution
* author__writes__paper

Lastly, the output files names and types (whats in the files) is also defined
For a homogeneous graph, only load the paper data.
For a heterogeneous graph, load all data.

Example:
python gnn_cite_graph_gen.py \
    -mg True \
    -outdir '/tmp/GNN2' \
    -papersScale 12 \
    -format "parquet" \
    -papersLabeledPercent 0.10 \
    -papersFeatureNoise 0.0001

Will output the following into /tmp/GNN2:
    author__affiliated_with__institution/
        edge.parquet

    author__writes__paper/
        edge.parquet

    paper__cites__paper/
        edge/
            part.0.parquet
            part.1.parquet
            ...
    paper/
        node_feat.npy
        node_label
        test_labels
        train_labels
        val_labels

    meta.json

The edge parquet files contain the src and dst arrays.

node_feat.npy contains the node features
and is of shape (# papers, num_features)

node_label (parquet) is primarily for debugging.  It is
a list of the node ids that are labeled.

test_labels (parquet) contains the indices and class labels (venue)
for the test data.  It is of shape (# test papers, 2)

train_labels (parquet) contains the indices and class labels (venue)
for the training data.  It is of shape (# train papers, 2)

val_labels (parquet) contains the indices and class labels (venue)
for the validation data.  It is of shape (# validation papers, 2)

meta.json has the following format:
{
    "paper": [paper_ix_start_incl, paper_ix_end_incl],
    "author": [author_ix_start_incl, author_ix_end_incl],
    "institution": [institution_ix_start_incl, institution_ix_end_incl],
}
The number of nodes for each type can be determined by
taking ix_end_incl - ix_start_incl + 1

Note: for MG, the number of dask workers can be adjusted by setting
DASK_NUM_WORKERS.

"""


import os
import argparse
import json

import numpy as np
import pandas as pd
import cudf
import dask_cudf

import rmm

from cugraph.generators import rmat
from cugraph.structure import NumberMap

from cugraph.dask.common.mg_utils import teardown_local_dask_cluster
import cugraph.dask.comms.comms as Comms
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

from typing import (
    Union,
    Tuple
)

# --- Some global attributes
paper_dir = "paper"
paper_cite_dir = "paper__cites__paper"
author_paper_dir = "author__writes__paper"
author_institute_dir = "author__affiliated_with__institution"


def setup_sg() -> None:

    # Set RMM to allocate all memory as managed memory
    rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
    assert rmm.is_initialized()


def setup_mg() -> None:
    print("Setup MG")
    visible_devices = ",".join([str(i) for i in range(1, 8)])
    cluster = LocalCUDACluster(
        protocol="tcp", rmm_managed_memory=True, CUDA_VISIBLE_DEVICES=visible_devices
    )
    client = Client(cluster)
    Comms.initialize(p2p=True)
    rmm.reinitialize(managed_memory=True)
    rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())

    return cluster, client


def stop_mg(client, cluster) -> None:
    teardown_local_dask_cluster(cluster, client)


def make_dir(d) -> None:
    """
    create the directory if it does not exists
    This is a separated funtion in case the directory should be
    deleted first and recreated (i.e. deleting all existing files)
    """
    if not os.path.exists(d):
        os.mkdir(d)


def setup_directories(basedir) -> None:
    """
    create the 4 directories
    """
    # Base
    make_dir(basedir)

    # Paper
    make_dir(os.path.join(basedir, paper_dir))

    # Paper Cites
    make_dir(os.path.join(basedir, paper_cite_dir))

    # Paper Author
    make_dir(os.path.join(basedir, author_paper_dir))

    # Author - Institution
    make_dir(os.path.join(basedir, author_institute_dir))


def save_data(
        args: argparse.Namespace,
        dir: str,
        file_name: str,
        data: Union[pd.DataFrame, cudf.DataFrame, dask_cudf.DataFrame]) -> None:
    """
    Saves pandas/cudf/dask_cudf data to the specified file.
    """
    dir_path = os.path.join(args.outdir, dir)

    if args.mg:
        f = os.path.join(dir_path, f"{file_name}")
    else:
        f = os.path.join(dir_path, f"{file_name}.{args.format}")
    print(f"\tsaving to {f}")

    if args.format == "parquet":
        data.to_parquet(f, index=False)
    else:
        data.to_csv(f, header=False, index=False)


def write_numpy(
        args: argparse.Namespace,
        arr: np.ndarray,
        directory: str,
        name: str) -> str:
    """
    Saves numpy data to the specified file.
    Returns the complete path to the written file.
    """
    output_path = os.path.join(
        os.path.join(args.outdir, directory),
        name
    )

    np.save(output_path, arr)
    return output_path


def create_rmat_dataset(
        scale: int,
        edgefactor: int = 16,
        seed: int = 42,
        mg: bool = False) -> Union[cudf.DataFrame, dask_cudf.DataFrame]:
    """
    Create data via RMAT and return a COO DataFrame.
    The RMAT paramaters match what Graph500 uses for {a,b,c} argumemnts.

        Parameters
        ----------
        scale: int
            The power of 2 number of nodes to create

        edgefactor: int, optional (default=16)
            The number of edge per node (average) to create
            The defaukt is 16 which match Graph500

        seed: int, optional (default=42)
            The seed to use in the random number generator

        mg: bool, optional (default=False)
            If True, R-MAT generation occurs across multiple GPUs. If False, only a
            single GPU is used.  Default is False (single-GPU)

        Returns
        ---------
        df : cudf.DataFrame or dask_cudf.DataFrame
            DataFrame with data in COO format ("src", "dst")

    """
    num_edges = (2 ** scale) * edgefactor
    rmat_df = rmat(
        scale,
        num_edges,
        0.57,  # from Graph500
        0.19,  # from Graph500
        0.19,  # from Graph500
        seed,
        clip_and_flip=False,
        scramble_vertex_ids=False,
        create_using=None,
        mg=mg,
    )

    clean_coo = NumberMap.renumber(rmat_df, src_col_names="src", dst_col_names="dst")[0]
    del rmat_df

    if mg:
        clean_coo = clean_coo.rename(
            columns={"renumbered_src": "src", "renumbered_dst": "dst"}
        )
    else:
        clean_coo.rename(
            columns={"renumbered_src": "src", "renumbered_dst": "dst"}, inplace=True
        )
    return clean_coo


def dec_df_update(
        df: Union[pd.DataFrame, cudf.DataFrame],
        num_items: int,
        col_name: str) -> Union[pd.DataFrame, cudf.DataFrame]:

    """ """
    df2 = df[df["count"] > 1]
    df2["count"] = df2["count"] - 1
    siz = len(df2)
    ids = np.random.randint(0, (num_items + 1), size=siz)
    df2[col_name] = ids
    return df2


# -----------------------------------------------------------------------
#
#
def create_paper_cites_data(args: argparse.Namespace) -> Tuple[int, int]:
    """
    Create an RMAT graph and write out the edges

    Returns the number of edges and nodes in the graph
    """

    print(f"making files under {paper_cite_dir}")
    print("\tCreate RMAT data and renumber")
    coo = create_rmat_dataset(
        scale=args.papersScale,
        edgefactor=args.papersEdgeFactor,
        seed=args.seed,
        mg=args.mg,
    )

    if args.mg:
        max_id = max(coo["src"].max().compute(), coo["dst"].max().compute())
    else:
        max_id = max(coo["src"].max(), coo["dst"].max())

    print("Save the papers cites graph (edge data)")
    print(f"\tnumber of edges {len(coo)}")
    print(f"\tnumber of nodes {max_id}")

    name = "edge"
    save_data(args, paper_cite_dir, name, coo)

    num_edges = len(coo)
    del coo

    return num_edges, max_id


# -----------------------------------------------------------------------
#  this is all done in Pandas in host memory
#
def create_papers_data(
        args: argparse.Namespace,
        num_papers: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates the paper features and selects the train/test/val splits.
    Returns the train/test/val splits.
    """

    num_labeled = int(num_papers * args.papersLabeledPercent)

    # create and empty DataFrame
    # if args.mg:
    #    nodes = pd.DataFrame()
    # else:
    #    nodes = cudf.DataFrame()

    nodes = pd.DataFrame()

    rng = np.random.default_rng(seed=args.seed)

    # create an np.array of random numbers and add to dataframe
    ran = rng.integers(0, num_papers, size=num_papers)
    nodes["rand"] = ran

    # now computed which are labeled
    # this is a simple look for whick rows have IDs smaller than the number of labeld
    # this is based on a true uniformed distributions, so results could be +/- a few
    nodes["label"] = 0
    nodes["label"].loc[nodes["rand"] < num_labeled] = 1
    nodes['venue'] = rng.integers(0, args.numClasses, size=num_papers, dtype='int64')

    # selected just the labeled nodes
    labeled_df = (nodes["label"].loc[nodes["label"] == 1]).to_frame().reset_index()

    # drop the "labeled" column since it is no longer needed
    labeled_df.drop(columns=["label"], inplace=True)

    # Save the list
    name = "node_label"
    save_data(args, paper_dir, name, labeled_df)

    perm = rng.permutation(len(labeled_df))
    partition_first = int(len(labeled_df) * args.validationPercent)
    partition_second = partition_first + int(len(labeled_df) * args.testPercent)
    val_ix = (
        labeled_df['index'].iloc[perm[:partition_first]]
        .reset_index(drop=True)
    )
    test_ix = (
        labeled_df['index'].iloc[perm[partition_first:partition_second]]
        .reset_index(drop=True)
    )
    train_ix = (
        labeled_df['index'].iloc[perm[partition_second:]]
        .reset_index(drop=True)
    )
    # some cleanup
    del ran
    del labeled_df

    nodes.drop(columns=["rand", "label"], inplace=True)

    # Generate the labels
    train_labels = nodes['venue'].loc[train_ix].reset_index(drop=True)
    train_df = pd.DataFrame({'node': train_ix, 'label': train_labels})

    test_labels = nodes['venue'].loc[test_ix].reset_index(drop=True)
    test_df = pd.DataFrame({'node': test_ix, 'label': test_labels})

    val_labels = nodes['venue'].loc[val_ix].reset_index(drop=True)
    val_df = pd.DataFrame({'node': val_ix, 'label': val_labels})

    save_data(args, paper_dir, 'train_labels', train_df)
    save_data(args, paper_dir, 'test_labels', test_df)
    save_data(args, paper_dir, 'val_labels', val_df)

    # --------------------
    # Now paper features
    num_f = args.papersNumFeatures

    for x in range(0, num_f):
        col_name = "feature_" + str(x)
        nodes[col_name] = nodes['venue'] * (x + 2.0)
    nodes.drop(columns=['venue'], inplace=True)

    # Save the features
    data = nodes.to_numpy()

    output_path = write_numpy(args, data, paper_dir, 'node_feat.npy')
    print(f'\tsaved features to {output_path}')

    return train_ix, test_ix, val_ix


def create_author_papers_data(
        args: argparse.Namespace,
        start_id: int,
        num_papers: int) -> Tuple[int, int]:
    """
    Returns the number of author->paper edges and number of authors.
    """

    num_authors = int(num_papers * args.authorPercent)
    avg_papers = args.authorAvgNumPapers

    print("Create Author to Papers edges")
    print(f"\tThere are {num_authors} authors")
    print(f"\tAvg papers per Author {avg_papers}")
    print(f"\tStarting ID will be {start_id}")

    low = 1

    # see if the data fits into one GPUs
    # - one GPU use cudf
    # - larger use pandas
    if (num_authors * avg_papers) > 2000000000:
        df = pd.DataFrame()
    else:
        df = cudf.DataFrame()

    a = np.random.normal(loc=avg_papers, scale=10, size=num_authors)
    df["count"] = a
    df["count"].loc[df["count"] < low] = low
    df.reset_index(inplace=True)

    df.rename(columns={"index": "author"}, inplace=True)

    df["author"] = df["author"] + start_id

    # add a random paper id
    paper_ids = np.random.randint(0, (num_papers + 1), size=num_authors)
    df["paper"] = paper_ids

    # what is the max 'count' - number of papers written?
    max_p = int(df["count"].max())

    auth_paper_array = [None] * max_p
    index = 0
    auth_paper_array[0] = df

    index += 1
    while index < max_p:
        auth_paper_array[index] = dec_df_update(
            auth_paper_array[index - 1], num_papers, "paper"
        )
        index += 1

    if isinstance(df, pd.DataFrame):
        data = pd.concat(auth_paper_array)
    elif args.mg:
        num_workers = len(Comms.get_workers())
        data = dask_cudf.concat(
            [
                dask_cudf.from_cudf(df, npartitions=2 * num_workers)
                for df in auth_paper_array
            ]
        ).persist()
    else:
        data = cudf.concat(auth_paper_array)

    del auth_paper_array
    del df
    if args.mg:
        data = data.drop(columns=['count'])
    else:
        data.drop(columns=["count"], inplace=True)

    print("Save the author writes paper data")
    name = "edge"
    save_data(args, author_paper_dir, name, data)
    num_edges = len(data)
    del data

    return num_edges, num_authors


# -----------------------------------------------------------------------
#
#
def create_author_institute_data(
        args: argparse.Namespace,
        start_id: int,
        num_authors: int) -> int:
    """
    Returns the number of author->institution edges and
    number of institutions.
    """

    num_institutes = args.numInstitutions
    avg_works = args.avgInstitutionsPerAuthor

    print(f"There are {num_institutes} institutions")
    print(f"There are {num_authors} authors")
    print(f"Avg works at {avg_works}")
    print(f"Starting ID will be {start_id}")

    # see if the data fits into one GPUs
    # - one GPU use cudf
    # - larger use pandas
    if (num_institutes * avg_works) > 2_000_000_000:
        df = pd.DataFrame()
    else:
        df = cudf.DataFrame()

    a = np.random.normal(loc=avg_works, scale=2, size=num_authors)
    df["count"] = a
    df.reset_index(inplace=True)
    df.rename(columns={"index": "author"}, inplace=True)
    df["author"] = df["author"] + start_id
    df = df[df["count"] > 0]
    s = len(df)

    # add a random paper id
    work_ids = np.random.randint(0, (num_institutes + 1), size=s)
    df["institution"] = work_ids

    # what is the max 'count' - number of insitutions?
    max_i = int(df["count"].max())

    auth_work_array = [None] * max_i
    index = 0
    auth_work_array[0] = df

    index += 1
    while index < max_i:
        auth_work_array[index] = dec_df_update(
            auth_work_array[index - 1], num_institutes, "institution"
        )
        index += 1

    if isinstance(df, pd.DataFrame):
        data = pd.concat(auth_work_array)
    elif args.mg:
        num_workers = len(Comms.get_workers())
        data = dask_cudf.concat(
            [
                dask_cudf.from_cudf(df, npartitions=2 * num_workers)
                for df in auth_work_array
            ]
        ).persist()
    else:
        data = cudf.concat(auth_work_array)

    del auth_work_array

    if args.mg:
        data = data.drop(columns=["count"])
    else:
        data.drop(columns=["count"], inplace=True)

    print("Save the author affiliated with institution")
    name = "edge"
    save_data(args, author_institute_dir, name, data)

    return len(data), num_institutes


###################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GNN synthetic data")

    # ---  Output
    parser.add_argument(
        "-outdir",
        type=str,
        default=None,
        help="Directory to store outputs",
        required=True,
    )

    parser.add_argument(
        "-format",
        type=str,
        default="parquet",
        help="output format.  Can be either parquet or csv",
        required=False,
    )

    # --- General
    parser.add_argument(
        "-seed",
        type=int,
        default=42,
        help="Random number seed",
        required=False,
    )

    # ---  Papers and Paper Cites Paper
    parser.add_argument(
        "-papersScale",
        type=int,
        default=None,
        help="Scale factor of the Papers-Cite_Papers graph",
        required=True,
    )

    parser.add_argument(
        "-papersEdgeFactor",
        type=int,
        default=15,
        help="Edge Factor - the average number of edges per node.  Default is 16",
        required=False,
    )

    parser.add_argument(
        "-papersLabeledPercent",
        type=float,
        default=0.05,
        help="Percent (float) of Paper nodes that are tagged as labeled",
        required=False,
    )

    parser.add_argument(
        "-papersNumFeatures",
        type=int,
        default=10,
        help="Number of paper node features (all are float values)",
        required=False,
    )

    parser.add_argument(
        "-papersFeatureNoise",
        type=float,
        default=0.00001,
        help="Percent (float) of noise in feature values of labeled nodes",
        required=False,
    )

    # ---  Author Write
    parser.add_argument(
        "-authorPercent",
        type=float,
        default=1.0,
        help="The number of Authors in relationship to the number of papers.  "
        "A value of 1 means that there is the same number of Authors and papers",
        required=False,
    )

    parser.add_argument(
        "-authorAvgNumPapers",
        type=int,
        default=3,
        help="The number papers an author writes",
        required=False,
    )

    # ---  Institutions
    parser.add_argument(
        "-numInstitutions",
        type=int,
        default=26000,
        help="The number of Institutions.",
        required=False,
    )

    parser.add_argument(
        "-avgInstitutionsPerAuthor",
        type=float,
        default=0.4,
        help="The number of Institutions.",
        required=False,
    )

    parser.add_argument(
        "-testPercent",
        type=float,
        default=0.1,
        help="Percentage of the labeled data used for testing",
        required=False,
    )

    parser.add_argument(
        "-validationPercent",
        type=float,
        default=0.05,
        help="Percentage of the labeled data used for testing",
        required=False
    )

    parser.add_argument(
        "-numClasses",
        type=float,
        default=150,
        help="Number of unique classes (venue labels)",
        required=False,
    )

    # --- Multi-GPU options
    parser.add_argument(
        "-mg",
        type=bool,
        default=False,
        help="Run Multi-GPU. This will create a local DASK cluster."
             " Single GPU is limited to Scale 26",
        required=False,
    )

    args = parser.parse_args()

    return args


# ------------------------------------------------


def main() -> None:

    args = parse_args()

    # --- Step 0: Setup/Init ---
    if args.mg:
        client, cluster = setup_mg()
    else:
        setup_sg()

    print(f"Running in MG mode is {args.mg}")

    # Create directories
    setup_directories(args.outdir)

    # Step 1: Create the Papers Cite graph
    num_paper_cites_paper_edges, last_paper_id = create_paper_cites_data(args)
    num_papers = last_paper_id + 1

    # --- Step 2: Papers info  (labels and features) ---
    train_ix, test_ix, val_ix = create_papers_data(args, num_papers)

    # --- Step 3:Author to Papers ---
    num_author_writes_paper_edges, num_auth = (
        create_author_papers_data(args, (last_paper_id + 1), num_papers)
    )

    # --- Step 4:Author to Institutions ---
    next_id = num_papers + num_auth + 1
    num_author_affil_institution_edges, num_inst = (
        create_author_institute_data(args, (next_id), num_auth)
    )

    with open(os.path.join(args.outdir, 'meta.json'), 'w') as out_json_file:
        json.dump(
            {
                'paper': [
                    0,
                    int(last_paper_id)
                ],
                'author': [
                    int(last_paper_id) + 1,
                    int(last_paper_id + num_auth)
                ],
                'institution': [
                    int(last_paper_id + num_auth) + 1,
                    int(last_paper_id + num_auth + num_inst) - 1
                ],
                'author__writes__paper': num_author_writes_paper_edges,
                'author__affiliated_with__institution':
                    num_author_affil_institution_edges,
                'paper__cites__paper': num_paper_cites_paper_edges,
            },
            out_json_file,

        )
    print("DONE")

    if args.mg:
        print("Stopping cluster")
        stop_mg(client, cluster)


if __name__ == "__main__":
    main()

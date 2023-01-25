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

gived a "base_dir", the following 4 subdirectroies are created
* paper
* paper__cites__paper
* author__affiliated_with__institution
* author__writes__paper

Lastely, the output files names and types (whats in the files) is also defined

Example:
python gnn_cite_graph_gen.py \
    -mg True \
    -outdir '/tmp/GNN2' \
    -papersScale 12 \
    -format "csv" \
    -papersLabeledPercent 0.10 \
    -papersFeatureNoise 0.0001


"""


import os
import ast
import argparse

import numpy as np
import pandas as pd

import rmm

import cugraph
from cugraph.generators import rmat
from cugraph.structure import NumberMap

import cudf as cudf_sg
import dask_cudf as cudf_mg
import cugraph.dask as dask_cugraph

import dask
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
from cugraph.dask.comms import comms as Comms


# --- Some global attributes
paper_dir = "/paper"
paper_cite_dir = "/paper__cites__paper"
author_paper_dir = "/author__writes__paper"
author_institute_dir = "/author__affiliated_with__institution"


def setup_sg() -> None:

    # Set RMM to allocate all memory as managed memory
    rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
    assert rmm.is_initialized()


def setup_mg():

    print("Setup MG")

    cluster = LocalCUDACluster(rmm_managed_memory=True)
    client = Client(cluster)
    Comms.initialize(p2p=True)

    return (client, cluster)


def stop_mg(client, cluster):
    Comms.destroy()
    client.close()
    cluster.close()


def make_dir(d):
    """
    create the directory if it does not exists
    This is a separated funtion in case the directory should be
    deleted first and recreated (i.e. deleting all existing files)
    """
    if not os.path.exists(d):
        os.mkdir(d)


def setup_directories(basedir):
    """
    create the 4 directories
    """
    # Base
    make_dir(basedir)

    # Paper
    make_dir(basedir + paper_dir)

    # Paper Cites
    make_dir(basedir + paper_cite_dir)

    # Paper Author
    make_dir(basedir + author_paper_dir)

    # Author - Institution
    make_dir(basedir + author_institute_dir)


def save_data(args, dir, file_name, data, mg=False):

    if mg:
        f = os.path.join(dir, f"{file_name}")
    else:
        f = os.path.join(dir, f"{file_name}.{args.format}")
    print(f"\tsaving to {f}")

    if format == "parquet":
        data.to_parquet(f, index=False)
    else:
        data.to_csv(f, header=False, index=False)


def create_base_labeled_features(num: int) -> np.array:
    # all values are floats
    return np.random.uniform(-1, 1, size=num)


def create_rmat_dataset(scale, edgefactor=16, seed=42, mg=False):
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


def dec_df_update(df, num_items, col_name):
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
def create_paper_cites_data(args: argparse.Namespace) -> int:
    """
    Create an RMAT graph and write out the edges

    Returns the number of nodes in the graph
    """

    print(f"making files under {paper_cite_dir}")
    print("\tCreate RMAT data and renumber")
    coo = create_rmat_dataset(
        scale=args.papersScale,
        edgefactor=args.papersEgefactor,
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

    dir = args.outdir + paper_cite_dir
    name = "edge"
    save_data(args, dir, name, coo, mg=args.mg)

    del coo

    return max_id


# -----------------------------------------------------------------------
#  this is all done in Pandas in host memory
#
def create_papers_data(args, num_papers):

    num_labeled = int(num_papers * args.papersLabeledPercent)

    # create and empty DataFrame
    # if args.mg:
    #    nodes = pd.DataFrame()
    # else:
    #    nodes = cudf_sg.DataFrame()

    nodes = pd.DataFrame()

    # create an np.array of random numbers and add to dataframe
    ran = np.random.randint(0, num_papers, size=num_papers)
    nodes["rand"] = ran

    # now computed which are labeled
    # this is a simple look for whick rows have IDs smaller than the number of labeld
    # this is based on a true uniformed distributions, so results could be +/- a few
    nodes["label"] = 0
    nodes["label"].loc[nodes["rand"] < num_labeled] = 1

    # selected just the labeld nodes
    labeled_df = (nodes["label"].loc[nodes["label"] == 1]).to_frame().reset_index()

    # drop the "labeled" column since it is no longer needed
    labeled_df.drop(columns=["label"], inplace=True)

    # Save the list
    dir = args.outdir + paper_dir
    name = "node-label." + args.format
    save_data(args, dir, name, labeled_df)

    # some cleanup
    del ran
    del labeled_df
    nodes.drop(columns=["rand"], inplace=True)

    # --------------------
    # Now paper features
    num_f = args.papersNumFeatures

    # define the base "labeld" data
    labeled_features = create_base_labeled_features(num_f)

    tmp_df = pd.DataFrame()

    for x in range(0, num_f):

        noise = np.random.uniform(
            (-1 * args.papersFeatureNoise), args.papersFeatureNoise, size=num_papers
        )
        tmp_df["noise"] = noise

        ran = np.random.uniform(-1, 1, size=num_papers)
        tmp_df["data"] = noise

        col_name = "feature_" + str(x)
        nodes[col_name] = tmp_df["noise"] + tmp_df["data"]

    del tmp_df

    # Save the features
    data = nodes.to_numpy()

    name = "node-feat.npy"
    dir = args.outdir + paper_dir
    f = os.path.join(dir, f"{name}")
    print(f"\tsaving features to {f}")
    np.save(f, data)


# -----------------------------------------------------------------------
#
#
def create_author_papers_data(args, start_id, num_papers) -> int:
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
        df = cudf_sg.DataFrame()

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
    else:
        data = cudf_sg.concat(auth_paper_array)

    del auth_paper_array
    del df
    data.drop(columns=["count"], inplace=True)

    print("Save the author writes paper data")
    dir = args.outdir + author_paper_dir
    name = "edge"
    save_data(args, dir, name, data)

    return num_authors


# -----------------------------------------------------------------------
#
#
def create_author_institute_data(args, start_id, num_authors):

    num_institutes = args.numInstitutions
    avg_works = args.avgInstitutionsPerAuthor

    print(f"There are {num_institutes} institutions")
    print(f"There are {num_authors} authors")
    print(f"Avg works at {avg_works}")
    print(f"Starting ID will be {start_id}")

    # see if the data fits into one GPUs
    # - one GPU use cudf
    # - larger use pandas
    if (num_institutes * avg_works) > 2000000000:
        df = pd.DataFrame()
    else:
        df = cudf_sg.DataFrame()

    if avg_works < 0:
        low = 0
        center = 0
    else:
        low = 1
        center = int(avg_works * num_institutes)

    a = np.random.normal(loc=center, scale=5, size=num_authors)
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
    else:
        data = cudf_sg.concat(auth_work_array)

    del auth_work_array

    data.drop(columns=["count"], inplace=True)

    print("Save the author writes works at institution")
    dir = args.outdir + author_institute_dir
    name = "edge"
    save_data(args, dir, name, data)

    return num_authors


###################################################


def arg_to_list(s: str):
    mylist = ast.literal_eval(s)
    if type(mylist) is not list:
        raise TypeError("Input is not in list format")
    return mylist


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
        "-papersEgefactor",
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

    # --- Multi-GPU options
    parser.add_argument(
        "-mg",
        type=bool,
        default=False,
        help="Run Multi-GPU. This will create a local DASK cluster. Single GPU is limited to Scale 26",
        required=False,
    )

    parser.add_argument(
        "-mgVisibleGPUs",
        type=arg_to_list,
        default=[],
        help="Visible GPUS in a MG setting",
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
    last_paper_id = create_paper_cites_data(args)
    num_papers = last_paper_id + 1

    # --- Step 2: Papers info  (labels and features) ---
    create_papers_data(args, num_papers)

    # --- Step 3:Author to Papers ---
    num_auth = create_author_papers_data(args, (last_paper_id + 1), num_papers)

    # --- Step 4:Author to Institutions ---
    next_id = num_papers + num_auth + 1
    create_author_institute_data(args, (next_id), num_auth)

    if args.mg:
        print("Stopping cluster")
        stop_mg(client, cluster)

    print("DONE")


if __name__ == "__main__":
    main()

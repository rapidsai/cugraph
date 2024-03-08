# cuGraph Sampling Benchmarks

## cuGraph Bulk Sampling

### Overview
The `cugraph_bulk_sampling.py` script runs the bulk sampler for a variety of datasets, including
both generated (rmat) datasets and disk (ogbn_papers100M, etc.) datasets.  It can also load
replicas of these datasets to create a larger benchmark (i.e. ogbn_papers100M x2).

### Arguments
The script takes a variety of arguments to control sampling behavior.
Required:
    --output_root
        The output root directory.  File/folder names are auto-generated.
        For instance, if the output root directory is /home/samples,
        the samples will be written to a new folder in /home/samples that
        contains information about the sampling run as well as the time
        of the run.

    --dataset_root
        The folder where datasets are stored.  Uses the format described
        in the input format section.

    --datasets
        Comma-separated list of datasets; can specify ogb or rmat (i.e. ogb_papers100M[2],rmat_22_16).
        For ogb datasets, can provide replication factor using brackets.
        Will attempt to read from dataset_root/<datset_name>.

Optional:
    --fanouts
        Comma-separated list of fanout values (i.e. [10, 25]).
        The default fanout is [10, 25].

    --batch_sizes
        Comma-separated list of batch sizes (i.e. 500, 1000).
        Defaults to "512,1024"

    --seeds_per_call_opts
        Comma-separated list of seeds per call.  Controls the number of input seed vertices processed
        in a single sampling call.
        Defaults to 524288

    --reverse_edges
        Whether to reverse the edges of the input edgelist. Should be set to False for PyG and True for DGL.
        Defaults to False (PyG).

    --dask_worker_devices
        Comma-separated list of the GPUs to assign to dask (i.e. "0,1,2").
        Defaults to just the default GPU (0).
        Changing this is strongly recommended in order to take advantage of all GPUs on the system.

    --random_seed
        Seed for random number generation.
        Defaults to '62'


### Input Format
The script expects its input data in the following format:
```
<top level directory>
|
|------ meta.json
|------ parquet
|------ |---------- <node type 0 (i.e. paper)>
|------ |---------- |---------------------------- [node_label.parquet]
|------ |---------- <node type 1 (i.e. author)>
|------ |---------- |---------------------------- [node_label.parquet]
...
|------ |---------- <edge type 0 (i.e. paper__cites__paper)>
|------ |---------- |------------------------------------------ edge_index.parquet
|------ |---------- <edge type 1 (i.e. author__writes__paper)>
|------ |---------- |------------------------------------------ edge_index.parquet
...

```

`node_label.parquet` only needs to be present for vertex types that have labeled
nodes. It consists of two columns, "node" which contains node ids, and "label",
which contains the labeled class of the node.

`edge_index.parquet` is required for all edge types.  It has two columns, `src`
and `dst`, representing the source and destination vertices of the edges in that
edge type's COO edge index.

`meta.json` is a json file containing metadata needed to properly process
the parquet files.  It must have the following format:
```
{
    "num_nodes": {
        "<node type 0 (i.e. paper)">: <# nodes of node type 0>,
        "<node type 1 (i.e. author)">: <# nodes of node type 1>,
        ...
    },
    "num_edges": {
        <edge type 0 (i.e. paper__cites__paper)>: <# edges of edge type 0>,
        <edge type 1 (i.e. author__writes__paper)>: <# edges of edge type 1>,
        ...
    }
}
```

### Output Meta
The script, in addition to the samples, will also output a file named `output_meta.json`.
This file contains various statistics about the sampling run, including the runtime,
as well as information about the dataset and system that the samples were produced from.

This metadata file can be used to gather the results from the sampling and training stages
together.

### Other Notes
For rmat datasets, you will need to generate your own bogus features in the training stage.
Since that is trivial, that is not done in this sampling script.

## cuGraph MNMG Training

### Overview
The script `run_train_job.sh` runs with the `sbatch` command to launch a series of slurm jobs.
First, for a given number of epochs, the script will produce samples for a given graph.
Then, the training process starts where samples are loaded and training iterations are
processed.

### Important Notes
Downloading the dataset files before running the slurm jobs is highly recommended.  Even though
the script will attempt to download the files if they are not available, this can often
lead to a timeout which will crash the scripts.  This applies regardless of whether you are training
with native PyG or cuGraph-PyG.  You can download data as follows:

```
from ogb.nodeproppred import NodePropPredDataset
dataset = NodePropPredDataset('ogbn-papers100M', root='/home/username/datasets')
```

For datasets other than ogbn-papers100M, you follow the same process but only change the dataset name.
The dataset will be correctly preprocessed when you run training.  In case you have a slow system, you
can also run preprocessing by running the training script on a single worker, which will avoid a timeout
which crashes the script.

The multi-GPU utilities are in `mg_utils` in the top level of the cuGraph repository.  You should either
copy them to this directory or symlink to them before running the scripts.

### Arguments
You will need to modify the bash scripts to run appopriately for your environment and
desired training workflow.  The standard sbatch arguments are at the top of the script, such as
job name, queue, etc.  These will need to be modified for your SLURM cluster.

Next are arguments for the container image (required),
and directories where the data and outputs are stored.  The directories default to subdirectories
of the current working directory.  But if there is a high-throughput storage system available,
using that storage for the samples and datasets is highly recommended.

Next are standard GNN training arguments such as `FANOUT`, `BATCH_SIZE`, etc.  You can also set
the number of training epochs here.  These are followed by the `REPLICATION_FACTOR` argument, which
can be used to create replications of the dataset for scale testing purposes.

The final two arguments are `FRAMEWORK` which can be either "cuGraphPyG" or "PyG", and `GPUS_PER_NODE`
which must be set to the correct value, even if this is provided by a SLURM argument.  If `GPUS_PER_NODE`
is not set to the correct number of GPUs, the script will hang indefinitely until it times out.  Mismatched
GPUs per node is currently unsupported by this script but should be possible in practice.

### Output
The results of training will be outputted to the logs directory with an `output.txt` file for each worker.
These will be overwritten upon each run.  Accuracy is only reported on rank 0.

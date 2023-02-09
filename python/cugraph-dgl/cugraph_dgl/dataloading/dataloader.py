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
import os
import shutil
import torch
import cugraph_dgl
import cupy as cp
import cudf
from cugraph.experimental import BulkSampler
from dask.distributed import default_client, Event
import dgl
from dgl.dataloading import WorkerInitWrapper, create_tensorized_dataset
from cugraph_dgl.dataloading import (
    HomogenousBulkSamplerDataset,
    HetrogenousBulkSamplerDataset,
)
from cugraph_dgl.dataloading.utils.extract_graph_helpers import (
    create_cugraph_graph_from_edges_dict,
)


class DataLoader(torch.utils.data.DataLoader):
    """
    Sampled graph data loader. Wrap a :class:`~cugraph_dgl.CuGraphStorage` and a
    :class:`~cugraph_dgl.dataloading.NeighborSampler` into
    an iterable over mini-batches of samples. cugraph_dgl's ``DataLoader`` extends
    PyTorch's ``DataLoader`` by handling creation and
    transmission of graph samples.
    """

    def __init__(
        self,
        graph: cugraph_dgl.CuGraphStorage,
        indices: torch.Tensor,
        graph_sampler: cugraph_dgl.dataloading.NeighborSampler,
        sampling_output_dir: str,
        batches_per_partition: int = 100,
        seeds_per_call: int = 400_000,
        device: torch.device = None,
        use_ddp: bool = False,
        ddp_seed: int = 0,
        batch_size: int = 1024,
        drop_last: bool = False,
        shuffle: bool = False,
        **kwargs,
    ):
        """
        Constructor for CuGraphStorage:
        -------------------------------
        graph :  CuGraphStorage
            The graph.
        indices : Tensor or dict[ntype, Tensor]
            The set of indices.  It can either be a tensor of
            integer indices or a dictionary of types and indices.
            The actual meaning of the indices is defined by the :meth:`sample` method of
            :attr:`graph_sampler`.
        graph_sampler : cugraph_dgl.dataloading.NeighborSampler
            The subgraph sampler.
        sampling_output_dir: str
            Output directory to share sampling results in
        batches_per_partition: int
            The number of batches of sampling results to write/read
        seeds_per_call: int
            The number of seeds to sample at once
        device : device context, optional
            The device of the generated MFGs in each iteration, which should be a
            PyTorch device object (e.g., ``torch.device``).
            By default this returns the tenors on device with the current
            cuda context
        use_ddp : boolean, optional
            If True, tells the DataLoader to split the training set for each
            participating process appropriately using
            :class:`torch.utils.data.distributed.DistributedSampler`.
            Overrides the :attr:`sampler` argument of
            :class:`torch.utils.data.DataLoader`.
        ddp_seed : int, optional
            The seed for shuffling the dataset in
            :class:`torch.utils.data.distributed.DistributedSampler`.
            Only effective when :attr:`use_ddp` is True.
        batch_size: int,
        kwargs : dict
            Key-word arguments to be passed to the parent PyTorch
            :py:class:`torch.utils.data.DataLoader` class. Common arguments are:
                - ``batch_size`` (int): The number of indices in each batch.
                - ``drop_last`` (bool): Whether to drop the last incomplete
                                        batch.
                - ``shuffle`` (bool): Whether to randomly shuffle the
                                      indices at each epoch
        Examples
        --------
        To train a 3-layer GNN for node classification on a set of nodes
        ``train_nid`` on a homogeneous graph where each node takes messages
        from 15 neighbors on the first layer, 10 neighbors on the second, and
        5 neighbors on the third:
        >>> sampler = cugraph_dgl.dataloading.NeighborSampler([15, 10, 5])
        >>> dataloader = cugraph_dgl.dataloading.DataLoader(
        ...     g, train_nid, sampler,
        ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=0)
        >>> for input_nodes, output_nodes, blocks in dataloader:
        ...     train_on(input_nodes, output_nodes, blocks)
        **Using with Distributed Data Parallel**
        If you are using PyTorch's distributed training (e.g. when using
        :mod:`torch.nn.parallel.DistributedDataParallel`),
        you can train the model by turning
        on the `use_ddp` option:
        >>> sampler = cugraph_dgl.dataloading.NeighborSampler([15, 10, 5])
        >>> dataloader = cugraph_dgl.dataloading.DataLoader(
        ...     g, train_nid, sampler, use_ddp=True,
        ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=0)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     for input_nodes, output_nodes, blocks in dataloader:
        ...
        """

        self.ddp_seed = ddp_seed
        self.use_ddp = use_ddp
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.graph_sampler = graph_sampler
        worker_init_fn = WorkerInitWrapper(kwargs.get("worker_init_fn", None))
        self.other_storages = {}
        self.epoch_number = 0
        self._batch_size = batch_size
        self._sampling_output_dir = sampling_output_dir
        self._batches_per_partition = batches_per_partition
        self._seeds_per_call = seeds_per_call

        indices = _dgl_idx_to_cugraph_idx(indices, graph)

        self.tensorized_indices_ds = create_tensorized_dataset(
            indices, batch_size, drop_last, use_ddp, ddp_seed, shuffle
        )
        if len(graph.ntypes) <= 1:
            self.cugraph_dgl_dataset = HomogenousBulkSamplerDataset(
                num_batches=len(self.tensorized_indices_ds),
                total_number_of_nodes=graph.total_number_of_nodes,
                edge_dir=self.graph_sampler.edge_dir,
            )
        else:
            etype_id_to_etype_str_dict = {v: k for k, v in graph._etype_id_dict.items()}

            self.cugraph_dgl_dataset = HetrogenousBulkSamplerDataset(
                num_batches=len(self.tensorized_indices_ds),
                num_nodes_dict=graph.num_nodes_dict,
                etype_id_dict=etype_id_to_etype_str_dict,
                etype_offset_dict=graph._etype_offset_d,
                ntype_offset_dict=graph._ntype_offset_d,
                edge_dir=self.graph_sampler.edge_dir,
            )

        if use_ddp:
            worker_info = torch.utils.data.get_worker_info()
            client = default_client()
            event = Event("cugraph_dgl_load_mg_graph_event")
            if worker_info.id == 0:
                G = create_cugraph_graph_from_edges_dict(
                    edges_dict=graph._edges_dict,
                    etype_id_dict=graph._etype_id_dict,
                    edge_dir=graph_sampler.edge_dir,
                )
                client.publish_dataset(cugraph_dgl_mg_graph_ds=G)
                event.set()
            else:
                if event.wait(timeout=1000):
                    G = client.get_dataset(G, "cugraph_dgl_mg_graph_ds")
                else:
                    raise RuntimeError(
                        f"Fetch cugraph_dgl_mg_graph_ds to worker_id {worker_info.id}",
                        "from worker_id 0 failed",
                    )
            self._rank = worker_info.id
        else:
            G = create_cugraph_graph_from_edges_dict(
                edges_dict=graph._edges_dict,
                etype_id_dict=graph._etype_id_dict,
                edge_dir=graph_sampler.edge_dir,
            )
            self._rank = 0
        self._cugraph_graph = G

        super().__init__(
            self.cugraph_dgl_dataset,
            batch_size=None,
            worker_init_fn=worker_init_fn,
            collate_fn=lambda x: x,  # Hack to prevent collating
            **kwargs,
        )

    def __iter__(self):
        output_dir = os.path.join(
            self._sampling_output_dir, "epoch_" + str(self.epoch_number)
        )
        _clean_directory(output_dir)

        # Todo: Figure out how to get rank
        rank = self._rank
        bs = BulkSampler(
            output_path=output_dir,
            batch_size=self._batch_size,
            graph=self._cugraph_graph,
            batches_per_partition=self._batches_per_partition,
            seeds_per_call=self._seeds_per_call,
            rank=rank,
            fanout_vals=self.graph_sampler._reversed_fanout_vals,
            with_replacement=self.graph_sampler.replace,
        )
        if self.shuffle:
            self.tensorized_indices_ds.shuffle()

        batch_df = create_batch_df(self.tensorized_indices_ds)
        bs.add_batches(batch_df, start_col_name="start", batch_col_name="batch_id")
        bs.flush()
        output_dir = output_dir + f"/rank={rank}/"
        self.cugraph_dgl_dataset.set_input_directory(output_dir)
        self.epoch_number = self.epoch_number + 1
        return super().__iter__()


def get_batch_id_series(n_output_rows: int, batch_size: int):
    num_batches = (n_output_rows + batch_size - 1) // batch_size
    print(f"Number of batches = {num_batches}".format(num_batches))
    batch_ar = cp.arange(0, num_batches).repeat(batch_size)
    batch_ar = batch_ar[0:n_output_rows].astype(cp.int32)
    return cudf.Series(batch_ar)


def create_batch_df(dataset: torch.Tensor):
    batch_id_ls = []
    indices_ls = []
    for batch_id, b_indices in enumerate(dataset):
        if isinstance(b_indices, dict):
            b_indices = torch.cat(list(b_indices.values()))
        batch_id_ar = cp.full(shape=len(b_indices), fill_value=batch_id, dtype=cp.int32)
        batch_id_ls.append(batch_id_ar)
        indices_ls.append(b_indices)

    batch_id_ar = cp.concatenate(batch_id_ls)
    indices_ar = cp.asarray(torch.concatenate(indices_ls))
    batches_df = cudf.DataFrame(
        {
            "start": indices_ar,
            "batch_id": batch_id_ar,
        }
    )
    return batches_df


def _dgl_idx_to_cugraph_idx(idx, cugraph_gs):
    if not isinstance(idx, dict):
        if len(cugraph_gs.ntypes) > 1:
            raise dgl.DGLError(
                "Must specify node type when the graph is not homogeneous."
            )
        return idx
    else:
        return {k: cugraph_gs.dgl_n_id_to_cugraph_id(n, k) for k, n in idx.items()}


def _clean_directory(path):
    """param <path> could either be relative or absolute."""
    if os.path.isfile(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains

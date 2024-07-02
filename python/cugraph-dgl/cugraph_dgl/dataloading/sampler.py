# Copyright (c) 2024, NVIDIA CORPORATION.
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

from typing import Iterator, Dict, Tuple, List, Union

import cugraph_dgl
from cugraph_dgl.nn import SparseGraph
from cugraph_dgl.typing import TensorType, DGLSamplerOutput
from cugraph_dgl.dataloading.utils.sampling_helpers import (
    create_homogeneous_sampled_graphs_from_tensors_csc,
)

from cugraph.gnn import DistSampleReader

from cugraph.utilities.utils import import_optional

torch = import_optional("torch")
dgl = import_optional("dgl")


class SampleReader:
    """
    Iterator that processes results from the cuGraph distributed sampler.
    """

    def __init__(self, base_reader: DistSampleReader, output_format: str = "dgl.Block"):
        """
        Constructs a new SampleReader.

        Parameters
        ----------
        base_reader: DistSampleReader
            The reader responsible for loading saved samples produced by
            the cuGraph distributed sampler.
        """
        self.__output_format = output_format
        self.__base_reader = base_reader
        self.__num_samples_remaining = 0
        self.__index = 0

    @property
    def output_format(self) -> str:
        return self.__output_format

    def __next__(self) -> DGLSamplerOutput:
        if self.__num_samples_remaining == 0:
            # raw_sample_data is already a dict of tensors
            self.__raw_sample_data, start_inclusive, end_inclusive = next(
                self.__base_reader
            )

            self.__decoded_samples = self._decode_all(self.__raw_sample_data)
            self.__num_samples_remaining = end_inclusive - start_inclusive + 1
            self.__index = 0

        out = self.__decoded_samples[self.__index]
        self.__index += 1
        self.__num_samples_remaining -= 1
        return out

    def _decode_all(self) -> List[DGLSamplerOutput]:
        raise NotImplementedError("Must be implemented by subclass")

    def __iter__(self) -> DGLSamplerOutput:
        return self


class HomogeneousSampleReader(SampleReader):
    """
    Subclass of SampleReader that reads DGL homogeneous output samples
    produced by the cuGraph distributed sampler.
    """

    def __init__(self, base_reader: DistSampleReader, output_format: str = "dgl.Block"):
        """
        Constructs a new HomogeneousSampleReader

        Parameters
        ----------
        base_reader: DistSampleReader
            The reader responsible for loading saved samples produced by
            the cuGraph distributed sampler.
        output_format: str
            The output format for blocks (either "dgl.Block" or
            "cugraph_dgl.nn.SparseGraph").
        """
        super().__init__(base_reader, output_format=output_format)

    def __decode_csc(
        self, raw_sample_data: Dict[str, "torch.Tensor"]
    ) -> List[DGLSamplerOutput]:
        return create_homogeneous_sampled_graphs_from_tensors_csc(
            raw_sample_data, output_format=self.output_format
        )

    def __decode_coo(
        self, raw_sample_data: Dict[str, "torch.Tensor"]
    ) -> List[DGLSamplerOutput]:
        raise NotImplementedError(
            "COO format is currently unsupported in the non-dask API"
        )

    def _decode_all(
        self, raw_sample_data: Dict[str, "torch.Tensor"]
    ) -> List[DGLSamplerOutput]:
        if "major_offsets" in raw_sample_data:
            return self.__decode_csc(raw_sample_data)
        else:
            return self.__decode_coo(raw_sample_data)


class Sampler:
    """
    Base sampler class for all cugraph-DGL samplers.
    """

    def __init__(self, sparse_format: str = "csc", output_format="dgl.Block"):
        """
        Parameters
        ----------
        sparse_format: str
            Optional (default = "coo").
            The sparse format of the emitted sampled graphs.
            Currently, only "csc" is supported.
        output_format: str
            Optional (default = "dgl.Block")
            The output format of the emitted sampled graphs.
            Can be either "dgl.Block" (default), or "cugraph_dgl.nn.SparseGraph".
        """

        if sparse_format != "csc":
            raise ValueError("Only CSC format is supported at this time")

        self.__sparse_format = sparse_format
        self.__output_format = output_format

    @property
    def output_format(self):
        return self.__output_format

    @property
    def sparse_format(self):
        return self.__sparse_format

    def sample(
        self, g: cugraph_dgl.Graph, indices: Iterator["torch.Tensor"], batch_size: int = 1
    ) -> Iterator[
        Tuple["torch.Tensor", "torch.Tensor", List[Union[SparseGraph, "dgl.Block"]]]
    ]:
        """
        Samples the graph.

        Parameters
        ----------
        g: cugraph_dgl.Graph
            The graph being sampled.
        indices: TensorType
            The node ids of seed nodes where sampling will initiate from.
        batch_size: int
            The number of seed nodes per batch.

        Returns
        -------
        Iterator[DGLSamplerOutput]
            Iterator over batches.  The returned tuples are in standard
            DGL format: (input nodes, output nodes, blocks) where input
            nodes are the renumbered input nodes, output nodes are
            the renumbered output nodes, and blocks are the output graphs
            for each hop.
        """

        raise NotImplementedError("Must be implemented by subclass")

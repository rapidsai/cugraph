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

import os

from math import ceil


import cupy

from cugraph.utilities.utils import MissingModule
from cugraph.gnn.data_loading.dist_io import DistSampleReader

from cugraph.gnn.data_loading.bulk_sampler_io import create_df_from_disjoint_arrays

from typing import Iterator, Tuple, Dict

torch = MissingModule("torch")


class DistSampleWriter:
    def __init__(
        self,
        directory: str,
        *,
        batches_per_partition: int = 256,
        format: str = "parquet",
    ):
        """
        Parameters
        ----------
        directory: str (required)
            The directory where samples will be written.  This
            writer can only write to disk.
        batches_per_partition: int (optional, default=256)
            The number of batches to write in a single file.
        format: str (optional, default='parquet')
            The file format of the output files containing the
            sampled minibatches.  Currently, only parquet format
            is supported.
        """
        if format != "parquet":
            raise ValueError("Invalid format (currently supported: 'parquet')")

        self.__format = format
        self.__directory = directory
        self.__batches_per_partition = batches_per_partition

    @property
    def _format(self):
        return self.__format

    @property
    def _directory(self):
        return self.__directory

    @property
    def _batches_per_partition(self):
        return self.__batches_per_partition

    def get_reader(
        self, rank: int
    ) -> Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]:
        """
        Returns an iterator over sampled data.
        """

        # currently only disk reading is supported
        return DistSampleReader(self._directory, format=self._format, rank=rank)

    def __write_minibatches_coo(self, minibatch_dict):
        has_edge_ids = minibatch_dict["edge_id"] is not None
        has_edge_types = minibatch_dict["edge_type"] is not None
        has_weights = minibatch_dict["weight"] is not None

        if minibatch_dict["renumber_map"] is None:
            raise ValueError(
                "Distributed sampling without renumbering is not supported"
            )

        # Quit if there are no batches to write.
        if len(minibatch_dict["batch_id"]) == 0:
            return

        fanout_length = (len(minibatch_dict["label_hop_offsets"]) - 1) // len(
            minibatch_dict["batch_id"]
        )

        for p in range(
            0, int(ceil(len(minibatch_dict["batch_id"]) / self.__batches_per_partition))
        ):
            partition_start = p * (self.__batches_per_partition)
            partition_end = (p + 1) * (self.__batches_per_partition)

            label_hop_offsets_array_p = minibatch_dict["label_hop_offsets"][
                partition_start * fanout_length : partition_end * fanout_length + 1
            ]

            batch_id_array_p = minibatch_dict["batch_id"][partition_start:partition_end]
            start_batch_id = batch_id_array_p[0]

            input_offsets_p = minibatch_dict["input_offsets"][
                partition_start : (partition_end + 1)
            ]
            input_index_p = minibatch_dict[input_offsets_p[0] : input_offsets_p[-1]]

            start_ix, end_ix = label_hop_offsets_array_p[[0, -1]]
            majors_array_p = minibatch_dict["majors"][start_ix:end_ix]
            minors_array_p = minibatch_dict["minors"][start_ix:end_ix]
            edge_id_array_p = (
                minibatch_dict["edge_id"][start_ix:end_ix]
                if has_edge_ids
                else cupy.array([], dtype="int64")
            )
            edge_type_array_p = (
                minibatch_dict["edge_type"][start_ix:end_ix]
                if has_edge_types
                else cupy.array([], dtype="int32")
            )
            weight_array_p = (
                minibatch_dict["weight"][start_ix:end_ix]
                if has_weights
                else cupy.array([], dtype="float32")
            )

            # create the renumber map offsets
            renumber_map_offsets_array_p = minibatch_dict["renumber_map_offsets"][
                partition_start : partition_end + 1
            ]

            renumber_map_start_ix, renumber_map_end_ix = renumber_map_offsets_array_p[
                [0, -1]
            ]

            renumber_map_array_p = minibatch_dict["renumber_map"][
                renumber_map_start_ix:renumber_map_end_ix
            ]

            results_dataframe_p = create_df_from_disjoint_arrays(
                {
                    "majors": majors_array_p,
                    "minors": minors_array_p,
                    "map": renumber_map_array_p,
                    "label_hop_offsets": label_hop_offsets_array_p,
                    "weight": weight_array_p,
                    "edge_id": edge_id_array_p,
                    "edge_type": edge_type_array_p,
                    "renumber_map_offsets": renumber_map_offsets_array_p,
                    "input_index": input_index_p,
                    "input_offsets": input_offsets_p,
                }
            )

            end_batch_id = start_batch_id + len(batch_id_array_p) - 1
            rank = minibatch_dict["rank"] if "rank" in minibatch_dict else 0

            full_output_path = os.path.join(
                self.__directory,
                f"batch={rank:05d}.{start_batch_id:08d}-"
                f"{rank:05d}.{end_batch_id:08d}.parquet",
            )

            results_dataframe_p.to_parquet(
                full_output_path,
                compression=None,
                index=False,
                force_nullable_schema=True,
            )

    def __write_minibatches_csr(self, minibatch_dict):
        has_edge_ids = minibatch_dict["edge_id"] is not None
        has_edge_types = minibatch_dict["edge_type"] is not None
        has_weights = minibatch_dict["weight"] is not None

        if minibatch_dict["renumber_map"] is None:
            raise ValueError(
                "Distributed sampling without renumbering is not supported"
            )

        # Quit if there are no batches to write.
        if len(minibatch_dict["batch_id"]) == 0:
            return

        fanout_length = (len(minibatch_dict["label_hop_offsets"]) - 1) // len(
            minibatch_dict["batch_id"]
        )

        for p in range(
            0, int(ceil(len(minibatch_dict["batch_id"]) / self.__batches_per_partition))
        ):
            partition_start = p * (self.__batches_per_partition)
            partition_end = (p + 1) * (self.__batches_per_partition)

            label_hop_offsets_array_p = minibatch_dict["label_hop_offsets"][
                partition_start * fanout_length : partition_end * fanout_length + 1
            ]

            batch_id_array_p = minibatch_dict["batch_id"][partition_start:partition_end]
            start_batch_id = batch_id_array_p[0]

            input_offsets_p = minibatch_dict["input_offsets"][
                partition_start : (partition_end + 1)
            ]
            input_index_p = minibatch_dict[input_offsets_p[0] : input_offsets_p[-1]]
            edge_inverse_p = (
                minibatch_dict["edge_inverse"][
                    (input_offsets_p[0] * 2) : (input_offsets_p[-1] * 2)
                ]
                if "edge_inverse" in minibatch_dict
                else None
            )

            # major offsets and minors
            (
                major_offsets_start_incl,
                major_offsets_end_incl,
            ) = label_hop_offsets_array_p[[0, -1]]

            start_ix, end_ix = minibatch_dict["major_offsets"][
                [major_offsets_start_incl, major_offsets_end_incl]
            ]

            major_offsets_array_p = minibatch_dict["major_offsets"][
                major_offsets_start_incl : major_offsets_end_incl + 1
            ]

            minors_array_p = minibatch_dict["minors"][start_ix:end_ix]
            edge_id_array_p = (
                minibatch_dict["edge_id"][start_ix:end_ix]
                if has_edge_ids
                else cupy.array([], dtype="int64")
            )
            edge_type_array_p = (
                minibatch_dict["edge_type"][start_ix:end_ix]
                if has_edge_types
                else cupy.array([], dtype="int32")
            )
            weight_array_p = (
                minibatch_dict["weight"][start_ix:end_ix]
                if has_weights
                else cupy.array([], dtype="float32")
            )

            # create the renumber map offsets
            renumber_map_offsets_array_p = minibatch_dict["renumber_map_offsets"][
                partition_start : partition_end + 1
            ]

            renumber_map_start_ix, renumber_map_end_ix = renumber_map_offsets_array_p[
                [0, -1]
            ]

            renumber_map_array_p = minibatch_dict["renumber_map"][
                renumber_map_start_ix:renumber_map_end_ix
            ]

            results_dataframe_p = create_df_from_disjoint_arrays(
                {
                    "major_offsets": major_offsets_array_p,
                    "minors": minors_array_p,
                    "map": renumber_map_array_p,
                    "label_hop_offsets": label_hop_offsets_array_p,
                    "weight": weight_array_p,
                    "edge_id": edge_id_array_p,
                    "edge_type": edge_type_array_p,
                    "renumber_map_offsets": renumber_map_offsets_array_p,
                    "input_index": input_index_p,
                    "input_offsets": input_offsets_p,
                    "edge_inverse": edge_inverse_p,
                }
            )

            end_batch_id = start_batch_id + len(batch_id_array_p) - 1
            rank = minibatch_dict["rank"] if "rank" in minibatch_dict else 0

            full_output_path = os.path.join(
                self.__directory,
                f"batch={rank:05d}.{start_batch_id:08d}-"
                f"{rank:05d}.{end_batch_id:08d}.parquet",
            )

            results_dataframe_p.to_parquet(
                full_output_path,
                compression=None,
                index=False,
                force_nullable_schema=True,
            )

    def write_minibatches(self, minibatch_dict):
        if (minibatch_dict["majors"] is not None) and (
            minibatch_dict["minors"] is not None
        ):
            self.__write_minibatches_coo(minibatch_dict)
        elif (minibatch_dict["major_offsets"] is not None) and (
            minibatch_dict["minors"] is not None
        ):
            self.__write_minibatches_csr(minibatch_dict)
        else:
            raise ValueError("invalid columns")

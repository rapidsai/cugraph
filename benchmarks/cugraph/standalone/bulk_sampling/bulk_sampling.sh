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

export RAPIDS_NO_INITIALIZE="1"
export CUDF_SPILL="1"
export LIBCUDF_CUFILE_POLICY=OFF


dataset_name=$1
dataset_root=$2
output_root=$3
batch_sizes=$4
fanouts=$5
reverse_edges=$6

rm -rf $output_root
mkdir -p $output_root

# Change to 2 in Selene
gpu_per_replica=4
#--add_edge_ids \

# Expand to 1, 4, 8 in Selene
for i in 1,2,3,4:
do 
    for replication in 2;
    do
        dataset_name_with_replication="${dataset_name}[${replication}]"
        dask_worker_devices=$(seq -s, 0 $((gpu_per_replica*replication-1)))
        echo "Sampling dataset = $dataset_name_with_replication on devices = $dask_worker_devices"
        python3 cugraph_bulk_sampling.py --datasets $dataset_name_with_replication \
                --dataset_root $dataset_root \
                --batch_sizes $batch_sizes \
                --output_root $output_root \
                --dask_worker_devices $dask_worker_devices \
                --fanouts $fanouts \
                --batch_sizes $batch_sizes \
                --reverse_edges
    done
done
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

import cudf
import cupy
import cugraph
from cugraph.experimental.datasets import karate
from cugraph.experimental import BulkSampler

import tempfile

def test_bulk_sampler_simple():
    el = karate.get_edgelist().reset_index().rename(columns={'index':'eid'})
    el['eid'] = el['eid'].astype('int32')
    el['etp'] = cupy.int32(0)
    print(el)
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(el, source='src', destination='dst', edge_attr=['wgt', 'eid', 'etp'], legacy_renum_only=True)

    tempdir_object = tempfile.TemporaryDirectory()
    bs = BulkSampler(
        output_path='/tmp/samples2',
        graph=G,
        fanout_vals=[2,2],
        with_replacement=False
    )

    batches = cudf.DataFrame({
        'start':cudf.Series([0,5,10], dtype='int32'),
        'batch':cudf.Series([0,0,1], dtype='int32'),
    })

    bs.add_batches(batches, start_col_name='start', batch_col_name='batch')
    bs.flush()

    
    assert False
    #assert len(cudf.read_parquet(tempdir_object.name)) == len(batches)
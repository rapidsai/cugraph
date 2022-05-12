# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
from pathlib import Path
from itertools import product

import pytest
import cudf


RAPIDS_DATASET_ROOT_DIR = os.getenv("RAPIDS_DATASET_ROOT_DIR", "../datasets")
RAPIDS_DATASET_ROOT_DIR_PATH = Path(RAPIDS_DATASET_ROOT_DIR)


def read_csv_file(csv_file, weights_dtype="float32"):
    return cudf.read_csv(
        csv_file,
        delimiter=" ",
        dtype=["int32", "int32", weights_dtype],
        header=None,
    )


def genFixtureParamsProduct(*args):
    """
    Returns the cartesian product of the param lists passed in. The lists must
    be flat lists of pytest.param objects, and the result will be a flat list
    of pytest.param objects with values and meta-data combined accordingly. A
    flat list of pytest.param objects is required for pytest fixtures to
    properly recognize the params. The combinations also include ids generated
    from the param values and id names associated with each list. For example:

    genFixtureParamsProduct( ([pytest.param(True, marks=[pytest.mark.A_good]),
                               pytest.param(False, marks=[pytest.mark.A_bad])],
                              "A"),
                             ([pytest.param(True, marks=[pytest.mark.B_good]),
                               pytest.param(False, marks=[pytest.mark.B_bad])],
                              "B") )

    results in fixture param combinations:

    True, True   - marks=[A_good, B_good] - id="A=True,B=True"
    True, False  - marks=[A_good, B_bad]  - id="A=True,B=False"
    False, True  - marks=[A_bad, B_good]  - id="A=False,B=True"
    False, False - marks=[A_bad, B_bad]   - id="A=False,B=False"

    Simply using itertools.product on the lists would result in a list of
    sublists of individual param objects (ie. not "merged"), which would not be
    recognized properly as params for a fixture by pytest.

    NOTE: This function is only needed for parameterized fixtures.
    Tests/benchmarks will automatically get this behavior when specifying
    multiple @pytest.mark.parameterize(param_name, param_value_list)
    decorators.
    """
    # Ensure each arg is a list of pytest.param objs, then separate the params
    # and IDs.
    paramLists = []
    ids = []
    paramType = pytest.param().__class__
    for (paramList, id) in args:
        for i in range(len(paramList)):
            if not isinstance(paramList[i], paramType):
                paramList[i] = pytest.param(paramList[i])
        paramLists.append(paramList)
        ids.append(id)

    retList = []
    for paramCombo in product(*paramLists):
        values = [p.values[0] for p in paramCombo]
        marks = [m for p in paramCombo for m in p.marks]
        comboid = ",".join(
            ["%s=%s" % (id, p.values[0]) for (p, id) in zip(paramCombo, ids)]
        )
        retList.append(pytest.param(values, marks=marks, id=comboid))
    return retList

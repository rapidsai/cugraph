# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

RAPIDS_DATASET_ROOT_DIR = os.getenv(
    "RAPIDS_DATASET_ROOT_DIR", os.path.join(os.path.dirname(__file__), "../datasets")
)
RAPIDS_DATASET_ROOT_DIR_PATH = Path(RAPIDS_DATASET_ROOT_DIR)


def gen_fixture_params(*param_values):
    """
    Returns a list of pytest.param objects suitable for use as fixture
    parameters created by merging the values in each tuple into individual
    pytest.param objects.

    Each tuple can contain multiple values or pytest.param objects. If pytest.param
    objects are given, the marks and ids are also merged.

    If ids is specicified, it must either be a list of string ids for each
    combination passed in, or a callable that accepts a list of values and
    returns a string.

    gen_fixture_params( (pytest.param(True, marks=[pytest.mark.A_good], id="A=True"),
                         pytest.param(False, marks=[pytest.mark.B_bad], id="B=False")),
                        (pytest.param(False, marks=[pytest.mark.A_bad], id="A=False"),
                         pytest.param(True, marks=[pytest.mark.B_good], id="B=True")),
                       )


    results in fixture param combinations:

    True, False  - marks=[A_good, B_bad]  - id="A=True,B=False"
    False, False - marks=[A_bad, B_bad]   - id="A=False,B=True"
    """
    fixture_params = []
    param_type = pytest.param().__class__  #

    for vals in param_values:
        new_param_values = []
        new_param_marks = []
        new_param_ids = []
        for val in vals:
            if isinstance(val, param_type):
                new_param_values += val.values
                new_param_marks += val.marks
                new_param_ids.append(val.id)
            else:
                new_param_values += val
                new_param_ids.append(str(val))
        fixture_params.append(
            pytest.param(
                new_param_values, marks=new_param_marks, id="-".join(new_param_ids)
            )
        )
    return fixture_params


def gen_fixture_params_product(*args):
    """
    Returns the cartesian product of the param lists passed in. The lists must
    be flat lists of pytest.param objects, and the result will be a flat list
    of pytest.param objects with values and meta-data combined accordingly. A
    flat list of pytest.param objects is required for pytest fixtures to
    properly recognize the params. The combinations also include ids generated
    from the param values and id names associated with each list. For example:

    gen_fixture_params_product( ([pytest.param(True, marks=[pytest.mark.A_good]),
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
    for (paramList, paramId) in args:
        paramListCopy = paramList[:]  # do not modify the incoming lists!
        for i in range(len(paramList)):
            if not isinstance(paramList[i], paramType):
                paramListCopy[i] = pytest.param(paramList[i])
        paramLists.append(paramListCopy)
        ids.append(paramId)

    retList = []
    for paramCombo in product(*paramLists):
        values = [p.values[0] for p in paramCombo]
        marks = [m for p in paramCombo for m in p.marks]
        id_strings = []
        for (p, paramId) in zip(paramCombo, ids):
            # Assume paramId is either a string or a callable
            if isinstance(paramId, str):
                id_strings.append("%s=%s" % (paramId, p.values[0]))
            else:
                id_strings.append(paramId(p.values[0]))
        comboid = ",".join(id_strings)
        retList.append(pytest.param(values, marks=marks, id=comboid))
    return retList

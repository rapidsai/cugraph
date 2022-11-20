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


def gen_fixture_params(*param_values, ids=None):
    """
    Returns a list of pytest.param objects suitable for use as fixture
    parameters created by merging the values in each tuple into individual
    pytest.param objects.

    Each tuple can contain multiple values or pytest.param objects. If pytest.param
    objects are given, the marks are also merged but any ids part of the
    pytest.param object are ignored.

    If ids is specicified, it must either be a list of string ids for each
    combination passed in, or a callable that accepts a list of values and
    returns a string.

    gen_fixture_params( (pytest.param(True, marks=[pytest.mark.A_good]),
                         pytest.param(False, marks=[pytest.mark.B_bad])),
                        (pytest.param(False, marks=[pytest.mark.A_bad]),
                         pytest.param(True, marks=[pytest.mark.B_good])),
                        ids=["combo1", "combo2"] )

    results in fixture param combinations:

    True, False  - marks=[A_good, B_bad]  - id="combo1"
    False, False - marks=[A_bad, B_bad]   - id="combo2"
    """
    fixture_params = []
    param_type = pytest.param().__class__  #
    ids_is_list = isinstance(ids, list)

    if ids_is_list and (ids is not None) and (len(ids) < len(param_values)):
        raise ValueError("ids list length < number of param values")

    for (vals_idx, vals) in enumerate(param_values):
        new_param_values = []
        new_param_marks = []
        new_param_id = ""
        for val in vals:
            if isinstance(val, param_type):
                new_param_values += val.values
                new_param_marks += val.marks
            else:
                new_param_values += val

        if ids_is_list:
            new_param_id = ids[vals_idx]
        elif ids is not None:
            new_param_id = ids(new_param_values)
        else:
            new_param_id = None

        fixture_params.append(
            pytest.param(new_param_values, marks=new_param_marks, id=new_param_id)
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

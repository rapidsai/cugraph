# pytest customizations specific to these benchmarks
import sys
from os import path
import importlib


def pytest_addoption(parser):
    parser.addoption("--no-rmm-reinit", action="store_true", default=False,
                     help="Do not reinit RMM to run benchmarks with different"
                          " managed memory and pool allocator options.")


def pytest_sessionstart(session):
    # if the --no-rmm-reinit option is given, import the benchmark's "params"
    # module and change the FIXTURE_PARAMS accordingly.
    if session.config.getoption("no_rmm_reinit"):
        paramsPyFile = path.join(path.dirname(path.abspath(__file__)),
                                 "params.py")

        # A simple "import" statement will not find the modules here (unless if
        # this package is on the import path) since pytest evaluates this from
        # a different location.
        spec = importlib.util.spec_from_file_location("params", paramsPyFile)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        module.FIXTURE_PARAMS = module.NO_RMMREINIT_FIXTURE_PARAMS

        # If "benchmarks.params" is registered in sys.modules, all future
        # imports of the module will simply refer to this one.
        sys.modules["benchmarks.params"] = module

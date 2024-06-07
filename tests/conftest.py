from pathlib import Path

import pytest

from projectionizer.version import VERSION

from utils import CIRCUIT_CONFIG_FILE, fake_circuit_config

MOCK_PARAMS = {
    "FloatParameter": 1.0,
    "IntParameter": 1,
    "ListParameter": [],
    "Parameter": "fake",
    "PathParameter": Path(),
    "TaskParameter": None,
    "BoolParameter": True,
}


@pytest.fixture(name="tmp_confdir")
def fixture_tmp_confdir(tmp_path):
    """Set up a temporary config directory with a circuit config file."""
    fake_circuit_config(tmp_path)
    return tmp_path


@pytest.fixture(name="MockTask")
def fixture_MockTask(tmp_confdir, request):
    """Create a mock task from the definition.

    Automatically generates mock values for all the parameters in the task and sets the
    "circuit_config" and "folder" based on the "tmp_confdir" fixture.

    To use in a test, just use a decorator, inherit it in your test class and overwrite
    parameters/methods normally.

    Example:
    >>> @pytest.mark.MockTask(cls=TaskYouWishToMock)
    ... def test_TaskyouWishToMock(MockTask):
    ...
    ...     class TestTaskYouWishToMock(MockTask):
    ...         parameter_a = "I want an actual value here for a test"  # override parameters
    ...
    ...         def input(self):  # override method
    ...             pass
    """
    cls = request.keywords[request.fixturename].kwargs["cls"]

    class MockTask(cls):
        pass

    def _get_fake_value(param_name):
        if param_name == "folder":
            return tmp_confdir
        if param_name == "circuit_config":
            return tmp_confdir / CIRCUIT_CONFIG_FILE
        if param_name == "projectionizer_version":
            return VERSION
        if param_name == "module_archive":
            return "unstable"
        if param_name == "morphology_type":
            return "asc"

        param = getattr(cls, param_name)
        return MOCK_PARAMS[param.__class__.__name__]

    for param in cls.get_param_names():
        setattr(MockTask, param, _get_fake_value(param))

    return MockTask

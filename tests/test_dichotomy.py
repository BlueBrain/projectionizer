import json

import luigi
import pytest
from luigi import FloatParameter, LocalTarget, PathParameter, Task, build

import projectionizer.dichotomy as test_module
from projectionizer import step_3_write

from utils import as_iterable


class LinearTask(Task):
    """Test class"""

    folder = PathParameter(absolute=True)
    param = FloatParameter()

    def run(self):
        with self.output().open("w") as outf:
            json.dump({"result": 20 + self.param}, outf)

    def output(self):
        return LocalTarget(f"{self.folder}/result-param-{self.param}.json")


def test_simple(tmp_confdir):
    task = LinearTask(param=-5, folder=tmp_confdir)
    task.run()
    with task.output().open() as inputf:
        assert json.load(inputf)["result"] == 15


@pytest.mark.MockTask(cls=test_module.SynapseCountMeanMinimizer)
def test_parameter_sharing(MockTask):
    """Test that the shared SONATA parameters are correctly shared and passed forward.

    I.e., check that shared parameter values are actually shared and those that should differ,
    actually differ between different projectionizer tasks.
    """
    config = {
        "mtype": "test_mtype",
        "node_file_name": "fake_nodes.h5",
        "edge_file_name": "fake_edges.h5",
        "node_population": "test_node_pop",
        "edge_population": "test_edge_pop",
    }
    for param, value in config.items():
        setattr(MockTask, param, value)

    def _check_params(task):
        for param, expected_value in config.items():
            if hasattr(task, param):
                assert getattr(task, param) == expected_value

        try:
            for subtask in as_iterable(task.requires()):
                _check_params(subtask)
        except luigi.parameter.MissingParameterException:
            # At this point we are wandering off from SONATA related tasks
            pass

    class MockSynapseCountMeanMinimizer(MockTask):
        def requires(self):
            return self.clone(step_3_write.RunAll)

    _check_params(MockSynapseCountMeanMinimizer())


class MismatchLinearTask(Task):
    """The task whose value must be minimized"""

    folder = PathParameter(absolute=True)
    target = FloatParameter()
    param = FloatParameter(default=0)

    def run(self):
        params = {"param": self.param, "folder": self.folder}
        task = yield self.clone(LinearTask, **params)
        with task.open() as inputf:
            with self.output().open("w") as outputf:
                json.dump({"error": json.load(inputf)["result"] - self.target}, outputf)

    def output(self):
        return LocalTarget(f"{self.folder}/MismatchLinearTask-{self.param}.json")


@pytest.mark.MockTask(cls=test_module.Dichotomy)
def test_dichotomy(MockTask):
    class TestDichotomy(MockTask):
        MinimizationTask = MismatchLinearTask
        target = -27
        target_margin = 0.5
        min_param = -123
        max_param = 456
        max_loop = 57

    res = build([TestDichotomy()], local_scheduler=True)
    assert res


@pytest.mark.MockTask(cls=test_module.Dichotomy)
def test_dichotomy_failed(MockTask):
    """Test dichotomy not converging fast enough
    leading to maximum number of iteration reached"""

    class TestDichotomy(MockTask):
        MinimizationTask = MismatchLinearTask
        target = 27
        target_margin = 5
        min_param = 123
        max_param = 456
        max_loop = 3

    res = build([TestDichotomy()], local_scheduler=True)
    assert not res

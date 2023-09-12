import json

import pytest
from luigi import FloatParameter, LocalTarget, Parameter, Task, build

import projectionizer.dichotomy as test_module


class LinearTask(Task):
    """Test class"""

    folder = Parameter()
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


class MismatchLinearTask(Task):
    """The task whose value must be minimized"""

    folder = Parameter()
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

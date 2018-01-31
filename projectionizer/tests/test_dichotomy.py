import json
import os
import time

from luigi import FloatParameter, LocalTarget, Parameter, Task, build, run
from luigi.contrib.simulate import RunAnywayTarget
from nose.tools import ok_
from numpy.testing import assert_allclose, assert_equal

from projectionizer.dichotomy import Dichotomy
from projectionizer.luigi_utils import JsonTask
from projectionizer.tests.mocks import class_with_dummy_params, dummy_params
from projectionizer.tests.utils import setup_tempdir


class LinearTask(Task):
    '''Test class'''
    param = FloatParameter()
    folder = Parameter()

    def run(self):
        print('run')
        print('self.output().path: {}'.format(self.output().path))
        with self.output().open('w') as outf:
            json.dump({'result': 20 + self.param}, outf)

    def output(self):
        return LocalTarget('{}/result-param-{}.json'.format(self.folder,
                                                            self.param))


def test_simple():
    with setup_tempdir('test_utils') as tmp_folder:
        task = LinearTask(param=-5, folder=tmp_folder)
        task.run()
        with task.output().open() as inputf:
            assert_equal(json.load(inputf)['result'], 15)


class MismatchLinearTask(JsonTask):
    '''The task whose value must be minimized'''
    target = FloatParameter()
    param = FloatParameter(default=0)

    def run(self):
        task = yield self.clone(LinearTask, param=self.param)
        with task.open() as inputf:
            with self.output().open('w') as outputf:
                json.dump({'error': json.load(inputf)['result'] - self.target},
                          outputf)


class TestDichotomy(Task):
    def requires(self):
        return self.clone(Dichotomy, **dummy_params())

    def run(self):
        with self.input().open() as inputf:
            assert_allclose(json.load(inputf)['param'], -47, atol=0.5)
        self.output().done()

    def output(self):
        return RunAnywayTarget(self)


def test_dichotomy():

    with setup_tempdir('test_utils') as tmp_folder:
        res = run(['TestDichotomy',
                   '--module', 'projectionizer.tests.test_dichotomy',
                   '--local-scheduler',
                   '--Dichotomy-MinimizationTask', 'MismatchLinearTask',
                   '--Dichotomy-target', '-27',
                   '--Dichotomy-target-margin', '0.5',
                   '--Dichotomy-min-param', '-123',
                   '--Dichotomy-max-param', '456',
                   '--Dichotomy-max-loop', '57',
                   '--Dichotomy-folder', tmp_folder])
        ok_(res)


def test_dichotomy_failed():
    '''Test dichotomy not converging fast enough
    leading to maximum number of iteration reached'''
    with setup_tempdir('test_utils') as tmp_folder:
        params = dummy_params()
        params.update({'MinimizationTask': MismatchLinearTask,
                       'target': 27,
                       'target_margin': 5,
                       'min_param': 123,
                       'max_param': 456,
                       'max_loop': 3,
                       'folder': tmp_folder})

        res = build([Dichotomy(**params)], local_scheduler=True)
        ok_(not res)

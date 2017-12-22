import json
import os

from nose.tools import ok_
from numpy.testing import assert_allclose, assert_equal

from luigi import FloatParameter, LocalTarget, Parameter, Task, run
from luigi.contrib.simulate import RunAnywayTarget
from mocks import class_with_dummy_params, dummy_params
from projectionizer.dichotomy import Dichotomy
from projectionizer.luigi_utils import JsonTask
from utils import setup_tempdir


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


def test_dichotomy():
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

import json

from luigi import FloatParameter, LocalTarget, Parameter, Task, build
from nose.tools import ok_
from numpy.testing import assert_equal

from projectionizer.dichotomy import Dichotomy

from utils import setup_tempdir


class LinearTask(Task):
    '''Test class'''
    folder = Parameter()
    param = FloatParameter()

    def run(self):
        print('run')
        print('self.output().path: {}'.format(self.output().path))
        with self.output().open('w') as outf:
            json.dump({'result': 20 + self.param}, outf)

    def output(self):
        return LocalTarget('{}/result-param-{}.json'.format(self.folder,
                                                            self.param))


def test_simple():
    with setup_tempdir('test_dichotomy') as tmp_folder:
        task = LinearTask(param=-5, folder=tmp_folder)
        task.run()
        with task.output().open() as inputf:
            assert_equal(json.load(inputf)['result'], 15)


class MismatchLinearTask(Task):
    '''The task whose value must be minimized'''
    folder = Parameter()
    target = FloatParameter()
    param = FloatParameter(default=0)

    def run(self):
        task = yield self.clone(LinearTask, param=self.param, folder=self.folder)
        with task.open() as inputf:
            with self.output().open('w') as outputf:
                json.dump({'error': json.load(inputf)['result'] - self.target},
                          outputf)

    def output(self):
        return LocalTarget('{}/MismatchLinearTask-{}.json'.format(self.folder,
                                                                  self.param))


def test_dichotomy():
    with setup_tempdir('test_dichotomy') as tmp_folder:
        params = {'MinimizationTask': MismatchLinearTask,
                  'target': -27,
                  'target_margin': 0.5,
                  'min_param': -123,
                  'max_param': 456,
                  'max_loop': 57,
                  'folder': tmp_folder,
                  }

        res = build([Dichotomy(**params)], local_scheduler=True)
        ok_(res)


def test_dichotomy_failed():
    '''Test dichotomy not converging fast enough
    leading to maximum number of iteration reached'''
    with setup_tempdir('test_dichotomy') as tmp_folder:
        params = {'MinimizationTask': MismatchLinearTask,
                  'target': 27,
                  'target_margin': 5,
                  'min_param': 123,
                  'max_param': 456,
                  'max_loop': 3,
                  'folder': tmp_folder,
                  }

        res = build([Dichotomy(**params)], local_scheduler=True)
        ok_(not res)

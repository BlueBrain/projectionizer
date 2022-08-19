import json
import os

from luigi import FloatParameter, LocalTarget, Parameter, Task, build

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
            assert json.load(inputf)['result'] == 15


class MismatchLinearTask(Task):
    '''The task whose value must be minimized'''
    folder = Parameter()
    target = FloatParameter()
    param = FloatParameter(default=0)

    def run(self):
        params = dict(param=self.param, folder=self.folder)
        task = yield self.clone(LinearTask, **params)
        with task.open() as inputf:
            with self.output().open('w') as outputf:
                json.dump({'error': json.load(inputf)['result'] - self.target},
                          outputf)

    def output(self):
        return LocalTarget('{}/MismatchLinearTask-{}.json'.format(self.folder,
                                                                  self.param))

DEFAULT_PARAMS = {
    'physiology_path': 'a/fake/path',
    'n_total_chunks': 1,
    'sgid_offset': 1,
    'oversampling': 1,
    'layers': 'fake_layers',
    }

def test_dichotomy():
    with setup_tempdir('test_dichotomy') as tmp_folder:
        params = dict(DEFAULT_PARAMS)
        params.update({'MinimizationTask': MismatchLinearTask,
                       'target': -27,
                       'target_margin': 0.5,
                       'min_param': -123,
                       'max_param': 456,
                       'max_loop': 57,
                       'folder': tmp_folder,
                       'circuit_config': os.path.join(tmp_folder, 'CircuitConfig'),
                       })
        res = build([Dichotomy(**params)], local_scheduler=True)
        assert res


def test_dichotomy_failed():
    '''Test dichotomy not converging fast enough
    leading to maximum number of iteration reached'''
    with setup_tempdir('test_dichotomy') as tmp_folder:
        params = dict(DEFAULT_PARAMS)
        params.update({'MinimizationTask': MismatchLinearTask,
                       'target': 27,
                       'target_margin': 5,
                       'min_param': 123,
                       'max_param': 456,
                       'max_loop': 3,
                       'folder': tmp_folder,
                       'circuit_config': os.path.join(tmp_folder, 'CircuitConfig'),
                       })

        res = build([Dichotomy(**params)], local_scheduler=True)
        assert not res

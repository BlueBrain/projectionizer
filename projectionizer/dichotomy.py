'''Correct oversampling binary search module'''
import json
import os

from luigi import Parameter, FloatParameter, IntParameter, TaskParameter, Task
from luigi.local_target import LocalTarget
from projectionizer.analysis import DoAll
from projectionizer.luigi_utils import CommonParams, JsonTask, camel2spinal_case, FolderTask
from projectionizer.step_0_sample import FullSample, SampleChunk
from projectionizer.step_3_write import SynapseCountPerConnectionL4PC


class Dichotomy(Task):
    '''Binary search to find the parameter that satisfies the target value'''
    folder = Parameter()
    target = FloatParameter()
    target_margin = FloatParameter(default=1)
    min_param = FloatParameter()
    max_param = FloatParameter()
    max_loop = IntParameter()
    MinimizationTask = TaskParameter()

    def run(self):
        start, end = self.min_param, self.max_param

        for _ in range(self.max_loop):
            param = 0.5 * (start + end)
            error = yield self.clone(self.MinimizationTask,
                                     param=param,
                                     folder=sub_folder(self.folder, param))

            with error.open() as inputf:
                result = json.load(inputf)['error']
            if abs(result) < self.target_margin:
                with self.output().open('w') as outf:
                    json.dump({'param': param}, outf)
                    return

            if result > 0:
                end = param
            else:
                start = param
        raise Exception('Maximum number of iteration reached')

    def output(self):
        name = camel2spinal_case(type(self).__name__)
        return LocalTarget('{}/{}.json'.format(self.folder, name))

    def requires(self):
        return FolderTask(folder=self.folder)


def sub_folder(base_folder, param):
    '''Return the directory for the given param '''
    return os.path.join(base_folder, 'param_{}'.format(param))


class TargetMismatch(JsonTask):
    '''The task whose value must be minimized'''
    target = FloatParameter()
    param = FloatParameter(default=0)

    def run(self):  # pragma: no cover
        task = yield self.clone(SynapseCountPerConnectionL4PC, oversampling=self.param)
        with task.open() as inputf:
            with self.output().open('w') as outputf:
                json.dump({'error': json.load(inputf)['result'] - self.target},
                          outputf)


class SynapseCountL4PCMeanMinimizer(CommonParams):
    '''Dichotomy applied to approaching the correct number of synapses per connection
    in L4PC cells'''
    target = FloatParameter()
    target_margin = FloatParameter(default=1)
    min_param = FloatParameter()
    max_param = FloatParameter()
    max_loop = IntParameter(default=20)

    def requires(self):  # pragma: no cover
        '''Start by generating a large number of synapses first'''
        return [self.clone(SampleChunk,
                           chunk_num=chunk_num,
                           oversampling=self.max_param,
                           folder=sub_folder(self.folder, self.max_param))
                for chunk_num in range(self.n_total_chunks)] + \
            [self.clone(Dichotomy, MinimizationTask=TargetMismatch)]

    def run(self):  # pragma: no cover
        dichotomy = self.input()[-1]
        with dichotomy.open() as inputf:
            param = json.load(inputf)['param']

        folder = sub_folder(self.folder, param)
        yield self.clone(FullSample, from_chunks=True, folder=folder)
        yield self.clone(DoAll, oversampling=param, folder=folder)

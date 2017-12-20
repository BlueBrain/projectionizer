'''Correct oversampling binary search module'''
import json
import os

import pandas as pd

from luigi import (FloatParameter, IntParameter, Parameter, Task,
                   TaskParameter)
from luigi.local_target import LocalTarget
from projectionizer.analysis import DoAll
from projectionizer.step_0_sample import SampleChunk
from projectionizer.step_3_write import SynapseCountPerConnectionL4PC
from projectionizer.utils import write_feather, load
from projectionizer.luigi_utils import CommonParams, JsonTask


class LinearTask(Task):
    '''Test class'''
    param = FloatParameter()

    def run(self):
        with self.output().open('w') as outf:
            json.dump({'result': 20 + self.param}, outf)

    def output(self):
        return LocalTarget('result-param-{}.json'.format(self.param))


class CloneTask(CommonParams):
    '''Clone sample and clone a task'''
    ClonedTask = TaskParameter()
    from_folder = Parameter()
    fraction = FloatParameter()
    chunk_num = IntParameter(default=-1)

    def requires(self):
        if self.chunk_num >= 0:
            return self.clone(self.ClonedTask, folder=self.from_folder, chunk_num=self.chunk_num)
        return self.clone(self.ClonedTask, folder=self.from_folder)

    def run(self):
        # pylint: disable=maybe-no-member
        df = load(self.input().path)
        write_feather(self.output().path, df.sample(frac=self.fraction))

    def output(self):
        name = os.path.basename(self.input().path)
        return LocalTarget(os.path.join(self.folder, name))


def try_makedir(folder):
    '''Try to create a folder'''
    try:
        os.makedirs(folder)
    except OSError:
        pass


class Dichotomy(JsonTask):
    '''Binary search to find the parameters that satisfies the connectivity_target parameter'''
    connectivity_target = FloatParameter()
    target_margin = FloatParameter(default=1)
    min_param = FloatParameter()
    max_param = FloatParameter()
    max_loop = IntParameter(default=20)

    def sub_folder(self, param):
        '''Return the directory for the given param '''
        return os.path.join(self.folder, 'param_{}'.format(param))

    def requires(self):
        '''Generating synapses with maximum oversampling'''
        folder = self.sub_folder(self.max_param)
        try_makedir(folder)
        return [self.clone(SampleChunk,
                           chunk_num=chunk_num,
                           oversampling=self.max_param,
                           folder=folder)
                for chunk_num in range(self.n_total_chunks)]

    def run(self):
        start, end = self.min_param, self.max_param
        original_oversampling = end

        for _ in range(self.max_loop):
            param = 0.5 * (start + end)
            folder = self.sub_folder(param)
            try_makedir(folder)
            chunks = list()
            for chunk_num in range(self.n_total_chunks):
                chunk = self.clone(CloneTask,
                                   ClonedTask=SampleChunk,
                                   chunk_num=chunk_num,
                                   fraction=float(param / original_oversampling),
                                   from_folder=self.sub_folder(self.max_param),
                                   oversampling=param,
                                   folder=folder)
                yield chunk
                chunks.append(load(chunk.output().path))

            job = self.clone(SynapseCountPerConnectionL4PC,
                             oversampling=param,
                             folder=folder)
            yield job
            with job.output().open() as inputf:
                job_result = json.load(inputf)
            if abs(self.connectivity_target - job_result['result']) < self.target_margin:
                write_feather('{}/full_sample.feather'.format(folder), pd.concat(chunks))
                yield self.clone(DoAll,
                                 oversampling=param,
                                 folder=folder)
                with self.output().open('w') as outf:
                    json.dump({'param': param}, outf)
                    return

            if job_result['result'] > self.connectivity_target:
                end = param
            else:
                start = param
        raise Exception('Maximum number of iteration reached')

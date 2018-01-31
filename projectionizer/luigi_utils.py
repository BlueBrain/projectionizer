'''Luigi related utils'''
import os
import re

from luigi import (Config, FloatParameter, IntParameter, Parameter, Task,
                   TaskParameter)
from luigi.local_target import LocalTarget
from projectionizer.utils import load, write_feather


def cloned_tasks(this, tasks):
    '''Utils function for self.requires()
    Returns: clone and returns a list of required tasks'''
    return [this.clone(task) for task in tasks]


def _camel_case_to_spinal_case(name):
    '''Camel case to snake case'''
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1).lower()


class FolderTask(Task):
    '''Simple dependency task to create missing folders'''
    folder = Parameter()

    def run(self):
        os.makedirs(self.folder)

    def output(self):
        return LocalTarget(self.folder)


class CommonParams(Config):
    """Paramaters that must be passed to all Task"""
    circuit_config = Parameter()
    folder = Parameter()
    geometry = Parameter()
    n_total_chunks = IntParameter()
    sgid_offset = IntParameter()
    oversampling = FloatParameter()

    # S1HL/S1 region parameters
    voxel_path = Parameter(default='')
    prefix = Parameter(default='')

    extension = None

    def output(self):
        name = _camel_case_to_spinal_case(self.__class__.__name__)
        if hasattr(self, 'chunk_num'):
            return LocalTarget('{}/{}-{}.{}'.format(self.folder,
                                                    name,
                                                    getattr(self, 'chunk_num'),
                                                    self.extension))
        return LocalTarget('{}/{}.{}'.format(self.folder, name, self.extension))

    def requires(self):
        return FolderTask(folder=self.folder)


class CsvTask(CommonParams):
    '''Task returning a CSV file'''
    extension = 'csv'


class FeatherTask(CommonParams):
    '''Task returning a feather file'''
    extension = 'feather'


class JsonTask(CommonParams):
    '''Task returning a JSON file'''
    extension = 'json'


class NrrdTask(CommonParams):
    '''Task returning a Nrrd file'''
    extension = 'nrrd'


class CloneTask(CommonParams):
    '''Clone a FeatherTask'''
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
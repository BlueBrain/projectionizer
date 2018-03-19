'''Luigi related utils'''
import os
import re
import tempfile

from luigi import (Config, FloatParameter, IntParameter, Parameter, Task,
                   )
from luigi.contrib.simulate import RunAnywayTarget
from luigi.local_target import LocalTarget


def cloned_tasks(this, tasks):
    '''Utils function for self.requires()
    Returns: clone and returns a list of required tasks'''
    return [this.clone(task) for task in tasks]


def camel2spinal_case(name):
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
        name = camel2spinal_case(self.__class__.__name__)
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


class RunAnywayTargetTempDir(RunAnywayTarget):
    '''Override tmp directory location for RunAnywayTarget

    RunAnywayTarget uses a directory in /tmp for keeping state,
    so if two different users try and launch a task that uses
    this target, it fails.  By using this target, the directory
    is under the user's control, and thus there won't be conflicts
    '''
    def __init__(self, task_obj, base_dir):
        self.temp_dir = tempfile.mkdtemp(dir=base_dir)
        super(RunAnywayTargetTempDir, self).__init__(task_obj)

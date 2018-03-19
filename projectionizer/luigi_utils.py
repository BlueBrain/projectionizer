'''Luigi related utils'''
import os
import re

import pkg_resources

from luigi import (Config, FloatParameter, IntParameter, Parameter, Task,
                   )
from luigi.contrib.simulate import RunAnywayTarget
from luigi.local_target import LocalTarget


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
    layers = Parameter()  # list of pairs of (layer name, thickness), starting at 'bottom'

    # S1HL/S1 region parameters
    voxel_path = Parameter(default='')
    prefix = Parameter(default='')

    # hex parameters
    hex_side = FloatParameter(default=0)  # size of a hexagon side
    # size of apron around the hexagon, so that there aren't edge effects when assigning
    # synapses to fibers
    hex_apron_size = FloatParameter(default=50)
    # path to CSV with two columns; x/z: location of fibers
    hex_fiber_locations = Parameter(default='rat_fibers.csv')

    extension = None

    def output(self):
        name = camel2spinal_case(self.__class__.__name__)
        target = '{}/{}.{}'.format(self.folder, name, self.extension)
        if hasattr(self, 'chunk_num'):
            target = '{}/{}-{}.{}'.format(
                self.folder, name, getattr(self, 'chunk_num'), self.extension)
        return LocalTarget(target)

    def requires(self):
        return FolderTask(folder=self.folder)

    @staticmethod
    def load_data(path):
        '''completely unqualified paths are loaded from the templates directory'''
        if '/' in path:
            return path
        else:
            templates = pkg_resources.resource_filename('projectionizer', 'templates')
            return os.path.join(templates, path)


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
        self.temp_dir = os.path.join(base_dir, 'luigi-tmp')
        super(RunAnywayTargetTempDir, self).__init__(task_obj)

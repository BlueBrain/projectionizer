'''Luigi related utils'''
import os
import re

import pkg_resources

from luigi import (Config, FloatParameter, IntParameter, Parameter, Task,
                   ListParameter,
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
    n_total_chunks = IntParameter()
    sgid_offset = IntParameter()
    oversampling = FloatParameter()
    layers = ListParameter()  # list of pairs of (layer name, thickness), starting at 'bottom'
    target_mtypes = ListParameter(default=['L4_PC', 'L4_UPC', 'L4_TPC', ])  # list of mtypes
    regions = ListParameter(default=[])

    # S1HL/S1 region parameters
    voxel_path = Parameter(default='')

    # hex parameters
    # bounding box for apron around the hexagon, so that there aren't edge effects when assigning
    # synapses to fibers
    # ListParameter can not default to None without further problems with luigi
    hex_apron_bounding_box = ListParameter(default=[])
    # path to CSV with two columns; x/z: location of fibers
    hex_fiber_locations = Parameter(default=None)

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

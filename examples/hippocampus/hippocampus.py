#!/usr/bin/env python
'''
module load brainbuilder
brainbuilder syn2 concat -o O1_ca1_20191017_projections.syn2 SYN2/*.syn2

module load synapsetool
syn-tool order-synapses O0_ca1_20191017.syn2 O0_ca1_20191017_sorted.syn2

module load brainbuilder
brainbuilder sonata from-syn2 -o O0_ca1_20191017_sorted.sonata O0_ca1_20191017_sorted.syn2
'''

from glob import glob
import logging
import os
import shutil
import sys

import click
import numpy as np
import pandas as pd
from projectionizer import (
    afferent_section_position,
    hippocampus,
    utils,
    version,
    write_sonata as sonata
)

from bluepy import Circuit

L = logging.getLogger(__name__)

ASSIGN_PATH = 'ASSIGNMENT'
MERGE_PATH = 'MERGE'
SONATA_PATH = 'SONATA'
SYN2_PATH = 'SYN2'
CHUNK_SIZE = 100000000

SEGMENT_START_COLS = hippocampus.SEGMENT_START_COLS
SEGMENT_END_COLS = hippocampus.SEGMENT_END_COLS
ASSIGNED_COLS = [
    'sgid',
    'tgid',
    'section_id',
    'segment_id',
    'synapse_offset',
    'section_pos',
    'section_type',
    'x',
    'y',
    'z',
]

REQUIRED_PATH = click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True)

PROJECTION_TEMPLATE = '''\
Projection {name}
{{
    Path {path}
    Source {source}
}}
'''


class DataFrameWrapper():
    '''Wrapper for the separately saved pandas.DataFrame columns.

    To have the separate columns to act as if they were in same DataFrame (for write_sonata).
    '''

    def __init__(self, path, distance_soma):
        self._path = path
        self._length = None
        self._distance_soma = distance_soma

    @property
    def sgid_path_distance(self):
        return pd.Series(np.full(len(self), self._distance_soma),
                         name='sgid_path_distance',
                         dtype=np.float32)

    @property
    def columns(self):
        return ASSIGNED_COLS

    def __len__(self):
        if self._length is None:
            self._length = len(self[ASSIGNED_COLS[0]])

        return self._length

    def _get_column_path(self, column):
        return os.path.join(self._path, column + '.feather')

    def __getattr__(self, attribute):
        if attribute in self.columns:
            return pd.read_feather(self._get_column_path(attribute))[attribute]
        return self.__getattribute__(attribute)

    def __getitem__(self, item):
        return self.__getattr__(item)


def _create_dir(path):
    '''Create a dir if it doesn't exist'''
    if not os.path.exists(path):
        os.makedirs(path)


def _path_exists(path):
    exists = os.path.exists(path)
    if exists:
        L.info('Already have %s, skipping', path)

    return exists


@click.group()
@click.version_option(version.VERSION)
@click.option('-c', '--config', type=REQUIRED_PATH, required=True)
@click.option('-o', '--output', required=True)
@click.option('-v', '--verbose', count=True)
@click.pass_context
def cli(ctx, config, verbose, output):
    '''Hippocampus projections generation'''
    L.setLevel((logging.WARNING,
                logging.INFO,
                logging.DEBUG)[min(verbose, 2)])
    if verbose > 3:
        import multiprocessing.util
        multiprocessing.util.log_to_stderr(logging.DEBUG)

    _create_dir(output)
    orig_config = os.path.join(output, os.path.basename(config))
    if os.path.exists(orig_config):
        orig_config_contents = utils.load(orig_config)
        config_contents = utils.load(config)

        if config_contents != orig_config_contents:
            sys.exit('Config at %s already exists, and is different.  '
                     'Please delete it or pick another directory' % orig_config)
    else:
        shutil.copy(config, output)

    ctx.obj['config'] = utils.load(config)

    ctx.obj['output'] = output


@cli.command()
@click.option('--region', required=True)
@click.pass_context
def full_sample(ctx, region):
    '''fully sample all the regions'''
    config, output = ctx.obj['config'], ctx.obj['output']
    atlas = Circuit(config['circuit_config']).atlas
    index_path = config['spatial_index_path']

    assert region in config['region_percentages']

    region_map = atlas.load_region_map()
    brain_regions = atlas.load_data('brain_regions')

    region_ids = region_map.find('@' + region, attr='acronym')

    for id_ in region_ids:
        L.debug('Sampling %s[%s]', region, id_)
        hippocampus.full_sample_parallel(brain_regions, region, id_, index_path, output)


def _pick_synapse_locations(segs, dist, count):
    picked = np.random.choice(
        len(segs),
        size=count,
        replace=True,
        p=dist)
    segs = segs.iloc[picked].reset_index(drop=True)

    alpha = np.random.random_sample((len(segs), 1)).astype(np.float32)
    segs['synapse_offset'] = alpha[:, 0] * segs['segment_length'].values

    starts = segs[SEGMENT_START_COLS].values
    ends = segs[SEGMENT_END_COLS].values

    locations = (alpha * starts + (1. - alpha) * ends).astype(np.float32)
    locations = pd.DataFrame(locations, columns=utils.XYZ, index=segs.index)

    segs.drop(SEGMENT_START_COLS + SEGMENT_END_COLS, axis=1, inplace=True)

    segs = segs.join(locations)

    L.debug('Subsample: %s', len(segs))

    return segs


def _assign_sgid(syns, sgid_start, sgid_count):
    '''assign source gids to syns'''
    if sgid_count == 1:
        syns['sgid'] = sgid_start
    else:
        sgids = np.arange(sgid_start, sgid_start + sgid_count)
        syns['sgid'] = np.random.choice(sgids, size=len(syns), replace=True)

    return syns


def _assign(cells, morph_class, count, samples_path, output, region, sgid_start, sgid_count, config):
    output = os.path.join(output, ASSIGN_PATH)
    _create_dir(output)

    L.debug('Assign doing: %s[%s] with %d synapses', region, morph_class, count)

    segs = [utils.read_feather(path)
            for path in glob(os.path.join(samples_path, region + '_*.feather'))]
    segs = pd.concat(segs, sort=False, ignore_index=True).join(cells['morph_class'], on='tgid')
    segs = segs[segs['morph_class'] == morph_class].drop(['morph_class'], axis=1)

    if len(segs) == 0:
        return

    dist = utils.normalize_probability(segs.segment_length.values)

    for i, size in enumerate([CHUNK_SIZE] * (count // CHUNK_SIZE) + [(count % CHUNK_SIZE)]):
        path = os.path.join(output, '%s_%s_%05d.feather' % (region, morph_class, i))
        if not _path_exists(path):
            with utils.delete_file_on_exception(path):
                syns = _pick_synapse_locations(segs, dist, size)
                syns = _assign_sgid(syns, sgid_start, sgid_count)
                morphs = utils.get_morphs_for_nodes(
                    config['target_node_path'],
                    config['target_node_population'],
                    config['morphology_path'],
                    config['morphology_type'],
                )
                syns['section_pos'] = afferent_section_position.compute_positions(syns, morphs)
                utils.write_feather(path, syns)


@cli.command()
@click.option('--region', required=True)
@click.pass_context
def assign(ctx, region):
    '''create correct count of synapses, assign their sgid

    This is performed separately for INT and PYR synapses
    '''
    config, output = ctx.obj['config'], ctx.obj['output']

    cells = Circuit(config['circuit_config']).cells.get()

    synapse_count = config['synapse_count_per_type']
    int_count = int(config['region_percentages'][region] *
                    synapse_count['INT'] *
                    np.count_nonzero(cells.morph_class == 'INT'))
    pyr_count = int(config['region_percentages'][region] *
                    synapse_count['PYR'] *
                    np.count_nonzero(cells.morph_class == 'PYR'))

    sgid_start, sgid_count = config['sgid_start'], config['sgid_count']

    sample_path = os.path.join(output, hippocampus.SAMPLE_PATH)
    _assign(cells, 'INT', int_count, sample_path, output, region, sgid_start, sgid_count, config)
    _assign(cells, 'PYR', pyr_count, sample_path, output, region, sgid_start, sgid_count, config)


@cli.command()
@click.pass_context
def merge(ctx):
    '''merge the data in the assigned files column by column (one file per column)'''
    config, output = ctx.obj['config'], ctx.obj['output']
    paths = glob(os.path.join(output, ASSIGN_PATH, '*.feather'))

    output = os.path.join(output, MERGE_PATH)
    _create_dir(output)

    for col in ASSIGNED_COLS:
        filepath = os.path.join(output, f'{col}.feather')
        if not _path_exists(filepath):
            L.debug('Merging column %s', col)
            data = pd.concat((utils.read_feather(f, columns=[col])
                              for f in paths), sort=False, ignore_index=False)

            L.debug('Writing %s', filepath)
            with utils.delete_file_on_exception(filepath):
                utils.write_feather(filepath, data)


@cli.command()
@click.pass_context
def write_sonata(ctx):
    '''Write out the h5 synapse files (for nodes and edges), corresponding to the assignment'''
    config, output = ctx.obj['config'], ctx.obj['output']
    column_path = os.path.join(output, MERGE_PATH)

    output = os.path.join(output, SONATA_PATH)
    _create_dir(output)

    syns = DataFrameWrapper(column_path, config['distance_soma'])
    node_path = os.path.join(output, 'nodes.h5')
    edge_path = os.path.join(output, 'nonparameterized-edges.h5')

    if not _path_exists(node_path):
        node_population = config.get('node_population_name', 'projections')

        L.debug('Writing %s', node_path)
        with utils.delete_file_on_exception(node_path):
            sonata.write_nodes(syns, node_path, node_population, 'virtual', keep_offset=True)

    if not _path_exists(edge_path):
        edge_population = config.get('edge_population_name', 'projections')

        L.debug('Writing %s', edge_path)
        with utils.delete_file_on_exception(edge_path):
            sonata.write_edges(syns, edge_path, edge_population, keep_offset=True)


if __name__ == '__main__':
    cli(obj={})

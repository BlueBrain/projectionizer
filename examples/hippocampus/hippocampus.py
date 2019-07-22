#!/usr/bin/env python
'''
module load brainbuilder
brainbuilder syn2 concat -o O1_ca1_20191017_projections.syn2 SYN2/*.syn2

module load synapsetool
syn-tool order-synapses O0_ca1_20191017.syn2 O0_ca1_20191017_sorted.syn2

module load brainbuilder
brainbuilder sonata from-syn2 -o O0_ca1_20191017_sorted.sonata O0_ca1_20191017_sorted.syn2
'''

from functools import partial
from glob import glob
import logging
import os
import shutil
import sys

import yaml

import click
import numpy as np
import pandas as pd
from projectionizer import hippocampus, utils, version, write_syn2 as syn2, write_sonata as sonata

from bluepy import Circuit

L = logging.getLogger(__name__)

ASSIGN_PATH = 'ASSIGNMENT'
MERGE_PATH = 'MERGE'
SONATA_PATH = 'SONATA'
SYN2_PATH = 'SYN2'
CHUNK_SIZE = 100000000

SEGMENT_START_COLS = ['segment_x1', 'segment_y1', 'segment_z1', ]
SEGMENT_END_COLS = ['segment_x2', 'segment_y2', 'segment_z2', ]
SEGMENT_COLUMNS = sorted(['section_id', 'segment_id', 'segment_length', ] +
                         SEGMENT_START_COLS + SEGMENT_END_COLS +
                         ['tgid']
                         )

ASSIGNED_COLS = ['section_id', 'segment_id', 'tgid', 'synapse_offset', 'x', 'y', 'z', 'sgid']

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

    def __len__(self):
        if self._length is None:
            self._length = len(self[ASSIGNED_COLS[0]])

        return self._length

    def _get_column_path(self, column):
        return os.path.join(self._path, column + '.feather')

    def __getattr__(self, attribute):
        if attribute in ASSIGNED_COLS:
            return pd.read_feather(self._get_column_path(attribute))[attribute]
        return self.__getattribute__(attribute)

    def __getitem__(self, item):
        return self.__getattr__(item)


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

    if not os.path.exists(output):
        os.makedirs(output)

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
    index_path = os.path.dirname(config['circuit_config'])

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
    # NOTE: Could this be the reason for the projections and target file mismatch (0 vs 1-based)?
    # Should first sgid be sgid_start and not sgid_start + 1?
    # e.g., 100000...100100 vs 100001...100101
    if sgid_count == 1:
        syns['sgid'] = sgid_start + 1
    else:
        sgids = np.arange(sgid_start + 1, sgid_start + sgid_count + 1)
        syns['sgid'] = np.random.choice(sgids, size=len(syns), replace=True)

    return syns


def _assign(cells, morph_class, count, samples_path, output, region, sgid_start, sgid_count):
    output = os.path.join(output, ASSIGN_PATH)
    if not os.path.exists(output):
        os.makedirs(output)

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
        if os.path.exists(path):
            L.info('Already have %s, skipping', path)
            continue

        with utils.delete_file_on_exception(path):
            syns = _pick_synapse_locations(segs, dist, size)
            syns = _assign_sgid(syns, sgid_start, sgid_count)
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
    _assign(cells, 'INT', int_count, sample_path, output, region, sgid_start, sgid_count)
    _assign(cells, 'PYR', pyr_count, sample_path, output, region, sgid_start, sgid_count)


@cli.command()
@click.pass_context
def merge(ctx):
    '''merge the data in the assigned files column by column (one file per column)'''
    output = ctx.obj['output']
    paths = glob(os.path.join(output, ASSIGN_PATH, '*.feather'))

    output = os.path.join(output, MERGE_PATH)
    if not os.path.exists(output):
        os.makedirs(output)

    for col in ASSIGNED_COLS:
        filepath = os.path.join(output, f'{col}.feather')
        if os.path.exists(filepath):
            L.info('Already have %s, skipping', filepath)
            continue

        L.debug('Merging column %s', col)
        data = pd.concat((utils.read_feather(f, columns=[col])
                          for f in paths), sort=False, ignore_index=False)

        L.debug('Writing %s', filepath)
        with utils.delete_file_on_exception(filepath):
            utils.write_feather(filepath, data)


def _write_syn2(output_path, source_path, cells, synapse_parameters):
    L.debug('Loading: %s', source_path)
    syns = (utils.read_feather(source_path)
            .join(cells[['morph_class', 'mtype', ]], on='tgid')
            )

    syns['delay'] = 1  # default delay, as specified by Armando

    syns['synapse_type_name'] = (syns.morph_class == 'INT').astype(np.uint8)

    synapse_data = {'type_0': {'physiology': synapse_parameters['PYR']},
                    'type_1': {'physiology': synapse_parameters['INT']},
                    }
    for i, name in enumerate(synapse_parameters):
        if name in ('INT', 'PYR', ):
            continue
        params = synapse_parameters[name]
        assert 'mtypes' in params, 'Must list the mtypes for special parameters'
        mask = np.isin(syns.mtype.values, list(params['mtypes']))
        syns['synapse_type_name'].values[mask] = i + 2
        synapse_data['type_%d' % (i + 2)] = {'physiology': params}

    synapse_data_creator = partial(syn2.create_synapse_data, synapse_data=synapse_data)
    syn2.write_synapses(syns, output_path, synapse_data_creator)


@cli.command()
@click.pass_context
def write_sonata(ctx):
    '''Write out the h5 synapse files (for nodes and edges), corresponding to the assignment'''
    config, output = ctx.obj['config'], ctx.obj['output']
    column_path = os.path.join(output, MERGE_PATH)

    output = os.path.join(output, SONATA_PATH)
    if not os.path.exists(output):
        os.makedirs(output)

    node_path = os.path.join(output, 'nodes.h5')
    edge_path = os.path.join(output, 'nonparameterized-edges.h5')

    syns = DataFrameWrapper(column_path, config['distance_soma'])

    node_population = config.get('node_population_name', 'projections')
    edge_population = config.get('edge_population_name', 'projections')

    L.debug('Writing %s', node_path)
    sonata.write_nodes(syns, node_path, node_population, 'virtual', keep_offset=True)
    L.debug('Writing %s', edge_path)
    sonata.write_edges(syns, edge_path, edge_population, keep_offset=True)


@cli.command()
@click.option('--region', required=True)
@click.pass_context
def write_syn2(ctx, region):
    '''Write out the syn2 synapse file, corresponding to the assignment'''
    config, output = ctx.obj['config'], ctx.obj['output']

    cells = Circuit(config['circuit_config']).cells.get()

    paths = glob(os.path.join(output, ASSIGN_PATH, region + '*.feather'))

    output = os.path.join(output, SYN2_PATH)
    if not os.path.exists(output):
        os.makedirs(output)

    synapse_parameters = config['synapse_parameters']
    for source_path in paths:
        output_path = os.path.join(output, os.path.basename(source_path)[:-8] + '.syn2')
        if os.path.exists(output_path):
            L.info('Already have %s, skipping', output_path)
            continue

        _write_syn2(output_path, source_path, cells, synapse_parameters)


@cli.command()
@click.option('--prefix', required=True)
@click.pass_context
def create_targets(ctx, prefix):
    '''create projection blocks suitable for the CircuitConfig/BlueConfig'''
    output = ctx.obj['output']

    blocks = ''
    for path in sorted(glob(os.path.join(output, SYN2_PATH, '*.syn2'))):
        path = os.path.abspath(path)
        source = os.path.basename(path).split('.')[0]
        name = '%s_%s' % (prefix, source)
        blocks += PROJECTION_TEMPLATE.format(name=name, path=path, source=source)

    output_path = os.path.join(output, 'targets')
    with open(output_path, 'w') as fd:
        fd.write(blocks)
    click.echo('Wrote targets to: %s' % output_path)


if __name__ == '__main__':
    cli(obj={})
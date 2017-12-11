'''
Step 2: pruning:  Take the (tgid, sgid) connections, and remove groups to achieve desired
distribution shape
'''

import logging

import luigi
import numpy as np
import pandas as pd
from bluepy.v2.circuit import Circuit
from scipy.stats import norm  # pylint: disable=no-name-in-module

from projectionizer.step_0_sample import SampleChunkTask
from projectionizer.step_1_assign import FiberAssignementTask
from projectionizer.utils import CommonParams, _write_feather, load, load_all

L = logging.getLogger(__name__)


class GroupByConnectionTask(CommonParams):
    """Returns a DataFrame containing the number of synapses per connection for each tuple mtype,
    neuron ID, fiber ID: (mtype, tgid, sgid)"""
    chunk_num = luigi.IntParameter()

    def requires(self):
        return self.clone(SampleChunkTask), self.clone(FiberAssignementTask)

    def run(self):
        synapses, sgids = load_all(self.input())

        synapses.rename(columns={'gid': 'tgid'}, inplace=True)
        sgids.rename(columns={'0': 'sgid'}, inplace=True)
        mtypes = Circuit(self.circuit_config).cells.get(properties='mtype')
        assert len(synapses) == len(sgids), \
            'len(synapses): {} != len(sgids): {}'.format(len(synapses), len(sgids))
        tgid_sgid_mtype = synapses[['tgid']].join(sgids).join(mtypes, on='tgid')
        res = tgid_sgid_mtype[['mtype', 'tgid', 'sgid']]
        _write_feather(self.output().path, res)

    def output(self):
        name = '{}/group-by-connection_{}.feather'.format(self.folder, self.chunk_num)
        return luigi.local_target.LocalTarget(name)


class ReduceGroupByConnectionTask(CommonParams):
    """Merge the group-by of all chunks"""

    def requires(self):
        return [self.clone(GroupByConnectionTask, chunk_num=i) for i in range(self.n_total_chunks)]

    def run(self):
        dfs = load_all(self.input())
        fat = pd.concat(dfs, ignore_index=True)
        res = fat.groupby(['mtype', 'sgid', 'tgid']).size().reset_index()
        _write_feather(self.output().path, res)

    def output(self):
        name = '{}/reduce-group-by-connection.feather'.format(self.folder)
        return luigi.local_target.LocalTarget(name)


def find_cutoff_mean_per_mtype(value_count, synaptical_fraction):
    '''the cutoff means should be proportional for all mytpes, reuse the method from
    the original projectionizer

    TODO: describe it better
    '''
    n_synapse_per_bin = np.array([value * count for value, count in value_count.iteritems()],
                                 dtype=float)
    x = np.cumsum(n_synapse_per_bin) / np.sum(n_synapse_per_bin)
    return np.interp(synaptical_fraction, xp=x, fp=value_count.index)


class CutoffMeans(CommonParams):
    '''For each mtype, find its unique cutoff_mean

    Args:
        synapses(DataFrame): must have columns 'mtype', 'tgid', 'sgid'
        synaptical_fraction(float): the wanted fraction of synapse below the cutoff

    Returns:
        dict(mtype -> cutoff_mean) where cutoff_mean is the value for which the cumulated number
        of synapses belonging to all connection having at maximum "cutoff_mean" synapses represents
        "synaptical_fraction" of the total number of synapse

    From thalamocortical_ps_s2f.py
        compute cutoff by inverse interpolation of target fraction on cumulative syncount
        distribution approximation: assumes hard cutoff, i.e. does not account for moments beyond
        mean.  Should be OK if dist is fairly symmetrical.
    '''

    def requires(self):
        return self.clone(ReduceGroupByConnectionTask)

    def run(self):
        # pylint thinks load() isn't returning a DataFrame
        # pylint: disable=maybe-no-member
        mtype_sgid_tgid = load(self.input().path)
        grouped_by_mtypes = mtype_sgid_tgid.groupby('mtype')  # pylint:disable=maybe-no-member
        not_emtpy_mtypes = [mtype_df
                            for mtype_df in grouped_by_mtypes
                            if not mtype_df[1].empty]
        mtypes, dfs = zip(*not_emtpy_mtypes)

        fraction_to_remove = 1 - 1 / self.oversampling
        print('fraction_to_remove: {}'.format(fraction_to_remove))
        cutoffs = [find_cutoff_mean_per_mtype(df.loc[:, '0'].value_counts(sort=False).sort_index(),
                                              fraction_to_remove)
                   for df in dfs]
        res = pd.DataFrame({'mtype': pd.Series(mtypes, dtype='category'),
                            'cutoff': cutoffs,
                            })
        _write_feather(self.output().path, res)

    def output(self):
        name = '{}/cutoff-means.feather'.format(self.folder)
        return luigi.local_target.LocalTarget(name)


class ChooseConnectionsToKeep(CommonParams):
    '''
    Args:
        cutoff_var(float):
    '''
    cutoff_var = luigi.FloatParameter()

    def requires(self):
        return self.clone(CutoffMeans), self.clone(ReduceGroupByConnectionTask)

    def run(self):
        '''Based on the frequency of mtypes, and the synapses/connection frequency,
        probabilistically remove *connections* (ie: groups of synapses in a (sgid, tgid) pair
        '''
        # pylint thinks load_all() isn't returning a DataFrame
        # pylint: disable=maybe-no-member
        cutoff_means, mtype_sgid_tgid = load_all(self.input())

        df = mtype_sgid_tgid.merge(cutoff_means, how='left', on='mtype')
        df['random'] = np.random.random(size=len(df))
        df['proba'] = norm.cdf(df.loc[:, '0'], df['cutoff'], self.cutoff_var)
        df['kept'] = df['random'] < df['proba']
        _write_feather(self.output().path, df)

    def output(self):
        name = '{}/choose-connections-to-keep.feather'.format(self.folder)
        return luigi.local_target.LocalTarget(name)


class PruneChunk(CommonParams):
    '''
    Args:
        chunk_num(float):
    '''
    chunk_num = luigi.IntParameter()

    def requires(self):
        return (self.clone(ChooseConnectionsToKeep),
                self.clone(SampleChunkTask),
                self.clone(FiberAssignementTask),
                )

    def run(self):
        # pylint thinks load_all() isn't returning a DataFrame
        # pylint: disable=maybe-no-member
        connections, sample, sgids = load_all(self.input())
        sample.rename(columns={'gid': 'tgid'}, inplace=True)
        is_kept = connections[['sgid', 'tgid', 'kept']]
        assert len(sgids) == len(sample)
        fat = sample.join(sgids).merge(is_kept, how='left', on=['tgid', 'sgid'])
        pruned = fat[fat['kept']]
        _write_feather(self.output().path, pruned.drop('kept', axis=1))

    def output(self):
        name = '{}/pruned_chunk_{}.feather'.format(self.folder, self.chunk_num)
        return luigi.local_target.LocalTarget(name)


class ReducePrune(CommonParams):
    '''Load all pruned chunks, and concat them together
    '''

    def requires(self):
        return [self.clone(PruneChunk, chunk_num=i) for i in range(self.n_total_chunks)]

    def run(self):
        synapses = pd.concat(load_all(self.input())).rename(
            columns={'Segment.ID': 'segment_id', 'Section.ID': 'section_id'})

        synapses['sgid'] += self.sgid_offset
        synapses["syn_ids"] = np.zeros_like(synapses["tgid"], dtype=np.int32)
        synapses.set_index(['tgid', 'sgid'], inplace=True)
        synapses.sort_index(inplace=True)

        syn_ids = [np.arange(synapses.loc[(tgid, ), :].shape[0], dtype=np.int32)
                   for tgid in synapses.index.levels[0]]
        synapses["syn_ids"] = np.concatenate(syn_ids)
        # TODO: Set real values for location and neurite_type
        synapses['location'] = 1
        synapses['neurite_type'] = 1
        synapses.reset_index(inplace=True)
        _write_feather(self.output().path, synapses)

    def output(self):
        name = '{}/pruned_reshaped.feather'.format(self.folder)
        return luigi.local_target.LocalTarget(name)

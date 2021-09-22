"""
Step 2: pruning:  Take the (tgid, sgid) connections, and remove groups to achieve desired
distribution shape
"""

import logging

import luigi
import numpy as np
import pandas as pd
from bluepy import Circuit
from scipy.stats import norm  # pylint: disable=no-name-in-module

from projectionizer import luigi_utils, step_0_sample, step_1_assign, straight_fibers
from projectionizer import synapses as syns
from projectionizer.utils import load, load_all, write_feather

L = logging.getLogger(__name__)


class GroupByConnection(luigi_utils.FeatherTask):
    """Returns a DataFrame containing the number of synapses per connection for each tuple mtype,
    neuron ID, fiber ID: (mtype, tgid, sgid)"""

    chunk_num = luigi.IntParameter()

    def requires(self):  # pragma: no cover
        return (
            self.clone(step_0_sample.SampleChunk),
            self.clone(step_1_assign.FiberAssignment),
        )

    def run(self):  # pragma: no cover
        synapses, sgids = load_all(self.input())

        synapses.rename(columns={"gid": "tgid"}, inplace=True)
        mtypes = Circuit(self.circuit_config).cells.get(properties="mtype")
        assert len(synapses) == len(sgids), "len(synapses): {} != len(sgids): {}".format(
            len(synapses), len(sgids)
        )
        tgid_sgid_mtype = synapses[["tgid"]].join(sgids).join(mtypes, on="tgid")
        res = tgid_sgid_mtype[["mtype", "tgid", "sgid"]]
        write_feather(self.output().path, res)


class ReduceGroupByConnection(luigi_utils.FeatherTask):
    """Merge the group-by of all chunks"""

    def requires(self):  # pragma: no cover
        return [self.clone(GroupByConnection, chunk_num=i) for i in range(self.n_total_chunks)]

    def run(self):  # pragma: no cover
        dfs = load_all(self.input())
        res = (
            pd.concat(dfs, ignore_index=True)
            .groupby(["mtype", "sgid", "tgid"], observed=True)
            .size()
            .reset_index(name="connection_size")
        )
        write_feather(self.output().path, res)


def find_cutoff_mean_per_mtype(value_count, synaptical_fraction):
    """the cutoff means should be proportional for all mytpes

    Args:
        value_count(pd.Series): histogram: index is syns/connection,
        value is number pathways with the index syns/connection
        synaptical_fraction(float): fraction to be removed

    The method comes from the original projectionizer
    """
    n_synapse_per_bin = value_count.index.values.astype(float) * value_count.values
    x = np.cumsum(n_synapse_per_bin) / np.sum(n_synapse_per_bin)
    return np.interp(synaptical_fraction, xp=x, fp=value_count.index)


def calculate_cutoff_means(mtype_sgid_tgid, oversampling):
    """for all the mtypes, calculate the cutoff mean"""
    not_empty_mtypes = [
        mtype_df for mtype_df in mtype_sgid_tgid.groupby("mtype") if not mtype_df[1].empty
    ]
    mtypes, dfs = zip(*not_empty_mtypes)

    fraction_to_remove = 1.0 - 1.0 / oversampling

    cutoffs = []
    for df in dfs:
        value_count = df.connection_size.value_counts(sort=False).sort_index()
        cutoff = find_cutoff_mean_per_mtype(value_count, fraction_to_remove)
        cutoffs.append(cutoff)

    res = pd.DataFrame({"mtype": pd.Series(mtypes, dtype="category"), "cutoff": cutoffs})
    return res


class CutoffMeans(luigi_utils.FeatherTask):
    """For each mtype, find its unique cutoff_mean

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
    """

    def requires(self):  # pragma: no cover
        return self.clone(ReduceGroupByConnection)

    def run(self):  # pragma: no cover
        # pylint thinks load() isn't returning a DataFrame
        # pylint: disable=maybe-no-member
        mtype_sgid_tgid = load(self.input().path)
        res = calculate_cutoff_means(mtype_sgid_tgid, self.oversampling)
        write_feather(self.output().path, res)


class ChooseConnectionsToKeep(luigi_utils.FeatherTask):
    """
    Args:
        cutoff_var(float):
    """

    cutoff_var = luigi.FloatParameter()

    def requires(self):  # pragma: no cover
        return self.clone(CutoffMeans), self.clone(ReduceGroupByConnection)

    def run(self):  # pragma: no cover
        """Based on the frequency of mtypes, and the synapses/connection frequency,
        probabilistically remove *connections* (ie: groups of synapses in a (sgid, tgid) pair
        """
        # pylint thinks load_all() isn't returning a DataFrame
        # pylint: disable=maybe-no-member
        cutoff_means, mtype_sgid_tgid = load_all(self.input())

        df = mtype_sgid_tgid.merge(cutoff_means, how="left", on="mtype")
        if self.oversampling > 1.0:
            df["random"] = np.random.random(size=len(df))
            df["proba"] = norm.cdf(df.connection_size, df["cutoff"], self.cutoff_var)
            df["kept"] = df["random"] < df["proba"]
        else:
            df["kept"] = True
        write_feather(self.output().path, df)


class PruneChunk(luigi_utils.FeatherTask):
    """Write out connections to keep for a subset of the samples (ie: chunk)

    Args:
        chunk_num(int): which chunk
        additive_path_distance(float): distance added to sgid_path_distance,
        can be used to add delay to cope with neurodamus requiring a minimum
        delay > dt

    Note: this also assigns the 'sgid_path_distance'
    """

    chunk_num = luigi.IntParameter()
    additive_path_distance = luigi.FloatParameter(default=0.0)

    def requires(self):  # pragma: no cover
        return (
            self.clone(task)
            for task in [
                ChooseConnectionsToKeep,
                step_0_sample.SampleChunk,
                step_1_assign.FiberAssignment,
                step_1_assign.VirtualFibersNoOffset,
            ]
        )

    def run(self):  # pragma: no cover
        # pylint thinks load_all() isn't returning a DataFrame
        # pylint: disable=maybe-no-member
        connections, sample, sgids, fibers = load_all(self.input())
        sample.rename(columns={"gid": "tgid"}, inplace=True)

        is_kept = connections[["sgid", "tgid", "kept"]]
        assert len(sgids) == len(sample)

        fat = sample.join(sgids).merge(is_kept, how="left", on=["tgid", "sgid"])
        pruned_no_apron = (
            pd.merge(
                fat[fat["kept"]], fibers[~fibers.apron][["apron"]], left_on="sgid", right_index=True
            )
            .drop(["kept", "apron"], axis=1)
            .reset_index(drop=True)
        )

        distance = straight_fibers.calc_pathlength_to_fiber_start(
            pruned_no_apron[list("xyz")].values,
            fibers[list("xyzuvw")].iloc[pruned_no_apron["sgid"]].values,
        )
        pruned_no_apron["sgid_path_distance"] = distance + self.additive_path_distance

        write_feather(self.output().path, pruned_no_apron)


class ReducePrune(luigi_utils.FeatherTask):
    """Load all pruned chunks, and concat them together"""

    def requires(self):  # pragma: no cover
        return [self.clone(PruneChunk, chunk_num=i) for i in range(self.n_total_chunks)]

    def run(self):  # pragma: no cover
        synapses = pd.concat(load_all(self.input())).rename(
            columns={"Segment.ID": "segment_id", "Section.ID": "section_id"}
        )

        # TODO: Set real values for location and neurite_type
        synapses["location"] = 1
        synapses["neurite_type"] = 1
        synapses["sgid"] += self.sgid_offset

        syns.organize_indices(synapses)
        write_feather(self.output().path, synapses)

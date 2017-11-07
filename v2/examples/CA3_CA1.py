import luigi
import numpy as np
import pandas as pd
import yaml
from bluepy.v2.circuit import Circuit
from bluepy.v2.enums import Cell, Section, Segment
from luigi import FloatParameter, Parameter
from luigi.local_target import LocalTarget
from scipy.spatial.distance import cdist
from scipy.stats import norm
from voxcell import RegionMap, VoxelData

from projectionizer.utils import (CommonParams, _write_feather, choice,
                                  cloned_tasks, load, load_all,
                                  normalize_probability)


class PreGIDs(CommonParams):
    def run(self):
        res = pd.DataFrame({'sgid': np.arange(10, 15)})
        _write_feather(self.output().path, res)

    def output(self):
        return LocalTarget('{}/pre_gids.feather'.format(self.folder))


class PostGIDs(CommonParams):
    def run(self):
        res = pd.DataFrame({'tgid': np.arange(20, 30)})
        _write_feather(self.output().path, res)

    def output(self):
        return LocalTarget('{}/post_gids.feather'.format(self.folder))


def non_nil_poisson_distrib(poisson_lambda, size):
    '''Draw non-zero numbers from a poisson distribution'''
    proba_zero = np.exp(-poisson_lambda)
    res = np.random.poisson(poisson_lambda, size=int(size / (1 - 2 * proba_zero)))
    res = res[res > 0]
    if len(res) >= size:
        return res[:size]
    return np.append(res, non_nil_poisson_distrib(poisson_lambda, size - len(res)))


class ChoosePostSynapticGIDs(CommonParams):
    sigma = FloatParameter(default=100)
    poisson_lambda = FloatParameter(default=1.5)

    def requires(self):
        return cloned_tasks(self, [PreGIDs, PostGIDs])

    def run(self):
        """ Choose some gids from `post_gids` to be connected to `pre_gid`. """
        pre_gids, post_gids = load_all(self.input())
        cells = Circuit(self.circuit_config).cells()
        pos = [Cell.X, Cell.Y, Cell.Z]
        dist = cdist(cells.loc[pre_gids.sgid][pos], cells.loc[post_gids.tgid][pos])
        prob = norm.pdf(dist, 0, self.sigma)
        tgid_idx = choice(prob)
        assigned_tgids = post_gids.iloc[tgid_idx].reset_index(drop=True)
        res = pre_gids.join(assigned_tgids)
        res['synapse_counts'] = non_nil_poisson_distrib(self.poisson_lambda, len(res))
        _write_feather(self.output().path, res)

    def output(self):
        return LocalTarget('{}/connections.feather'.format(self.folder))


def get_segment_infos(gid, circuit, atlas, layer_map):
    '''Returns: a DataFrame with data about each segment for the given neurom.
    DataFrame columns are [Segment.LENGTH, Segment.REGION,
                           Section.NeuriteType, Section.BRANCH_ORDER]'''
    segment_features = circuit.morph.segment_features(
        gid, [Segment.LENGTH, Segment.REGION], atlas=atlas
    )
    segment_features[Segment.REGION] = segment_features[Segment.REGION].map(layer_map)
    segment_features.dropna(inplace=True)

    section_features = circuit.morph.section_features(
        gid, [Section.NEURITE_TYPE, Section.BRANCH_ORDER]
    )
    post_segment_info = segment_features.join(section_features)
    post_segment_info.reset_index(inplace=True)
    return post_segment_info


def choose_segments(n_synapses, segment_info, depth_profile):
    """Choose segment"""
    prob = np.zeros(len(segment_info))
    for r_name, r_prob in yaml.load(depth_profile).items():
        mask = (segment_info[Segment.REGION] == r_name).values
        if mask.any():
            prob[mask] = r_prob / np.count_nonzero(mask)
    prob = normalize_probability(prob)
    idx = np.random.choice(segment_info.index, n_synapses, p=prob, replace=True)
    return segment_info.iloc[idx].reset_index(drop=True)


class BuildConnectivity(CommonParams):
    atlas = Parameter()
    region_map = Parameter()
    depth_profile = Parameter()

    def requires(self):
        return self.clone(ChoosePostSynapticGIDs)

    def run(self):
        """
            Generate synapses between `pre_gid` and `post_gid`.

            Return the result as pandas.DataFrame with synapse parameters.
        """
        circuit = Circuit(self.circuit_config)
        atlas = VoxelData.load_nrrd(self.atlas)
        region_map = RegionMap.from_json(self.region_map)

        connections = load(self.input().path)
        layers = ['SLM', 'SO', 'SP', 'SR']
        layer_map = {_id: layer for layer in layers for _id in region_map.ids(layer)}

        all_segments = list()
        for post_gid, df in connections.groupby('tgid'):
            post_segment_info = get_segment_infos(post_gid, circuit, atlas, layer_map)

            n_synapses = sum(df.synapse_counts)
            print('n_synapses: {}'.format(n_synapses))
            segments = choose_segments(n_synapses, post_segment_info, self.depth_profile)
            dup = np.repeat(df.sgid, df.synapse_counts)
            segments['sgid'] = dup.values
            print(type(segments))
            print(segments)
            print(post_gid)
            segments['tgid'] = post_gid

            # TODO: remove this line
            segments.drop(Section.NEURITE_TYPE, axis=1, inplace=True)

            all_segments.append(segments)

        res = pd.concat(all_segments).convert_objects()
        print(res)
        _write_feather(self.output().path, res)

    def output(self):
        return LocalTarget('{}/synapses.feather'.format(self.folder))

"""Step 0; sample segments from circuit to be used as potential synapses
"""
import json

import numpy as np
import pandas as pd
from bluepy import Circuit
from luigi import BoolParameter, FloatParameter, IntParameter, ListParameter

from projectionizer.luigi_utils import FeatherTask, JsonTask, NrrdTask
from projectionizer.sscx import (
    recipe_to_relative_height_and_density,
    recipe_to_relative_heights_per_layer,
)
from projectionizer.sscx_hex import get_mask_bounding_box
from projectionizer.synapses import build_synapses_default, pick_synapses
from projectionizer.utils import load, load_all, mask_by_region, write_feather


class VoxelSynapseCount(NrrdTask):  # pragma: no cover
    """Generate the VoxelData containing the number
    of segment to be sampled in each voxel"""

    oversampling = FloatParameter()

    def requires(self):
        return self.clone(Height), self.clone(SynapseDensity)

    def run(self):
        height, synapse_density = load_all(self.input())
        res = build_synapses_default(height, synapse_density, self.oversampling)
        res.save_nrrd(self.output().path)


class Height(NrrdTask):  # pragma: no cover
    """return a VoxelData instance w/ all the layer-wise relative heights for given region_name

    distance is defined as from the voxel to the bottom of L6, voxels
    outside of region_name are set to nan

    Args:
        region_name(str): name to look up in atlas
        path(str): path to where nrrd files are, must include 'brain_regions.nrrd'
    """

    def run(self):
        regions = self.get_regions()
        atlas = Circuit(self.circuit_config).atlas
        mask = mask_by_region(regions, atlas)
        distance = atlas.load_data("[PH]y")

        if len(self.hex_apron_bounding_box):
            mask = get_mask_bounding_box(distance, mask, self.hex_apron_bounding_box)

        distance.raw[np.invert(mask)] = np.nan
        distance = recipe_to_relative_heights_per_layer(distance, atlas, self.layers)

        distance.save_nrrd(self.output().path)


class SampleChunk(FeatherTask):
    """Split the big sample into chunks"""

    chunk_num = IntParameter()

    def requires(self):
        return self.clone(FullSample)

    def run(self):
        # pylint thinks load() isn't returning a DataFrame
        # pylint: disable=maybe-no-member
        full_sample = load(self.input().path)
        chunk_size = int((len(full_sample) / self.n_total_chunks) + 1)
        start, end = self.chunk_num * chunk_size, (self.chunk_num + 1) * chunk_size
        chunk_df = full_sample.iloc[start:end]
        write_feather(self.output().path, chunk_df)


class FullSample(FeatherTask):  # pragma: no cover
    """Sample segments from circuit"""

    from_chunks = BoolParameter(default=False)

    def requires(self):
        if self.from_chunks:
            return [self.clone(SampleChunk, chunk_num=i) for i in range(self.n_total_chunks)]
        return self.clone(VoxelSynapseCount)

    def run(self):
        if self.from_chunks:
            # pylint: disable=maybe-no-member
            chunks = load(self.input().path)
            write_feather(self.output().path, pd.concat(chunks))
        else:
            # pylint: disable=maybe-no-member
            voxels = load(self.input().path)
            synapses = pick_synapses(self.segment_index_path, voxels)

            synapses.rename(columns={"gid": "tgid"}, inplace=True)
            write_feather(self.output().path, synapses)


class SynapseDensity(JsonTask):  # pragma: no cover
    """Return the synaptic density profile"""

    density_params = ListParameter()

    def run(self):
        res = [
            recipe_to_relative_height_and_density(
                self.layers,
                data["low_layer"],
                data["low_fraction"],
                data["high_layer"],
                data["high_fraction"],
                data["density_profile"],
            )
            for data in self.density_params  # pylint: disable=not-an-iterable
        ]
        with self.output().open("w") as outfile:
            json.dump(res, outfile)

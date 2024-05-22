"""Step 0; sample segments from circuit to be used as potential synapses
"""

import json

import numpy as np
import pandas as pd
import spatial_index.experimental
from bluepy import Circuit
from luigi import FloatParameter, IntParameter, ListParameter

from projectionizer.luigi_utils import FeatherTask, JsonTask, NrrdTask
from projectionizer.sscx import (
    get_mask_bounding_box,
    recipe_to_relative_height_and_density,
    recipe_to_relative_heights_per_layer,
)
from projectionizer.synapses import build_synapses_default, pick_synapses
from projectionizer.utils import XYZ, load, load_all, mask_by_region, write_feather


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


class VoxelOrder(FeatherTask):
    """Reorganize voxel order optimally for SpatialIndex."""

    def requires(self):  # pragma: no cover
        return self.clone(VoxelSynapseCount)

    def run(self):
        voxels = load(self.input().path)
        idx = np.transpose(np.nonzero(voxels.raw))
        xyzs = voxels.indices_to_positions(idx)
        order = spatial_index.experimental.space_filling_order(xyzs)
        voxel_indices = pd.DataFrame(idx[order], columns=XYZ)

        write_feather(self.output().path, voxel_indices)


class SampleChunk(FeatherTask):
    """Split the big sample into chunks"""

    chunk_num = IntParameter()

    def requires(self):  # pragma: no cover
        return self.clone(VoxelOrder), self.clone(VoxelSynapseCount)

    def _get_voxel_indices(self):
        """Helper function to get voxels' indices."""
        voxel_order = load(self.input()[0].path)
        chunk_size = int((len(voxel_order) / self.n_total_chunks) + 1)
        start = self.chunk_num * chunk_size
        end = start + chunk_size
        return voxel_order[XYZ].to_numpy()[start:end]

    def _get_xyzs_count(self):
        """Helper function to get voxels' positions and synapse counts."""
        voxels = load(self.input()[1].path)
        indices = self._get_voxel_indices()
        min_xyzs = voxels.indices_to_positions(indices)
        max_xyzs = min_xyzs + voxels.voxel_dimensions
        counts = voxels.raw[tuple(indices.T)]

        return np.hstack((min_xyzs, max_xyzs, counts[:, np.newaxis]))

    def run(self):  # pragma: no cover
        xyzs_count = self._get_xyzs_count()

        synapses = pick_synapses(self.segment_index_path, xyzs_count)
        synapses.rename(columns={"gid": "tgid"}, inplace=True)

        write_feather(self.output().path, synapses)


class FullSample(FeatherTask):  # pragma: no cover
    """Sample segments from circuit"""

    def requires(self):
        return [self.clone(SampleChunk, chunk_num=i) for i in range(self.n_total_chunks)]

    def run(self):
        # pylint: disable=maybe-no-member
        chunks = load_all(self.input())
        write_feather(self.output().path, pd.concat(chunks))


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

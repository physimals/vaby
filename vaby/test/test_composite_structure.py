"""
Tests of data structure classes
"""
import numpy as np
import nibabel as nib

import pytest

from vaby.data import CorticalSurface, Volume, CompositeStructure
from vaby.utils import NP_DTYPE

def test_surf_plus_volume():
    SHAPE = [10, 10, 10, 5]
    VOXEL_SIZES = [2.5, 3.5, 4.5]
    vol_data = np.random.normal(size=SHAPE).astype(NP_DTYPE)
    vol = Volume(vol_data, voxel_sizes=VOXEL_SIZES)

    surf = CorticalSurface(
        "gii/HCA6002236_V1_MR.L.white.native.surf.gii",
        "gii/HCA6002236_V1_MR.L.pial.native.surf.gii",
        name="L",
        projector="vaby_proj.h5"
    )

    struc = CompositeStructure([surf, vol])
    model2data, data2model = struc.get_projection(vol)
    struc_data = data2model(vol.srcdata.flat)
    assert struc_data.shape[0] == struc.size

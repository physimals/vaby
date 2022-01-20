"""
Tests of data structure classes
"""
import numpy as np
import nibabel as nib

import pytest

from vaby.data import CorticalSurface, Volume
from vaby.utils import NP_DTYPE

def test_ctor_only():
    surf = CorticalSurface("gii/HCA6002236_V1_MR.L.white.native.surf.gii", "gii/HCA6002236_V1_MR.L.pial.native.surf.gii", name="L")

@pytest.mark.skip(reason="slow")
def test_create_projector():
    surf = CorticalSurface("gii/HCA6002236_V1_MR.L.white.native.surf.gii", "gii/HCA6002236_V1_MR.L.pial.native.surf.gii", name="L")
 
    SHAPE = [10, 10, 10]
    VOXEL_SIZES = [2.5, 3.5, 4.5]
    vol_data = np.random.normal(size=SHAPE)
    vol = Volume(vol_data, voxel_sizes=VOXEL_SIZES)
    surf.get_projection(vol)

def test_load_projector():
    surf = CorticalSurface(
        "gii/HCA6002236_V1_MR.L.white.native.surf.gii",
        "gii/HCA6002236_V1_MR.L.pial.native.surf.gii",
        name="L",
        projector="vaby_proj.h5"
    )

    SHAPE = [10, 10, 10, 5]
    VOXEL_SIZES = [2.5, 3.5, 4.5]
    vol_data = np.random.normal(size=SHAPE).astype(NP_DTYPE)
    vol = Volume(vol_data, voxel_sizes=VOXEL_SIZES)

    surf2vol, vol2surf = surf.get_projection(vol)
    surf_data = vol2surf(vol.srcdata.flat)
    assert surf_data.shape[0] == surf.size


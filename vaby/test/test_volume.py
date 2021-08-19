"""
Tests of data structure classes
"""
import numpy as np
import nibabel as nib

from vaby.data import get_data_structure, Volume, Surface

def test_volume_3d():
    SHAPE = [10, 10, 10]
    vol_data = np.random.normal(size=SHAPE)
    vol = Volume(vol_data)
    assert(list(vol.shape) == SHAPE)
    assert(vol.size == 1000)
    assert(vol.n_tpts == 1)
    assert(np.allclose(vol.mask_vol, np.ones(SHAPE, dtype=int)))
    assert(np.allclose(vol.data_flat[..., 0], vol_data.flatten()))
    assert(isinstance(vol.nii, nib.Nifti1Image))
    assert(np.allclose(vol.voxel_sizes, [1.0, 1.0, 1.0]))

def test_volume_3d_masked():
    SHAPE = [10, 10, 10]
    vol_data = np.random.normal(size=SHAPE)
    mask = np.random.randint(0, 2, size=SHAPE)
    vol = Volume(vol_data, mask=mask)
    assert(list(vol.shape) == SHAPE)
    assert(vol.size == np.count_nonzero(mask))
    assert(vol.n_tpts == 1)
    assert(np.allclose(vol.mask_vol, mask))
    assert(np.allclose(vol.data_flat[..., 0], vol_data[mask == 1]))
    assert(isinstance(vol.nii, nib.Nifti1Image))
    assert(np.allclose(vol.voxel_sizes, [1.0, 1.0, 1.0]))

def test_volume_3d_vox_sizes():
    SHAPE = [10, 10, 10]
    VOXEL_SIZES = [2.5, 3.5, 4.5]
    vol_data = np.random.normal(size=SHAPE)
    vol = Volume(vol_data, voxel_sizes=VOXEL_SIZES)
    assert(np.allclose(vol.voxel_sizes, VOXEL_SIZES))

def test_volume_3d_nii():
    SHAPE = [10, 10, 10]
    VOXEL_SIZES = [2.5, 3.5, 4.5]
    vol_data = np.random.normal(size=SHAPE)
    affine = np.zeros((4, 4), dtype=np.float32)
    for d in range(3): affine[d, d] = VOXEL_SIZES[d]
    nii = nib.Nifti1Image(vol_data, affine=affine)
    vol = Volume(nii=nii)
    assert(list(vol.shape) == SHAPE)
    assert(vol.size == 1000)
    assert(vol.n_tpts == 1)
    assert(np.allclose(vol.mask_vol, np.ones(SHAPE, dtype=int)))
    assert(np.allclose(vol.data_flat[..., 0], vol_data.flatten()))
    assert(vol.nii == nii)
    assert(np.allclose(vol.voxel_sizes, VOXEL_SIZES))

def test_volume_3d_nii_masked():
    SHAPE = [10, 10, 10]
    VOXEL_SIZES = [2.5, 3.5, 4.5]
    vol_data = np.random.normal(size=SHAPE)
    mask = np.random.randint(0, 2, size=SHAPE)
    affine = np.zeros((4, 4), dtype=np.float32)
    for d in range(3): affine[d, d] = VOXEL_SIZES[d]
    nii = nib.Nifti1Image(vol_data, affine=affine)
    vol = Volume(nii=nii, mask=mask)
    assert(list(vol.shape) == SHAPE)
    assert(vol.size == np.count_nonzero(mask))
    assert(vol.n_tpts == 1)
    assert(np.allclose(vol.mask_vol, mask))
    assert(np.allclose(vol.data_flat[..., 0], vol_data[mask == 1]))
    assert(vol.nii == nii)
    assert(np.allclose(vol.voxel_sizes, VOXEL_SIZES))

def _categorize(vid, shape):
    faceid = int(vid / (shape[0]*shape[1]))
    vid = vid % (shape[0]*shape[1])
    rowid = int(vid / shape[0])
    colid = vid % shape[0]
    edge_cats = [faceid in (0, shape[2]-1), rowid in (0, shape[1]-1), colid in (0, shape[0]-1)]
    print(vid, faceid, rowid, colid, edge_cats)
    corner = sum(edge_cats) == 3
    edge = sum(edge_cats) == 2
    face = sum(edge_cats) == 1
    return corner, edge, face
    
def test_volume_3d_adjacency_matrix():
    SHAPE = [10, 10, 10]
    vol_data = np.random.normal(size=SHAPE)
    vol = Volume(vol_data)
    adj = vol.adj_matrix.todense()
    assert(list(adj.shape) == [1000, 1000])
    # Interior voxels + face voxels + edge voxels + corner voxels
    total_neighbours = 8*8*8*6 + 6*8*8*5 + 8*12*4 + 8*3
    assert(np.sum(adj) == total_neighbours)
    # Do a few spot checks
    assert(adj[155, 156] == 1) # Interior neighbours
    assert(adj[8, 9] == 1)     # Edge/corner
    assert(adj[9, 10] == 0)    # corner/next face
    assert(adj[5, 15] == 1)    # edge/next row
    assert(adj[95, 105] == 0)  # edge/next layer

def test_volume_3d_laplacian_matrix():
    SHAPE = [5, 5, 5]
    vol_data = np.random.normal(size=SHAPE)
    vol = Volume(vol_data)
    l = vol.laplacian.todense()
    assert(list(l.shape) == [125, 125])
    for vid in range(125):
        corner, edge, face = _categorize(vid, SHAPE)
        if corner:
            assert(l[vid, vid] == -3)
        elif edge:
            assert(l[vid, vid] == -4)
        elif face:
            assert(l[vid, vid] == -5)
        else:
            assert(l[vid, vid] == -6)

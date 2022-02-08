"""
VABY - Volume structure models
"""
from types import SimpleNamespace

import numpy as np
import nibabel as nib
from scipy import sparse
import tensorflow as tf

from ..utils import NP_DTYPE
from .base import DataStructure

class Volume(DataStructure):
    """
    A data volume with optional mask

    Volume-specific attributes:
      - shape: 3D shape of volume
      - mask: 3D binary mask volume
      - voxel_sizes: Voxel dimensions in mm
      - srcdata.nii: Nibabel image of source data, if supplied as Nibabel image, None otherwise
      - srcdata.vol: 3D or 4D Numpy array of source data used to create the structure
      - srcdata.n_tpts: Number of timepoints in source data
      - srcdata.flat: Masked source data as 2D Numpy array
    """
    def __init__(self, vol_data=None, mask=None, nii=None, voxel_sizes=None, file_ext=".nii.gz", **kwargs):
        DataStructure.__init__(self, file_ext=file_ext, **kwargs)
        self.log.info("Volumetric data structure: %s" % self.name)
        self.log.info(" - File extension: %s" % self.file_ext)

        # In principle we could create a structure without source data, but for now it's hard to see
        # a reason why you'd want to do that
        if vol_data is None and nii is None:
            raise ValueError("No source data supplied - need either Numpy array or Nifti image")
        elif vol_data is None:
            vol_data = nii.get_fdata().astype(NP_DTYPE)
        else:
            if isinstance(vol_data, str):
                nii = nib.load(vol_data)
                vol_data = nii.get_fdata()
            vol_data = vol_data.astype(NP_DTYPE)

        # Use data supplied to define shape of structure
        if vol_data.ndim > 4:
            raise ValueError("Source data has too many dimensions: %i (max 4)" % vol_data.ndim)
        elif vol_data.ndim == 3:
            self.log.info(" - Source data is 3D - interpreting as 3D volume")
            vol_data = np.expand_dims(vol_data, -1)
        elif vol_data.ndim != 4:
            self.log.info(" - Source data is %iD - interpreting last dimension as timeseries"  % vol_data.ndim)
            while vol_data.ndim < 4:
                vol_data = np.expand_dims(vol_data, -2)

        self.shape = list(vol_data.shape[:3])
        self.srcdata = SimpleNamespace()
        self.srcdata.nii = nii
        self.srcdata.vol = vol_data
        self.srcdata.n_tpts = vol_data.shape[3]
        self.log.info(" - Shape: %s (Source data contained %i volumes)", self.shape, self.srcdata.n_tpts)

        # Handle mask if supplied
        if mask is None:
            self.log.info(" - No mask supplied")
            mask = np.ones(self.shape, dtype=int)
        elif isinstance(mask, str):
            mask = nib.load(mask).get_fdata().astype(int)
            if self.shape != list(mask.shape):
                raise ValueError("Mask has different 3D shape to data: %s vs %s" % (self.shape, mask.shape))

        self.mask = mask
        self.srcdata.flat = self.srcdata.vol[self.mask > 0]
        self.size = self.srcdata.flat.shape[0]
        self.log.info(" - Masked volume contains %i voxels", self.size)

        # Handle voxel sizes
        if self.srcdata.nii is not None:
            self.voxel_sizes = self.srcdata.nii.header['pixdim'][1:4]
        else:
            if voxel_sizes is not None:
                self.voxel_sizes = voxel_sizes
            else:
                self.log.warning("Voxel sizes not provided for Numpy array input - assuming 1mm isotropic")
                self.voxel_sizes = [1.0, 1.0, 1.0]
            self.srcdata.nii = nib.Nifti1Image(self.srcdata.vol, np.diag(list(self.voxel_sizes) + [1.0]))
        self.log.info(" - Voxel sizes: %s", self.voxel_sizes)

        # Calculate adjacency matrix and Laplacian
        self._calc_adjacency_matrix()
        self._calc_laplacian()

    def get_projection(self, data_space):
        try:
            self.check_compatible(data_space)
            
            def _identity_projection(tensor):
                return tensor

            return (_identity_projection, _identity_projection)
        except:
            import traceback
            traceback.print_exc()
            raise NotImplementedError("Projection between different volume spaces")

    def check_compatible(self, struc):
        """
        Check another supplied data structure represents the same underlying data space
        """
        DataStructure.check_compatible(self, struc)

        if self.shape != struc.shape:
            raise ValueError("Volumetric data structures do not have matching shape: %s vs %s" % (self.shape, struc.shape))

        if not np.allclose(self.voxel_sizes, struc.voxel_sizes):
            raise ValueError("Volumetric data structures do not have matching voxel sizes: %s vs %s" % (self.voxel_sizes, struc.voxel_sizes))

        if not np.all(self.mask == struc.mask):
            raise ValueError("Volumetric data structures do not have same mask")

    def load_data(self, data, **kwargs):
        from . import get_data_structure # Avoid circular import
        data_struc = get_data_structure(data, **kwargs)
        data_struc.check_compatible(self)
        return data_struc.srcdata.flat

    def nibabel_image(self, data):
        shape = self.shape
        if data.ndim > 1:
            shape = list(shape) + [data.shape[1]]
        ndata = np.zeros(shape, dtype=np.float)
        ndata[self.mask > 0] = data
        if self.srcdata.nii is not None:
            return nib.Nifti1Image(ndata, None, header=self.srcdata.nii.header)
        else:
            if self.voxel_sizes is None:
                self.log.warning("Voxel sizes not available - assuming 1mm isotropic")
                self.voxel_sizes = [1.0, 1.0, 1.0]
            affine = np.diag(list(self.voxel_sizes) + [1.0,])
            return nib.Nifti1Image(ndata, affine)

    def _calc_adjacency_matrix(self):
        """
        Calculate adjacency matrix for voxels

        The adjacency matrix value at (x, y) is 1 if voxels x and y
        are nearest neighbours, 0 otherwise. Voxel indices refer to
        the masked, flattened array so 0 is the first unmasked voxel
        and the result is a square sparse COO matrix of size
        ``self.size``.

        These are required for spatial priors and in practice do not
        take long to calculate so we provide them as a matter of course
        """
        def add_if_unmasked(x, y, z, masked_indices, nns):
            # Check that potential neighbour is not masked and if so
            # add it to the list of nearest neighbours
            idx  = masked_indices[x, y, z]
            if idx >= 0:
                nns.append(idx)

        # Generate a Numpy array which contains -1 for voxels which
        # are not in the mask, and for those which are contains the
        # voxel index, starting at 0 and ordered in row-major ordering
        # Note that the indices are for unmasked voxels only so 0 is
        # the index of the first unmasked voxel, 1 the second, etc.
        # Note that Numpy uses (by default) C-style row-major ordering
        # for voxel indices so the the Z co-ordinate varies fastest
        masked_indices = np.full(self.shape, -1, dtype=int)
        nx, ny, nz = tuple(self.shape)
        voxel_idx = 0
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if self.mask[x, y, z] > 0:
                        masked_indices[x, y, z] = voxel_idx
                        voxel_idx += 1

        # Now generate the nearest neighbour lists.
        voxel_nns = []
        indices_nn = []
        voxel_idx = 0
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if self.mask[x, y, z] > 0:
                        nns = []
                        if x > 0: add_if_unmasked(x-1, y, z, masked_indices, nns)
                        if x < nx-1: add_if_unmasked(x+1, y, z, masked_indices, nns)
                        if y > 0: add_if_unmasked(x, y-1, z, masked_indices, nns)
                        if y < ny-1: add_if_unmasked(x, y+1, z, masked_indices, nns)
                        if z > 0: add_if_unmasked(x, y, z-1, masked_indices, nns)
                        if z < nz-1: add_if_unmasked(x, y, z+1, masked_indices, nns)
                        voxel_nns.append(nns)
                        # For TensorFlow sparse tensor
                        for nn in nns:
                            indices_nn.append([voxel_idx, nn])
                        voxel_idx += 1

        # Edge case of no neighbours (e.g. single voxel often used in testing)
        if len(indices_nn) == 0:
            values = ([], [[], []])
        else:
            values = (np.ones(len(indices_nn)), (np.array(indices_nn).T))

        self.adj_matrix = sparse.coo_matrix(
            values,
            shape=[self.size, self.size], 
            dtype=NP_DTYPE
        )

        assert not (self.adj_matrix.tocsr()[np.diag_indices(self.size)] != 0).max()

    def _calc_laplacian(self):
        """
        Calculate Laplacian matrix.

        This is a spatial smoothing operator used to implement spatial priors.
        For a volume, off-diagonal elements (x, y) are 1 if voxels x and y are
        nearest neighbourd, 0 otherwise. The diagonal elements are the negative
        of the number of neighbours a voxel has. Voxel indices refer to
        the masked, flattened array so 0 is the first unmasked voxel
        and the result is a square sparse COO matrix of size
        ``self.size``.

        Note the sign convention is negatives on the diagonal, and positive values
        off diagonal.
        """
        lap = self.adj_matrix.todok(copy=True)
        lap[np.diag_indices(lap.shape[0])] = -lap.sum(1).T
        assert lap.sum(1).max() == 0, 'Unweighted Laplacian matrix'
        self.laplacian = lap
        #self.laplacian = self._scipy_to_tf_sparse(lap)

class PartialVolumes(Volume):
    """
    Volumetric structure in which every voxel has a partial volume

    The source data is assumed to define the PVs unless a separate PV map is given.
    The PVs of each voxel affect the projection of data onto the structure.
    """
    def __init__(self, pv_vol=None, **kwargs):
        Volume.__init__(self, **kwargs)
        if pv_vol is None:
            if self.srcdata.n_tpts != 1:
                raise ValueError("Partial volume map is 4D")

            self.pvs = self.srcdata.flat
        else:
            if isinstance(pv_vol, str):
                pv_vol = nib.load(pv_vol).get_fdata()
            if list(pv_vol.shape) != list(self.shape):
                raise ValueError("Partial volume map does not have the same shape as the underlying volume")

            self.pvs = pv_vol[self.mask]

    def get_projection(self, data_space):
        try:
            self.check_compatible(data_space)
            def _model2data(tensor):
                return tensor * self.pvs

            def _data2model(tensor):
                return tensor

            return _model2data, _data2model
        except:
            raise NotImplementedError("Projection between different volume spaces")

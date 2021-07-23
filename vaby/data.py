"""
VABY - Data model
"""
import math
import collections

import six
import numpy as np
import nibabel as nib
from scipy import sparse

from .utils import LogBase, NP_DTYPE

class DataModel(LogBase):
    """
    Encapsulates information about the data volume being modelled

    Two spaces are defined: 
    
     - Voxel space which is a volumetric space in which the measured
       data is defined
     - Model space which defines a set of *nodes* on which the data is modelled
       and which may be volumetric, surface based or a combination of both.
       Model space may be divided into structures (e.g. WM/GM, R/L hemispheres)
       with each structure modelled in a different way. The definition of model
       space also includes the mapping between nodes and voxels.
       
    Measured data
    -------------

    :ivar data_vol: Measured data as a 4D volume
    :ival shape: List containing 3D shape of measured data
    :ival n_tpts: Number of timepoints in data
    :ival mask_vol: Binary mask as a 3D volume. If none provided, a simple
          array of ones matching the data will be generate
    :ivar data: Measured data as a 2D array (voxels, timepoints). If
          a mask is provided only masked voxels are included
    :ival n_voxels: Number of voxels in the data that are included the mask
    :ival voxel_sizes: Sizes of voxels in mm as a sequence of 3 values

    :param data: Measured data as a filename, Numpy array or Nifti image
    :param mask: Optional mask as a filename, Numpy array or Nifti image
    :param voxel_sizes: Voxel sizes as sequence of 3 values in mm. For use
           when the data is supplied as a Numpy array (not a Nifti image)
    :param model_space: Sequence of dictionaries defining the structures
           in model space. Keys are ``name``: Structure name, ``type``:
           ``volume`` or ``surface``, ``data`` : Filename or Numpy array.
           For a volume, this should be a mask compatible with the measured
           data defining the voxels that should be treated as nodes. For
           a surface this should be a GIFTI geometry file defining the
           surface nodes. If not specified, a volume model space is
           created matched with the measured data and mask.

    This includes its overall dimensions, mask (if provided), 
    and neighbouring voxel lists
    """

    def __init__(self, data, mask=None, **kwargs):
        LogBase.__init__(self)

        ### Data space
        self.nii, self.data_vol = self._load_data(data)
        while self.data_vol.ndim < 4:
            self.data_vol = self.data_vol[np.newaxis, ...]

        self.shape = list(self.data_vol.shape)[:3]
        self.n_tpts = self.data_vol.shape[3]
        self.voxel_sizes = self.nii.header['pixdim'][1:4]

        # If there is a mask load it and use it to mask the data
        if mask is not None:
            _mask_nii, self.mask_vol = self._load_data(mask)
            if self.shape != list(self.mask_vol.shape):
                raise ValueError("Mask has different shape to main data: %s vs %s" % (self.shape, self.mask_vol.shape))
        else:
            self.mask_vol = np.ones(self.shape, dtype=np.int)

        self.data_flat = self.data_vol[self.mask_vol > 0]
        self.n_voxels = self.data_flat.shape[0]
        self._calc_adjacency_matrix()

        ### Model space
        model_structures = kwargs.get("model_structures", None)
        if model_structures is None:
            self.model_structures = [{
                "name" : "voxels",
                "type" : "volume",
                "size" : self.n_voxels
            },]
            self.n_nodes = self.n_voxels
        else:
            for structure in model_structures:
                self.log.info("Creating model space structure: %s" % structure["name"])
                raise NotImplementedError()

        if kwargs.get("initial_posterior", None):
            self.post_init = self._get_posterior_data(kwargs["initial_posterior"])
        else:
            self.post_init = None

        self._calc_laplacian()

    def _calc_adjacency_matrix(self):
        """
        Generate adjacency matrix for voxel nearest neighbours.

        Note the result will be a square sparse COO matrix of size 
        n_voxels so index 0 refers to the first un-masked voxel.

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
        masked_indices = np.full(self.shape, -1, dtype=np.int)
        nx, ny, nz = tuple(self.shape)
        voxel_idx = 0
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if self.mask_vol[x, y, z] > 0:
                        masked_indices[x, y, z] = voxel_idx
                        voxel_idx += 1

        # Now generate the nearest neighbour lists.
        voxel_nns = []
        indices_nn = []
        voxel_idx = 0
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if self.mask_vol[x, y, z] > 0:
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
            shape=2*[self.n_voxels], 
            dtype=np.float32
        )

        assert not (self.adj_matrix.tocsr()[np.diag_indices(self.n_voxels)] != 0).max()

    def _calc_laplacian(self):
        """
        Laplacian matrix. Note the sign convention is negatives
        on the diagonal, and positive values off diagonal. 
        """
        lap = self.adj_matrix.todok(copy=True)
        lap[np.diag_indices(lap.shape[0])] = -lap.sum(1).T
        assert lap.sum(1).max() == 0, 'Unweighted Laplacian matrix'
        self.laplacian = lap.tocoo()

    def nodes_to_voxels_ts(self, tensor, pv_sum=True):
        return tensor

    def nodes_to_voxels(self, tensor, pv_sum=True):
        return tensor

    def voxels_to_nodes(self, tensor, pv_sum=True):
        return tensor

    def voxels_to_nodes_ts(self, tensor, pv_sum=True):
        return tensor

    def nifti_image(self, data):
        """
        :return: A nibabel.Nifti1Image for some, potentially masked, output data
        """
        shape = self.shape
        if data.ndim > 1:
            shape = list(shape) + [data.shape[1]]
        ndata = np.zeros(shape, dtype=np.float)
        ndata[self.mask_vol > 0] = data
        return nib.Nifti1Image(ndata, None, header=self.nii.header)

    def posterior_data(self, mean, cov):
        """
        Get voxelwise data for the full posterior

        We use the Fabber method of saving the upper triangle of the
        covariance matrix concatentated with a column of means and
        an additional 1.0 value to make it square.

        Note that if some of the posterior is factorized or 
        covariance is not being inferred some or all of the covariances
        will be zero.
        """
        if cov.shape[0] != self.n_voxels or mean.shape[0] != self.n_voxels:
            raise ValueError("Posterior data has %i voxels - inconsistent with data model containing %i unmasked voxels" % (cov.shape[0], self.n_voxels))

        num_params = mean.shape[1]
        vols = []
        for row in range(num_params):
            for col in range(row+1):
                vols.append(cov[:, row, col])
        for row in range(num_params):
            vols.append(mean[:, row])
        vols.append(np.ones(mean.shape[0]))
        return np.array(vols).transpose((1, 0))

    def get_voxel_data(self, data, **kwargs):
        """
        Get data in voxel space
        
        The data must be compatible with the main voxel data set. If it is
        a volume, it must have the same volumetric shape. If there is a mask,
        it will be applied to the volume. If it is a flattened array it must
        match the flattened masked data array.
        
        :param data: Either a string containing the filename of a supported file, a Nibabel image or a Numpy array
        :return: Numpy array shape [n_voxels, n_tpts] or [n_voxels] if not a timeseries
        """
        nii, vol = self._load_data(data)

        if vol.ndim >= 3:
            # If 3d/4d volume, use the mask to flatten it provided the shape is correct
            if self.shape != list(vol.shape):
                raise ValueError("get_voxel_data: Data volume has different shape to main data: %s vs %s" % (self.shape, vol.shape))
            data_flat = vol[self.mask_vol > 0]
        else:
            # If 1d/2d, first dim should correspond to the the number of voxels
            if self.n_voxels != vol.shape[0]:
                raise ValueError("get_voxel_data: Data number of voxels does not match main data: %s vs %s" % (self.n_voxels, vol.shape[0]))
            data_flat = vol
    
        if nii is not None:
            voxel_sizes = nii.header['pixdim'][1:4]
            if not np.allclose(voxel_sizes, self.voxel_sizes):
                self.log.warn("get_voxel_data: Data voxel sizes does not match main data: %s vs %s", self.voxel_sizes, voxel_sizes)

        return data_flat

    def _load_data(self, data):
        if isinstance(data, six.string_types):
            nii = nib.load(data)
            if data.endswith(".nii") or data.endswith(".nii.gz"):
                data_vol = nii.get_fdata()
            elif data.endswith(".gii"):
                # FIXME
                raise NotImplementedError()
            self.log.info("Loaded data from %s", data)
        elif isinstance(data, nib.Nifti1Image):
            data_vol = data.get_fdata()
            nii = data
        else:
            data_vol = data
            nii = None
        return nii, data_vol.astype(NP_DTYPE)

    def _get_posterior_data(self, post_data):
        if isinstance(post_data, six.string_types):
            return self._posterior_from_file(post_data)
        elif isinstance(post_data, collections.Sequence):
            return tuple(post_data)
        else:
            raise TypeError("Invalid data type for initial posterior: should be filename or tuple of mean, covariance")

    def _posterior_from_file(self, fname):
        """
        Read a Nifti file containing the posterior saved using --save-post
        and extract the covariance matrix and the means
        
        This can then be used to initialize a new run - note that
        no checking is performed on the validity of the data in the MVN
        other than it is the right size.
        """
        post_data = nib.load(fname).get_data()
        if post_data.ndim !=4:
            raise ValueError("Posterior input file '%s' is not 4D" % fname)
        if list(post_data.shape[:3]) != list(self.shape):
            raise ValueError("Posterior input file '%s' has shape %s - inconsistent with mask shape %s" % (fname, post_data.shape[:3], self.shape))

        post_data = post_data[self.mask_vol > 0]
        nvols = post_data.shape[1]
        self.log.info("Posterior image contains %i volumes" % nvols)

        n_params = int((math.sqrt(1+8*float(nvols)) - 3) / 2)
        nvols_recov = (n_params+1)*(n_params+2) / 2
        if nvols != nvols_recov:
            raise ValueError("Posterior input file '%s' has %i volumes - not consistent with upper triangle of square matrix" % (fname, nvols))
        self.log.info("Posterior image contains %i parameters", n_params)
        
        cov = np.zeros((self.n_voxels, n_params, n_params), dtype=np.float32)
        mean = np.zeros((self.n_voxels, n_params), dtype=np.float32)
        vol_idx = 0
        for row in range(n_params):
            for col in range(row+1):
                cov[:, row, col] = post_data[:, vol_idx]
                cov[:, col, row] = post_data[:, vol_idx]
                vol_idx += 1
        for row in range(n_params):
            mean[:, row] = post_data[:, vol_idx]
            vol_idx += 1
        if not np.all(post_data[:, vol_idx] == 1):
            raise ValueError("Posterior input file '%s' - last volume does not contain 1.0", fname)

        self.log.info("Posterior mean shape: %s, cov shape: %s", mean.shape, cov.shape)
        return mean, cov

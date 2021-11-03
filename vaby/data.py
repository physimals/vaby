"""
VABY - Data model
"""
import math
import collections
from nibabel import volumeutils

import six
import numpy as np
import nibabel as nib
from scipy import sparse

from .utils import LogBase, NP_DTYPE

def get_data_structure(data, **kwargs):
    """
    Factory method to return an instance of DataStructure
    
    :param data: Source of data, either filename, Numpy array, Nifti1Image or GiftiImage
    
    Keyword args:
     - 
    """
    if isinstance(data, six.string_types):
        data = nib.load(data)

    if isinstance(data, np.ndarray):
        return Volume(data, **kwargs)
    elif isinstance(data, nib.Nifti1Image):
        return Volume(nii=data, **kwargs)
    elif isinstance(data, nib.GiftiImage):
        return Surface(gii=data, **kwargs)

class DataStructure(LogBase):
    """
    Single data structure, volumetric or surface
    """
    def __init__(self):
        LogBase.__init__(self)

class Volume(DataStructure):
    """
    A data volume with optional mask

    Attributes:
      - vol_data
      - shape
      - size
      - n_tpts
      - mask_vol
      - data_flat
      - nii
      - voxel_sizes
    """
    def __init__(self, vol_data=None, mask=None, nii=None, voxel_sizes=None, **kwargs):
        DataStructure.__init__(self)
        self.log.info("Volumetric data structure:")
        self.voxel_sizes = voxel_sizes

        if vol_data is None and nii is None:
            raise ValueError("No data supplied - need either Numpy array or Nifti image")
        elif vol_data is None:
            vol_data = nii.get_fdata().astype(np.float32)

        if vol_data.ndim > 4:
            raise ValueError("Data has too many dimensions: %i (max 4)" % vol_data.ndim)
        self.vol_data = vol_data
        while self.vol_data.ndim < 4:
            self.vol_data = self.vol_data[..., np.newaxis]
        self.shape = list(self.vol_data.shape[:3])
        self.n_tpts = self.vol_data.shape[3]
        self.log.info(" - 3D shape %s, %i volumes", self.shape, self.n_tpts)

        if mask is None:
            self.log.info(" - No mask supplied")
            mask = np.ones(self.shape)
        elif isinstance(mask, six.string_types):
            mask = nib.load(mask).get_fdata().astype(np.int)

        self.mask_vol = mask
        if self.shape != list(self.mask_vol.shape):
            raise ValueError("Mask has different shape to data: %s vs %s" % (self.shape, self.mask_vol.shape))
        self.data_flat = self.vol_data[self.mask_vol > 0]
        self.size = self.data_flat.shape[0]
        self.log.info(" - Masked data contains %i voxels", self.size)
        if nii is not None:
            self.nii = nii
            self.voxel_sizes = self.nii.header['pixdim'][1:4]
        else:
            if self.voxel_sizes is None:
                self.log.warning("Voxel sizes not provided for Numpy array input - assuming 1mm isotropic")
                self.voxel_sizes = [1.0, 1.0, 1.0]
            affine = np.diag(list(self.voxel_sizes) + [1.0,])
            self.nii = nib.Nifti1Image(self.vol_data, affine)
        self.log.info(" - Voxel sizes: %s", self.voxel_sizes)

        self._calc_adjacency_matrix()
        self._calc_laplacian()
    
    def check_compatible(self, struc):
        """
        Check the supplied data structure represents the same underlying data space
        """
        if type(self) != type(struc):
            raise ValueError("Data structure of type %s does not match structure of type %s" % (type(self), type(struc)))
        
        if self.shape != struc.shape:
            raise ValueError("Data structures do not have matching shape: %s vs %s" % (self.shape, struc.shape))

        if not np.allclose(self.voxel_sizes, struc.voxel_sizes):
            raise ValueError("Data structures do not have matching voxel sizes: %s vs %s" % (self.voxel_sizes, struc.voxel_sizes))

    def nifti_image(self, data):
        """
        :param data: Data array defined in this data space, i.e. a 1D Numpy array
                     or 2D array if multi-volume data
        :return: nibabel.Nifti1Image
        """
        shape = self.shape
        if data.ndim > 1:
            shape = list(shape) + [data.shape[1]]
        ndata = np.zeros(shape, dtype=np.float)
        ndata[self.mask_vol > 0] = data
        return nib.Nifti1Image(ndata, None, header=self.nii.header)

    def _calc_adjacency_matrix(self):
        """
        Generate adjacency matrix for voxel nearest neighbours.

        Note the result will be a square sparse COO matrix of size 
        ``self.size`` so index 0 refers to the first un-masked voxel.

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
            shape=2*[self.size], 
            dtype=np.float32
        )

        assert not (self.adj_matrix.tocsr()[np.diag_indices(self.size)] != 0).max()

    def _calc_laplacian(self):
        """
        Laplacian matrix. Note the sign convention is negatives
        on the diagonal, and positive values off diagonal. 
        """
        lap = self.adj_matrix.todok(copy=True)
        lap[np.diag_indices(lap.shape[0])] = -lap.sum(1).T
        assert lap.sum(1).max() == 0, 'Unweighted Laplacian matrix'
        self.laplacian = lap.tocoo()

class Surface(DataStructure):
    """
    A surface data structure
    """

    def __init__(self, gii, **kwargs):
        DataStructure.__init__(self)
        self.gii = gii
        self._calc_adjacency_matrix()
        self._calc_laplacian()

    def _calc_adjacency_matrix(self):
        pass

    def _calc_laplacian(self):
        pass

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

    def __init__(self, data, **kwargs):
        LogBase.__init__(self)

        ### Data space
        self.data_space = get_data_structure(data, **kwargs)
        self.n_voxels = self.data_space.size
        self.n_tpts = self.data_space.n_tpts
        self.shape = self.data_space.shape
        self.mask_vol = self.data_space.mask_vol
        self.data_flat = self.data_space.data_flat
        self.adj_matrix = self.data_space.adj_matrix
        self.laplacian = self.data_space.laplacian

        # Compatibility
        self.data_flattened = self.data_flat
        self.n_unmasked_voxels = self.n_voxels

        ### Model space
        model_structures = kwargs.get("model_structures", None)
        if model_structures is None:
            self.model_structures = [{
                "name" : "voxels",
                "type" : "volume",
                "size" : self.data_space.size,
                "shape" : self.data_space.shape,
                "mask" : self.data_space.mask_vol,
            },]
            self.n_nodes = self.data_space.size
        else:
            for structure in model_structures:
                #self.log.info("Creating model space structure: %s" % structure["name"])
                #nii, vol = self._load_data(structure["data"])
                #structure["type"] = volume
                #structure["shape"] = vol.shape
                #structure["mask"] = mask_vol
                #self.model_structures.append(structure)
                raise NotImplementedError()

        if kwargs.get("initial_posterior", None):
            self.post_init = self._get_posterior_data(kwargs["initial_posterior"])
        else:
            self.post_init = None

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
        return self.data_space.nifti_image(data)

    def get_voxel_data(self, data, **kwargs):
        """
        Get data in acquisition space
        
        The data must be compatible with the main voxelwise input data set. If it is
        a volume, it must have the same volumetric shape. If there is a mask,
        it will be applied to the volume. If it is a flattened array it must
        match the flattened masked data array.
        
        :param data: Either a string containing the filename of a supported file, a Nibabel image or a Numpy array
        :return: Numpy array shape [n_voxels, n_tpts] or [n_voxels] if not a timeseries
        """
        data_struc = get_data_structure(data, **kwargs)
        data_struc.check_compatible(self.data_space)
        return data_struc.data_flat

    def encode_posterior(self, mean, cov):
        """
        :return: a voxelwise data array containing the mean and covariance for the posterior

        We use the Fabber method of serializing the upper triangle of the
        covariance matrix concatentated with a column of means and
        an additional 1.0 value to make it square.

        Note that if some of the posterior is factorized or 
        covariance is not being inferred some or all of the covariances
        will be zero.
        """
        if cov.shape[0] != self.n_nodes or mean.shape[0] != self.n_nodes:
            raise ValueError("Posterior data has %i nodes - inconsistent with data model containing %i unmasked voxels" % (cov.shape[0], self.n_voxels))

        num_params = mean.shape[1]
        vols = []
        for row in range(num_params):
            for col in range(row+1):
                vols.append(cov[:, row, col])
        for row in range(num_params):
            vols.append(mean[:, row])
        vols.append(np.ones(mean.shape[0]))
        return np.array(vols).transpose((1, 0))

    def decode_posterior(self, post_data):
        """
        Convert possibly encoded posterior data array into tuple of mean, covariance
        """
        if isinstance(post_data, collections.Sequence):
            return tuple(post_data)
        else:
            # FIXME posterior should be defined in model space not data space
            post_data_arr = self.get_voxel_data(post_data)
            nvols = post_data_arr.shape[1]
            self.log.info("Posterior image contains %i volumes" % nvols)

            n_params = int((math.sqrt(1+8*float(nvols)) - 3) / 2)
            nvols_recov = (n_params+1)*(n_params+2) / 2
            if nvols != nvols_recov:
                raise ValueError("Posterior input file '%s' has %i volumes - not consistent with upper triangle of square matrix" % (fname, nvols))
            self.log.info("Posterior image contains %i parameters", n_params)
            
            cov = np.zeros((self.n_nodes, n_params, n_params), dtype=np.float32)
            mean = np.zeros((self.n_nodes, n_params), dtype=np.float32)
            vol_idx = 0
            for row in range(n_params):
                for col in range(row+1):
                    cov[:, row, col] = post_data_arr[:, vol_idx]
                    cov[:, col, row] = post_data_arr[:, vol_idx]
                    vol_idx += 1
            for row in range(n_params):
                mean[:, row] = post_data_arr[:, vol_idx]
                vol_idx += 1
            if not np.all(post_data_arr[:, vol_idx] == 1):
                raise ValueError("Posterior input file '%s' - last volume does not contain 1.0", fname)

            self.log.info("Posterior mean shape: %s, cov shape: %s", mean.shape, cov.shape)
            return mean, cov

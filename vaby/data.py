"""
VABY - Data model
"""
import math
import collections
import os
from nibabel import volumeutils
from types import SimpleNamespace

import six
import numpy as np
import nibabel as nib
from scipy import sparse

from .utils import LogBase, NP_DTYPE

def get_data_structure(data, **kwargs):
    """
    Factory method to return an instance of DataStructure
    
    :param data: Source of data, either filename, Numpy array, Nifti1Image or GiftiImage
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
    """"
    A data structure
    
    Each data structure has the following attributes:
     - size: Total number of independent nodes in the structure (e.g. surface nodes, unmasked voxels)
     - adj_matrix: Sparse matrix dimension (size, size) containing 1 where nodes are regarded as connected, 0 otherwise
     - laplacian: Sparse matrix dimension (size, size) containing Laplacian for spatial smoothing
     - file_ext: Standard extension for saved files (e.g. .nii.gz or .gii)
    """
    def __init__(self, **kwargs):
        LogBase.__init__(self)
        self.file_ext = kwargs.get("file_ext", "")
        self.name = kwargs.get("name", "data")

    def check_compatible(self, struc):
        """
        Check another supplied data structure represents the same underlying data space
        """
        if type(self) != type(struc):
            raise ValueError("Data structure of type %s does not match structure of type %s" % (type(self), type(struc)))
        
        if self.size != struc.size:
            raise ValueError("Data structure of size %i does not match structure of type %i" % (self.size, struc.size))

    def load_data(self, data, **kwargs):
        """
        Get data from a file, Numpy array or Nibabel object and check it is compatible with this data space

        :param data: Either a string containing the filename of a supported file, a Nibabel image or a Numpy array
        :return: 1D or 2D Numpy array shape (size, [n_tpts]) containing loaded data
        """
        raise NotImplementedError()

    def nibabel_image(self, data):
        """
        Generate a Nibabel image for arbitrary data defined in this data space

        :param data: Data array defined in this data space, i.e. a 1D Numpy array
                     or 2D array if multi-volume data
        :return: Appropriate type of Nibabel image, e.g. nibabel.Nifti1Image
        """
        raise NotImplementedError()

    def save_data(self, data, name, outdir="."):
        self.nibabel_image(data).to_filename(os.path.join(outdir, name) + self.file_ext)

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
            vol_data = nii.get_fdata().astype(np.float32)

        # Use data supplied to define shape of structure
        if vol_data.ndim > 4:
            raise ValueError("Source data has too many dimensions: %i (max 4)" % vol_data.ndim)
        if vol_data.ndim == 1:
            self.log.info(" - Source data is 1D - interpreting as single voxel timeseries")

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
            mask = np.ones(self.shape)
        elif isinstance(mask, six.string_types):
            mask = nib.load(mask).get_fdata().astype(np.int)

        self.mask = mask
        if self.shape != list(self.mask.shape):
            raise ValueError("Mask has different 3D shape to data: %s vs %s" % (self.shape, self.mask.shape))
        self.srcdata.flat = self.srcdata.vol[self.mask > 0]
        self.size = self.srcdata.flat.shape[0]
        self.log.info(" - Masked volume contains %i voxels", self.size)

        # Handle voxel sizes
        if self.srcdata.nii is not None:
            self.voxel_sizes = self.srcdata.nii.header['pixdim'][1:4]
        elif voxel_sizes is not None:
            self.voxel_sizes = voxel_sizes
        else:
            self.log.warning("Voxel sizes not provided for Numpy array input - assuming 1mm isotropic")
            self.voxel_sizes = [1.0, 1.0, 1.0]
        self.log.info(" - Voxel sizes: %s", self.voxel_sizes)

        # Calculate adjacency matrix and Laplacian
        self._calc_adjacency_matrix()
        self._calc_laplacian()

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
        """
        :param gii: nibabel.Gifti1Image containing surface structure data
        """
        DataStructure.__init__(self)
        self.gii = gii
        self._calc_adjacency_matrix()
        self._calc_laplacian()

    def load_data(self, data, **kwargs):
        raise NotImplementedError()

    def nibabel_image(self, data):
        raise NotImplementedError()
        #for hemi, hslice in zip(data_model.projector.iter_hemis, 
        #                        data_model.iter_hemi_slicers): 
        #    d = sdata[hslice]
        #    p = path_gen(name_base, hemi.side)
        #    g = data_model.gifti_image(d, hemi.side)
        #    nib.save(g, p)

    def _calc_adjacency_matrix(self):
        raise NotImplementedError()

    def _calc_laplacian(self):
        raise NotImplementedError()

class CompositeDataStructure(DataStructure):
    """
    A data structure with multiple named parts

    The adjacency and laplacian matrix are block-diagonal copies of the source structures with
    zeros elsewhere (i.e. the separate structures are not considered to be connected)

    Attributes:
     - parts: Sequence of (name, DataStructure) instances
    """

    def __init__(self, parts, **kwargs):
        """
        :param parts: Sequence of (name, DataStructure) instances
        """
        DataStructure.__init__(self)
        self.parts = parts
        self.size = sum([p.size for p in parts])
        self._calc_adjacency_matrix()
        self._calc_laplacian()

    def _calc_adjacency_matrix(self):
        raise NotImplementedError()

    def _calc_laplacian(self):
        raise NotImplementedError()

    def save_data(self, data, name, outdir="."):
        for struct_name, struct in self.parts:
            struct.save_data(data, name + "_" + struct_name, outdir)

class DataModel(LogBase):
    """
    Encapsulates information about the physical structure of the source data and how it
    is being modelled

    Two spaces are defined: 

    :ival data_space: DataStructure defining acquisition data space
    :ival model_space: DataStructure defining modelling data space. This may be
                       identical to data space, a different kind of space (e.g.
                       surface), or a composite space containing multiple 
                       independent structures.
    """

    def __init__(self, data, **kwargs):
        LogBase.__init__(self)

        ### Acquisition data space
        self.data_space = get_data_structure(data, name="acquisition", **kwargs)

        ### Model space
        model_structures = kwargs.get("model_structures", None)
        if model_structures is None:
            self.model_space = get_data_structure(data, name="model", **kwargs)
        else:
            struc_list = []
            for name, src in model_structures:
                self.log.info("Creating model space structure: %s" % name)
                if not isinstance(src, DataStructure):
                    struc_list.append((name, get_data_structure(src)))
                else:
                    struc_list.append((name, src))
            self.model_space = CompositeDataStructure(model_structures)

        if kwargs.get("initial_posterior", None):
            raise NotImplementedError()
            #self.post_init = self._get_posterior_data(kwargs["initial_posterior"])
        else:
            self.post_init = None

    def model_to_data(self, tensor, pv_sum=True):
        """
        Convert model space data into source data space

        FIXME assuming spaces are the same
        """
        return tensor

    def data_to_model(self, tensor, pv_sum=True):
        """
        Convert source data space data into model space

        FIXME assuming spaces are the same
        """
        return tensor

    def encode_posterior(self, mean, cov):
        """
        Encode the posterior mean and covariance as a single timeseries

        We use the Fabber method of serializing the upper triangle of the
        covariance matrix concatentated with a column of means and
        an additional 1.0 value to make it square.

        Note that if some of the posterior is factorized or 
        covariance is not being inferred some or all of the covariances
        will be zero.

        :return: a nodewise data array containing the mean and covariance for the posterior
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

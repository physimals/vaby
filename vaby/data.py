"""
VABY - Data model
"""
import math
import collections
import os
from types import SimpleNamespace
from google.protobuf.reflection import ParseMessage

import six
import numpy as np
import nibabel as nib
from scipy import sparse
import tensorflow as tf

try:
    import toblerone
    import regtricks
except ImportError:
    toblerone = None

from .utils import LogBase, NP_DTYPE, TF_DTYPE

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
        return CorticalSurface(**kwargs)

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

    def get_projection(self, data_space):
        """
        Get the projections between this space and the data acquisition space

        :param data_space: Volume defining the acquisition data space
        :return: Tuple of callables (data_to_model, model_to_data) which may be called
                 passing a tensor in data/model space and returning a tensor in model/data
                 space
        """
        raise NotImplementedError()

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
            vol_data = nii.get_fdata().astype(NP_DTYPE)
        else:
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
        elif isinstance(mask, six.string_types):
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

    def _identity_projection(self, tensor, pv_sum):
        return tensor

    def get_projection(self, data_space):
        try:
            self.check_compatible(data_space)
            return (self._identity_projection, self._identity_projection)
        except:
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
        self.laplacian = lap.tocoo()

class CorticalSurface(DataStructure):
    """
    A cortical surface data structure

    This consists of one or two hemispheres defined by inner (white matter) and
    outer (pial) surfaces. These two surfaces contain corresponding tri-meshes
    that define triangular voxels each of which defines a node in the structure.
    """

    def __init__(self, white, pial, **kwargs):
        """
        :param left: Tuple of (LWS, LPS) where LWS and LPS are filenames or nibabel.Gifti1Image instances
                     containing the white and pial surfaces for the left hemisphere
        :param left: Tuple of (LWS, LPS) where LWS and LPS are filenames or nibabel.Gifti1Image instances
                     containing the white and pial surfaces for the right hemisphere
        """
        DataStructure.__init__(self, **kwargs)
        if toblerone is None:
            raise RuntimeError("Toblerone not installed - cannot create cortical surface structure")

        if self.name not in ("L", "R"):
            raise ValueError("For now, the names of cortical surfaces must be either 'L' or 'R'")

        from toblerone.classes import Hemisphere, Surface
        self.hemisphere = Hemisphere(Surface(white, self.name + "WS"), Surface(pial, self.name + "PS"), self.name)
        self.size = self.hemisphere.n_points
        self.projector = kwargs.get("projector", None)
        self.proj_tensors = None
        if isinstance(self.projector, str):
            self.log.info(f"Loading projector from {self.projector}")
            self.projector = toblerone.Projector.load(self.projector)

        self.adj_matrix = self.hemisphere.adjacency_matrix()
        self.laplacian = self.hemisphere.mesh_laplacian()

    def get_projection(self, data_space):
        if self.projector is None:
            self.log.info("Generating projector - this may take some time...")
            self.projector = toblerone.Projector(self.hemisphere, regtricks.ImageSpace(data_space.srcdata.nii), factor=10, cores=8)
            self.projector.save("vaby_proj.h5")
            self.log.info("Projector generated")

        if self.projector.spc != regtricks.ImageSpace(data_space.srcdata.nii):
            raise ValueError("Projector supplied is not defined on same image space as acquisition data")

        if self.proj_tensors is None:
            proj_matrices = {
                "n2v" : self.projector.surf2vol_matrix(edge_scale=True).astype(NP_DTYPE),
                "n2v_noedge" : self.projector.surf2vol_matrix(edge_scale=False).astype(NP_DTYPE),
                "v2n" : self.projector.vol2surf_matrix(edge_scale=True).astype(NP_DTYPE),
                "v2n_noedge" : self.projector.vol2surf_matrix(edge_scale=False).astype(NP_DTYPE),
            }

            if data_space.size!= proj_matrices["n2v"].shape[0]:
                raise ValueError('Acquisition data size does not match projector')
            #if self.projector.n_surf_nodes != n2v.shape[1]:
            #    raise ValueError('Mask size does not match projector')

            # Knock out voxels from projection matrices that are not in the mask
            # and convert to sparse tensors
            self.proj_tensors = {}
            vox_inds = np.flatnonzero(data_space.mask)
            for name, mat in proj_matrices.items():
                if name.startswith("n2v"):
                    masked_mat = mat.tocsr()[vox_inds, :].tocoo()
                else:
                    masked_mat = mat.tocsr()[:, vox_inds].tocoo()
                self.proj_tensors[name] = tf.SparseTensor(
                    indices=np.array([masked_mat.row, masked_mat.col]).T,
                    values=masked_mat.data,
                    dense_shape=masked_mat.shape,
                )

        def _surf2vol(tensor, pv_sum=False):
            if pv_sum:
                return tf.sparse.sparse_dense_matmul(self.proj_tensors["n2v"], tensor)
            else:
                return tf.sparse.sparse_dense_matmul(self.proj_tensors["n2v_noedge"], tensor)

        def _vol2surf(tensor, pv_sum=False):
            if pv_sum:
                return tf.sparse.sparse_dense_matmul(self.proj_tensors["v2n"], tensor)
            else:
                return tf.sparse.sparse_dense_matmul(self.proj_tensors["v2n_noedge"], tensor)

        return (_surf2vol, _vol2surf)

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

class CompositeDataStructure(DataStructure):
    """
    A data structure with multiple named parts

    The adjacency and laplacian matrix are block-diagonal copies of the source structures with
    zeros elsewhere (i.e. the separate structures are not considered to be connected).

    Attributes:
     - parts: Sequence of (name, DataStructure) instances
     - slices: Slice object for each part to extract model nodes relevant to that part
    """

    def __init__(self, parts, **kwargs):
        """
        :param parts: Sequence of DataStructure instances
        """
        DataStructure.__init__(self)
        self.parts = parts
        self.size = sum([p.size for p in parts])
        self.slices = []
        start = 0
        for p in self.parts:
            self.slices.append(slice(start, start+p.size))
            start += p.size
        self.projectors = []
        self.adj_matrix = sparse.block_diag([p.adj_matrix for p in self.parts]).astype(NP_DTYPE)
        self.laplacian = sparse.block_diag([p.laplacian for p in self.parts]).astype(NP_DTYPE)

    def get_projection(self, data_space):
        if not self.projectors:
            self.projectors = [p.get_projection(data_space) for p in self.parts]

        def model2data(tensor, pv_sum=False):
            tensor_data = []
            for proj, slc in zip(self.projectors, self.slices):
                tensor_data.append(proj[0](tensor[slc, ...], pv_sum)) # [V, T]
            return sum(tensor_data) # [V, T]

        def data2model(tensor, pv_sum=False):
            tensor_model = []
            for proj in self.projectors:
                tensor_model.append(proj[1](tensor, pv_sum)) # [w, T]
            return tf.concat(tensor_model, axis=0) # [W, T]

        return model2data, data2model

    def save_data(self, data, name, outdir="."):
        for struct, slc in zip(self.parts, self.slices):
            struct.save_data(data[slc, ...], name + "_" + struct.name, outdir)

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
    :ival projector: Tuple of callables that can convert data between acquisition
                     and model space. The first converts model space tensors to
                     acquisition space, the second goes the other way.
    """

    def __init__(self, data, **kwargs):
        LogBase.__init__(self)

        ### Acquisition data space
        self.data_space = get_data_structure(data, name="acquisition", **kwargs)

        ### Model space
        model_structures = kwargs.get("model_structures", None)
        if model_structures is None:
            self.log.info(" - Model space is same as acquisition space")
            self.model_space = self.data_space
        else:
            struc_list = []
            for name, src in model_structures.items():
                self.log.info("Creating model space structure: %s" % name)
                if not isinstance(src, DataStructure):
                    struc_list.append((name, get_data_structure(**src)))
                else:
                    struc_list.append((name, src))
            self.model_space = CompositeDataStructure(struc_list)

        self.projector = self.model_space.get_projection(self.data_space)

        if kwargs.get("initial_posterior", None):
            raise NotImplementedError()
            #self.post_init = self._get_posterior_data(kwargs["initial_posterior"])
        else:
            self.post_init = None

    def model_to_data(self, tensor, pv_sum=True):
        """
        Convert model space data into source data space
        """
        return self.projector[0](tensor, pv_sum)

    def data_to_model(self, tensor, pv_sum=True):
        """
        Convert source data space data into model space
        """
        return self.projector[1](tensor, pv_sum)

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
        if cov.shape[0] != self.model_space.size or mean.shape[0] != self.model_space.size:
            raise ValueError("Posterior data has %i nodes - inconsistent with model containing %i nodes" % (cov.shape[0], self.model_space.size))

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
                raise ValueError("Posterior input has %i volumes - not consistent with upper triangle of square matrix" % nvols)
            self.log.info("Posterior image contains %i parameters", n_params)
            
            cov = np.zeros((self.model_space.size, n_params, n_params), dtype=NP_DTYPE)
            mean = np.zeros((self.model_space.size, n_params), dtype=NP_DTYPE)
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
                raise ValueError("Posterior input file - last volume does not contain 1.0")

            self.log.info("Posterior mean shape: %s, cov shape: %s", mean.shape, cov.shape)
            return mean, cov

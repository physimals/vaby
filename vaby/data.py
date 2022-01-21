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

def get_data_structure(**kwargs):
    """
    Factory method to return an instance of DataStructure
    
    Either the class name may be direcly specified using the 'type' kwarg, or
    a data filename/object may be provided via the 'data' kwarg
    """
    if "type" in kwargs:
        classname = kwargs.pop("type")
        cls = globals().get(classname, None)
        if cls is None:
            raise ValueError("No such data structure: %s" % classname)
        return cls(**kwargs)
    elif "data" in kwargs:
        data = kwargs["data"]
        if isinstance(data, six.string_types):
            data = nib.load(data)

        if isinstance(data, np.ndarray):
            return Volume(data, **kwargs)
        elif isinstance(data, nib.Nifti1Image):
            return Volume(nii=data, **kwargs)
        #elif isinstance(data, nib.GiftiImage):
        #    return SimpleSurface(gii=data, **kwargs)
        else:
            raise ValueError("Unable to create model structure from data type %s" % type(data))
    else:
        raise ValueError("Unable to create model structure - neither data nor structure type given")

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
        Get the projections between this space and the data acquisition space.

        This method should only be called once for a given data space - hence any
        expensive calculation of the projector can be done in the method (but ideally
        not in the returned callables)

        :param data_space: Volume defining the acquisition data space
        :return: Tuple of callables (model_to_data, data_to_model) which may be called
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
    def __init__(self, vol_data=None, mask=None, pv_vol=None, nii=None, voxel_sizes=None, file_ext=".nii.gz", **kwargs):
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
            def _model2data(tensor, pv_sum=False):
                return tensor * self.pvs

            def _data2model(tensor, pv_sum=False):
                # FIXME how is this defined?
                return tensor

            return _model2data, _data2model
        except:
            raise NotImplementedError("Projection between different volume spaces")

class CorticalSurface(DataStructure):
    """
    A cortical surface data structure

    This consists two surfaces, inner (white matter) and outer (pial). These surfaces
    contain corresponding tri-meshes that define triangular voxels each of which represents
    a node in the structure.

    Multiple cortical surface structures can be combined in a CompositeStructure, e.g.
    right and left hemispheres, potentially additionally with volumetric subcortical white matter
    """

    def __init__(self, white, pial, file_ext=".gii", **kwargs):
        """
        :param left: Tuple of (LWS, LPS) where LWS and LPS are filenames or nibabel.Gifti1Image instances
                     containing the white and pial surfaces for the left hemisphere
        :param left: Tuple of (LWS, LPS) where LWS and LPS are filenames or nibabel.Gifti1Image instances
                     containing the white and pial surfaces for the right hemisphere
        """
        DataStructure.__init__(self, file_ext=file_ext, **kwargs)
        if toblerone is None:
            raise RuntimeError("Toblerone not installed - cannot create cortical surface structure")
        from toblerone.classes import Hemisphere, Surface

        if self.name not in ("L", "R"):
            raise ValueError("For now, the names of cortical surfaces must be either 'L' or 'R'")

        self.hemisphere = Hemisphere(Surface(white, self.name + "WS"), Surface(pial, self.name + "PS"), self.name)
        self.size = self.hemisphere.n_points
        self.projector = kwargs.get("projector", None)
        if isinstance(self.projector, str):
            self.log.info(f"Loading projector from {self.projector}")
            self.projector = toblerone.Projector.load(self.projector)

        self.adj_matrix = self.hemisphere.adjacency_matrix()
        self.laplacian = self.hemisphere.mesh_laplacian()

    def get_projection(self, data_space):
        if self.projector is None:
            self.log.info("Generating projector - this may take some time...")
            projector = toblerone.Projector(self.hemisphere, regtricks.ImageSpace(data_space.srcdata.nii), factor=10, cores=8)
            projector.save("vaby_proj.h5")
            self.log.info("Projector generated")
        else:
            if self.projector.spc != regtricks.ImageSpace(data_space.srcdata.nii):
                raise ValueError("Projector supplied is not defined on same image space as acquisition data")
            projector = self.projector

        proj_matrices = {
            "n2v" : projector.surf2vol_matrix(edge_scale=True).astype(NP_DTYPE),
            "n2v_noedge" : projector.surf2vol_matrix(edge_scale=False).astype(NP_DTYPE),
            "v2n" : projector.vol2surf_matrix(edge_scale=True).astype(NP_DTYPE),
            "v2n_noedge" : projector.vol2surf_matrix(edge_scale=False).astype(NP_DTYPE),
        }

        if data_space.size != proj_matrices["n2v"].shape[0]:
            raise ValueError('Acquisition data size does not match projector')

        # Knock out voxels from projection matrices that are not in the mask
        # and convert to sparse tensors
        proj_tensors = {}
        vox_inds = np.flatnonzero(data_space.mask)
        for name, mat in proj_matrices.items():
            if name.startswith("n2v"):
                masked_mat = mat.tocsr()[vox_inds, :].tocoo()
            else:
                masked_mat = mat.tocsr()[:, vox_inds].tocoo()
            proj_tensors[name] = tf.SparseTensor(
                indices=np.array([masked_mat.row, masked_mat.col]).T,
                values=masked_mat.data,
                dense_shape=masked_mat.shape,
            )

        def _surf2vol(tensor, pv_sum=False):
            is_vector = tf.rank(tensor) < 2
            if is_vector:
                tensor = tf.expand_dims(tensor, -1)

            if pv_sum:
                proj = proj_tensors["n2v"]
            else:
                proj = proj_tensors["n2v_noedge"]

            print("_surf2vol: %s %s" % (proj.shape, tensor.shape))
            ret = tf.sparse.sparse_dense_matmul(proj, tensor)
            if is_vector:
                ret = tf.reshape(ret, [-1])

            return ret

        def _vol2surf(tensor, pv_sum=False):
            is_vector = tf.rank(tensor) < 2
            if is_vector:
                tensor = tf.expand_dims(tensor, -1)

            if pv_sum:
                proj = proj_tensors["v2n"]
            else:
                proj = proj_tensors["v2n_noedge"]

            ret = tf.sparse.sparse_dense_matmul(proj, tensor)
            if is_vector:
                ret = tf.reshape(ret, [-1])

            return ret

        return (_surf2vol, _vol2surf)

    def load_data(self, data, **kwargs):
        raise NotImplementedError()

    def nibabel_image(self, data):
        if data.shape[0] != self.size:
            raise ValueError("Incorrect data shape for surface")

        meta = {'Description': f'{self.name} cortex parameter estimates produced by vaby'}
        arr = nib.gifti.GiftiDataArray(
            data.astype(np.float32),
            intent='NIFTI_INTENT_ESTIMATE',
            datatype='NIFTI_TYPE_FLOAT32',
            meta=meta)
        return nib.GiftiImage(darrays=[arr])

class CompositeStructure(DataStructure):
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
        start_idx = 0
        for p in self.parts:
            self.slices.append(slice(start_idx, start_idx+p.size))
            start_idx += p.size
        self.adj_matrix = sparse.block_diag([p.adj_matrix for p in self.parts]).astype(NP_DTYPE)
        self.laplacian = sparse.block_diag([p.laplacian for p in self.parts]).astype(NP_DTYPE)

    def get_projection(self, data_space):
        projectors = [p.get_projection(data_space) for p in self.parts]

        def model2data(tensor, pv_sum=False):
            tensor_data = []
            for proj, slc in zip(projectors, self.slices):
                tensor_data.append(proj[0](tensor[slc, ...], pv_sum)) # [V, T]
            return sum(tensor_data) # [V, T]

        def data2model(tensor, pv_sum=False):
            tensor_model = []
            for proj in projectors:
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
        self.data_space = get_data_structure(data=data, name="acquisition", **kwargs)

        ### Model space
        model_structures = kwargs.get("model_structures", None)
        if model_structures is None:
            self.log.info(" - Model space is same as acquisition space")
            self.model_space = self.data_space
        else:
            struc_list = []
            for struc in model_structures:
                if isinstance(struc, DataStructure):
                    self.log.info("Found model space structure: %s" % struc.name)
                    struc_list.append(struc)
                else:
                    self.log.info("Creating model space structure")
                    struc_list.append(get_data_structure(**struc))
            self.model_space = CompositeStructure(struc_list)

        self.projector = self.model_space.get_projection(self.data_space)

        if kwargs.get("initial_posterior", None):
            raise NotImplementedError()
            #self.post_init = self._get_posterior_data(kwargs["initial_posterior"])
        else:
            self.post_init = None

    def model_to_data(self, tensor, pv_sum=False):
        """
        Convert model space data into source data space
        """
        return self.projector[0](tensor, pv_sum)

    def data_to_model(self, tensor, pv_sum=False):
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

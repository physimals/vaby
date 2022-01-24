"""
VABY - Basic structure models
"""
import os

from scipy import sparse
import tensorflow as tf

from ..utils import LogBase, NP_DTYPE

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

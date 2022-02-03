"""
VABY - Base class for structure models
"""
import os

from ..utils import LogBase

class DataStructure(LogBase):
    """"
    A data structure
    
    Each data structure has the following attributes:
     - size: Total number of independent nodes in the structure (e.g. surface nodes, unmasked voxels)
     - adj_matrix: Scipy sparse matrix dimension (size, size) containing 1 where nodes are regarded as connected, 0 otherwise
     - laplacian: Scipy sparse matrix dimension (size, size) containing Laplacian for spatial smoothing
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
                 passing a 2D tensor in data/model space and returning a 2D tensor 
                 in model/data space. Handling of 1D tensors (vectors) and 3D timeseries
                 tensors is handled in the data model.
        """
        raise NotImplementedError()

    def check_compatible(self, struc):
        """
        Check another supplied data structure represents the same underlying data space
        """
        if not isinstance(self, type(struc)):
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
        """
        Save data defined in this data space
        
        :param data: Data array defined in this data space, i.e. a 1D Numpy array
                     or 2D array if multi-volume data
        :param name: Name for saved data
        :param outdir: Output directory
        """
        self.nibabel_image(data).to_filename(os.path.join(outdir, name) + self.file_ext)

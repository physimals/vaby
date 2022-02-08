"""
VABY - Class defining the structure of model space
"""
from scipy import sparse
import tensorflow as tf

from .base import DataStructure
from ..utils import NP_DTYPE

class ModelSpace(DataStructure):
    """
    Model space is defined by a sequence of DataStructure forming disjoint parts

    The adjacency and laplacian matrix are block-diagonal copies of the source structures with
    zeros elsewhere (i.e. the separate structures are not considered to be connected).

    Attributes:
     - parts: Sequence of DataStructure instances
     - num_strucs: Number of structures
     - slices: Slice object for each part to extract model nodes relevant to that part
    """

    def __init__(self, parts, **kwargs):
        """
        :param parts: Sequence of DataStructure instances
        """
        DataStructure.__init__(self, name="modelspace")
        self.parts = parts
        self.size = sum([p.size for p in parts])
        self.num_strucs = len(self.parts)
        self.slices = []
        start_idx = 0
        for p in self.parts:
            self.slices.append(slice(start_idx, start_idx+p.size))
            start_idx += p.size
        self.adj_matrix = sparse.block_diag([p.adj_matrix for p in self.parts]).astype(NP_DTYPE)
        self.laplacian = sparse.block_diag([p.laplacian for p in self.parts]).astype(NP_DTYPE)

    def get_projection(self, data_space):
        projectors = [p.get_projection(data_space) for p in self.parts]

        def model2data(tensor):
            tensor_data = []
            for proj, slc in zip(projectors, self.slices):
                tensor_data.append(proj[0](tensor[slc, ...])) # [V, T]
            return sum(tensor_data) # [V, T]

        def data2model(tensor):
            tensor_model = []
            for proj in projectors:
                tensor_model.append(proj[1](tensor)) # [w, T]
            return tf.concat(tensor_model, axis=0) # [W, T]

        return model2data, data2model

    def split(self, tensor, axis):
        """
        Split a tensor into sub-structure parts

        :param tensor: Tensor whose first dimension is nodes
        :return: Mapping of structure name : tensor
        """
        slices = [slice(None)] * int(tf.rank(tensor))
        ret = {}
        for struc, slc in zip(self.parts, self.slices):
            slices[axis] = slc
            ret[struc.name] = tensor[slices]
        return ret

    def save_data(self, data, name, outdir="."):
        for struct, slc in zip(self.parts, self.slices):
            struct.save_data(data[slc, ...], name + "_" + struct.name, outdir)

"""
VABY - Class defining the structure of model space
"""
from scipy import sparse
import numpy as np
import tensorflow as tf

from .base import DataStructure
from ..utils import NP_DTYPE, TF_DTYPE

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

    @tf.function
    def model2data(self, tensor, data_space):
        tensor_data = tf.TensorArray(TF_DTYPE, size=self.num_strucs)
        for idx, part in enumerate(self.parts):
            pt = part.model2data(tensor[self.slices[idx], ...], data_space)
            tensor_data = tensor_data.write(idx, pt) # [V, T]
        return tf.reduce_sum(tensor_data.stack(), axis=0) # [V, T]

    @tf.function
    def data2model(self, tensor, data_space):
        tensor_model = tf.TensorArray(TF_DTYPE, size=self.num_strucs, infer_shape=False)
        for idx, part in enumerate(self.parts):
            pt = part.data2model(tensor, data_space)
            tensor_model = tensor_model.write(idx, pt) # [w, T]
        return tensor_model.concat() # [W, T]

    def split(self, tensor, axis=0):
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

    def load_data(self, fname, **kwargs):
        # FIXME this isn't going to work with multiple structures because of filenames
        full_data = None
        for struct, slc in zip(self.parts, self.slices):
            part_data = struct.load_data(fname, **kwargs)
            if full_data is None:
                full_data = np.zeros((self.size, part_data.shape[1]), dtype=NP_DTYPE)
            full_data[slc, :] = part_data
        return full_data

    def save_data(self, data, name, **kwargs):
        for struct, slc in zip(self.parts, self.slices):
            struct.save_data(data[slc, ...], name + "_" + struct.name, **kwargs)

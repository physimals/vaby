"""
VABY - General utility functions
"""
import logging

import numpy as np
import tensorflow as tf

NP_DTYPE = np.float32
TF_DTYPE = tf.float32

def ValueList(value_type):
    """
    Class used with argparse for options which can be given as a space or comma separated list
    """
    def _call(value):
        return [value_type(v) for v in value.replace(",", " ").split()]
    return _call

class LogBase(object):
    """
    Base class that provides a named log
    """
    def __init__(self, **kwargs):
        self.log = logging.getLogger(type(self).__name__)

def scipy_to_tf_sparse(scipy_sparse):
    """Converts a scipy sparse matrix to TF representation"""

    spmat = scipy_sparse.tocoo()
    return tf.SparseTensor(
        indices=np.array([
            spmat.row, spmat.col]).T,
        values=spmat.data.astype(NP_DTYPE), 
        dense_shape=spmat.shape, 
    )
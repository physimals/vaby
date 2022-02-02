"""
VABY - Data model
"""
import math
import collections

import numpy as np
import tensorflow as tf

from .utils import LogBase, NP_DTYPE
from .structures import get_data_structure, DataStructure, CompositeStructure

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
    MODEL_SPACE = "model space"
    DATA_SPACE = "data space"

    def __init__(self, data, **kwargs):
        LogBase.__init__(self)

        ### Acquisition data space
        self.log.info("Creating acquisition data structure")
        self.data_space = get_data_structure(data=data, name="acquisition", **kwargs)

        ### Model space
        self.log.info("Creating model inference data structure")
        model_structures = kwargs.get("model_structures", None)
        if model_structures is None:
            self.log.info(" - Model structure is same as acquisition structure")
            self.model_space = self.data_space
        else:
            struc_list = []
            for struc in model_structures:
                if isinstance(struc, DataStructure):
                    self.log.info(" - Found model structure: %s" % struc.name)
                    struc_list.append(struc)
                else:
                    self.log.info(" - Creating model structure")
                    struc_list.append(get_data_structure(**struc))
            self.model_space = CompositeStructure(struc_list)

        self.model2data, self.data2model = self.model_space.get_projection(self.data_space)

        if kwargs.get("initial_posterior", None):
            raise NotImplementedError()
            #self.post_init = self._get_posterior_data(kwargs["initial_posterior"])
        else:
            self.post_init = None

    def _change_space(self, projector, tensor, pv_sum=False):
        """
        Convert model space data into source data space
        """
        vector_input = tf.rank(tensor) < 2
        timeseries_input = tf.rank(tensor) == 3
        if vector_input:
            tensor = tf.expand_dims(tensor, -1)
        
        if timeseries_input:
            ret = []
            for t in range(tensor.shape[2]):
                ret.append(projector(tensor[..., t], pv_sum))
            ret = tf.stack(ret, axis=2)
        else:
            ret = projector(tensor, pv_sum)

        if vector_input:
            ret = tf.reshape(ret, [-1])

        return ret

    def model_to_data(self, tensor, pv_sum=False):
        """
        Convert model space data into source data space
        """
        return self._change_space(self.model2data, tensor, pv_sum)

    def data_to_model(self, tensor, pv_sum=False):
        """
        Convert source data space data into model space
        """
        return self._change_space(self.data2model, tensor, pv_sum)

    def save_model_data(self, data, name, output, save_model=True, save_native=False, **kwargs):
        if save_model:
            self.model_space.save_data(data, name, output)
        if save_native:
            self.data_space.save_data(self.model_to_data(data), name, output)

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

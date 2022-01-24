"""
VABY - Data model
"""
import math
import collections

import numpy as np

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

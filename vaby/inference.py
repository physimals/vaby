"""
VABY - Base class for implementing a VB inference method
"""
import os

import numpy as np

from .utils import LogBase, NP_DTYPE, makedirs

class InferenceMethod(LogBase):
    """
    Base class for infererence method (analytic VB, stochastic VB)

    :ivar data_model: Model instance to be fitted to some data
    :ivar fwd_model: Model instance to be fitted to some data
    """

    def __init__(self, data_model, fwd_model, **kwargs):
        LogBase.__init__(self)

        self.data_model = data_model
        self.fwd_model = fwd_model

        # Make sure time points has dimension for voxelwise variation even if it is the same for all voxels
        self.tpts = self.fwd_model.tpts().astype(NP_DTYPE)
        if self.tpts.ndim == 1:
            self.tpts = self.tpts.reshape(1, -1)

        # Consistency check
        if self.tpts.shape[0] > 1 and self.tpts.shape[0] != self.n_nodes:
            raise ValueError("Time points has %i nodes, but data has %i" % (self.tpts.shape[0], self.n_nodes))
        if self.tpts.shape[1] != self.n_tpts:
            raise ValueError("Time points has length %i, but data has %i volumes" % (self.tpts.shape[1], self.n_tpts))

        # What type of average to log
        log_avg = kwargs.get("log_avg", "mean").lower().strip()
        if log_avg == "mean":
            self.log_avg = np.mean
        elif log_avg == "median":
            self.log_avg = np.median
        else:
            raise ValueError("Unknown log average: %s" % log_avg)
        
    @property
    def params(self):
        """
        Model parameters
        """
        return self.fwd_model.params

    @property
    def n_params(self):
        """
        Number of model paramters
        """
        return len(self.params)

    @property
    def n_all_params(self):
        """
        Number of model and noise paramters
        """
        return self.n_params + 1

    @property
    def data(self):
        """
        Source data we are fitting to with shape [V, T]
        """
        return self.data_model.data_space.srcdata.flat

    @property
    def n_voxels(self):
        """
        Number of data voxels that will be used for estimation 
        """
        return self.data_model.data_space.size
 
    @property
    def n_tpts(self):
        """
        Number of data voxels that will be used for estimation 
        """
        return self.data_model.data_space.srcdata.n_tpts

    @property
    def n_nodes(self):
        """
        Number of positions at which *model* parameters will be estimated
        """
        return self.data_model.model_space.size

    def save(self, state, runtime=None, **kwargs):
        """
        Save output to directory and/or in-memory dictionary

        :param runtime: Runtime in seconds

        Keyword arguments can include:
            :param outdir: Output directory
            :param outdict: Dictionary to store output data (images in Nifti/Gifti format)
        """
        outdir = kwargs.get("outdir", None)
        outdict = kwargs.get("outdict", None)
        if outdir:
            makedirs(outdir, exist_ok=True)
        
        # Write out parameter mean and variance images
        # FIXME pv_scale for variance / std?
        mean = state["model_mean"]
        variances = state["model_var"]
        for idx, param in enumerate(self.params):
            if kwargs.get("save_mean", False):
                self.data_model.save_model_data(mean[idx], "mean_%s" % param.name, pv_scale=param.pv_scale, **kwargs)
            if kwargs.get("save_var", False):
                self.data_model.save_model_data(variances[idx], "var_%s" % param.name, **kwargs)
            if kwargs.get("save_std", False):
                self.data_model.save_model_data(np.sqrt(variances[idx]), "std_%s" % param.name, **kwargs)

        if kwargs.get("save_noise", False):
            if kwargs.get("save_mean", False):
                self.data_model.data_space.save_data(state["noise_mean"], "mean_noise", **kwargs)
            if kwargs.get("save_var", False):
                self.data_model.data_space.save_data(state["noise_var"], "var_noise", **kwargs)
            if kwargs.get("save_std", False):
                self.data_model.data_space.save_data(np.sqrt(state["noise_var"]), "std_noise", **kwargs)

        # Write out modelfit
        if kwargs.get("save_model_fit", False):
            self.data_model.save_model_data(state["modelfit"], "modelfit", pv_scale=True, **kwargs)

        # Total model space partial volumes in data space
        if kwargs.get("save_total_pv", False):
            self.data_model.data_space.save_data(self.data_model.dataspace_pvs, "total_pv", **kwargs)

        # Write out voxelwise free energy (and history if required)
        if kwargs.get("save_free_energy", False):
            self.data_model.model_space.save_data(self.free_energy_vox, "free_energy", **kwargs)
        if kwargs.get("save_free_energy_history", False):
            self.data_model.model_space.save_data(self.history["free_energy_vox"], "free_energy_history", **kwargs)

        # Write out voxelwise parameter history
        if kwargs.get("save_param_history", False):
            for idx, param in enumerate(self.params):
                self.data_model.model_space.save_data(self.history["model_mean"][idx], "mean_%s_history" % param.name, pv_scale=param.pv_scale, **kwargs)

        # Write out posterior - not including noise since defined in different spaces
        if kwargs.get("save_posterior", False):
            post_data = self.data_model.encode_posterior(state["post_mean"], state["post_cov"])
            self.log.debug("Posterior data shape: %s", post_data.shape)
            self.data_model.model_space.save_data(post_data, "posterior", **kwargs)
            post_data = self.data_model.encode_posterior(state["noise_mean"], state["noise_var"])
            self.data_model.model_space.save_data(post_data, "noise_posterior", **kwargs)

        # Write out runtime
        if runtime and kwargs.get("save_runtime", False):
            if outdir:
                with open(os.path.join(outdir, "runtime"), "w") as runtime_f:
                    runtime_f.write("%f\n" % runtime)
            if outdict:
                outdict["runtime"] = runtime

        # Write out input data
        if kwargs.get("save_input_data", False):
            self.data_model.data_space.save_data(self.data_model.data_space.srcdata.flat, "input_data", **kwargs)

        if outdir:
            self.log.info("Output written to: %s", outdir)

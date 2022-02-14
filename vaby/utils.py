"""
VABY - General utility functions
"""
import logging
import logging.config
import os

import numpy as np
import tensorflow as tf

# Standard data types for Numpy and Tensorflow. We use single
# precision by default as memory demands can be quite high
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
        self.tpts = self.fwd_model.tpts()
        if self.tpts.ndim == 1:
            self.tpts = self.tpts.reshape(1, -1)
        self.tpts = self.tpts.astype(NP_DTYPE) # FIXME is this necessary

        # Consistency check
        if self.tpts.shape[0] > 1 and self.tpts.shape[0] != self.n_nodes:
            raise ValueError("Time points has %i nodes, but data has %i" % (self.tpts.shape[0], self.n_nodes))
        if self.tpts.shape[1] != self.n_tpts:
            raise ValueError("Time points has length %i, but data has %i volumes" % (self.tpts.shape[1], self.n_tpts))

    @property
    def data(self):
        """
        Source data we are fitting to with shape [V, T]
        """
        return self.data_model.data_space.srcdata.flat

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
    def n_nodes(self):
        """
        Number of positions at which *model* parameters will be estimated
        """
        return self.data_model.model_space.size

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

    def finalize(self):
        """
        Make output tensors into Numpy arrays
        """
        for attr in ("model_mean", "model_var", "noise_mean", "noise_var", "modelfit"):
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).numpy())

    def save(self, outdir, rt=None, **kwargs):
        """
        Save output to directory

        :param outdir: Output directory
        :param rt: Runtime in seconds
        """
        makedirs(outdir, exist_ok=True)
        
        # Write out parameter mean and variance images
        # FIXME pv_scale for variance / std?
        mean = self.model_mean
        variances = self.model_var
        for idx, param in enumerate(self.params):
            if kwargs.get("save_mean", False):
                self.data_model.save_model_data(mean[idx], "mean_%s" % param.name, outdir, pv_scale=param.pv_scale, **kwargs)
            if kwargs.get("save_var", False):
                self.data_model.save_model_data(variances[idx], "var_%s" % param.name, outdir, **kwargs)
            if kwargs.get("save_std", False):
                self.data_model.save_model_data(np.sqrt(variances[idx]), "std_%s" % param.name, outdir, **kwargs)

        if kwargs.get("save_noise", False):
            if kwargs.get("save_mean", False):
                self.data_model.data_space.save_data(self.noise_mean, "mean_noise", outdir)
            if kwargs.get("save_var", False):
                self.data_model.data_space.save_data(self.noise_var, "var_noise", outdir)
            if kwargs.get("save_std", False):
                self.data_model.data_space.save_data(np.sqrt(self.noise_var), "std_noise", outdir)

        # Write out modelfit
        if kwargs.get("save_model_fit", False):
            self.data_model.save_model_data(self.modelfit, "modelfit", outdir, pv_scale=True, **kwargs)

        # Total model space partial volumes in data space
        if kwargs.get("save_total_pv", False):
            self.data_model.data_space.save_data(self.data_model.dataspace_pvs, "total_pv", outdir)

        # Write out voxelwise free energy (and history if required)
        if kwargs.get("save_free_energy", False):
            self.data_model.model_space.save_data(self.free_energy_vox, "free_energy", outdir)
        if kwargs.get("save_free_energy_history", False):
            self.data_model.model_space.save_data(self.history["free_energy_vox"], "free_energy_history", outdir)

        # Write out voxelwise parameter history
        if kwargs.get("save_param_history", False):
            for idx, param in enumerate(self.params):
                self.data_model.model_space.save_data(self.history["model_mean"][idx], "mean_%s_history" % param.name, outdir, pv_scale=param.pv_scale, **kwargs)

        # Write out posterior
        if kwargs.get("save_post", False):
            post_data = self.data_model.encode_posterior(self.all_mean, self.all_cov)
            self.log.debug("Posterior data shape: %s", post_data.shape)
            self.data_model.model_space.save_data(post_data, "posterior", outdir)

        # Write out runtime
        if rt and kwargs.get("save_runtime", False):
            with open(os.path.join(outdir, "runtime"), "w") as runtime_f:
                runtime_f.write("%f\n" % rt)

        # Write out input data
        if kwargs.get("save_input_data", False):
            self.data_model.data_space.save_data(self.data_model.data_space.srcdata.flat, "input_data", outdir)

        self.log.info("Output written to: %s", outdir)

def makedirs(data_vol, exist_ok=False):
    """
    Make directories, optionally ignoring them if they already exist
    """
    try:
        os.makedirs(data_vol)
    except OSError as exc:
        import errno
        if not exist_ok or exc.errno != errno.EEXIST:
            raise

def setup_logging(outdir=".", **kwargs):
    """
    Set the log level, formatters and output streams for the logging output

    By default this goes to <outdir>/logfile at level INFO
    """
    # First we clear all loggers from previous runs
    for logger_name in list(logging.Logger.manager.loggerDict.keys()) + ['']:
        logger = logging.getLogger(logger_name)
        logger.handlers = []

    if kwargs.get("log_config", None):
        # User can supply a logging config file which overrides everything else
        logging.config.fileConfig(kwargs["log_config"])
    else:
        # Set log level on the root logger to allow for the possibility of 
        # debug logging on individual loggers
        level = kwargs.get("log_level", "info")
        if not level:
            level = "info"
        level = getattr(logging, level.upper(), logging.INFO)
        logging.getLogger().setLevel(level)

        if outdir and kwargs.get("save_log", False):
            # Send the log to an output logfile
            makedirs(outdir, True)
            logfile = os.path.join(outdir, "logfile")
            logging.basicConfig(filename=logfile, filemode="w", level=level)

        if kwargs.get("log_stream", None) is not None:
            # Can also supply a stream to send log output to as well (e.g. sys.stdout)
            extra_handler = logging.StreamHandler(kwargs["log_stream"])
            extra_handler.setFormatter(logging.Formatter('%(levelname)s : %(message)s'))
            logging.getLogger().addHandler(extra_handler)

def runtime(runnable, *args, **kwargs):
    """
    Record how long it took to run something

    :return: Tuple of runtime (s), normal return value
    """
    import time
    start_time = time.time()
    ret = runnable(*args, **kwargs)
    end_time = time.time()
    if ret is not None:
        return (end_time - start_time), ret
    else:
        return (end_time - start_time)

def scipy_to_tf_sparse(scipy_sparse):
    """Converts a scipy sparse matrix to TF representation"""

    spmat = scipy_sparse.tocoo()
    return tf.SparseTensor(
        indices=np.array([
            spmat.row, spmat.col]).T,
        values=spmat.data.astype(NP_DTYPE), 
        dense_shape=spmat.shape, 
    )
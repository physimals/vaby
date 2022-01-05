"""
VABY - General utility functions
"""
import argparse
import logging
import logging.config
import os
import re
import sys

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

class ArgumentParser(argparse.ArgumentParser):
    """
    ArgumentParser for VABY options
    """

    PARAM_OPTIONS = {
        "dist" : str,
        "prior_mean" : float,
        "prior_var" : float,
        "prior_type" : str,
        "post_mean" : float,
        "post_var" : float,
    }

    def __init__(self, **kwargs):
        argparse.ArgumentParser.__init__(self, prog="vaby", add_help=False, **kwargs)

        group = self.add_argument_group("Main Options")
        group.add_argument("--data", help="Timeseries input data")
        group.add_argument("--mask", help="Optional voxel mask")
        group.add_argument("--post-init", dest="post_init_fname",
                         help="Initialize posterior from data file saved using --output-post")
        group.add_argument("--model", dest="model_name", help="Model name")
        group.add_argument("--method", help="Inference method", choices=["avb", "svb"])
        group.add_argument("--output", help="Output folder name", default="vaby_out")
        group.add_argument("--log-level", help="Logging level", default="info")
        group.add_argument("--log-config",
                         help="Optional logging configuration file, overrides --log-level")
        group.add_argument("--help", action="store_true", default=False, help="Display help")
        
        group = self.add_argument_group("Output options")
        group.add_argument("--save-var",
                         help="Save parameter variance",
                         action="store_true", default=False)
        group.add_argument("--save-std",
                         help="Save parameter standard deviation",
                         action="store_true", default=False)
        group.add_argument("--save-param-history",
                         help="Save parameter history by epoch",
                         action="store_true", default=False)
        group.add_argument("--save-noise",
                         help="Save noise parameter",
                         action="store_true", default=False)
        group.add_argument("--save-free-energy",
                         help="Save free energy",
                         action="store_true", default=False)
        group.add_argument("--save-free-energy-history",
                         help="Save free energy history by iteration",
                         action="store_true", default=False)
        group.add_argument("--save-model-fit",
                         help="Save model fit",
                         action="store_true", default=False)
        group.add_argument("--save-post", "--save-posterior",
                         help="Save full posterior distribution",
                         action="store_true", default=False)

        group = self.add_argument_group("Options for method=avb")
        group.add_argument("--max-iterations", help="Max number of iterations", type=int, default=20)
        group.add_argument("--use-adam", action="store_true", default=False,
                         help="Directly maximise free energy using Adam optimizer")
        group.add_argument("--init-leastsq",
                         help="Do an initial least-squares fit before optimizing full free energy cost",
                         action="store_true", default=False)

        group = self.add_argument_group("Options for method=svb")
        group.add_argument("--epochs",
                         help="Number of training epochs",
                         type=int, default=100)
        group.add_argument("--learning-rate", "--lr",
                         help="Initial learning rate",
                         type=float, default=0.1)
        group.add_argument("--batch-size", "--bs",
                         help="Batch size. If not specified data will not be processed in batches",
                         type=int)
        group.add_argument("--sample-size", "--ss",
                         help="Sample size for drawing samples from posterior",
                         type=int, default=20)
        group.add_argument("--max-trials",
                         help="Number of epochs without improvement in the cost before reducing the learning rate",
                         type=int, default=50)
        group.add_argument("--no-covar", 
                         dest="infer_covar",
                         help="Do not infer a full covariance matrix",
                         action="store_false", default=True)
        group.add_argument("--force-num-latent-loss",
                         help="Force numerical calculation of the latent loss function",
                         action="store_true", default=False)
        group.add_argument("--allow-nan",
                         dest="suppress_nan",
                         help="Do not suppress NaN values in posterior",
                         action="store_false", default=True)      
   
    def parse_args(self, argv=None, namespace=None):
        # Parse built-in fixed options but skip unrecognized options as they may be
        #  model-specific option or parameter-specific optionss.
        options, extras = argparse.ArgumentParser.parse_known_args(self, argv, namespace)

        # Now we should know the model, so we can add it's options and parse again
        if options.model_name:
            group = self.add_argument_group("%s model options" % options.model_name.upper())
            for model_option in get_model_class(options.model_name).OPTIONS:
                kwargs = {
                    "help" : model_option.desc,
                    "type" : model_option.type,
                    "default" : model_option.default,
                }
                if model_option.units:
                    kwargs["help"] += " (%s)" % model_option.units
                if model_option.default is not None:
                    kwargs["help"] += " - default %s" % str(model_option.default)
                else:
                    kwargs["help"] += " - no default"

                if model_option.type == bool:
                    kwargs["action"] = "store_true"
                    kwargs.pop("type")
                group.add_argument(*model_option.clargs, **kwargs)
            options, extras = argparse.ArgumentParser.parse_known_args(self, argv, namespace)

        if options.help:
            self.print_help()
            sys.exit(0)

        # Support arguments of the form --param-<param name>-<param option>
        # (e.g. --param-ftiss-mean=4.4 --param-delttiss-prior-type M)
        param_arg = re.compile(r"--param-(\w+)-([\w-]+)")
        options.param_overrides = {}
        consume_next_arg = None
        for arg in extras:
            if consume_next_arg:
                if arg.startswith("-"):
                    raise ValueError("Value for parameter option cannot start with - : %s" % arg)
                param, thing = consume_next_arg
                options.param_overrides[param][thing] = self.PARAM_OPTIONS[thing](arg)
                consume_next_arg = None
            else:
                kv = arg.split("=", 1)
                key = kv[0]
                match = param_arg.match(key)
                if match:
                    param, thing = match.group(1), match.group(2)

                    # Use underscore for compatibility with kwargs
                    thing = thing.replace("-", "_")
                    if thing not in self.PARAM_OPTIONS:
                        raise ValueError("Unrecognized parameter option: %s" % thing)

                    if param not in options.param_overrides:
                        options.param_overrides[param] = {}
                    if len(kv) == 2:
                        options.param_overrides[param][thing] = self.PARAM_OPTIONS[thing](kv[1])
                    else:
                        consume_next_arg = (param, thing)
                else:
                    raise ValueError("Unrecognized argument: %s" % arg)

        return options

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
        self.tpts = self.tpts.astype(np.float32)

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

    def save(self, output, rt, **kwargs):
        """
        Save output to directory

        FIXME Currently tailored for AVB - needs to be generic
        """
        makedirs(output, exist_ok=True)
        params = [p.name for p in self.params]
        
        # Write out parameter mean and variance images
        mean = self.model_mean
        variances = self.model_var
        for idx, param in enumerate(params):
            if kwargs.get("save_mean", False):
                self.data_model.model_space.save_data(mean[idx], "mean_%s" % param, output)
            if kwargs.get("save_var", False):
                self.data_model.model_space.save_data(variances[idx], "var_%s" % param, output)
            if kwargs.get("save_std", False):
                self.data_model.model_space.save_data(np.sqrt(variances[idx]), "std_%s" % param, output)

        if kwargs.get("save_noise", False):
            if kwargs.get("save_mean", False):
                self.data_model.data_space.save_data(self.noise_mean, "mean_noise", output)
            if kwargs.get("save_var", False):
                self.data_model.data_space.save_data(self.noise_var, "var_noise", output)
            if kwargs.get("save_std", False):
                self.data_model.data_space.save_data(np.sqrt(self.noise_var), "std_noise", output)

        # Write out voxelwise free energy (and history if required)
        if kwargs.get("save_free_energy", False):
            self.data_model.model_space.save_data(self.free_energy_vox, "free_energy", output)
        if kwargs.get("save_free_energy_history", False):
            self.data_model.model_space.save_data(self.history["free_energy_vox"], "free_energy_history", output)

        # Write out voxelwise parameter history
        if kwargs.get("save_param_history", False):
            for idx, param in enumerate(params):
                self.data_model.model_space.save_data(self.history["model_mean"][idx], "mean_%s_history" % param, output)

        # Write out modelfit
        if kwargs.get("save_model_fit", False):
            self.data_model.model_space.save_data(self.modelfit, "modelfit", output)

        # Write out posterior
        if kwargs.get("save_post", False):
            post_data = self.data_model.encode_posterior(self.all_mean, self.all_cov)
            self.log.debug("Posterior data shape: %s", post_data.shape)
            self.data_model.model_space.save_data(post_data, "posterior", output)

        # Write out runtime
        if kwargs.get("save_runtime", False):
            with open(os.path.join(output, "runtime"), "w") as runtime_f:
                runtime_f.write("%f\n" % rt)

        # Write out input data
        if kwargs.get("save_input_data", False):
            self.data_model.data_space.save_data(self.data_model.data_space.srcdata.flat, "input_data", output)

        self.log.info("Output written to: %s", output)

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

        if kwargs.get("save_log", False):
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
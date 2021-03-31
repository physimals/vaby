"""
Base class for a forward model whose parameters are to be fitted
"""
import pkg_resources
import collections

import numpy as np

import tensorflow as tf
   
from .utils import LogBase, ValueList

MODELS = {
}

_models_loaded = False

def get_model_class(model_name):
    """
    Get a model class by name

    :param model_name: Name of the model
    :return Model class (not instantiated)
    :raise: ValueError if model not found
    """
    global _models_loaded
    if not _models_loaded:
        for model in pkg_resources.iter_entry_points('vaby.models'):
            MODELS[model.name] = model.load()
        _models_loaded = True

    model_class = MODELS.get(model_name, None)
    if model_class is None:
        raise ValueError("No such model: %s" % model_name)

    return model_class

class ModelOption:
    """
    A model option

    :attr attr_name: Name of the attribute to be created on the model object
    :attr desc: Description of the option
    :attr clargs: Sequence of possible command line arguments
                  (attr_name with _ replaced with - if not specified)
    :attr default: Default value (None if not specified)
    :attr units: Units as a string
    :attr type: Python data type
    """
    def __init__(self, attr_name, desc, **kwargs):
        self.attr_name = attr_name
        self.desc = desc
        self.clargs = kwargs.get("clargs", ["--%s" % attr_name.replace("_", "-")])
        self.default = kwargs.get("default", None)
        self.units = kwargs.get("units", None)
        self.type = kwargs.get("type", str)

class Model(LogBase):
    """
    A forward model

    :attr data_model: DataModel associated with the model. This defines the shape and
                      geometric relationship of the data being modelled
    :attr params:     Sequence of ``Parameter`` objects
    """
    OPTIONS = [
        ModelOption("dt", "Time separation between volumes", type=float, default=1.0),
        ModelOption("t0", "Time offset for first volume", type=float, default=0.0),
    ]

    def __init__(self, data_model, **options):
        LogBase.__init__(self)
        self.data_model = data_model
        self.params = []
        for option in self.OPTIONS:
            setattr(self, option.attr_name, options.get(option.attr_name, option.default))

    @property
    def nparams(self):
        """
        Number of parameters in the model
        """
        return len(self.params)

    def param_idx(self, name):
        """
        :return: the index of a named parameter
        """
        for idx, param in enumerate(self.params):
            if param.name == name:
                return idx
        raise ValueError("Parameter not found in model: %s" % name)

    def tpts(self):
        """
        Get the full set of timeseries time values

        By default this is a linear space using the attributes ``t0`` and ``dt``.
        Some models may have time values fixed by some other configuration.

        :return: Either a 1D Numpy array of shape [n_tpts] or a 2D Numpy array of shape
                 [N, T] for nodewise timepoints
        """
        return np.linspace(self.t0, self.t0+self.data_model.n_tpts*self.dt, num=self.data_model.n_tpts, endpoint=False)

    def evaluate(self, params, tpts):
        """
        Evaluate the model

        :param t: Time values to evaluate the model at, supplied as a tensor of shape
                  [1x1xB] (if time values at each voxel are identical) or [Vx1xB]
                  otherwise.
        :param params Sequence of parameter values arrays, one for each parameter.
                      Each array is WxSx1 tensor where W is the number of parameter vertices and
                      S is the number of samples per parameter. This
                      may be supplied as a PxVxSx1 tensor where P is the number of
                      parameters.

        :return: [VxSxB] tensor containing model output at the specified time values
                 for each voxel, and each sample (set of parameter values).
        """
        raise NotImplementedError("evaluate")

    def test_data(self, tpts, params_map):
        """
        Generate test data by evaluating the model on known parameter values
        with optional added noise

        FIXME this is non-functional at present.

        :param tpts: 1xN or MxN tensor of time values (possibly varying by voxel)
        :param params_map: Mapping from parameter name either a single parameter
                           value or a sequence of M parameter values. The special
                           key ``noise_sd``, if present, should containing the
                           standard deviation of Gaussian noise to add to the
                           output.
        :return If noise is present, a tuple of two MxN Numpy arrays. The first
                contains the 'clean' output data without noise, the second
                contains the noisy data. If noise is not present, only a single
                array is returned.
        """
        param_values = {}
        for idx, param in enumerate(self.params):
            if param.name not in params_map:
                raise IndexError("Required parameter not found: %s" % param.name)
            elif isinstance(params_map[param.name], (float, int)):
                value_sequence = np.reshape([params_map[param.name]], (1, 1))
            elif isinstance(params_map[param.name], collections.Sequence):
                value_sequence = np.reshape(params_map[param.name], (-1, 1))
            else:
                raise ValueError("Unsupported value for parameter '%s': %s" % (param.name, params_map[param.name]))

            param_values[param.name] = value_sequence

        max_num_values = max([len(param_values[name]) for name in param_values.keys()])

        param_values_array = np.zeros((len(self.params), max_num_values, len(tpts)))
        for name, values in param_values.items():

            if len(values) != 1 and len(values) != max_num_values:
                raise ValueError("Parameter %s has wrong number of values: %i (expected %i)" %
                                 (param.name, len(values), max_num_values))
            else:
                param_values[idx, :, :] = values

        with tf.Session():
            clean = self.evaluate(param_values, tpts).eval()
            if "noise_sd" in params_map:
                np.random.seed(1)
                noisy = np.random.normal(clean, params_map["noise_sd"])
                return clean, noisy
            else:
                return clean

    def log_config(self, log=None):
        """
        Write model configuration to a log stream
        
        :param: log Optional logger to use - defaults to class instance logger
        """
        if log is None:
            log = self.log
        log.info("Model: %s", str(self))
        for option in self.OPTIONS:
            log.info(" - %s: %s", option.desc, str(getattr(self, option.attr_name)))

"""
VABY - Models based on linear combinations of exponentials
"""
import tensorflow as tf

from vaby import __version__
from vaby.model import Model, ModelOption
from vaby.parameter import get_parameter

class MultiExpModel(Model):
    """
    Exponential decay with multiple independent decay rates and amplitudes
    """

    OPTIONS = Model.OPTIONS + [
        ModelOption("num_exps", "Number of exponentials", type=int, default=1),
    ]

    def __init__(self, data_model, **options):
        Model.__init__(self, data_model, **options)
        for idx in range(self.num_exps):
            self.params += [
                get_parameter("amp%i" % (idx+1), 
                              dist="LogNormal", mean=1.0, 
                              prior_var=1e6, post_var=1.5, 
                              post_init=self._init_amp,
                              **options),
                get_parameter("r%i" % (idx+1), 
                              dist="LogNormal", mean=1.0, 
                              prior_var=1e6, post_var=1.5,
                              **options),
            ]


    def _init_amp(self, _param, _t, data):
        return tf.reduce_max(data, axis=1) / self.num_exps, None

    def evaluate(self, params, tpts):
        ret = None
        for idx in range(self.num_exps):
            amp = params[2*idx]
            r = params[2*idx+1]
            contrib = amp * tf.exp(-r * tpts)
            if ret is None:
                ret = contrib
            else:
                ret += contrib
        return ret

    def __str__(self):
        return "Multi exponential model with %i exponentials: %s" % (self.num_exps, __version__)

class ExpModel(MultiExpModel):
    """
    Single exponential decay model
    """
    OPTIONS = Model.OPTIONS

    def __init__(self, data_model, **options):
        MultiExpModel.__init__(self, data_model, num_exps=1, **options)

    def __str__(self):
        return "Exponential model: %s" % __version__

class BiExpModel(MultiExpModel):
    """
    Exponential decay with two independent decay rates and amplitudes
    """
    OPTIONS = Model.OPTIONS
    
    def __init__(self, data_model, **options):
        MultiExpModel.__init__(self, data_model, num_exps=2, **options)

    def __str__(self):
        return "Bi-Exponential model: %s" % __version__

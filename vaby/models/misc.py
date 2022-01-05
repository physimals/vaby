"""
VABY - Miscellaneous models mostly for testing
"""
import tensorflow as tf

from vaby import __version__
from vaby.model import Model, ModelOption
from vaby.parameter import get_parameter

class ConstantModel(Model):
    """
    Model which generates a constant signal
    """

    def __init__(self, data_model, **options):
        Model.__init__(self, data_model, **options)
        self.params += [
            get_parameter("mu", desc="Constant signal",
                          dist="Normal", mean=0.0,
                          prior_var=1e6, post_var=1.0, 
                          **options),
        ]

    def evaluate(self, params, tpts):
        return params[0] * tf.ones_like(tpts)

    def __str__(self):
        return "Constant signal model: %s" % __version__

class PolyModel(Model):
    """
    Model which generates a signal from a polynomial
    """

    @staticmethod
    def options():
        return Model.options() + [
            ModelOption("degree", "Maximum power degree in the polynomial", type=int, default=2),
        ]   

    def __init__(self, data_model, **options):
        Model.__init__(self, data_model, **options)
        for idx in range(self.degree+1):
            self.params.append(
                get_parameter(f"c%(idx)i" % idx, desc=f"Coefficient of x^%(idx)i",
                              dist="Normal", mean=0.0,
                              prior_var=1e6, post_var=1.0, 
                              **options),
            )

    def evaluate(self, params, tpts):
        ret = None
        for idx in range(self.degree):
            c = params[idx]
            contrib = c * tf.pow(tpts, idx)
            if ret is None:
                ret = contrib
            else:
                ret += contrib
        return ret

    def __str__(self):
        return "Polynomial model: %s" % __version__

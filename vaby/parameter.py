"""
VABY - Model parameters

A parameter is a variable in the model which can be fitted to the data.
It has a prior distribution and an initial posterior. 
"""
import tensorflow as tf
   
from .utils import LogBase
from . import dist

def get_parameter(name, **kwargs):
    """
    Factory method to create an instance of a parameter

    Keyword arguments:
         - ``prior`` Name of the prior distribution. Other keyword arguments beginning with ``prior``
                     are passed to the ``dist.get_dist`` factory function to construct the prior
                     distribution.
         - ``prior_type`` Code letter giving type of prior to be used, e.g. normal, spatial, ARD...
         - ``post`` Name of the posterior distribution. Other keyword arguments beginning with ``post``
                     are passed to the ``dist.get_dist`` factory function to construct the posterior
                     distribution.
         - ``post_type`` Type of posterior, e.g. nodewise, global...
         - ``param_overrides`` Dictionary keyed by parameter name. If a key for this parameter is found in
                               the ``param_overrides`` dictionary, it's value will be taken as an additional
                               dictionary of keyword arguments to override those defined in ``**kwargs``
    """
    custom_kwargs = kwargs.pop("param_overrides", {}).get(name, {})
    kwargs.update(custom_kwargs)

    kwargs["prior"] = dist.get_dist(prefix="prior", **kwargs)
    kwargs["post"] = dist.get_dist(prefix="post", **kwargs)

    # FIXME: Because when var = 1 the LogNormal gives inf latent cost during training
    if (kwargs.get('dist') == 'LogNormal') and (kwargs.get('var') == 1.0):
        raise ValueError('LogNormal distribution cannot have initial var = 1.0') 

    return Parameter(name, **kwargs)

class Parameter(LogBase):
    """
    A standard model parameter
    """

    def __init__(self, name, **kwargs):
        """
        Constructor

        :param name: Parameter name

        Keyword arguments (optional):
         - ``desc`` Text description
         - ``prior`` ``Dist`` instance giving the parameter's prior distribution
         - ``prior_type`` Code letter giving type of prior to be used, e.g. normal, spatial, ARD...
         - ``post`` ``Dist`` instance giving the parameter's initial posterior distribution. The parameters
                    of the posterior will be inferred however the distibution type is fixed
         - ``post_init`` Callable which will be used to initialize the posterior from the data
         - ``post_type`` Type of posterior, e.g. nodewise, global...
         - ``pv_scale`` If True, parameter scales with total partial volume, e.g. perfusion, 
                        False otherwise (e.g. delay times, tissue properties like T1, T2...).
        """
        LogBase.__init__(self)

        custom_kwargs = kwargs.pop("param_overrides", {}).get(name, {})
        kwargs.update(custom_kwargs)

        self.name = name
        self.desc = kwargs.get("desc", "No description given")
        self.prior_dist = kwargs.get("prior")
        self.prior_type = kwargs.get("prior_type", "N")
        self.post_dist = kwargs.get("post", self.prior_dist)
        self.post_init = kwargs.get("post_init", None)
        self.post_type = kwargs.get("post_type", "vertexwise")
        self.pv_scale = kwargs.get("pv_scale", False)

    def __str__(self):
        return "Parameter: %s (%s)" % (self.name, self.desc)

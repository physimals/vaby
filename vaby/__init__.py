"""
Core classes for Variational Bayesian inference on timeseries data
"""
try:
    from ._version import __version__, __timestamp__
except ImportError:
    __version__ = "Unknown version"
    __timestamp__ = "Unknown timestamp"

from .data import DataModel
from .inference import InferenceMethod
from .model import Model, get_model_class
from .main import run
from .utils import NP_DTYPE, TF_DTYPE

__all__ = [   "__version__",
    "__timestamp__",
    "DataModel",
    "InferenceMethod",
    "Model",
    "get_model_class",
    "run",
    "NP_DTYPE",
    "TF_DTYPE",
]

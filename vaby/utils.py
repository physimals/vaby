"""
VABY - General utility functions
"""
import logging
import logging.config
import os

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
    return (end_time - start_time), ret

def scipy_to_tf_sparse(scipy_sparse):
    """Converts a scipy sparse matrix to TF representation"""

    spmat = scipy_sparse.tocoo()
    return tf.SparseTensor(
        indices=np.array([
            spmat.row, spmat.col]).T,
        values=spmat.data.astype(NP_DTYPE), 
        dense_shape=spmat.shape, 
    )
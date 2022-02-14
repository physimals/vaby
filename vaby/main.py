"""
Implementation of command line tool for SVB

Examples::

    svb --data=asldata.nii.gz --mask=bet_mask.nii.gz
        --model=aslrest --epochs=200 --output=svb_out
"""
import sys
import logging

from .model import get_model_class
from .utils import setup_logging, runtime
from .cli import ArgumentParser
from .data import DataModel

from . import __version__

def main():
    """
    Command line tool entry point
    """
    try:
        arg_parser = ArgumentParser()
        options = arg_parser.parse_args()

        if not options.data:
            raise ValueError("Input data not specified")
        if not options.model_name:
            raise ValueError("Model name not specified")

        # Fixed for CL tool
        options.save_mean = True
        options.save_runtime = True
        options.save_log = True

        welcome = "Welcome to VABY %s" % __version__
        print(welcome)
        print("=" * len(welcome))
        runtime, _ = run(log_stream=sys.stdout, **vars(options))
        print("FINISHED - runtime %.3fs" % runtime)
    except (RuntimeError, ValueError) as exc:
        sys.stderr.write("ERROR: %s\n" % str(exc))
        import traceback
        traceback.print_exc()

def run(data, model_name, output=None, method="avb", **kwargs):
    """
    Run model fitting on a data set

    :param data: File name of 4D NIFTI data set containing data to be fitted
    :param model_name: Name of model we are fitting to
    :param output: output directory, will be created if it does not exist. If not
                   specified no filesystem output will be generated
    :param method: Inference method (avb, svb)

    All keyword arguments are passed to constructor of the model, the ``Svb``
    object and the ``Svb.train`` method.
    """
    setup_logging(output, **kwargs)
    log = logging.getLogger(__name__)

    # Initialize the data model which contains data dimensions, number of time
    # points, list of unmasked voxels, etc
    data_model = DataModel(data, **kwargs)
    
    # Create the generative model
    fwd_model = get_model_class(model_name)(data_model, **kwargs)

    # Check that any parameter overrides actually match parameters in the model
    #assert_param_overrides_used(fwd_model.params, kwargs)

    if method == "avb":
        from vaby_avb import Avb, __version__
        vb = Avb(data_model, fwd_model, **kwargs)
    elif method == "svb":
        from vaby_svb import Svb, __version__
        vb = Svb(data_model, fwd_model, **kwargs)
    else:
        raise ValueError("Unknown inference method: %s" % method)

    log.info("%s %s", method.upper(), __version__)
    rt = runtime(vb.run, **kwargs)
    log.info("DONE: %.3fs", rt)
    if output:
        vb.save(output, rt, **kwargs)

    return rt, vb

if __name__ == "__main__":
    main()

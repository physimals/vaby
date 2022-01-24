"""
VABY - Command line interface
"""
import argparse
import re
import sys

from .model import get_model_class

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
        group.add_argument("--mask", help="Optional voxel mask for volumetric data")
        group.add_argument("--surface", help="Surface geometry data for surface based input data")
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

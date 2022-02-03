import numpy as np
import nibabel as nib

from .base import DataStructure
from .volume import Volume, PartialVolumes
from .surface import SimpleSurface, CorticalSurface
from .model import ModelSpace

def get_data_structure(**kwargs):
    """
    Factory method to return an instance of DataStructure
    
    Either the class name may be direcly specified using the 'type' kwarg, or
    a data filename/object may be provided via the 'data' kwarg
    """
    if "type" in kwargs:
        classname = kwargs.pop("type")
        cls = globals().get(classname, None)
        if cls is None:
            raise ValueError("No such data structure: %s" % classname)
        return cls(**kwargs)
    elif "data" in kwargs:
        data = kwargs.pop("data")
        if isinstance(data, str):
            data = nib.load(data)

        if isinstance(data, np.ndarray):
            return Volume(data, **kwargs)
        elif isinstance(data, nib.Nifti1Image):
            return Volume(nii=data, **kwargs)
        elif isinstance(data, nib.GiftiImage):
            tsdata = data.darrays[0].data
            geomdata = kwargs.get("surface", None)
            if not geomdata:
                raise ValueError("Surface based input data provided but no surface geometry file supplied")
            return SimpleSurface(tsdata, geomdata=geomdata, **kwargs)
        else:
            raise ValueError("Unable to create model structure from data type %s" % type(data))
    else:
        raise ValueError("Unable to create model structure - neither data nor structure type given")

__all__ = [
    "DataStructure",
    "Volume",
    "PartialVolumes",
    "SimpleSurface",
    "CortialSurface",
    "ModelSpace",
]
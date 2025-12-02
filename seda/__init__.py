# import the submodule object
from . import input_parameters
from . import plots
from . import utils
from . import models
from . import chi2_fit
from . import bayes_fit
from . import phy_params
from .synthetic_photometry import synthetic_photometry
from .spectral_indices import spectral_indices

try:
	from ._version import __version__
except (ImportError, ModuleNotFoundError):  
	__version__ = "0.0.0+local"

# defines the explicit public API of package's top-level namespace, so
# from seda import *
# will import the names below
__all__ = [
    "input_parameters",
    "plots",
    "utils",
    "models",
    "chi2_fit",
    "bayes_fit",
    "phy_params",
    "synthetic_photometry",
    "spectral_indices",
]

print(f'\n    SEDA v{__version__} package imported')
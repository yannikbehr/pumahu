import pkg_resources


def get_data(filename):
    return pkg_resources.resource_filename(__name__, filename)

from .forward_model import Forwardmodel
from .mcmc import ns_sampling 
from .syn_model import SynModel
from .uks import UnscentedKalmanSmoother
from .data import LakeData, get_MetService_wind, get_outflow

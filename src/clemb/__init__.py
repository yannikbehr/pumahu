from .clemb import *

import pkg_resources


def get_data(filename):
    return pkg_resources.resource_stream(__name__, filename)

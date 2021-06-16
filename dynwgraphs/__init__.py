"""Top-level package for dynwgraphs."""

__author__ = """Domenico Di Gangi"""
__email__ = 'digangidomenico@gmail.com'
__version__ = '0.1.0'



import os
import numpy as np
import pkg_resources

from .dirSpW1_dynNets import dirSpW1_staNet
from .dirSpW1_dynNets import dirSpW1_dynNet_SD
from .dirBin1_dynNets import dirBin1_staNet
from .dirBin1_dynNets import dirBin1_dynNet_SD 



# load test data
stream =  pkg_resources.resource_stream(__name__, os.path.join("..", "tests", "test_data", "dir_w_test_data.npz"))

_test_w_data_ = dict(np.load(stream))
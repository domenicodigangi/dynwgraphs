"""Top-level package for dynwgraphs."""

__author__ = """Domenico Di Gangi"""
__email__ = 'digangidomenico@gmail.com'
__version__ = '0.1.0'


import os
import numpy as np

# Import custom exceptions
from .exceptions import (
    DynWGraphsError,
    ParameterError,
    ShapeError,
    ModelNotReadyError,
    DataError,
    OptimizationError,
    IdentificationError,
)

# from .dirSpW1_dynNets import dirSpW1_staNet
# from .dirSpW1_dynNets import dirSpW1_dynNet_SD
# from .dirBin1_dynNets import dirBin1_staNet
# from .dirBin1_dynNets import dirBin1_dynNet_SD

__all__ = [
    'DynWGraphsError',
    'ParameterError',
    'ShapeError',
    'ModelNotReadyError',
    'DataError',
    'OptimizationError',
    'IdentificationError',
] 



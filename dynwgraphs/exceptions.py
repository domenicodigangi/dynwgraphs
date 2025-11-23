#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom exceptions for dynwgraphs package.

@author: Domenico Di Gangi, <digangidomenico@gmail.com>
"""


class DynWGraphsError(Exception):
    """Base exception for all dynwgraphs errors."""
    pass


class ParameterError(DynWGraphsError):
    """Exception raised for errors in parameter specification or initialization."""
    pass


class ShapeError(DynWGraphsError):
    """Exception raised when tensor or matrix shapes are incompatible."""
    pass


class ModelNotReadyError(DynWGraphsError):
    """Exception raised when attempting to use a model feature that is not yet implemented."""
    pass


class DataError(DynWGraphsError):
    """Exception raised for errors in data format or validity."""
    pass


class OptimizationError(DynWGraphsError):
    """Exception raised when optimization fails or encounters issues."""
    pass


class IdentificationError(DynWGraphsError):
    """Exception raised when parameter identification constraints are violated."""
    pass


class SecurityError(DynWGraphsError):
    """Exception raised for security-related issues such as loading untrusted files."""
    pass

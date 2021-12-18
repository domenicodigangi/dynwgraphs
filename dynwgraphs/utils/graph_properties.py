#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from enum import Enum


class MatrixSymmetry(Enum):
    """
    Matrix property can be associated with a Model or data
    """
    uninf = 1 # All elements are equal
    undir = 2 # Symmetric matrix
    dir = 3 # No symmetry
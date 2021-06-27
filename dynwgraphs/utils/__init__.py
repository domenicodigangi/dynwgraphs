#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Friday June 18th 2021

"""

import os
import numpy as np
import pkg_resources
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load test data
stream =  pkg_resources.resource_stream(__name__, os.path.join( "test_data", "dir_w_test_data.npz"))

_test_w_data_ = dict(np.load(stream))


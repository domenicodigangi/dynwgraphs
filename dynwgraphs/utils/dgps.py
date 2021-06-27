#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday June 16th 2021

"""


"""
functions needed in different parts of the module
"""

from . import _test_w_data_
from .tensortools import tens
import torch
import numpy as np

def rand_steps(start_val, end_val, Nsteps, T, rand=True):
    out = np.zeros(T)
    if Nsteps > 1:
        heights = np.linspace(start_val, end_val, Nsteps)
        if rand:
            np.random.shuffle(heights)

        Tstep = T // Nsteps
        last = 0
        for i in range(Nsteps - 1):
            out[last:last + Tstep] = heights[i]
        last += Tstep
        out[last:] = heights[-1]

    elif Nsteps == 1:
        out = start_val

    return tens(out)

def dgpAR(mu, B, sigma, T, N=1, minMax=None, scaling = "uniform"):

    w = mu * (1-B)
    path = torch.randn(N, T) * sigma
    path[:, 0] = mu

    for t in range(1, T):
        path[:, t] = w + B*path[:, t-1] + path[:, t]

    if minMax is not None:
        min = minMax.min(dim=1)
        max = minMax.max(dim=1)
        if scaling == "uniform":
            minPath = path.min(dim=1)
            maxPath = path.max(dim=1)
            Δ = (max-min)/(maxPath - minPath)
            rescPath = min + (path - minPath)*Δ
        elif scaling == "nonlinear":
              rescPath = min + (max - min) * 1/(1 + torch.exp(path))

    else:
        rescPath = path
    return rescPath


def gen_test_net(N, density):
    Y_mat = torch.rand(N,N)
    Y_mat[Y_mat > density] = 0
    Y_mat *= 10000
    return Y_mat

def get_test_w_seq(avg_weight = 1e5):    
    Y_T = tens(_test_w_data_["Y_T"])
    X_T = tens( _test_w_data_["X_T"])

    # scale weights s.t. the average weight is the input one
    Y_T = Y_T /Y_T.mean() * avg_weight

    X_scalar_T = tens(X_T).unsqueeze(1).unsqueeze(1).permute((3,2,1,0))
    X_matrix_T = Y_T[:, :, :-1].log().unsqueeze(2)
    X_matrix_T[~torch.isfinite(X_matrix_T)] = 0
    Y_T = Y_T[:,:,:-1]

    N = Y_T.shape[0]
    T = Y_T.shape[2]

    
    X_T_multi = X_matrix_T.repeat_interleave(2, dim=2)
    X_T_multi[:, :, 1, :] = torch.range(0, T-1) * torch.ones(N,N).unsqueeze(dim=2)
    return Y_T, X_scalar_T, X_T_multi











#
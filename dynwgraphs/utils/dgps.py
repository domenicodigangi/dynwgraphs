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
from dynwgraphs.dirGraphs1_dynNets import  dirBin1_sequence_ss


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


def sample_phi_dgp_ar(model, phi_um, B, sigma, T):

        N = phi_um.shape[0]
        phi_T_sample_mat = torch.zeros(N, T)
        for i in range(N):
            phi_T_sample_mat[i, :] = dgpAR(phi_um[i], B, sigma, T)

        phi_T_sample_list = model.par_matrix_T_to_list(phi_T_sample_mat)

        for t in range(T):
            #as first step, identify phi_io
            phi_T_sample_list[t] = model.identify_phi_io(phi_T_sample_list[t])

        return phi_T_sample_list


def get_dgp_model(dgp_par, Y_reference=None, beta_tv = tens([False]).bool(), X_type = None ):

    N = dgp_par["N"]
    T = dgp_par["T"]

    if Y_reference is None:
        raise Exception("default choice yet to be defined")
    else:
        assert N == Y_reference.shape[0]

    
        
    mod_tmp = dirBin1_sequence_ss(torch.zeros(N,N,T)) 

    if dgp_par["model"] =="dirBin1":

        # set reference values for dgp 
        phi_0 = mod_tmp.start_phi_from_obs(Y_reference)
        dist_par_un_T_dgp = None
    else:
        raise
        

    if dgp_par["dgp_phi"]["is_tv"]:    
        if dgp_par["dgp_phi"]["type"] == "AR":
            B = dgp_par["dgp_phi"]["B"]
            sigma = dgp_par["dgp_phi"]["sigma"]
            phi_T_dgp = sample_phi_dgp_ar(mod_tmp, phi_0, B, sigma, T)
        else:
            raise
    else:
        phi_T_dgp = [phi_0]


    
    # dgp for beta
    if "dgp_beta" in dgp_par:
        beta_tv = dgp_par["dgp_beta"]["is_tv"]
        size_beta_t = dgp_par["dgp_beta"]["size_beta_t"]

        if dgp_par["dgp_beta"]["X_type"] == "uniform":
            if len(beta_tv) != 1:
                raise
            x_T = np.random.standard_normal(T)
            X_T = tens(np.tile(x_T, (N, N, 1, 1)))
        else:
            raise

        beta_0 = dgp_par["dgp_beta"]["beta_0"]  
        if dgp_par["dgp_beta"]["is_tv"]:    
            if dgp_par["dgp_beta"]["type"] == "AR":
                B = dgp_par["dgp_beta"]["B"]
                sigma = dgp_par["dgp_beta"]["sigma"]
                beta_T_dgp = mod_tmp.par_matrix_T_to_list(dgpAR(beta_0, B, sigma, T))
            else:
                raise
        else:
            beta_T_dgp = [beta_0]

    else:
        X_T = None
        size_beta_t = None
        beta_tv = None

    
    mod_dgp = dirBin1_sequence_ss(torch.zeros(N,N,T), X_T=X_T, beta_tv = beta_tv, size_beta_t = size_beta_t) 

    mod_dgp.phi_T = phi_T_dgp
    mod_dgp.beta_T = beta_T_dgp
    mod_dgp.dist_par_un_T = dist_par_un_T_dgp

    mod_dgp.Y_T = mod_dgp.sample_Y_T_from_par_list(T, mod_dgp.phi_T, X_T = mod_dgp.X_T, beta_T=mod_dgp.beta_T, dist_par_un_T=mod_dgp.dist_par_un_T)
    
    mod_dgp.set_par_dict_to_save()
    return mod_dgp










#
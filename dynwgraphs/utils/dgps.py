#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday June 16th 2021

"""


"""
functions needed in different parts of the module
"""

from logging import Logger

from mlflow.tracking.fluent import end_run
from . import _test_w_data_
from .tensortools import splitVec, strIO_from_tens_T, tens
import torch
import numpy as np
from dynwgraphs.dirGraphs1_dynNets import  dirBin1_sequence_ss, dirSpW1_SD, dirSpW1_sequence_ss
import itertools
import copy
import logging
logger = logging.getLogger(__name__)


def get_default_tv_dgp_par(dgp_type):
    if dgp_type== "AR":
        dgp_par = {"type":"AR", "B":0.98, "sigma":0.1, "is_tv":True}
    
    return dgp_par


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


def sample_par_vec_dgp_ar(model, unc_mean, B, sigma, T, identify=True):

        N = unc_mean.shape[0]
        par_T_sample_mat = torch.zeros(N, T)
        for i in range(N):
            par_T_sample_mat[i, :] = dgpAR(unc_mean[i], B, sigma, T)

        par_T_sample_list = model.par_tens_T_to_list(par_T_sample_mat)

        if identify:
            for t in range(T):
                #as first step, identify phi_io
                par_T_sample_list[t] = model.identify_phi_io(par_T_sample_list[t])

        return par_T_sample_list


def get_dgp_model(N, T, model,  n_ext_reg, size_beta, type_dgp_phi, type_dgp_beta, all_beta_tv,  Y_reference=None):

    dgp_par = {"N": N, "T": T}

    dgp_phi = get_default_tv_dgp_par(type_dgp_phi)
    
 
    dgp_par["phi"] = dgp_phi
    if n_ext_reg == 0:
        pass
    elif n_ext_reg == 1:
        # combinations of tv and static regression coefficients are not yet allowed
        dgp_beta = get_default_tv_dgp_par(type_dgp_beta)
        
        dgp_beta["is_tv"] = [all_beta_tv for p in range(n_ext_reg)]
        dgp_beta["X_type"] = "uniform"
        if size_beta=="one":
            dgp_beta["size_beta_t"] = 1
        elif size_beta=="N":
            dgp_beta["size_beta_t"] = N
        elif size_beta=="2N":
            dgp_beta["size_beta_t"] = 2*N
        else:
            raise
        dgp_beta["beta_0"] = torch.ones(1,1)
   
     
    else:
        raise
 
    dgp_par["beta"] = dgp_beta


    if Y_reference is None:
        Y_T_test, _, _ =  get_test_w_seq(avg_weight=1e3)
        Y_reference = (Y_T_test[:N,:N,0]>0).float()

    else:
        assert N == Y_reference.shape[0]  

    if model =="dirBin1":     
        mod_tmp = dirBin1_sequence_ss(torch.zeros(N,N,T)) 
        dist_par_un_T_dgp = None

    elif model =="dirSpW1":    
        mod_tmp = dirSpW1_sequence_ss(torch.zeros(N,N,T))
        dist_par_un_T_dgp = [torch.ones(1)]    

    # set reference values for dgp 
    if "phi_0" in dgp_phi:
        phi_0 = dgp_phi["phi_0"]
    else:
        phi_0 = mod_tmp.start_phi_from_obs(Y_reference)

    if dgp_phi["is_tv"]:    
        if dgp_phi["type"] == "AR":
            B = dgp_phi["B"]
            sigma = dgp_phi["sigma"]
            phi_T_dgp = sample_par_vec_dgp_ar(mod_tmp, phi_0, B, sigma, T)
        else:
            raise
    else:
        phi_T_dgp = [phi_0]
    
    # dgp for beta
    if n_ext_reg >0:
        if n_ext_reg >1:
            raise

        beta_tv = dgp_beta["is_tv"]
        size_beta_t = dgp_beta["size_beta_t"]

        #sample regressors
        if dgp_beta["X_type"] == "uniform":
            if len(beta_tv) != 1:
                raise
            x_T = np.random.standard_normal(T)
            X_T = tens(np.tile(x_T, (N, N, 1, 1)))
        else:
            raise

        # sample reg coeff
        beta_0 = dgp_beta["beta_0"]
        if beta_0.shape[0] != dgp_beta["size_beta_t"]:
            raise 

        if any(dgp_beta["is_tv"]):
            if not  all(dgp_beta["is_tv"]):
                raise    
            if dgp_beta["type"] == "AR":
                B = dgp_beta["B"]
                sigma = dgp_beta["sigma"]
                if dgp_beta["size_beta_t"] == 1:
                    beta_T_dgp = mod_tmp.par_tens_T_to_list(dgpAR(beta_0, B, sigma, T).unsqueeze(dim=1))
                elif dgp_beta["size_beta_t"] >= 1:
                    beta_T_dgp = sample_par_vec_dgp_ar(mod_tmp, beta_0, B, sigma, T)
                    beta_T_dgp = [b.unsqueeze(1) for b in beta_T_dgp]
             
            else:
                raise
        else:
            beta_T_dgp = [beta_0]
    else:
        X_T = None
        size_beta_t = 1
        beta_tv = None
        beta_T_dgp = None


    if model =="dirBin1":    
        mod_dgp = dirBin1_sequence_ss(torch.zeros(N,N,T), X_T=X_T, beta_tv = beta_tv, size_beta_t = size_beta_t) 
    
    elif model =="dirSpW1":    

        mod_dgp = dirSpW1_sequence_ss(torch.zeros(N,N,T), X_T=X_T, beta_tv = beta_tv, size_beta_t = size_beta_t) 

    mod_dgp.phi_T = phi_T_dgp
    mod_dgp.beta_T = beta_T_dgp
    mod_dgp.dist_par_un_T = dist_par_un_T_dgp

    mod_dgp.set_par_dict_to_save()

    return mod_dgp, dgp_par, Y_reference










#
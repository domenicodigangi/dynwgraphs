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
    Y_T = Y_T[:, :, :-1]

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


def get_mod_and_par(N, T, model, dgp_set_dict,  Y_reference=None):

    if Y_reference is None:
        Y_T_test, _, _ =  get_test_w_seq(avg_weight=1e3)
        Y_reference = (Y_T_test[:N, :N, 0] > 0).float()

    else:
        assert N == Y_reference.shape[0]  

    if model == "dirBin1":
        mod_tmp = dirBin1_sequence_ss(torch.zeros(N, N, T)) 
        dist_par_un_T_dgp = None

    elif model == "dirSpW1":
        mod_tmp = dirSpW1_sequence_ss(torch.zeros(N, N, T))
        dist_par_un_T_dgp = [torch.ones(1)]    

    phi_dgp_type, phi_0, B_phi, sigma_phi = dgp_set_dict["type_tv_dgp_phi"]

    # set reference values for dgp
    if phi_0 == "ref_mat":
        phi_0 = mod_tmp.start_phi_from_obs(Y_reference)
    elif phi_dgp_type == "const_unif":
        phi_0 = float(phi_0)
    else:
        raise

    if phi_dgp_type == "const_unif":
        if model =="dirBin1":
            exp_a = phi_0
            phi_0_i = 0.5*torch.log(tens(exp_a /(1 - exp_a)))
            phi_0 = torch.ones(2*N)*phi_0_i
            phi_T_dgp = sample_par_vec_dgp_ar(mod_tmp, phi_0, 0, 0, T)
            phi_tv =True
        else:
            raise

    elif phi_dgp_type == "AR":
        phi_T_dgp = sample_par_vec_dgp_ar(mod_tmp, phi_0, B_phi, sigma_phi, T)
        phi_tv =True
    else:
        raise


    # dgp for coefficients of ext reg beta
    if dgp_set_dict["n_ext_reg"] > 0:
        if dgp_set_dict["n_ext_reg"] > 1:
            raise
        #sample regressors
        X_cross_type, X_dgp_type, unc_mean_X, B_X, sigma_X = dgp_set_dict["type_tv_dgp_ext_reg"]

        beta_tv = dgp_set_dict["beta_tv"]
        size_beta_t = dgp_set_dict["size_beta_t"]

        if X_cross_type == "uniform":
            if len(beta_tv) != 1:
                raise
            
            if X_dgp_type == "AR":
                x_T = dgpAR(unc_mean_X, B_X, sigma_X, T).unsqueeze(dim=1)
            X_T = tens(np.tile(x_T, (N, N, 1, 1)))
        else:
            raise

        # sample reg coeff
        beta_dgp_type, unc_mean_beta, B_beta, sigma_beta = dgp_set_dict["type_tv_dgp_beta"]
        print(unc_mean_beta)
        if unc_mean_beta is None:
            unc_mean_beta = 1 + torch.randn(dgp_set_dict["size_beta_t"], dgp_set_dict["n_ext_reg"])
            if unc_mean_beta.shape[0] != dgp_set_dict["size_beta_t"]:
                raise
        elif type(unc_mean_beta) == float:
            unc_mean_beta = unc_mean_beta +  torch.randn(dgp_set_dict["size_beta_t"], dgp_set_dict["n_ext_reg"])
        else:
            raise

        if any(beta_tv):
            if not all(beta_tv):
                raise
            if beta_dgp_type == "AR":
                if dgp_set_dict["size_beta_t"] == 1:
                    beta_T_dgp = mod_tmp.par_tens_T_to_list(dgpAR(unc_mean_beta, B_beta, sigma_beta, T).unsqueeze(dim=1))
                elif dgp_set_dict["size_beta_t"] >= 1:
                    beta_T_dgp = sample_par_vec_dgp_ar(mod_tmp, unc_mean_beta, B_beta, sigma_beta, T)
                    beta_T_dgp = [b.unsqueeze(1) for b in beta_T_dgp]
            else:
                raise
        else:
            beta_T_dgp = [unc_mean_beta]
    else:
        X_T = None
        size_beta_t = 1
        beta_tv = None
        beta_T_dgp = None


    if model =="dirBin1":    
        mod_dgp = dirBin1_sequence_ss(torch.zeros(N, N, T), X_T=X_T, phi_tv=phi_tv, beta_tv=beta_tv, size_beta_t=size_beta_t) 
    
    elif model =="dirSpW1":    

        mod_dgp = dirSpW1_sequence_ss(torch.zeros(N, N, T), X_T=X_T, phi_tv=phi_tv, beta_tv=beta_tv, size_beta_t=size_beta_t) 

    mod_dgp.phi_T = phi_T_dgp
    mod_dgp.beta_T = beta_T_dgp
    mod_dgp.dist_par_un_T = dist_par_un_T_dgp

    mod_dgp.set_par_dict_to_save()

    return mod_dgp, Y_reference










#
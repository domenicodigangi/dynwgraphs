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
from .tensortools import splitVec, strIO_from_tens_T, tens, size_from_str
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
    X_T_multi[:, :, 1, :] = torch.arange(0, T) * torch.ones(N,N).unsqueeze(dim=2)
    return Y_T, X_scalar_T, X_T_multi


def sample_par_vec_dgp_ar(model, unc_mean, B, sigma, T, identify=True):

        N = unc_mean.shape[0]
        par_T_sample_mat = torch.zeros(N, T)
        for i in range(N):
            par_T_sample_mat[i, :] = dgpAR(unc_mean[i], B, sigma, T)

        par_T_sample_list = model.par_tens_T_to_list(par_T_sample_mat)

        if identify:
            model.identify_par_seq_T(par_T_sample_list)

        return par_T_sample_list


def get_dgp_mod_and_par(N, T, dgp_set_dict,  Y_reference=None):

    bin_or_w = dgp_set_dict["bin_or_w"]
    T_train = dgp_set_dict["T_train"]
    size_phi_t = size_from_str(dgp_set_dict["size_phi_t"], N)
    if Y_reference is None:
        Y_T_test, _, _ = get_test_w_seq(avg_weight=1e3)
        if bin_or_w == "w":
            Y_reference = (Y_T_test[:N, :N, 0]).float()
        elif bin_or_w == "bin":
            Y_reference = (Y_T_test[:N, :N, 0] > 0).float()

    else:
        assert N == Y_reference.shape[0]  

    if bin_or_w == "bin":
        mod_tmp = dirBin1_sequence_ss(torch.zeros(N, N, T), size_phi_t=size_phi_t) 
        dist_par_un_T_dgp = None

    elif bin_or_w == "w":
        mod_tmp = dirSpW1_sequence_ss(torch.zeros(N, N, T), size_phi_t=size_phi_t)
        dist_par_un_T_dgp = [torch.ones(1)]    

    phi_dgp_type, phi_0, B_phi, sigma_phi = dgp_set_dict["phi_set_dgp_type_tv"]

    # set reference values for dgp
    if dgp_set_dict["size_phi_t"] == "2N":
        if phi_0 == "ref_mat":
            phi_0 = mod_tmp.start_phi_from_obs(Y_reference)
        elif phi_dgp_type[:11] == "const_unif_":
            exp_a = float(phi_dgp_type[11:])
            phi_0 = torch.randn(size_phi_t) + torch.ones(size_phi_t)*0.5*torch.log(tens(exp_a /(1 - exp_a)))
        else:
            raise
    elif dgp_set_dict["size_phi_t"] in ["0", None, "None"]:
        phi_0 = None
    else:
        raise
    phi_tv = dgp_set_dict["phi_tv"]

    if phi_0 is not None:
        if phi_tv:
            if phi_dgp_type == "AR":
                phi_T_dgp = sample_par_vec_dgp_ar(mod_tmp, phi_0, B_phi, sigma_phi, T)
            else:
                raise
        else:
            phi_T_dgp = [phi_0]
    else:
        phi_T_dgp = None

    # dgp for coefficients of ext reg beta
    if dgp_set_dict["n_ext_reg"] > 0:
        reg_list = []
        beta_list = []
        for n in range(dgp_set_dict["n_ext_reg"]):

            #sample regressors
            X_cross_type, X_dgp_type, unc_mean_X, B_X, sigma_X = dgp_set_dict["ext_reg_dgp_set_type_tv"]

            beta_tv = dgp_set_dict["beta_tv"]
            size_beta_t = size_from_str(dgp_set_dict["size_beta_t"], N)

            if X_cross_type == "uniform":
                if X_dgp_type == "AR":
                    x_T = dgpAR(unc_mean_X, B_X, sigma_X, T).unsqueeze(dim=1)
                X_T = tens(np.tile(x_T, (N, N, 1, 1)))
            elif X_cross_type == "link_specific":
                if X_dgp_type == "AR":
                    X_T = torch.zeros(N, N, 1, T)
                    for i in range(N):
                        for j in range(N):
                            if i!=j:
                                X_T[i, j, 0, :] = dgpAR(unc_mean_X, B_X, sigma_X, T).unsqueeze(dim=1)

            # sample reg coeff
            beta_dgp_type, unc_mean_beta, B_beta, sigma_beta = dgp_set_dict["beta_set_dgp_type_tv"]
            if n == 1:
                unc_mean_beta = dgp_set_dict["beta_set_dgp_type_tv_un_mean_2"]

            if unc_mean_beta is None:
                unc_mean_beta = 1 + torch.randn(size_beta_t, 1)
                if unc_mean_beta.shape[0] != size_beta_t:
                    raise
            elif type(unc_mean_beta) in [float, int]:
                if size_beta_t > 1:
                    unc_mean_beta = unc_mean_beta +  torch.randn(size_beta_t, 1)
                else:
                    unc_mean_beta = unc_mean_beta * torch.ones(size_beta_t, 1)
            else:
                raise

            if beta_tv[n]:
                if not all(beta_tv):
                    raise
                if beta_dgp_type == "AR":
                    if size_beta_t == 1:
                        beta_T_dgp = mod_tmp.par_tens_T_to_list(dgpAR(unc_mean_beta, B_beta, sigma_beta, T).unsqueeze(dim=1))
                    elif size_beta_t >= 1:
                        beta_T_dgp = sample_par_vec_dgp_ar(mod_tmp, unc_mean_beta, B_beta, sigma_beta, T)
                        beta_T_dgp = [b.unsqueeze(1) for b in beta_T_dgp]
                else:
                    raise
            else:
                beta_T_dgp = [unc_mean_beta]

            reg_list.append(X_T)
            beta_list.append(beta_T_dgp)

        #collpse matrices and betas for different regr 
        X_T = torch.cat(reg_list, dim=2)
        beta_T_dgp = []
        for t, beta_0 in enumerate(beta_list[0]):
            all_beta_n_at_time_t = [beta_list[n][t] for n in range(len(beta_list))]
            beta_t_dgp = torch.cat(all_beta_n_at_time_t, dim=1)
            beta_T_dgp.append(beta_t_dgp)


    else:
        X_T = None
        size_beta_t = 1
        beta_tv = None
        beta_T_dgp = None


    if bin_or_w == "bin":    
        mod_dgp = dirBin1_sequence_ss(torch.zeros(N, N, T), X_T=X_T, size_phi_t=size_phi_t, phi_tv=phi_tv, beta_tv=beta_tv, size_beta_t=size_beta_t, T_train=T_train) 
    
    elif bin_or_w == "w":    

        mod_dgp = dirSpW1_sequence_ss(torch.zeros(N, N, T), X_T=X_T, size_phi_t=size_phi_t, phi_tv=phi_tv, beta_tv=beta_tv, size_beta_t=size_beta_t, T_train=T_train) 

    mod_dgp.inds_never_obs_w
    mod_dgp.phi_T = phi_T_dgp
    mod_dgp.beta_T = beta_T_dgp
    mod_dgp.dist_par_un_T = dist_par_un_T_dgp

    mod_dgp.set_par_dict_to_opt_and_save()

    return mod_dgp, Y_reference










#
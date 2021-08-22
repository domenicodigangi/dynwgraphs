#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday July 10th 2021

"""





# %% import packages
from typing import NamedTuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import dynwgraphs
from dynwgraphs.utils.tensortools import tens, splitVec, strIO_from_mat
from dynwgraphs.dirGraphs1_dynNets import  dirBin1_sequence_ss, dirBin1_SD
import importlib

from torch.functional import split
importlib.reload(dynwgraphs)

# %%

from dynwgraphs.utils.dgps import get_test_w_seq

Y_T, X_T_scalar, X_T_matrix =  get_test_w_seq(avg_weight=1e3)
X_T_test = X_T_matrix
N, _, T = Y_T.shape

# %% Test single snapshot estimates of  phi_t
model = dirBin1_sequence_ss(Y_T, X_T=X_T_test, size_beta_t=1, avoid_ovflw_fun_flag=True, size_phi_t = "2N") # 'lognormal')
model.opt_options_ss_t["max_opt_iter"] = 50

t=1
model.check_exp_vals(t)
model.estimate_ss_t(t, est_phi=True, est_beta = True, est_dist_par=False)

model.check_exp_vals(t)
model.get_time_series_latent_par()


# %% Test sequence of single snapshot estimates of  phi_T
model = dirBin1_sequence_ss(Y_T, X_T=X_T_test, avoid_ovflw_fun_flag=True, size_beta_t=1, beta_tv=[True, True]) # 'lognormal')
model.opt_options_ss_seq["max_opt_iter"] = 20
model.opt_options_ss_seq["opt_n"] = "LBFGS"


model.estimate_ss_seq_joint()
model.get_time_series_latent_par()
model.identify_sequence()

t=56
phi_t = model.phi_T[t]
beta_t = model.beta_T[t][:,model.reg_cross_unique]
x_t = model.X_T[0,0,model.reg_cross_unique,t]

phi_t_identified = model.identify_phi_io(phi_t)

phi_t_identified, beta_id = model.identify_phi_io_beta(phi_t_identified, beta_t, x_t )    



# %% Test Score driven estimates of  phi_T
model = dirBin1_SD(Y_T, size_phi_t="2N", phi_tv = True, avoid_ovflw_fun_flag=True, rescale_SD=True) # 'lognormal')

model = dirBin1_SD(Y_T, size_phi_t="2N", phi_tv = False, avoid_ovflw_fun_flag=True, rescale_SD=True) # 'lognormal')

model = dirBin1_SD(Y_T, X_T=X_T_test[:,:,0:1,:], size_phi_t="2N", phi_tv = False,  size_beta_t = "2N", beta_tv=[ False], avoid_ovflw_fun_flag=True, rescale_SD=False) # 'lognormal')

model = dirBin1_SD(Y_T, X_T=X_T_test[:,:,0:1,:], size_phi_t="2N", phi_tv = False,  size_beta_t = "one", beta_tv=[True], avoid_ovflw_fun_flag=True, rescale_SD=False) # 'lognormal')


model.opt_options_sd["max_opt_iter"] = 21

optimizer = model.estimate()

model.beta_T

#%%

model.sd_stat_par_un_phi["w"].requires_grad
model.sd_stat_par_un_beta["w"].requires_grad
model.par_l_to_opt[0]


model.get_time_series_latent_par()

model.get_phi_T()

model.get_unc_mean(model.sd_stat_par_un_phi)

model.roll_sd_filt()


model.plot_phi_T()

model.un2re_A_par(model.sd_un_phi["A"])
model.un2re_B_par(model.sd_un_phi["B"])


# %% Test sampling of dgp

from dynwgraphs.utils.dgps import get_test_w_seq
Y_T_test, _, _ =  get_test_w_seq(avg_weight=1e3)

N = 50
T = 100
B = 0.98
sigma = 0.1

mod_dgp = dirBin1_SD(torch.zeros(N,N,T)) 

Y = (Y_T_test[:,:,0]>0).float()
phi_0 = mod_dgp.start_phi_from_obs(Y)
mod_dgp.phi_T = sample_phi_dgp_ar(phi_0, B, sigma, T)
Y_T_sampled = mod_dgp.sample_Y_T_from_par_list(T, mod_dgp.phi_T)



mod_est = dirBin1_SD(Y_T_sampled)
mod_est_ = dirBin1_SD(Y_T_sampled)
mod_est_.init_phi_T_from_obs()

mod_est.estimate_sd()

i=11
mod_dgp.plot_phi_T(i=i)
mod_est.plot_phi_T(i=i)
mod_est_.plot_phi_T(i=i)

mod_est.plot_sd_par()


# %%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday June 26th 2021

"""



# %% import packages
from typing import NamedTuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import dynwgraphs
from dynwgraphs.utils.tensortools import tens, splitVec
from dynwgraphs.dirGraphs1_dynNets import  dirSpW1_sequence_ss, dirSpW1_SD
import importlib

from torch.functional import split
importlib.reload(dynwgraphs)

# %%

from dynwgraphs.utils.dgps import get_test_w_seq

Y_T, X_T_scalar, X_T_matrix =  get_test_w_seq(avg_weight=1e3)
X_T_test = X_T_matrix
N, _, T = Y_T.shape

# %% Test single snapshot estimates of  phi_t
model = dirSpW1_sequence_ss(Y_T, X_T=X_T_test, size_beta_t=1, avoid_ovflw_fun_flag=True, distr = 'gamma') # 'lognormal')
model.opt_options_ss_t["max_opt_iter"] = 100


model.estimate_ss_t(1, est_phi=True, est_beta = True, est_dist_par=False)
model.get_seq_latent_par()


# %% Test sequence of single snapshot estimates of  phi_T
model = dirSpW1_sequence_ss(Y_T, X_T=X_T_test, avoid_ovflw_fun_flag=True, distr = 'gamma', size_beta_t=1, beta_tv=[True, True]) # 'lognormal')
model.opt_options_ss_seq["max_opt_iter"] = 10
model.opt_options_ss_seq["opt_n"] = "ADAM"

model.rescale_SD

model.estimate_ss_seq_joint()
model.get_seq_latent_par()
model.identify_sequence()

t=56
phi_t = model.phi_T[t]
beta_t = model.beta_T[t][:,model.reg_cross_unique]
x_t = model.X_T[0,0,model.reg_cross_unique,t]

phi_t_identified = model.identify_phi_io(phi_t)

phi_t_identified, beta_id = model.identify_phi_io_beta(phi_t_identified, beta_t, x_t )    



# %% Test Score driven estimates of  phi_T
model = dirSpW1_SD(Y_T, avoid_ovflw_fun_flag=True, distr = 'gamma', rescale_SD=True) # 'lognormal')

model = dirSpW1_SD(Y_T, X_T=X_T_test[:,:,0:1,:], beta_tv=[ False], avoid_ovflw_fun_flag=True, distr = 'gamma', rescale_SD=False) # 'lognormal')

model.opt_options_sd["max_opt_iter"] = 20
model.opt_options_sd["opt_n"] = "ADAM"



optimizer = model.estimate_sd()

model.get_seq_latent_par()

model.get_unc_mean(model.sd_stat_par_un_phi)

model.roll_sd_filt()


model.plot_phi_T()

model.un2re_A_par(model.sd_un_phi["A"])
model.un2re_B_par(model.sd_un_phi["B"])


# %%

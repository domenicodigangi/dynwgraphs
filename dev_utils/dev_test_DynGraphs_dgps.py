

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday July 10th 2021

"""

#%% import packages

import torch
import matplotlib.pyplot as plt
import dynwgraphs
from dynwgraphs.utils.tensortools import tens, splitVec, strIO_from_mat
from dynwgraphs.dirGraphs1_dynNets import  dirBin1_sequence_ss, dirBin1_SD
import importlib

from torch.functional import split
importlib.reload(dynwgraphs)

#%%

from dynwgraphs.utils.dgps import get_test_w_seq

dgp_phi = {"type":"AR", "B":0.98, "sigma":0.1, "is_tv":True}
dgp_beta = {"type":"AR", "B":0.98, "sigma":0.1, "is_tv":[True], "X_type":"uniform", "size_beta_t": 1, "beta_0":1}
dgp_dist_par = None

dgp_par = {"N" : 50, "T" : 100, "model":"dirBin1", "dgp_phi" : dgp_phi, "dgp_beta":dgp_beta, "dgp_dist_par":dgp_dist_par}



Y_T_test, _, _ =  get_test_w_seq(avg_weight=1e3)
Y_reference = (Y_T_test[:dgp_par["N"],:dgp_par["N"],0]>0).float()
mod_dgp = get_dgp_model(dgp_par, Y_reference=Y_reference)

mod_dgp.X_T
mod_dgp.beta_T[0].shape

mod_dgp.plot_beta_T()



plt.plot


#%% Test single snapshot estimates of  phi_t
model = dirBin1_sequence_ss(Y_T, X_T=X_T_test, size_beta_t=1, avoid_ovflw_fun_flag=True) # 'lognormal')
model.opt_options_ss_t["max_opt_iter"] = 50

t=1
model.check_exp_vals(t)
model.estimate_ss_t(t, est_phi=True, est_beta = True, est_dist_par=False)

model.check_exp_vals(t)
model.get_seq_latent_par()




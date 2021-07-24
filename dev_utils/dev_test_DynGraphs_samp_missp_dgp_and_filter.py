

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday July 10th 2021

"""

#%% import packages

import torch
import matplotlib.pyplot as plt
import numpy as np

import dynwgraphs
from dynwgraphs.utils.tensortools import tens, splitVec, strIO_from_mat
from dynwgraphs.dirGraphs1_dynNets import  dirBin1_sequence_ss, dirBin1_SD
import importlib

from torch.functional import split
importlib.reload(dynwgraphs)
from dynwgraphs.utils.dgps import get_test_w_seq, get_dgp_model, dgpAR

#%%


dgp_par = {"N" : 50, "T" : 100, "model":"dirBin1"}
dgp_phi = {"type":"AR", "B":0.98, "sigma":0.1, "is_tv":True}
dgp_beta = {"type":"AR", "B":0.98, "sigma":0.1, "is_tv":[False], "X_type":"uniform", "size_beta_t": 2*dgp_par["N"], "beta_0":torch.randn (2*dgp_par["N"],1)}
dgp_dist_par = None

dgp_par["dgp_phi"] = dgp_phi
dgp_par["dgp_beta"]= dgp_beta
dgp_par["dgp_dist_par"]= dgp_dist_par




Y_T_test, _, _ =  get_test_w_seq(avg_weight=1e3)
Y_reference = (Y_T_test[:dgp_par["N"],:dgp_par["N"],0]>0).float()
mod_dgp = get_dgp_model(dgp_par, Y_reference=Y_reference)

mod_dgp.X_T
mod_dgp.beta_T[0]

# mod_dgp.plot_beta_T()
mod_dgp.plot_phi_T()



#%% Test single snapshot estimates 
filt_kwargs = {"size_beta_t":dgp_beta["size_beta_t"], "X_T" : mod_dgp.X_T, "beta_tv":np.array([True])}
sim_args  = {"max_opt_iter": 1000, "tb_fold": "./tb_logs"}


mod_naive = dirBin1_SD(mod_dgp.Y_T, **filt_kwargs)
mod_naive.init_phi_T_from_obs()

mod_sd = dirBin1_SD(mod_dgp.Y_T, **filt_kwargs)

mod_sd.opt_options_sd["max_opt_iter"] = sim_args["max_opt_iter"]
_, h_par_opt = mod_sd.estimate_sd(tb_save_fold=sim_args["tb_fold"])


mod_ss = dirBin1_sequence_ss(mod_dgp.Y_T, **filt_kwargs, beta_start_val = 1)
mod_ss.opt_options_ss_seq["max_opt_iter"] = sim_args["max_opt_iter"]

_, h_par_opt = mod_ss.estimate_ss_seq_joint(tb_save_fold=sim_args["tb_fold"])
mod_ss.beta_T


#%%
i = 7

fig, ax = mod_dgp.plot_phi_T(i=i)
mod_ss.plot_phi_T(i=i, fig_ax = (fig, ax))
mod_sd.plot_phi_T(i=i, fig_ax = (fig, ax))

fig, ax = mod_dgp.plot_beta_T()
mod_sd.plot_beta_T(fig_ax = (fig, ax))
mod_ss.plot_beta_T(fig_ax = (fig, ax))




# %%
mod_dgp.plot_beta_T()
mod_sd.plot_beta_T()
mod_ss.plot_beta_T()
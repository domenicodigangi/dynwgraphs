

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
from dynwgraphs.utils.tensortools import strIO_from_tens_T, tens, splitVec, strIO_from_tens_T
from dynwgraphs.dirGraphs1_dynNets import  dirBin1_sequence_ss, dirBin1_SD, dirSpW1_SD, dirSpW1_sequence_ss
import importlib

importlib.reload(dynwgraphs)
from dynwgraphs.utils.dgps import get_test_w_seq, get_dgp_model, dgpAR

#%%

dgp_par = {"N" : 50, \
        "T" : 100, \
        "model":"dirBin1", \
        "n_ext_reg": 0, \
        "size_beta_t": "one", \
        "type_dgp_phi": "const_unif_0.5", \
        "type_dgp_beta": "AR", \
        "all_beta_tv": True}

mod_dgp_bin, dgp_par_bin, Y_reference_bin = get_dgp_model(**dgp_par)

mod_dgp_bin.phi_tv
mod_dgp_bin.sample_Y_T().mean()

#%%
dgp_par["model"] = "dirSpW1"
dgp_par["type_dgp_phi"] = "AR"


mod_dgp_w, dgp_par_w, Y_reference_w = get_dgp_model(**dgp_par)
mod_dgp_w.bin_mod = mod_dgp_bin


mod_dgp_bin.beta_T
mod_dgp_bin.size_beta_t

# mod_dgp_bin.plot_beta_T()
# mod_dgp_w.plot_beta_T()


mod_dgp_bin.Y_T = mod_dgp_bin.sample_Y_T() 
mod_dgp_w.Y_T = mod_dgp_w.sample_Y_T(mod_dgp_w.bin_mod.Y_T>0)



#%%


#%% Test estimates 
filt_kwargs = {"size_beta_t":mod_dgp_bin.size_beta_t, "X_T" : mod_dgp_bin.X_T, "beta_tv":mod_dgp_bin.beta_tv}



sim_args  = {"max_opt_iter": 11, "tb_fold": "./tb_logs"}

mod_ss_bin = dirBin1_sequence_ss(mod_dgp_bin.Y_T, **filt_kwargs, beta_start_val = 1)
mod_ss_bin.opt_options_ss_seq["max_opt_iter"] = sim_args["max_opt_iter"]
_, h_par_opt = mod_ss_bin.estimate_ss_seq_joint(tb_save_fold=sim_args["tb_fold"])


mod_naive_bin = dirBin1_SD(mod_dgp_bin.Y_T, **filt_kwargs)
mod_naive_bin.init_phi_T_from_obs()
mod_naive_bin.phi_T[1].requires_grad


mod_sd_bin = dirBin1_SD(mod_dgp_bin.Y_T, **filt_kwargs)

mod_sd_bin.opt_options_sd["max_opt_iter"] = sim_args["max_opt_iter"]
_, h_par_opt = mod_sd_bin.estimate_sd(tb_save_fold=sim_args["tb_fold"])


#%%
filt_kwargs = {"size_beta_t":mod_dgp_w.size_beta_t, "X_T" : mod_dgp_w.X_T, "beta_tv":mod_dgp_w.beta_tv}

sim_args  = {"max_opt_iter": 5000, "tb_fold": "./tb_logs"}

mod_ss_w = dirSpW1_sequence_ss(mod_dgp_w.Y_T, **filt_kwargs, beta_start_val = 1)
mod_ss_w.opt_options_ss_seq["max_opt_iter"] = sim_args["max_opt_iter"]
_, h_par_opt = mod_ss_w.estimate_ss_seq_joint(tb_save_fold=sim_args["tb_fold"])

# PER QUALCHE MOTIVO LA STIMA SD SBALLA per reti non troppo sparse! capire perchè e come inizializzarla. VA MEGLIO CON LBFGS PROVARE ALTRI ALGOS

mod_sd_w = dirSpW1_SD(mod_dgp_w.Y_T, **filt_kwargs)
mod_sd_w.opt_options_sd["max_opt_iter"] = sim_args["max_opt_iter"]
mod_sd_w.opt_options_sd["opt_n"] = "ADAMHD" 

_, h_par_opt = mod_sd_w.estimate_sd(tb_save_fold=sim_args["tb_fold"])

mod_sd_w.sd_stat_par_un_phi["A"]
mod_ss_w.dist_par_un_T
mod_sd_w.dist_par_un_T

mod_sd_w.phi_T
mod_sd_w.sd_stat_par_un_phi["B"]
mod_sd_w.sd_stat_par_un_phi["w"]

mod_sd_w.init_static_sd_from_obs()
mod_sd_w.get_unc_mean(mod_sd_w.sd_stat_par_un_phi)
mod_sd_w.link_dist_par(mod_sd_w.dist_par_un_T[0], mod_sd_w.N)


plt.plot(mod_sd_w.Y_T.sum(dim=(0,1)))
plt.plot((mod_sd_w.Y_T>0).sum(dim=(0,1)))

mod_sd_w.estimate_ss_t(1, True, True, False)

mod_sd_w.phi_T[1]
mod_sd_w.dist_par_un_T

#%%
i = 7

fig, ax = mod_dgp_bin.plot_phi_T(i=i)
mod_ss_bin.plot_phi_T(i=i, fig_ax = (fig, ax))
mod_sd_bin.plot_phi_T(i=i, fig_ax = (fig, ax))

fig, ax = mod_dgp_bin.plot_beta_T()
mod_ss_bin.plot_beta_T(fig_ax = (fig, ax))
mod_sd_bin.plot_beta_T(fig_ax = (fig, ax))

fig, ax = mod_dgp_w.plot_phi_T(i=i)
mod_ss_w.plot_phi_T(i=i, fig_ax = (fig, ax))
mod_sd_w.plot_phi_T(i=i, fig_ax = (fig, ax))

fig, ax = mod_dgp_w.plot_beta_T()
mod_ss_w.plot_beta_T(fig_ax = (fig, ax))
mod_sd_w.plot_beta_T(fig_ax = (fig, ax))


# %%
mod_dgp.plot_beta_T()
mod_sd.plot_beta_T()
mod_ss.plot_beta_T()
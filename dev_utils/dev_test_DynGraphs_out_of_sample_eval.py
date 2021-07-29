#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday July 28th 2021

"""

#%% import packages

import torch
import matplotlib.pyplot as plt
import numpy as np
import copy
import dynwgraphs
from dynwgraphs.utils.tensortools import strIO_from_tens_T, tens, splitVec, strIO_from_tens_T
from dynwgraphs.dirGraphs1_dynNets import  dirBin1_sequence_ss, dirBin1_SD, dirSpW1_SD, dirSpW1_sequence_ss
import importlib

importlib.reload(dynwgraphs)
from dynwgraphs.utils.dgps import get_test_w_seq, get_dgp_model, dgpAR

#%%

dgp_bin_set = {"N" : 50, \
        "T" : 100, \
        "model":"dirBin1", \
        "type_dgp_phi": "AR", \
        "n_ext_reg": 0, \
        "size_beta_t": "2N", \
        "type_dgp_beta": "AR", \
        "all_beta_tv": False}

mod_dgp_bin, dgp_par_bin, Y_reference_bin = get_dgp_model(**dgp_bin_set)

mod_dgp_bin.phi_tv
mod_dgp_bin.sample_Y_T().mean()

mod_dgp_bin.beta_T

dgp_w_set = copy.deepcopy(dgp_bin_set) 
dgp_w_set["model"] = "dirSpW1"
dgp_w_set["type_dgp_phi"] = "AR"
dgp_w_set["n_ext_reg"] = 0
dgp_w_set["size_beta_t"] = "2N"
dgp_w_set["all_beta_tv"] = False
dgp_w_set["type_dgp_beta"] = "AR"

mod_dgp_w, dgp_par_w, Y_reference_w = get_dgp_model(**dgp_w_set)
mod_dgp_w.bin_mod = mod_dgp_bin


mod_dgp_bin.beta_T
mod_dgp_bin.size_beta_t

# mod_dgp_bin.plot_beta_T()
# mod_dgp_w.plot_beta_T()


mod_dgp_bin.Y_T = mod_dgp_bin.sample_Y_T() 
mod_dgp_w.Y_T = mod_dgp_w.sample_Y_T(mod_dgp_w.bin_mod.Y_T>0)


#%% define filters 
filt_kwargs = {"size_beta_t":mod_dgp_bin.size_beta_t, "X_T" : mod_dgp_bin.X_T, "beta_tv":mod_dgp_bin.beta_tv}
sim_args  = {"max_opt_iter": 15000, "tb_fold": "./tb_logs"}


mod_ss_bin = dirBin1_sequence_ss(mod_dgp_bin.Y_T, **filt_kwargs, beta_start_val = 1)
mod_ss_bin.opt_options_ss_seq["max_opt_iter"] = sim_args["max_opt_iter"]
mod_sd_bin = dirBin1_SD(mod_dgp_bin.Y_T, **filt_kwargs)
mod_sd_bin.opt_options_sd["max_opt_iter"] = sim_args["max_opt_iter"]

filt_kwargs = {"size_beta_t":mod_dgp_w.size_beta_t, "X_T" : mod_dgp_w.X_T, "beta_tv":mod_dgp_w.beta_tv}
mod_ss_w = dirSpW1_sequence_ss(mod_dgp_w.Y_T, **filt_kwargs, beta_start_val = 1)

mod_sd_w = dirSpW1_SD(mod_dgp_w.Y_T, **filt_kwargs)
mod_sd_w.opt_options_sd["max_opt_iter"] = sim_args["max_opt_iter"]


#%%

import sklearn


# %%


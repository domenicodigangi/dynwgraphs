#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday July 10th 2021

"""





#%% import packages
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

#%%
import mlflow
from dynwgraphs.utils.dgps import get_test_w_seq

# %%

experiment_name = "binary directed sd filter missp dgp"
experiment = mlflow.get_experiment_by_name(experiment_name)
mlflow.set_experiment(experiment_name)

print("Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))


#%%

def get_dgp(dgp_type, dgp_par):
    N = dgp_par["N"]
    T = dgp_par["T"]
   
    Y_T_test, _, _ =  get_test_w_seq(avg_weight=1e3)
    mod_dgp = dirBin1_SD(torch.zeros(N,N,T)) 
    # set reference values for dgp 
    Y = (Y_T_test[:,:,0]>0).float()
    phi_0 = mod_dgp.start_phi_from_obs(Y)

    if dgp_par["type"] == "AR":
        mod_dgp.phi_T = mod_dgp.sample_phi_dgp_ar(phi_0, dgp_par["B"], dgp_par["sigma"], T)

        mod_dgp.Y_T = mod_dgp.sample_mats_from_par_lists(T, mod_dgp.phi_T)
    return mod_dgp

# define run's parameters
dgp_par = {"N" : 50, "T" : 100, "type" : "AR", "B" : 0.98, "sigma" : 0.1}


def sample_and_estimate(dgp_par, n_sample, max_opt_iter, avoid_ovflw_fun_flag):

    mlflow.log_param("n_sample", n_sample)
    mlflow.log_param("max_opt_iter", max_opt_iter)
    mlflow.log_param("avoid_ovflw_fun_flag", avoid_ovflw_fun_flag)

    mlflow.log_params(dgp_par)

    mod_sd = dirBin1_SD(Y_T_sampled)
    mod_naive = dirBin1_SD(Y_T_sampled)
    mod_naive.init_phi_T_from_obs()

    mod_sd.opt_options_sd["max_opt_iter"] = 5
    mod_sd.estimate_sd()

    phi_T_sd = mod_sd.par_list_to_matrix_T(mod_sd.phi_T)
    phi_T_naive = mod_naive.par_list_to_matrix_T(mod_naive.phi_T)
    phi_T_dgp = mod_dgp.par_list_to_matrix_T(mod_dgp.phi_T)


    mse_sd = (torch.square(phi_T_dgp - phi_T_sd)).mean().item()
    mse_naive = (torch.square(phi_T_dgp - phi_T_naive)).mean().item()

    metrics = {"mse_sd" : mse_sd, "mse_naive":mse_naive} 

    i=11
    mod_dgp.plot_phi_T(i=i)
    mod_sd.plot_phi_T(i=i)
    mod_naive.plot_phi_T(i=i)

    mod_sd.plot_sd_par()


    mlflow.log_metric("sd_mse", mse_sd)
    mlflow.log_metric("naive_mse", mse_naive)


if __name__ == "__main__":
    sample_and_estimate()



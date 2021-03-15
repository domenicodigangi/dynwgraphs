#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:56:03 2019

@author: domenico
"""



#%% test on data


import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
#os.chdir('./')
print(os.getcwd())
sys.path.append("./src/")
from utils import tens, strIO_from_mat

#load yearly world Trade networks
ld_data = np.load("./data/world_trade_network/world_trade_net_T.npz",
                  allow_pickle=True)
wtn_T, all_y, nodes= ld_data["wtn_T"], ld_data["all_y"], ld_data["nodes"]
dist_T, scaling_infl = ld_data["dist_T"], ld_data["scaling_infl"]

# Set unit measure and  rescale for inflation
unit_measure = 1e6
Y_T = torch.tensor(wtn_T * scaling_infl[:, 1])/unit_measure
A_T = tens(Y_T>0)
X_T = torch.tensor(dist_T/dist_T.mean())
N = A_T.shape[0]
T = A_T.shape[2]
SAVE_FOLD = './data/estimates_real_data/WTN'

from dirBin1_dynNets_old import dirBin1_staNet, dirBin1_dynNet_SD



t =T -1
Y_t = A_T[:, :, t]
X_t=X_T[:, :, t]
A_t = tens(Y_t>0)



#%% Test single snapshot estimates
model = dirBin1_dynNet_SD()
theta_ss_t, diag_iter =  model.estimate_ss_bin_t(A_t, max_mle_cycle=50, print_flag=True)
theta_ss_t, diag_iter =  model.estimate_ss_bin_t_MLE(A_t, print_flag=True)


#%% test single snapshot filter
# can take a long time
model = dirBin1_dynNet_SD()
theta_ss_est_T, diag_ss_T = model.ss_filt_bin(A_T, opt_steps=50, lRate=0.5, print_flag=True, mle_only=True)

#%% test estimate pf thetas given delta
# can take a long time
dim_delta = 1
delta = torch.ones(dim_delta)
model = dirBin1_X0_dynNet_SD()
theta_ss_reg_t, diag_ss_T = model.estimate_ss_bin_t(A_t, X_t=X_t, delta=delta,
                                                        opt_steps=500, lRate=0.5, print_flag=True)


# check that parameters est without reg are not as good
degIO = strIO_from_mat(A_t)
torch.mean(model.check_fitBin(theta_ss_reg_t, degIO,  X_t=X_t, delta=delta))
torch.mean(model.check_fitBin(theta_ss_t, degIO,  X_t=X_t, delta=delta))

theta_ss_est_T, diag_ss_T = model.ss_filt_bin(A_T[:,:,:3], opt_steps=250, lRate=0.5, print_flag=True, mle_only=True)


#%%test score driven version
model = dirBin1_X0_dynNet_SD()
B, A = torch.tensor([0.7, 0.7]), torch.ones(2) * 0.0001
wI, wO = 1 + torch.randn(N), torch.randn(N)
w = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001
dim_delta=N
delta = torch.zeros(dim_delta)
W_est, B_est, A_est, delta_est, diag = model.estimateBin_SD(A_T, X_T, dim_delta=dim_delta, Steps=2, lRate=0.05)

#%%

#%%































#
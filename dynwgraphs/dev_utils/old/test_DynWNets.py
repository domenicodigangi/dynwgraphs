#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:56:03 2019

@author: domenico
"""






"""
Simulate a known DGP and estimate the parameters multiple times
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
#os.chdir('./')
print(os.getcwd())
sys.path.append("./src/")
from dirSpW1_dynNets import dirSpW1_dynNet_SD

# test many estimates
N = 20
T = 50
learn_rate = 0.001

N_steps = 10000
N_sample = 150

N_BA = N
p_list = [0.2, 0.8]
dim_beta = (1)

all_W = np.zeros((2*N, N_sample, len(p_list)))
all_B = np.zeros((2*N_BA, N_sample, len(p_list)))
all_A = np.zeros((2*N_BA, N_sample, len(p_list)))
all_beta = np.zeros((dim_beta, N_sample, len(p_list)))

# Define DGP parameters -----------------------------------------------------
beta = 0.1*torch.ones(1)
B = torch.cat([torch.randn(N_BA)*0.03 + 0.9, torch.randn(N_BA)*0.03 + 0.8])
A = torch.cat([torch.randn(N_BA)*0.003 + 0.02, torch.randn(N_BA)*0.003 + 0.02])
wI, wO = 1 + torch.randn(N), torch.randn(N)
w = torch.cat((torch.ones(N)*wI, torch.ones(N)*wO)) * 0.001
alpha = torch.ones(1)
# generate random regressors
X_T = torch.randn(N, N, T)

# starting points for the optimization-------------------------------------------
B0 = torch.cat([torch.ones(N_BA) * 0.7, torch.ones(N_BA) * 0.7])
A0 = torch.cat([torch.ones(N_BA) * 0.0001, torch.ones(N_BA) * 0.0001])
wI, wO = 3 + torch.randn(N), torch.randn(N)
W0 = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001


model = dirSpW1_dynNet_SD()

w = model.identify(w)
W0 = model.identify(W0)

p_const = p_list[n_p]
p_t = p_mat_unif = torch.ones((N, N)) * p_const
p_T = model.seq_bin(p_mat_unif, T=T)
phi_T, Y_T = model.sd_dgp_w(w, B, A, p_T, beta=beta, X_T=X_T)


t=10
Y_mat = Y_T[:, :, t]*100000
phi = phi_T[:, t]*20
model.loglike_w_t(Y_mat, phi, beta=None, X_t=None, alpha=torch.ones(1), like_type=2)
model.loglike_w_t(Y_mat, phi, beta=None, X_t=None, alpha=torch.ones(1), like_type=1)
model.loglike_w_t(Y_mat, phi, beta=None, X_t=None, alpha=torch.ones(1), like_type=0)

W_est, B_est, A_est, beta_est, diag = model.estimate_W(Y_T, X_T, B0=B0, A0=A0, W0=W0,
                                                             dim_beta=dim_beta, Steps=N_steps,
                                                             lRate=learn_rate)

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
from dirSpW1_dynNets import dirSpW1_X0_dynNet_SD

# test many estimates
N = 20
T = 50
learn_rate = 0.025

N_steps = 8000
N_sample = 250

N_BA = N
p_list = [0.01, 0.2, 0.8]
dim_beta = (1)


SAVE_FOLD = './data/estimates_test'
file_path = SAVE_FOLD + '/dirSpW1_X0_dynNet_SD_est_test_lr_' + \
            str(learn_rate) + '_T_' + str(T) + '_N_steps_' + \
            str(N_steps) + '_dim_beta_' + str(dim_beta) + \
            '_Nsample_' + str(N_sample) + '_N_SDpars.npz'
print(file_path)

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


model = dirSpW1_X0_dynNet_SD()

w = model.identify(w)
W0 = model.identify(W0)

for n_p in range(len(p_list)):
    p_const = p_list[n_p]
    p_t = p_mat_unif = torch.ones((N, N)) * p_const
    p_T = model.seq_bin(p_mat_unif, T=T)
    for n in range(N_sample):
        print((n_p, n))
        phi_T, Y_T = model.sd_dgp_w(w, B, A, p_T, beta=beta, X_T=X_T)

        W_est, B_est, A_est, beta_est, diag = model.estimateW_SD(Y_T, X_T, B0=B0, A0=A0, W0=W0,
                                                             dim_beta=dim_beta, Steps=N_steps,
                                                             lRate=learn_rate)
        all_W[:, n, n_p] = W_est.detach().numpy()
        all_B[:, n, n_p] = B_est.detach().numpy()
        all_A[:, n, n_p] = A_est.detach().numpy()
        all_beta[:, n, n_p] = beta_est.detach().numpy()

print(file_path)
np.savez(file_path, all_W, all_B, all_A, all_beta, w, B, A, beta)









#%%
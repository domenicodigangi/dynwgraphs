#########
#Created Date: Saturday June 5th 2021
#Author: Domenico Di Gangi,  <digangidomenico@gmail.com>
#-----
#Last Modified: Sunday June 6th 2021 5:08:00 pm
#Modified By:  Domenico Di Gangi
#-----
#Description:
#-----
########


"""
Simulate  dirBin1 models with time varying parameters following a SD dgp and filter with SD model. save data
"""

import sys

import numpy as np
import torch

from dynwgraphs.utils import tens, strIO_from_mat
from dynwgraphs.dirBin1_dynNets import dirBin1_dynNet_SD

avoid_ovflw_fun_flag = True
rescale_score = False
distribution = 'bernoulli'

torch.manual_seed(2)
model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False)
#%%
N = 35
T = 100
N_sample = 2
N_steps_max = 15000

N_BA = N
n_reg = 2
n_beta_tv = 0
dim_beta = N
type_dgp = 'SD'

#define storage variables
w_sd_all = torch.zeros(2*N, N_sample)
B_sd_all = torch.zeros(2*N_BA, N_sample)
A_sd_all = torch.zeros(2*N_BA, N_sample)
Y_T_dgp_all = torch.zeros(N, N, T, N_sample)
diag_all = []

A_dgp = torch.cat([torch.ones(N_BA) * 0.05, torch.ones(N_BA) * 0.05])
B_dgp = torch.cat([torch.ones(N_BA) * 0.98, torch.ones(N_BA) * 0.98])
um_sd_par, um_degs = model.dgp_phi_var_size(N, degMin=10, degMax=25)
w_dgp = um_sd_par * (1-B_dgp)

for s in range(N_sample):
    # Sample the SD Dgp
    phi_T, beta_sd_T, Y_T_s = model.sd_dgp(w_dgp, B_dgp, A_dgp, p_T=None, beta_const=None, X_T=None, N=N, T=T)

    #score driven estimates
    B_0 = 0.9 * torch.ones(2*N)
    A_0 = 0.001 * torch.ones(2*N)
    mean_degs = strIO_from_mat(Y_T_s).mean(dim=1)
    nnz_ind = mean_degs > 0
    um_0 = torch.log(mean_degs)
    um_0[~ nnz_ind] = - 200

    w_0 = um_0 * (1-B_0)
    print(Y_T_s.sum())
    w_sd_s, B_sd_s, A_sd_s, dist_par_un_est,  diag = model.estimate_SD(Y_T_s,
                                                                       B0=B_0,
                                                                       A0=A_0,
                                                                       W0=w_0,
                                                                       max_opt_iter=N_steps_max,
                                                                       lr=0.01,
                                                                       print_flag=True, plot_flag=False,
                                                                        print_every=50)

    diag_all.append(diag)
    Y_T_dgp_all[:, :, :, s] = Y_T_s.clone()
    w_sd_all[:, s] = w_sd_s.clone().detach()
    B_sd_all[:, s] = B_sd_s.clone().detach()
    A_sd_all[:, s] = A_sd_s.clone().detach()
    print(B_sd_s)
    print(A_sd_s)
    print(s)


file_path = SAVE_FOLD + '/filter_sd_dgp_dirBin1' + \
            '_N_' + str(N) + '_T_' + str(T) + \
            '_N_steps_' + str(N_steps_max) + '_N_BA_' + str(N_BA) + \
            '_resc_score_' + str(rescale_score) + '_avoid_ovflw_fun_flag_' + str(avoid_ovflw_fun_flag) + \
            '_distr_' + distribution + \
            '_N_sample_' + str(N_sample) + \
            '_type_dgp_' + type_dgp + \
            '.npz'


print(file_path)
np.savez(file_path, w_dgp.detach(), B_dgp.detach(), A_dgp.detach(),
                    w_sd_all, B_sd_all, A_sd_all, Y_T_dgp_all)





# #%
# import matplotlib.pyplot as plt
# #plt.plot(diag)
# plt.close()
# plt.plot(Y_T_s.sum(dim=(0)).transpose(0, 1)[:, 5:])

#



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:56:03 2019

@author: domenico
"""


# %% import packages
from typing import NamedTuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import dynwgraphs
from dynwgraphs.utils.tensortools import tens, splitVec
from dynwgraphs.dirSpW1_dynNets_new import  dirSpW1_funs, dirSpW1_sequence_ss
import importlib
importlib.reload(dynwgraphs)

# %%
#load yearly world Trade networks
proj_home_path = "../../../"
ld_data = np.load(f"{proj_home_path}/data/world_trade_network/world_trade_net_T.npz", allow_pickle=True)

wtn_T, all_y, nodes= ld_data["wtn_T"], ld_data["all_y"], ld_data["nodes"]
dist_T, scaling_infl = ld_data["dist_T"], ld_data["scaling_infl"]

# Set unit measure and  rescale for inflation
unit_measure = 1e6

Y_T_all = tens(wtn_T * scaling_infl[:, 1])/unit_measure
Y_T = Y_T_all[:, :, 1:]

# X_T = tens(dist_T).log().unsqueeze(2)#torch.tensor((dist_T - dist_T.mean())/dist_T.std()).unsqueeze(2)
X_T = Y_T_all[:, :, :-1].log().unsqueeze(2)

X_T[~torch.isfinite(X_T)] = 0
N = Y_T.shape[0]
T = Y_T.shape[2]
t =T -12
Y_t = Y_T[:, :, t]
X_t=X_T[:, :, :, t]
A_t = Y_t>0
Y_tp1 = Y_T[:, :, t+1]

X_T_multi = X_T.repeat_interleave(2, dim=2)
X_T_multi[:, :, 1, :] += 1


# %% Test single snapshot estimates of  phi_t
model = dirSpW1_sequence_ss(Y_T, avoid_ovflw_fun_flag=True, distr = 'gamma') # 'lognormal')
beta_t = torch.zeros(N)
model.estimate_ss_t(t, est_phi=True, est_beta = False, est_dist_par=False)

# %% Test sequence of single snapshot estimates of  phi_T
model = dirSpW1_sequence_ss(Y_T, avoid_ovflw_fun_flag=True, distr = 'gamma') # 'lognormal')
beta_t = torch.zeros(N)
model.estimate_ss_seq_joint()


# %% Test starting points for phi
model = dirSpW1_sequence_ss(Y_T, avoid_ovflw_fun_flag=True, distr = 'gamma')
beta_X = (None, None)
phi_0 = model.start_phi_from_obs(Y_t)
model.cond_exp_Y(phi_0, *beta_X)
print((model.cond_exp_Y(phi_0, *beta_X)[Y_t>0].sum() - Y_t.sum())/ Y_t.sum())
print((model.cond_exp_Y(phi_0, *beta_X).sum() - Y_t.sum())/ Y_t.sum())
Y_t.sum()

# %%

wI, wO = 1 + torch.randn(N), torch.randn(N)
phi_0 = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001
for i in range(3):
    phi_ss_t, diag_iter =  model.estimate_ss_t(t, phi_0=phi_0, plot_flag=False, print_flag=False, max_opt_iter=5000,dist_par_un_t=torch.zeros(1))
    phi_0=phi_ss_t
    diag.append(diag_iter)
    print(model.check_tot_exp(Y_t, phi_ss_t) )
A_t = Y_t > 0
EYcond_mat = model.cond_exp_Y(phi_ss_t, beta=beta_t, X_t=X_t)
tmp = Y_t[A_t] / EYcond_mat[A_t]
torch.mean(torch.tensor(tmp<1, dtype=torch.float32))
plt.hist(tmp.detach(), 30, [0,2])

# %% Test joint ss estimates of phi and dist_par
model = dirSpW1_dynNet_SD()
dim_dist_par_un = N
dim_beta = N
n_reg = X_T.shape[2]
beta = torch.zeros(dim_beta, n_reg) * 1
phi_T_0 = model.start_phi_from_obs_T(Y_T)
phi_t_0 = phi_T_0[:, t]
par_ss_t, diag = model.estimate_ss_t( Y_t, X_t=X_T_multi[:, :, :, t], beta_t=beta, phi_0=phi_t_0,
                                      dist_par_un_t=model.dist_par_un_start_val(dim_dist_par_un), like_type=2,
                            est_dist_par=True, dim_dist_par_un=N, max_opt_iter=50, print_flag=True, print_every=1)

model.loglike_t(Y_t, phi_t_0, X_t=X_t, beta=beta)

# %% Test joint ss estimates of phi and dist_par and beta
model = dirSpW1_dynNet_SD()

par_ss_t, diag = model.estimate_ss_t( Y_t, X_t=X_t, beta_t=None, phi_0=None, dist_par_un_t=None, like_type=2,
                            est_dist_par=True, dim_dist_par_un=N, est_beta=True, dim_beta=1,
                                        max_opt_iter=50, print_flag=True)

# %% test estimate of sequence of phi given static diss par
model = dirSpW1_staNet(avoid_ovflw_fun_flag=True  )
dim_dist_par_un = N
phi_T_0 = model.start_phi_from_obs_T(Y_T)
dist_par_un = model.dist_par_un_start_val(dim_dist_par_un)
all_par_est_T, diag_T = model.ss_filt(Y_T, X_T=None, beta=None, phi_T_0=phi_T_0, dist_par_un=dist_par_un,
                                        est_dist_par=False, est_beta=False,
                                        max_opt_iter=4, opt_n=1, lr=0.01,
                                        print_flag=True, plot_flag=False, print_every=1)
# %% test estimate of sequence of phi given static diss par and static beta
model = dirSpW1_staNet(avoid_ovflw_fun_flag=True  )
dim_dist_par_un = N
dist_par_un = model.dist_par_un_start_val(dim_dist_par_un)
dim_beta = N
n_reg = X_T.shape[2]
beta = torch.zeros(dim_beta, n_reg) * 1
phi_T_0 = torch.randn(2*N,T)*0.0001# model.start_phi_from_obs_T(Y_T)


all_par_est_T, diag_T = model.ss_filt(Y_T, X_T=X_T_multi, beta=beta, phi_T_0=phi_T_0, dist_par_un=dist_par_un,
                                        est_dist_par=False, est_beta=False,
                                        max_opt_iter=20, opt_n=1, lr=0.01,
                                        print_flag=True, plot_flag=False, print_every=1)


# %% Tests joint estimate of const betas
model = dirSpW1_staNet(avoid_ovflw_fun_flag=True  )
dim_dist_par_un = N
dist_par_un = model.dist_par_un_start_val(dim_dist_par_un)
all_par_est_T, diag_T = model.ss_filt(Y_T, X_T=None, beta=None, phi_T_0=None, dist_par_un=dist_par_un,
                                        est_dist_par=False, est_beta=False,
                                        max_opt_iter=4, opt_n=1, lr=0.01,
                                        print_flag=True, plot_flag=False, print_every=10)

phi_ss_est_T = all_par_est_T[:2*N,:]
# test estimate of beta_t given phi_T and dist_par
dim_beta = 1
model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True  )
beta_t_est, diag_beta_t = model.estimate_beta_const_given_phi_T(Y_T, X_T_multi, phi_ss_est_T,
                                                                  dim_beta=dim_beta, dist_par_un=dist_par_un,
                                                                  max_opt_iter=10, print_flag=True, plot_flag=True)

# %% test estimate of  phi_T and constant beta and distr_par
model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True  )
phi_T, dist_par_un, beta, diag = \
model.ss_filt_est_beta_dist_par_const(Y_T, X_T=X_T_multi, beta=None, phi_T=None, dist_par_un=None, like_type=2,
                                      est_const_dist_par=True, dim_dist_par_un=1,
                                      est_const_beta=True, dim_beta=1,
                                      opt_large_steps=4, opt_n=1, max_opt_iter_phi=15, lr_phi=0.01,
                                      max_opt_iter_dist_par=15, lr_dist_par=0.01,
                                      max_opt_iter_beta=4, lr_beta=0.01,
                                      print_flag_phi=False, print_flag_dist_par=True, print_flag_beta=True,
                                      print_every=1)


# %%%
plt.close("all")
plt.plot(-np.array(diag))

model_reg = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True)
beta_t = -torch.ones(dim_beta,2)
beta_t[0,0] = 100
model_reg.regr_product(beta_t, X_T_multi[:,:,:,t])

model_reg.loglike_t(Y_T[:, :, t], phi_ss_est_T[:, t], X_t=X_T_multi[:, :, :, t], beta=beta_t)

phi_t = phi_ss_est_T[:, t]
X_t = X_T[:,:,:,t]
model_reg.loglike_t(Y_t, phi_t, beta_t, X_t)
EYcond_mat = model_reg.cond_exp_Y(phi_t, beta_t, X_t)
dist_par_un =1

A_t = Y_t > 0
"""version that uses built in torch likelihood"""
EYcond_mat = model_reg.cond_exp_Y(phi_t, beta=beta_t, X_t=X_t)
rate = torch.div(dist_par_un, EYcond_mat[A_t])
dist = torch.distributions.gamma.Gamma(dist_par_un, rate)
log_probs = dist.log_prob(Y_t[A_t])
torch.sum(log_probs)
log_probs.sum()

# %% Define Parameters for Score Driven Dynamics
N_max_opt_iter_max = 10000
N_max_opt_iter_each_iter = 200
N_iter = N_max_opt_iter_max//N_max_opt_iter_each_iter
N_BA = N

B = torch.cat([torch.ones(N_BA) * 0.95, torch.ones(N_BA) * 0.95])
A = torch.cat([torch.ones(N_BA) * 0.01, torch.ones(N_BA) * 0.01])
wI, wO = 1 + torch.randn(N), torch.randn(N)
W = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001
# %% test score
model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD = False)
model.distr = 'gamma'

phi_t = W
Y_t_test =Y_t# putZeroDiag(Y_t)
        # torch.randn(N,N) + 1000
dist_par_un =  torch.ones(1)
beta = torch.zeros(1)
X =  X_t
s_t_ad, _, _ = model.score_t(Y_t_test, phi_t, beta_t=beta, X_t=X, dist_par_un=dist_par_un, backprop_score=True)
s_t, _, _ = model.score_t(Y_t_test, phi_t, beta_t=beta, X_t=X, dist_par_un=dist_par_un, backprop_score=False)


plt.hist((s_t-s_t_ad).detach())
print((s_t-s_t_ad).abs().mean().item())#, ((s_t-s_t_ad)/s_t_ad)[s_t_ad>0].abs().max().item())
(s_t-s_t_ad).abs()

# %%  Test overflow limitations
model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True  )
model.loglike_sd_filt(W, B, A, Y_T, beta_const=None, X_T = None, dist_par_un=torch.ones(1))
model.loglike_sd_filt(W, B, A, Y_T, beta_const=None, X_T = None, dist_par_un=torch.ones(1))

phi_T, beta_sd_T = model.sd_filt(W, B, A, Y_T)
X_t = None
logl_T = 0
for t in range(T):
    Y_t = Y_T[:, :, t]
    phi_t = phi_T[:, t]
    logl_t = model.loglike_t(Y_t, phi_t)
    print(logl_t)
    logl_T = logl_T + logl_t


# %% test likelihood for numerical errors
phi=phi_t
Y_mat = Y_t
dist_par_un = torch.ones(1)

model.loglike_t(Y_mat, phi, beta=None, X_t=None, dist_par_un=torch.ones(1), like_type=2)
model.loglike_t(Y_mat, phi, beta=None, X_t=None, dist_par_un=torch.ones(1), like_type=1)
model.loglike_t(Y_mat, phi, beta=None, X_t=None, dist_par_un=torch.ones(1), like_type=0)

model.loglike_t(Y_t, phi_t)

A_mat = Y_mat > 0
log_EYcond_mat = model.cond_exp_Y(phi, ret_log=True)
# divide the computation of the loglikelihood in 4 pieces
tmp = (dist_par_un - 1) * torch.sum(torch.log(Y_mat[A_mat]))
tmp1 = - torch.sum(A_mat) * torch.lgamma(dist_par_un)
tmp2 = - dist_par_un * torch.sum(log_EYcond_mat)
# tmp3 = - torch.sum(torch.div(Y_mat[A_mat], torch.exp(log_EYcond_mat[A_mat])))
tmp3 = - torch.sum(torch.exp(torch.log(Y_mat[A_mat]) - log_EYcond_mat[A_mat]))
out = tmp + tmp1 + tmp2 + tmp3



# %% Tet score driven estimates
import utils
model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD = False )
model.backprop_sd = False

utils.tic()
W_est, B_est, A_est, dist_par_un_est, sd_par_0, diag = model.estimate_SD(Y_T, B0=B, A0=A, W0=W,
                                                                sd_par_0=None, init_filt_um=False,
                                                                max_opt_iter=20, lr=0.01, print_every=1,
                                                dim_dist_par_un=N, print_flag=True, plot_flag=False,
                                                                 est_dis_par_un=False)
utils.toc()

# %% Test SD loglike speed
import time
model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD = False )
model.distr = 'gamma'
model.backprop_sd = True
%timeit model.sd_filt(W, B, A, Y_T)


# %% Test Score Driven Estimates with static regressors
dim_beta=1
n_reg=X_T_multi.shape[2]
n_beta_tv = 0
dim_dist_par_un = 1

model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD = False )
model.distr = 'lognormal'
W_est, B_est, A_est, dist_par_un_est, beta_const_est, sd_par_0,  diag = model.estimate_SD_X0(Y_T, X_T=X_T_multi,
                                                                dim_beta=dim_beta,
                                                                n_beta_tv = n_beta_tv,
                                                                est_dis_par_un=False,
                                                                dim_dist_par_un=dim_dist_par_un,
                                                                B0=torch.cat((B, torch.zeros(n_beta_tv))),
                                                                A0=torch.cat((A, torch.zeros(n_beta_tv))),
                                                                W0=torch.cat((W, torch.zeros(n_beta_tv))),
                                                                beta_const_0= torch.zeros(dim_beta, n_reg-n_beta_tv),
                                                                max_opt_iter=5,
                                                                lr=0.01,
                                                                print_flag=True, plot_flag=False, print_every=1)

# %% Test Score Driven Estimates with time varying regressors
dim_beta=1
n_reg=X_T_multi.shape[2]
n_beta_tv = 1
dim_dist_par_un = N

model = dirSpW1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD = False )
model.distr = 'lognormal'
W_est, B_est, A_est, dist_par_un_est, beta_const_est, sd_par_0,  diag = model.estimate_SD_X0(Y_T, X_T=X_T_multi,
                                                                dim_beta=dim_beta,
                                                                n_beta_tv = n_beta_tv,
                                                                est_dis_par_un=True,
                                                                dim_dist_par_un=dim_dist_par_un,
                                                                B0=torch.cat((B, torch.zeros(n_beta_tv))),
                                                                A0=torch.cat((A, torch.zeros(n_beta_tv))),
                                                                W0=torch.cat((W, torch.zeros(n_beta_tv))),
                                                                beta_const_0= torch.zeros(dim_beta, n_reg-n_beta_tv),
                                                                max_opt_iter=20,
                                                                lr=0.01,
                                                                print_flag=True, plot_flag=False)



# %% Test log Normal version


# %% Test different score scalings

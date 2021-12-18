#########
#Created Date: Sunday June 6th 2021
#Author: Domenico Di Gangi,  <digangidomenico@gmail.com>
#-----
#Last Modified: Sunday June 6th 2021 7:28:04 pm
#Modified By:  Domenico Di Gangi
#-----
#Description:
#-----
########


# %% import packages
import numpy as np
import torch
import matplotlib.pyplot as plt

from dynwgraphs.dirBin1_dynNets import  dirBin1_staNet, dirBin1_dynNet_SD
from dynwgraphs.utils import splitVec, tens, putZeroDiag, optim_torch, gen_test_net, soft_lu_bound, soft_l_bound, degIO_from_mat, strIO_from_mat, tic, toc, rand_steps, dgpAR, putZeroDiag_T, tens, strIO_from_mat


# %% 
test_data = np.load("../tests/test_data/dir_w_test_data.npz")
Y_T = tens(test_data["Y_T"]>0)
X_T = tens(test_data["X_T"])

# %% Test single snapshot estimates of  phi
t=0
Y_t = Y_T[:, :, t]
model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=False)
model.distr = 'bernoulli'
phi_0 = model.start_phi_from_obs(Y_t, n_iter = 30)
phi_0 = model.identify(phi_0)
model.exp_A(phi_0).sum(dim=1)
Y_t.sum(dim=1)

model.check_tot_exp(Y_t, phi_0)

model.zero_deg_par_fun(Y_t, phi_0)
model.identify(model.set_zero_deg_par(Y_t, phi_0))

phi = phi_0.clone()
N = phi.shape[0] // 2
zero_deg_par_i, zero_deg_par_o = model.zero_deg_par_fun(Y_t, phi)
phi[:N][phi[:N] == 0] = zero_deg_par_i
phi[N:][phi[N:] == 0] = zero_deg_par_o

phi_ss_t, diag_iter =  model.estimate_ss_t(Y_t, phi_0=phi_0, plot_flag=False, print_flag=True, max_opt_iter=2500,
                                             dist_par_un_t=None, print_every=500)
print(model.check_tot_exp(Y_t, phi_ss_t) )

# %% test quick ss estimate
model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=False, rescale_SD = False )
degIO = strIO_from_mat(Y_t)

phi_t, diag = model.estimate_ss_t_bin_from_degIO(degIO, min_opt_iter=500,
                                    max_opt_iter=2500, opt_n=1, lr=0.01,
                                    print_flag=True, plot_flag=False, print_every=500)

Y_t.sum()
model.exp_A(phi_t).sum()
print(model.check_tot_exp(Y_t, phi_t))

# %% Test joint ss estimates of phi and beta
X_t = X_T[t].unsqueeze(1).unsqueeze(1)
model = dirBin1_dynNet_SD()
par_ss_t, diag = model.estimate_ss_t( Y_t, X_t=X_t, beta_t=None, phi_0=None, dist_par_un_t=None, like_type=2,
                                     est_beta=True, dim_beta=1,
                                        max_opt_iter=50, print_flag=True)

# %% test estimate of sequence of phi given static beta
model = dirBin1_staNet(avoid_ovflw_fun_flag=True)
dim_beta = N
n_reg = X_T.shape[2]
beta = torch.ones(dim_beta, n_reg) * 1
all_par_est_T, diag_T = model.ss_filt(Y_T, X_T=X_T_multi, beta=beta, phi_T_0=None,
                                         est_beta=False,
                                        max_opt_iter=15, opt_n=1, lr=0.01,
                                        print_flag=True, plot_flag=False, print_every=10)

plt.plot(diag_T)
# %% test estimate of beta constant given phi_T
model = dirBin1_staNet(avoid_ovflw_fun_flag=True)
all_par_est_T, diag_T = model.ss_filt(Y_T, X_T=None, beta=None, phi_T_0=None,
                                         est_beta=False,
                                        max_opt_iter=4, opt_n=1, lr=0.01,
                                        print_flag=True, plot_flag=False, print_every=10)

phi_ss_est_T = all_par_est_T[:2*N, :]
dim_beta = 1
model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True  )
beta_t_est, diag_beta_t = model.estimate_beta_const_given_phi_T(Y_T, X_T_multi, phi_ss_est_T.clone().detach(),
                                                                  dim_beta=dim_beta, dist_par_un=0,
                                                                  max_opt_iter=10, print_flag=True, plot_flag=True)

# %% test estimate of beta and phi_T
model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True  )
phi_T, dist_par_un, beta, diag = \
model.ss_filt_est_beta_const(Y_T, X_T=X_T_multi, beta=None, phi_T=None,
                                      est_const_beta=True, dim_beta=1,
                                      opt_large_steps=2, opt_n=1, max_opt_iter_phi=5, lr_phi=0.01,
                                      max_opt_iter_beta=4, lr_beta=0.01,
                                      print_flag_phi=False, print_flag_beta=True,
                                      print_every=1)



plt.close("all")
plt.plot(-np.array(diag))


# %% Define Parameters for Score Driven Dynamics
N_max_opt_iter_max = 10000
N_max_opt_iter_each_iter = 200
N_iter = N_max_opt_iter_max//N_max_opt_iter_each_iter
N_BA = N

B = torch.cat([torch.ones(N_BA) * 0.85, torch.ones(N_BA) * 0.85])
A = torch.cat([torch.ones(N_BA) * 0.1, torch.ones(N_BA) * 0.1])
wI, wO = 1 + torch.randn(N), torch.randn(N)
W = torch.cat((torch.ones(N) * wI, torch.ones(N) * wO)) * 0.001

# %% Test score
model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=True)
s_1 = model.score_t(Y_t, W, backprop_score=True, like_type=2)
s_2 = model.score_t(Y_t, W, backprop_score=False, like_type=2)

print((s_1[0] - s_2[0]).abs().sum())
print((s_1[1] - s_2[1]).abs().sum())

# %% test starting values
strIO_from_mat(model.exp_A(model.start_phi_form_obs(Y_t))) - strIO_from_mat(Y_t)
model.check_tot_exp(Y_t, model.start_phi_form_obs(Y_t), one_dim_out=False)


# %% Tet score driven estimates
model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False)
W_est, B_est, A_est, diag = model.estimate_SD(Y_T, B0=B, A0=A, W0=W, max_opt_iter=20, lr=0.01,
                                                dim_dist_par_un=1, print_flag=True, plot_flag=False,
                                                                 est_dis_par_un=False)

# %% Test Score Driven Estimates with static regressors
dim_beta=N
n_reg=X_T_multi.shape[2]
n_beta_tv = 0
dim_dist_par_un = N

model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD = False )
W_est, B_est, A_est, dist_par_un_est, beta_const_est,  diag = model.estimate_SD_X0(Y_T, X_T=X_T_multi,
                                                                dim_beta=dim_beta,
                                                                n_beta_tv = n_beta_tv,
                                                                B0=torch.cat((B, torch.zeros(n_beta_tv))),
                                                                A0=torch.cat((A, torch.zeros(n_beta_tv))),
                                                                W0=torch.cat((W, torch.zeros(n_beta_tv))),
                                                                beta_const_0= torch.zeros(dim_beta, n_reg-n_beta_tv),
                                                                max_opt_iter=50,
                                                                lr=0.01,
                                                                print_flag=True, plot_flag=False)



# %% Test sd dgp with time varying regressors
model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False)
N = 30
T = 100
N_max_opt_iter_max = 10000
N_max_opt_iter_each_iter = 200
N_iter = N_max_opt_iter_max//N_max_opt_iter_each_iter
N_BA = N
n_reg = 2
n_beta_tv = 1
dim_beta = 1

X_T = torch.randn(N, N, n_reg, T)
A = torch.cat([torch.ones(N_BA) * 0.01, torch.ones(N_BA) * 0.01])
B = torch.cat([torch.ones(N_BA) * 0.9, torch.ones(N_BA) * 0.9])
um_sd_par, um_degs = model.dgp_phi_var_size(N, degMin=5, degMax=25)
beta_const = torch.ones(dim_beta, n_reg - n_beta_tv )

if n_beta_tv * dim_beta > 0:
    A = torch.cat((A, torch.ones(n_beta_tv * dim_beta) * 0.01))
    B = torch.cat((B, torch.ones(n_beta_tv * dim_beta) * 0.01))
    um_sd_par = torch.cat((um_sd_par, torch.ones(n_beta_tv * dim_beta) * 1/(B[- n_beta_tv * dim_beta])))

W = um_sd_par * (1-B)

model.n_reg_beta_tv = n_beta_tv
model.dim_beta = dim_beta
phi_T, beta_sd_T, Y_T_dgp = model.sd_dgp(W, B, A, p_T=None, beta_const=beta_const, X_T=X_T, N=N, T=T)

Y_T_dgp



W_est, B_est, A_est, dist_par_un_est, beta_const_est,  diag = model.estimate_SD_X0(Y_T_dgp, X_T=X_T,
                                                        dim_beta=dim_beta,
                                                        n_beta_tv=n_beta_tv,
                                                        B0=B,
                                                        A0=A,
                                                        W0=W,
                                                        beta_const_0=beta_const,
                                                        max_opt_iter=20,
                                                        lr=0.01,
                                                        print_flag=True, plot_flag=False)

# %% Test missp dgps
N = 50
T = 100
N_sample = 10


model = dirBin1_dynNet_SD(avoid_ovflw_fun_flag=True, rescale_SD=False)
degb = tens([10, N-10])
um_phi, um_degs = model.dgp_phi_var_size(N, degMin=degb[0], degMax=degb[1])

Y_T_S_dgp, phi_T_dgp, X_T_dgp, beta_T_dgp = model.sample_from_dgps(N, T, N_sample, um_phi=um_phi, dgp_type='step',
                                                                    X_T=None, n_reg=2, n_reg_beta_tv=1,
                                                                    dim_beta=1, degb=degb)
plt.close()
plt.plot(phi_T_dgp[3, :])
plt.plot(phi_T_dgp[N+2, :])

plt.close()
plt.plot(Y_T_S_dgp[:, :, :, 0].sum(dim=0).transpose(0, 1)[:, 4])
Y_T_S_dgp[:, :, :, 0].sum(dim=0)

# %%











# %%



















#
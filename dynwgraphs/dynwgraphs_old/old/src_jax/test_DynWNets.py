#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:56:03 2019

@author: domenico
"""
from dirSpW1_dynNets import *

#%%#def estimate_ZA_SD_w(Y_T.p_mat_T):
#    N = Y_T.shape[0]
#    T = Y_T.shape[-1]
#    w0_un,B0_un,A0_un =  un_stat_par( npj.ones(N),0.9,0.01)
#    res1 = optimize.minimize(obj_fun, theta_start, method = 'BFGS', 
#	       options={'disp': False}, jac = jacobian_)
#    print("Convergence Achieved: ", res1.success)
#    print("Number of Function Evaluations: ", res1.nfev)
#%% Test static
    
import time
N = 50
density = 0.5
Y_mat = gen_test_net(N,density)
p_mat = npj.array(np.random.rand(N,N))
mu_mat = np.array(Y_mat,copy = True)
np.random.shuffle(mu_mat)
mu_mat = npj.array(mu_mat)
A = npj.array(np.greater(Y_mat,0))
alpha = 3

phi_opt,info =  estimate_cs_ZA_wt(Y_mat)
# test that the identification does not change the likelihood
loglike_w_t(Y_mat, identify(phi_opt)) - loglike_w_t(Y_mat, phi_opt)
phi_opt = identify(phi_opt)

# sample and estimate the same parameters
phi_dgp = phi_opt
Nsample = 5
mean_p_vals = (0.2,0.8)
mean_bias = []
dens = []
err_phi = []
for mean in mean_p_vals:
    b= 2
    a = b*mean/(1-mean)
    p_mat = npj.array(np.random.beta(a=a,b=b,size=(N,N)))
    store_phi = []
    mean_den = 0
    for i in range(Nsample):
        Y_sample = sample_ZA_gam(p_mat,cond_exp_Y(phi_dgp))
        mean_den += (np.sum(np.greater(Y_sample,0))/(N**2 - N))/Nsample
        phi_opt_iter,info = estimate_cs_ZA_wt(Y_sample)
        store_phi.append( np.array(identify(phi_opt_iter)))
    store_phi = np.array(store_phi)   
    dens.append(mean_den)
    # for each parameter compute the mean and standard deviation of the bias
    err_phi_iter = store_phi - np.array(phi_opt)
    err_phi.append( err_phi_iter)
    


#%%
from utils import hist_and_norm

hist_and_norm( err_phi[0].flatten())
hist_and_norm( err_phi[1].flatten())
#%%

dates_emid = pd.read_csv("/home/domenico/Dropbox/Dynamic_Networks/data/emid_data/csvFiles/")
all_trades = pd.read_csv("/home/domenico/Dropbox/Dynamic_Networks/data/emid_data/csvFiles/eMid_all_T.txt")








#%% Test static estimates

N = 20    
T = 100
p_const = 0.5

alpha = 1

p_t = p_mat_unif = npj.ones((N,N)) * p_const
p_T = seq_bin(p_mat_unif,T=T)

B,A =npj.ones(2)*0.9,npj.ones(2)*0.01
wI,wO = 1,1
w = np.ones(2*N); w[:N] = wI; w[N:] = wO;
w = npj.array(w)


density = 0.5
Y_t = gen_test_net(N,density)
phi_T,Y_T = sd_dgp_w(w,B,A,p_T)
 


    

# test differente forms of likelihood
def test(phi):
    return loglike_w_t_non_jit(Y_t,phi)
def test2(phi):
    return loglike_w_t_jittable(Y_t,phi)

phi_0 = npj.ones(2*N) * 0.1
print(test(phi_0)-test2(phi_0))
print(grad(test)(phi_0)- grad(test2)(phi_0))
print(hessian(test)(phi_0) - hessian(test2)(phi_0))

#Test static estimate
phi_opt,info =  estimate_cs_ZA_wt(Y_t)

import time
# for a compariso between jit and non jit modify the loglikelihood called in 
# estimate_cs_ZA_wt    
%timeit phi_opt,info =  estimate_cs_ZA_wt(Y_t)

#%% time the filter
import time

sd_filt_w(w,B,A,p_T,Y_T)

%timeit sd_filt_w(w,B,A,p_T,Y_T)
%timeit sd_filt_w_(w,B,A,p_T,Y_T)
 
 

#%% test the single snapshot filter of a SD dynamic

from matplotlib import pyplot as plt

phi_T = np.array(phi_T)
Y_T = np.array(Y_T)
s_T = np.vstack((np.sum(Y_T,axis=0),np.sum(Y_T,axis=1)))
ind = 0
fig, axs = plt.subplots(2, 1)
axs[0].plot(phi_T[ind,:])
axs[1].plot(s_T[ind,:])





phi_ss_T = ss_filt_w(Y_T)
#
ind = 7
fig, axs = plt.subplots(2, 1)
axs[0].plot(phi_T[ind,:],"-k")
axs[0].plot(phi_ss_T[ind,:],"--b")



#
#
#
#
#
#
#shape, scale = 2., 40.  # mean=4, std=2*sqrt(2)
#s = npj.random.gamma(shape, , 100)
#print(s.mean())
#import matplotlib.pyplot as plt
#import scipy.special as sps
#count, bins, ignored = plt.hist(s, 50, density=True)
#y = bins**(shape-1)*(npj.exp(-bins/scale) /
#                      (sps.gamma(shape)*scale**shape))
#plt.plot(bins, y, linewidth=2, color='r')
#plt.show()
#%%








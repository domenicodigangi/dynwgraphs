#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Saturday June 26th 2021

"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:56:03 2019

@author: domenico
"""


#%% import packages
from typing import NamedTuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import dynwgraphs
from dynwgraphs.utils.tensortools import tens, splitVec
from dynwgraphs.dirSpW1_dynNets_new import  dirSpW1_funs, dirSpW1_sequence_ss, dirSpW1_SD
import importlib
importlib.reload(dynwgraphs)

#%%

from dynwgraphs.utils.dgps import get_test_w_seq

Y_T, X_T_scalar, X_T_matrix =  get_test_w_seq(avg_weight=1e3)
X_T_test = X_T_matrix
N, _, T = Y_T.shape

#%% Test single snapshot estimates of  phi_t
model = dirSpW1_sequence_ss(Y_T, X_T=X_T_test, size_beta_t=1, ovflw_lm=True, distr = 'gamma') # 'lognormal')
model.opt_options_ss_t["max_opt_iter"] = 100

model.estimate_ss_t(1, est_phi=True, est_beta = True, est_dist_par=False)
model.get_par_data()
model.identify_phi_T()



#%% Test sequence of single snapshot estimates of  phi_T
model = dirSpW1_sequence_ss(Y_T, X_T=X_T_test, ovflw_lm=True, distr = 'gamma', size_beta_t=1, beta_tv=[True, True]) # 'lognormal')
model.opt_options_ss_seq["max_opt_iter"] = 100
model.opt_options_ss_seq["opt_n"] = "LBFGS"


model.estimate_ss_seq_joint()
model.get_par_data()


#%% Test Score driven estimates of  phi_T
model = dirSpW1_SD(Y_T, ovflw_lm=True, distr = 'gamma') # 'lognormal')
model.opt_options_sd["max_opt_iter"] = 100

model.get_par_data()
model.roll_sd_filt()
model.get_par_data()[0]
model.get_par_data()[0][:,1]
model.estimate_sd()


# %%

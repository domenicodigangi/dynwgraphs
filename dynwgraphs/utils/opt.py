#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday June 16th 2021

"""


"""
functions needed in different parts of the module
"""

import torch
from ..hypergrad import SGDHD, AdamHD
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from ..lbfgsnew import LBFGSNew


def store_and_print_opt_info(i, last_print_it, opt_steps, loss, unPar, rel_improv, grad_norm, lRate, diag, print_every,
                             roll_rel_im, no_improv_count, print_par, print_fun=None):

    grad_mean = unPar.grad.abs().mean().item()
    num_par = unPar.shape[0]
    #store info and Print them when required
    if print_fun is not None:
        fun_val = print_fun(unPar.clone().detach())
        diag.append((loss.item(), fun_val.item(), grad_norm, grad_mean, roll_rel_im))
    else:
        diag.append((loss.item(), grad_norm, grad_mean, roll_rel_im))
    tmp = unPar.data
    if i//print_every > last_print_it:
        last_print_it = i//print_every
        if print_fun is not None:
            print((i, opt_steps, loss.item(), grad_norm, grad_mean, rel_improv, roll_rel_im, no_improv_count,
                   num_par, lRate, fun_val))
        elif print_par:
            print((i, opt_steps, loss.item(), grad_norm, grad_mean, rel_improv, roll_rel_im, no_improv_count,
                   num_par, lRate, tmp))
        else:
            print((i, opt_steps, loss.item(), grad_norm, grad_mean, rel_improv, roll_rel_im, no_improv_count,
                    num_par, lRate))
    return last_print_it


def optim_torch(obj_fun_, unPar0, opt_steps=1000, opt_n=1, lRate=0.01, rel_improv_tol=5e-8, no_improv_max_count=50,
                min_n_iter=250, bandwidth=250, small_grad_th=1e-6,
                print_flag=False, print_every=10, print_par=False, print_fun=None, plot_flag=False):
    """given a function and a starting vector, run one of different pox optimizations"""
    unPar = unPar0.clone().detach()
    unPar.requires_grad = True

    optimizers = {"SGD" : torch.optim.SGD([unPar], lr=lRate, nesterov=False),
                  "ADAM": torch.optim.Adam([unPar], lr=lRate),
                  "SGDHD": SGDHD([unPar], lr=lRate, hypergrad_lr=1e-8),
                  "ADAMHD": AdamHD([unPar], lr=lRate, hypergrad_lr=1e-8),
                  "NESTEROV_1" : torch.optim.SGD([unPar], lr=lRate, momentum=0.5, nesterov=True),
                  "NESTEROV_2" : torch.optim.SGD([unPar], lr=lRate, momentum=0.7, nesterov=True),
                  "LBFGS" : torch.optim.LBFGS([unPar], lr=lRate),
                  "LBFGS_NEW" : LBFGSNew([unPar], lr=lRate, line_search_fn = True)}

    if type(opt_n) == int:
        optimizer = list(optimizers.values())[opt_n]
    elif type(opt_n) == str:
        optimizer = optimizers[opt_n]


    def closure():
        # if torch.is_grad_enabled():
        optimizer.zero_grad()
        loss = obj_fun_(unPar)
        # if loss.requires_grad:
        loss.backward()
        return loss
    last_print_it=0
    diag = []
    rel_im = np.ones(0)
    i = 0
    loss = closure()
    last_loss = loss.item()
    no_improv_flag = False
    small_grad_flag = False
    nan_flag = False
    no_improv_count = 0
    while (i <= opt_steps) and (not no_improv_flag) & (not small_grad_flag) & (not nan_flag):
        loss = optimizer.step(closure)

        # check the gradient's norm
        grad_norm = unPar.grad.norm().item()
        small_grad_flag = grad_norm < small_grad_th
        # check presence of nans in opt vector
        nan_flag = torch.isnan(unPar).any().item()
        # check improvement
        #print((i, loss.item()))
        rel_improv = (last_loss - loss.item())
        if not (loss.item() == 0):
            rel_improv = rel_improv/loss.abs().item()
        rel_im = np.append(rel_im, rel_improv)
        last_loss = loss.item()
        if i > min_n_iter:
            roll_rel_im = rel_im[-bandwidth:].mean()
            if roll_rel_im < rel_improv_tol:
                no_improv_count = no_improv_count + 1
            else:
                no_improv_count = 0
            if no_improv_count > no_improv_max_count:
                no_improv_flag = True
        else:
            roll_rel_im = rel_im.mean()

        last_print_it = store_and_print_opt_info(i, last_print_it, opt_steps, loss, unPar, rel_improv, grad_norm, lRate, diag,
                                                print_every, roll_rel_im, no_improv_count, print_par, print_fun=print_fun)
        i = i+1
    loss = closure()
    grad_norm = unPar.grad.norm().item()
    store_and_print_opt_info(i, last_print_it, opt_steps, loss, unPar, rel_improv, grad_norm, lRate, diag,
                             print_every, -99999.0, no_improv_count, print_par, print_fun=print_fun)

    if plot_flag:
        plt.figure()
        plt.plot(diag)
        plt.legend(legend[opt_n])

    unPar_est = unPar.clone()
    return unPar_est, diag























#
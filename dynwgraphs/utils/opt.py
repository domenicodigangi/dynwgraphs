#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Sunday June 27th 2021

"""


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
from torch.utils.tensorboard.summary import hparams
from ..hypergrad import SGDHD, AdamHD
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from pathlib import Path
from dynwgraphs.adahessian.image_classification.optim_adahessian import Adahessian

from ..lbfgsnew import LBFGSNew
from torch.utils.tensorboard import SummaryWriter

import logging
logger = logging.getLogger(__name__)

def store_and_print_opt_info(i, last_print_it, max_opt_iter, loss, unPar, rel_improv, grad_norm, lr, diag, print_every,
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
            print((i, max_opt_iter, loss.item(), grad_norm, grad_mean, rel_improv, roll_rel_im, no_improv_count,
                   num_par, lr, fun_val))
        elif print_par:
            print((i, max_opt_iter, loss.item(), grad_norm, grad_mean, rel_improv, roll_rel_im, no_improv_count,
                   num_par, lr, tmp))
        else:
            print((i, max_opt_iter, loss.item(), grad_norm, grad_mean, rel_improv, roll_rel_im, no_improv_count,
                    num_par, lr))
    return last_print_it


def get_rel_improv(prev_loss, loss):
    if (loss != 0): 
        return (prev_loss - loss)/np.abs(loss) 
    else:
        return prev_loss/np.abs(loss)

def optim_torch_old(obj_fun_, unPar0, max_opt_iter=1000, opt_n="ADAM", lr=0.01, rel_improv_tol=5e-8, no_improv_max_count=50,
                min_opt_iter=250, bandwidth=250, small_grad_th=1e-6,
                print_flag=False, print_every=10, print_par=False, print_fun=None, plot_flag=False):
    """given a function and a starting vector, run one of different pox optimizations"""
    unPar = unPar0.clone().detach()
    unPar.requires_grad = True

    optimizers = {"SGD" : torch.optim.SGD([unPar], lr=lr, nesterov=False),
                  "ADAM": torch.optim.Adam([unPar], lr=lr),
                  "SGDHD": SGDHD([unPar], lr=lr, hypergrad_lr=1e-8),
                  "ADAMHD": AdamHD([unPar], lr=lr, hypergrad_lr=1e-8),
                  "NESTEROV_1" : torch.optim.SGD([unPar], lr=lr, momentum=0.5, nesterov=True),
                  "NESTEROV_2" : torch.optim.SGD([unPar], lr=lr, momentum=0.7, nesterov=True),
                  "LBFGS" : torch.optim.LBFGS([unPar], lr=lr),
                  "LBFGS_NEW" : LBFGSNew([unPar], lr=lr, line_search_fn = True)}

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
    while (i <= max_opt_iter) and (not no_improv_flag) & (not small_grad_flag) & (not nan_flag):
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
        if i > min_opt_iter:
            roll_rel_im = rel_im[-bandwidth:].mean()
            if roll_rel_im < rel_improv_tol:
                no_improv_count = no_improv_count + 1
            else:
                no_improv_count = 0
            if no_improv_count > no_improv_max_count:
                no_improv_flag = True
        else:
            roll_rel_im = rel_im.mean()

        last_print_it = store_and_print_opt_info(i, last_print_it, max_opt_iter, loss, unPar, rel_improv, grad_norm, lr, diag,
                                                print_every, roll_rel_im, no_improv_count, print_par, print_fun=print_fun)
        i = i+1
    loss = closure()
    grad_norm = unPar.grad.norm().item()
    store_and_print_opt_info(i, last_print_it, max_opt_iter, loss, unPar, rel_improv, grad_norm, lr, diag,
                             print_every, -99999.0, no_improv_count, print_par, print_fun=print_fun)

    if plot_flag:
        plt.figure()
        plt.plot(diag)
        plt.legend(legend[opt_n])

    unPar_est = unPar.clone()
    return unPar_est, diag

def grad_norm_from_list(par_list):
    parameters = [p for p in par_list if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.linalg.norm(p.grad.detach()).to(device) for p in parameters]), 2.0).item()
    return total_norm

def optim_torch(obj_fun_, unParIn, max_opt_iter=1000, opt_n="ADAM", lr=0.01, rel_improv_tol=5e-8, no_improv_max_count=50, min_opt_iter=250, bandwidth=250, small_grad_th=1e-6, folder_name="runs", run_name=None, tb_log_flag=True, hparams_dict_in=None):
    """given a function and a starting vector, run one of different pox optimizations"""


    {""}    

    if isinstance(unParIn, list):
        unPar = [par for par in unParIn if par is not None]
    elif isinstance(unParIn, torch.ParameterDict):
        unPar = unParIn
    else:
        unPar= [unParIn]

    
    optimizers = {"SGD" : torch.optim.SGD(unPar, lr=lr, nesterov=False),
                  "ADAM": torch.optim.Adam(unPar, lr=lr),
                  "SGDHD": SGDHD(unPar, lr=lr, hypergrad_lr=1e-8),
                  "ADAMHD": AdamHD(unPar, lr=lr, hypergrad_lr=1e-8),
                  "NESTEROV_1" : torch.optim.SGD(unPar, lr=lr, momentum=0.5, nesterov=True),
                  "NESTEROV_2" : torch.optim.SGD(unPar, lr=lr, momentum=0.7, nesterov=True),
                  "LBFGS" : torch.optim.LBFGS(unPar, lr=lr),
                  "LBFGS_NEW" : LBFGSNew(unPar, lr=lr, line_search_fn = True),
                  "ADAHESSIAN" : Adahessian(unPar, lr=lr,  betas= (0.9, 0.999), weight_decay=0)}

    hparams_dict = {"optimizer" :opt_n,  "lr" :lr, "max_opt_iter" :max_opt_iter, "rel_improv_tol" :rel_improv_tol, "no_improv_max_count" :no_improv_max_count,  "min_opt_iter" :min_opt_iter, "bandwidth" :bandwidth, "n_learned_par" :sum((p.numel() for p in unPar))}

    if hparams_dict_in is not None:
        hparams_dict.update(hparams_dict_in)

    logger.info(f"starting optimization with {''.join([f'{key}:: {value}, ' for key, value in hparams_dict.items()])}")

    if tb_log_flag:
        comment = run_name + "".join([f"_{k}_{v}" for k, v in hparams_dict.items()])

        full_name = Path(folder_name) 

        writer = SummaryWriter(str(full_name), comment=comment)

    optimizer = optimizers[opt_n]


    def closure():
        optimizer.zero_grad()
        loss = obj_fun_()
        if opt_n == "ADAHESSIAN":
            loss.backward(create_graph=True)
        else:
            loss.backward()
        return loss


    last_print_it=0
    diag = []
    rel_im = np.ones(0)
    n_iter = 0
    loss = closure()
    last_loss = loss.item()
    no_improv_flag = False
    small_grad_flag = False
    nan_flag = False
    no_improv_count = 0
    for n_iter in range(max_opt_iter):
        
        terminate_flag = no_improv_flag | small_grad_flag | nan_flag

        if n_iter > min_opt_iter:
            if terminate_flag:
                break
        
        loss = optimizer.step(closure)

        # check the gradient's norm
        
        grad_norm = grad_norm_from_list(unPar) 
        small_grad_flag = grad_norm < small_grad_th
        # check presence of nans in opt vector
        nan_flag = np.any([torch.isnan(par).any().item() for par in unPar])
        # check improvement
        rel_improv = get_rel_improv(last_loss, loss.item())

        rel_im = np.append(rel_im, rel_improv)
        last_loss = loss.item()
        if n_iter > min_opt_iter:
            roll_rel_im = rel_im[-bandwidth:].mean()
            if roll_rel_im < rel_improv_tol:
                no_improv_count = no_improv_count + 1
            else:
                no_improv_count = 0
            if no_improv_count > no_improv_max_count:
                no_improv_flag = True
        else:
            roll_rel_im = rel_im.mean()

        
        if tb_log_flag:
            writer.add_scalar('Loss/value', loss.item(), n_iter)
            writer.add_scalar('Loss/roll_avg_rel_improv', roll_rel_im, n_iter)
            writer.add_scalar('Loss/roll_avg_rel_improv', grad_norm, n_iter)

        logger.info(f" iter {n_iter}, grad norm {'{:.3e}'.format(grad_norm)}, roll rel improv { '{:.3e}'.format( roll_rel_im)},  loss {'{:.5e}'.format(loss.item())}")

    final_loss = closure()
    logger.info(f"final loss {final_loss.item()}")
    if tb_log_flag:
        metric_dict = {"final_loss" :loss.item()}
        writer.add_hparams(hparams_dict, metric_dict)
       
        writer.flush()
        writer.close()

    return optimizer
























#
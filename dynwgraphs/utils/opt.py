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
from torch import nn
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


def get_rel_improv(prev_loss, loss):
    if (loss != 0): 
        return (prev_loss - loss)/np.abs(loss) 
    else:
        return prev_loss/np.abs(loss)

def grad_norm_from_list(par_list):
    parameters = [p for p in par_list if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.linalg.norm(p.grad.detach()).to(device) for p in parameters]), 2.0).item()
    return total_norm

def optim_torch(obj_fun_, unParIn, max_opt_iter=1000, opt_n="ADAM", lr=0.01, rel_improv_tol=5e-8, no_improv_max_count=10, min_opt_iter=50, bandwidth=10, small_grad_th=1e-3, tb_folder="tb_logs", tb_log_flag=True, hparams_dict_in=None, run_name="", log_interval=200, disable_logging=False, clip_grad_norm_to=None, clip_grad_val_to=1e3):
    """given a function and a starting vector, run one of different pox optimizations"""
    if disable_logging:
        logger.disabled = True
    else:
        logger.disabled = False

    # do not log tensorboard data for opt runs that are clearly tests
    if max_opt_iter <=5:
        tb_log_flag = False

  
    if isinstance(unParIn, list):
        unPar = unParIn # [par for par in unParIn if par is not None]
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
                  "ADAHESSIAN" : Adahessian(unPar, lr=lr,  betas= (0.9, 0.999), weight_decay=0)}

    hparams_dict = {"optimizer": opt_n,  "lr": lr, "n_par": sum((p.numel() for p in unPar))}

    opt_info_str = run_name + ''.join([f'{key}_{value}_' for key, value in hparams_dict.items()])

    if hparams_dict_in is not None:
        hparams_dict.update(hparams_dict_in)


    tb_log_str = "not saving tb logss"
    if tb_log_flag:
        tb_log_str = f"saving to tb logs to {tb_folder}"

        full_name = Path(tb_folder) / opt_info_str

        writer = SummaryWriter(str(full_name))

    optimizer = optimizers[opt_n]

    logger.info(f"OPT START: {opt_info_str}, INIT LOSS = {obj_fun_()}, {tb_log_str}")

    def closure():
        optimizer.zero_grad()
        loss = obj_fun_()
        if opt_n == "ADAHESSIAN":
            loss.backward(create_graph=True)
        else:
            loss.backward()

        if clip_grad_norm_to is not None:
            nn.utils.clip_grad_norm_(unPar, max_norm=clip_grad_norm_to, norm_type=2)
            nn.utils.clip_grad_value_(unPar, clip_grad_val_to)
        return loss


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
        try:
            loss = optimizer.step(closure)

            # check the gradient's norm
            
            grad_norm = grad_norm_from_list(unPar) 
            grad_max_vals = [p.max().item() for p in unPar] 
            grad_max = np.max(grad_max_vals)
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
                writer.add_scalar('Loss/grad_norm', grad_norm, n_iter)
                writer.add_scalar('Loss/grad_max', grad_norm, n_iter)

            if (n_iter % log_interval) == 0:
                logger.info(f" iter {n_iter}, g_norm {'{:.3e}'.format(grad_norm)}, g_max {'{:.3e}'.format(grad_max)}, roll rel impr { '{:.3e}'.format( roll_rel_im)},  loss {'{:.5e}'.format(loss.item())}")
        except:
            raise Exception(f"Error at iter {n_iter}, with parameters {unPar}")

    opt_metrics = {"actual_n_opt_iter": n_iter, "final_grad_norm": grad_norm, "final_grad_max": grad_max, "final_roll_improv": roll_rel_im, "final_loss": loss.item()}
    hparams_dict["max_opt_iter"]= max_opt_iter


    assert torch.isfinite(loss), f"loss is {loss}"


    final_loss = closure()
    if tb_log_flag:
        metric_dict = {"final_loss" :loss.item()}
        writer.add_hparams(hparams_dict, metric_dict)

       
        writer.flush()
        writer.close()

    # add prefix to dict keys
    hparams_dict = {f"{key}": val for key, val in hparams_dict.items()}

    logger.info(f"OPT TERMINATED {hparams_dict}. FINAL LOSS={final_loss.item()} EXIT FLAGS:  no_improv_flag = {no_improv_flag} ,  small_grad_flag = {small_grad_flag} , nan_flag = {nan_flag}, {opt_metrics}")


    return optimizer, hparams_dict, opt_metrics
























#
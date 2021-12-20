#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday December 15th 2021

"""

from dynwgraphs.tobit.tobit  import TobitModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.stats import norm
import itertools
import logging
logger = logging.getLogger(__name__)


def get_ij_giraitis_reg(i, j, Y_T, T_train, t_oos):
    y_ij = Y_T[i, j, 1:T_train+t_oos+1].numpy()
    y_ij_tm1 = Y_T[i, j, :T_train+t_oos].numpy()
    y_ji_tm1 = Y_T[j, i, :T_train+t_oos].numpy()
    y_i_sum_in_tm1 = Y_T[i, :, :T_train+t_oos].sum(0) - y_ij_tm1
    y_i_sum_out_tm1 = Y_T[:, i, :T_train+t_oos].sum(0) - y_ji_tm1
    y_j_sum_in_tm1 = Y_T[j, :, :T_train+t_oos].sum(0) - y_ji_tm1
    y_j_sum_out_tm1 = Y_T[:, j, :T_train+t_oos].sum(0) - y_ij_tm1
    y_sum_not_ij_tm1 = Y_T[:, :, :T_train+t_oos].sum((0,1)) - y_ij_tm1 - y_ji_tm1

    x_T = pd.DataFrame({"y_ij_tm1": y_ij_tm1, "y_i_sum_in_tm1": y_i_sum_in_tm1, "y_j_sum_in_tm1": y_j_sum_in_tm1, "y_i_sum_out_tm1": y_i_sum_out_tm1, "y_j_sum_out_tm1": y_j_sum_out_tm1, "y_sum_not_ij_tm1": y_sum_not_ij_tm1})

    y_T = pd.Series(y_ij)
    return x_T, y_T

def predict_kernel_tobit(x_T, y_T, T_train, t_oos, ker_type="gauss", bandwidth=20):

    x_train, y_train = x_T[:T_train], y_T[:T_train]
    tr = TobitModel()
    tr.fit(x_train, y_train, type=ker_type, bandwidth=bandwidth)
    pred = tr.predict(x_T.iloc[T_train:T_train+t_oos, : ].values)
    sigma = tr.sigma_
    pred_prob_zero = norm.cdf(0, loc=pred, scale=sigma)
    bin_pred = 1-pred_prob_zero
    pred[pred<0] = 0
    fract_nnz = (y_train.values > 0).mean()
    
    # assert bin_pred.shape == pred.shape
    return {"obs": y_T.iloc[T_train:T_train+t_oos].values, "pred": pred, "bin_pred": bin_pred, "fract_nnz": fract_nnz}


def predict_ZA_regression(x_T, y_T, T_train, t_oos, ker_type=None, bandwidth=None):

    x_train, y_train = x_T[:T_train], y_T[:T_train]
    y_train_bin = y_train > 0
    model_bin = LogisticRegression(max_iter=300)
    model_w = LinearRegression()
    model_bin.fit(x_train.values, y_train_bin.values)
    model_w.fit(x_train.values, y_train.values)
    pred = model_w.predict(x_T.iloc[T_train:T_train+t_oos, : ].values)
    bin_pred = model_bin.predict_proba(x_T.iloc[T_train:T_train+t_oos, : ].values)[0][1]
    pred[pred<0] = 0
    fract_nnz = (y_train.values > 0).mean()
    
    # assert bin_pred.shape == pred.shape
    return {"obs": y_T.iloc[T_train:T_train+t_oos].values, "pred": pred, "bin_pred": bin_pred, "fract_nnz": fract_nnz}



def get_obs_and_pred_giraitis_regr_whole_mat_nnz(Y_T, T_train, t_oos, pred_fun, ker_type="gauss", bandwidth=20, max_links=None, include_zero_oos_obs=True):
    if t_oos != 1:
        raise "to handle multi step ahead need to fix the check on non zero obs and maybe other parts"
    Y_T_train_nnz = (Y_T[:, :, 1:T_train] > 0 ).sum(axis=2)
    N = Y_T.shape[0]
    obs_vec = np.zeros(0)
    pred_vec = np.zeros(0)
    bin_pred_vec = np.zeros(0)
    fract_nnz_vec = np.zeros(0)
    counter = 0
    for i, j in itertools.product(range(N), range(N)):
        if i != j:
            if Y_T_train_nnz[i, j] != 0:
                x_T, y_T = get_ij_giraitis_reg(i, j, Y_T, T_train, t_oos)
                if (Y_T_train_nnz[i, j] > 0) & (Y_T_train_nnz[i, j] < T_train-1): # do not run if no obs are nnz
                    if (y_T.iloc[T_train+t_oos-1] != 0) | include_zero_oos_obs:
                        counter += 1
                        logger.info(f" running {i,j}")
                        res = pred_fun(x_T, y_T, T_train, t_oos, ker_type=ker_type, bandwidth=bandwidth)
                    
                        obs_vec = np.append(obs_vec, res["obs"])        
                        pred_vec = np.append(pred_vec, res["pred"]) 
                        bin_pred_vec = np.append(bin_pred_vec, res["bin_pred"]) 
                        fract_nnz_vec = np.append(fract_nnz_vec, res["fract_nnz"])

                        # assert bin_pred_vec.shape == pred_vec.shape
                        if max_links is not None:
                            if counter > max_links:
                                return fract_nnz_vec, obs_vec, pred_vec, bin_pred_vec

    return fract_nnz_vec, obs_vec, pred_vec, bin_pred_vec

def apply_t(t_0, Y_T, max_links, T_train, t_oos, pred_fun, ker_type="gauss", bandwidth=20):
    logger.info(f"eval forecast {t_0}")
    Y_T_train_oos = Y_T[:, :, t_0:t_0+T_train+t_oos+1]
    fract_nnz_vec, obs_vec, pred_vec, bin_pred_vec = get_obs_and_pred_giraitis_regr_whole_mat_nnz(Y_T_train_oos, T_train, t_oos, pred_fun, max_links=max_links, ker_type=ker_type, bandwidth=bandwidth)
    t0_vec = np.full(fract_nnz_vec.shape[0], t_0)
    
    return t0_vec, fract_nnz_vec, obs_vec, pred_vec, bin_pred_vec

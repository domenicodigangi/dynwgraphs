#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Domenico Di Gangi,  <digangidomenico@gmail.com>
Created on Wednesday June 23rd 2021

"""


"""

Functions for Zero Augmented models for sparse weighted dynamical networks, coded using pytorch
Author: Domenico Di Gangi


"""
from dynwgraphs import hypergrad
import torch
import sys
import numpy as np
import os
from matplotlib import pyplot as plt
from .utils.tensortools import splitVec, tens, putZeroDiag, putZeroDiag_T, soft_lu_bound, strIO_from_mat
from .utils.opt import optim_torch
from torch.autograd import grad
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

#--------------------------
# --------- Zero Augmented Static model functions
#self = dirSpW1_dynNet_SD(ovflw_lm=True, rescale_SD = False )

class dirGraphs_funs(nn.Module):
    """
    This class collects core methods used to model one observation of directed weighted sparse static networks, modelled with a zero augmented distribution, one  parameter per each node (hence the 1 in the name)
    """

    def __init__(self, ovflw_lm, distr,  size_dist_par_un_t, size_beta_t, like_type, phi_id_type):
        super().__init__()
        self.ovflw_lm = ovflw_lm
        self.distr = distr
        self.size_dist_par_un_t = size_dist_par_un_t
        self.size_beta_t = size_beta_t
        self.like_type = like_type
        self.ovflw_exp_L_limit = -50
        self.ovflw_exp_U_limit = 40
        self.phi_id_type=phi_id_type

    def check_reg_and_coeff_pres(self, X, beta):
        """
        Check that X and beta are either both None or both not None. 
        """
        if (X is not None) and (beta is not None):
            return True
        elif (X is None) and (beta is None):
            return False 
        elif (X is not None) and (beta is None):
            raise 
        elif (X is None) and (beta is not None):
            raise 

    def check_reg_and_coeff_shape(self, X_t, beta_t):
        """
        Check that the shape of regressors and coefficients are correct and agree at every time step
        """
        if not (X_t.dim() == 3):
            raise
        if not (X_t.shape[0] == X_t.shape[1] == self.N):
            raise
        if beta_t.dim() !=2:
            raise

        if X_t.shape[2] != beta_t.shape[1]:
            raise Exception(f"X_t.shape = {X_t.shape} , beta_t.shape = {beta_t.shape} ")

        if beta_t.shape[0] not in [1, self.N, 2*self.N]:
            raise Exception(f"beta_t.shape = {beta_t.shape}, the allowed numers of parameters  beta_t for each regressors are [1, N, 2*N]" )

    def regr_product(self, beta_t, X_t):
        """
        Given a matrix of regressors and a vector of parameters obtain a matrix that has to be summed to the matrix obtained from nodes specific unrestricted dynamical paramters
        """
        if beta_t.shape[0] == 1:  # one parameter for all links
            prod = beta_t * X_t
        elif beta_t.shape[0] == self.N:
            # one parameter for each node for each regressor, i.e. same effect on in and out connections
            prod = beta_t * beta_t.unsqueeze(1) * X_t
        elif beta_t.shape[0] == 2*self.N:
            # two parameters (in and out connections) for each node for each regressor
            prod = (beta_t[:self.N, :] * beta_t[self.N:, :].unsqueeze(1)) * X_t

        return prod.sum(dim=2)

    def cond_exp_Y(self, phi, beta, X_t, ret_log=False, ret_all=False):
        """
        given the parameters related with the weights, compute the conditional expectation of matrix Y.
        the conditioning is on each matrix element being greater than zero
        can have non zero elements on the diagonal!!
        """
        phi_i, phi_o = splitVec(phi)
        log_Econd_mat = phi_i + phi_o.unsqueeze(1)
        if self.check_reg_and_coeff_pres(X_t, beta):
            log_Econd_mat = log_Econd_mat + self.regr_product(beta, X_t)

        if self.ovflw_lm:
            """if required force the exponent to stay within overflow-safe bounds"""
            log_Econd_mat_restr = \
                soft_lu_bound(log_Econd_mat, l_limit=self.ovflw_exp_L_limit, u_limit=self.ovflw_exp_U_limit)
        else:
            log_Econd_mat_restr = log_Econd_mat

        if ret_all:
            return log_Econd_mat, log_Econd_mat_restr, torch.exp(log_Econd_mat_restr)
        elif ret_log:
            return log_Econd_mat_restr
            #  putZeroDiag(log_Econd_mat)
        else:
            return torch.exp(log_Econd_mat_restr)  
            #  putZeroDiag(torch.exp(log_Econd_mat))

    def identify_phi_io(self, phi):
        """ enforce an identification condition on the phi parameters 
        """
        # set the first in parameter to zero
        phi_i, phi_o = splitVec(phi)
        if self.phi_id_type == "first_in_zero":
            """set one to zero"""
            d = phi_i[0]
        elif self.phi_id_type == "in_sum_eq_out_sum":
            """set sum of in parameters equal to sum of out par """
            d = (phi_i.mean() - phi_o.mean())/2
        elif self.phi_id_type == "in_sum_zero":
            """set sum of in parameters to zero """
            d = phi_i.mean()
       
        phi_i_out = phi_i - d
        phi_o_out = phi_o + d
        return torch.cat((phi_i_out, phi_o_out))

    def identify_phi_io_beta(self, phi, beta, x):
        """ enforce an identification condition on the phi and beta parameters 
        """
        # set the first in parameter to zero
        phi_i, phi_o = splitVec(phi)
        if self.phi_id_type != "in_sum_zero":
            raise
        else:
            if (~ torch.isclose(phi_i.mean(), torch.zeros(1), atol=1e-4)).any() :
                raise
        
        # raise error for cases where the identification problem is not yet clear
        if beta.shape[0] >1 : 
            raise
        if self.reg_cross_unique.sum() >1 : 
            raise
        if beta.shape[1] >1:
            raise

        d = phi_o.mean()
        phi_i_out = phi_i
        if x !=0 :
            phi_o_out = phi_o - d
            beta_out = beta + d/x 
        else:
            phi_o_out = phi_o
            beta_out = beta 

        return torch.cat((phi_i_out, phi_o_out)), beta_out

    def start_phi_from_obs(self, Y_t):
        pass

    def dist_from_pars(self, phi, beta, X_t, dist_par_un, A_t=None):
        """
        return a pytorch distribution. it can be matrix valued (if no A_t is given) or vector valued
        """
        pass

    def loglike_t(self, Y_t, phi, beta, X_t, dist_par_un):
        """ The log likelihood of the zero augmented gamma network model as a function of the observations
        """
        pass

    def link_dist_par(self, dist_par_un, N, A_t=None):
        """
        take as input the unrestricted version of distribution parameters (the ones that we optimize)
        return their restricted versions, compatible with the different distributions
        """
        pass

    def dist_par_un_start_val(self):
        pass    
  
    def check_tot_exp(self, Y_t, phi_t, X_t, beta_t):
        pass



class dirGraphs_sequence_ss(dirGraphs_funs):
    """
    Single snapshot sequence
    """

    def __init__(self, Y_T, X_T=None, phi_tv=True, phi_par_init="fast_mle", ovflw_lm=True, distr='gamma',  phi_id_type="in_sum_zero", like_type=2,   size_dist_par_un_t = 1, dist_par_tv=False, size_beta_t = 1, beta_tv= tens([False]).bool(), data_name="", 
            opt_options_ss_t = {"opt_n" :"ADAM", "min_opt_iter" :200, "max_opt_iter" :5000, "lr" :0.01}, 
            opt_options_ss_seq = {"opt_n" :"ADAM", "min_opt_iter" :200, "max_opt_iter" :5000, "lr" :0.01}):

        super().__init__( ovflw_lm, distr,  size_dist_par_un_t, size_beta_t, phi_id_type=phi_id_type, like_type=like_type)
        
        self.ovflw_lm = ovflw_lm
        self.Y_T = Y_T
        self.N = Y_T.shape[0]
        self.T = Y_T.shape[2]
        self.X_T = X_T
        if self.X_T is None:
            self.n_reg = 0
        else:
            self.n_reg = X_T.shape[2]

        self.reg_cross_unique = self.check_regressors_cross_uniqueness()
        self.id_phi_beta_required = False#any(self.reg_cross_unique) and any(beta_tv)
        self.phi_tv = phi_tv
        self.phi_par_init = phi_par_init
        self.dist_par_tv = dist_par_tv
        self.beta_tv = beta_tv
        self.n_beta_tv = sum(beta_tv)

        self.opt_options_ss_t = opt_options_ss_t
        self.opt_options_ss_seq = opt_options_ss_seq
        
        self.data_name = data_name
        
        if not self.phi_tv:
            self.phi_T = [torch.zeros(self.N*2, requires_grad=True)]
            raise
        else:
            self.phi_T = [torch.zeros(self.N*2, requires_grad=True) for t in range(self.T)]

        if self.dist_par_tv:
            self.dist_par_un_T = [torch.zeros(size_dist_par_un_t, requires_grad=True) for t in range(self.T)]
        else:
            self.dist_par_un_T = [torch.zeros(size_dist_par_un_t, requires_grad=True)]
            
        if self.X_T is None:
            self.beta_T = None
            if any(beta_tv):
                raise
        else:
            if any(self.beta_tv):
                self.beta_T = [torch.zeros(size_beta_t, self.n_reg, requires_grad=True) for t in range(self.T)]
            else:
                self.beta_T = [torch.zeros(size_beta_t, self.n_reg, requires_grad=True)]


    def get_obs_t(self, t):
        Y_t = self.Y_T[:, :, t]
        if self.X_T is not None:
            X_t = self.X_T[:, :, :, t]
        else:
            X_t = None
        return Y_t, X_t
 
    def get_par_t(self, t):
        """
        If a paramter is time varying, return the parameter at the t-th time step, if not return the only time step present. 
        beta_T can be None, in that case return beta_t = None
        """
        if not (0 <= t <= self.T) : 
            raise Exception(f"Requested t = {t}, T = {self.T} ")

        if self.phi_tv:
            phi_t = self.phi_T[t]
        elif len(self.phi_T) == 1:
            phi_t = self.phi_T[0]
        else:
            raise

        if self.dist_par_tv:
            dist_par_un_t = self.dist_par_un_T[t]
        elif len(self.dist_par_un_T) == 1:
            dist_par_un_t = self.dist_par_un_T[0]
        else:
            raise

        if self.beta_T is not None:
            if len(self.beta_T) == self.T:
                beta_t = self.beta_T[t]
            elif len(self.beta_T) == 1:
                beta_t = self.beta_T[0]
            else:
                raise
        else:
            beta_t = None

        return phi_t, dist_par_un_t, beta_t

    def get_seq_latent_par(self):
        phi_T_data = torch.stack([p.data for p in self.phi_T], dim=1)
        if self.beta_T is not None:
            beta_T_data = torch.stack([p.data for p in self.beta_T], dim=1)
        else:
            beta_T_data = None

        dist_par_un_T_data = torch.stack([p.data for p in self.dist_par_un_T], dim=1)

        return phi_T_data, dist_par_un_T_data, beta_T_data

    def identify_sequence(self):    
        for t in range(self.T):
            phi_t = self.phi_T[t][:]
            phi_t_identified = self.identify_phi_io(phi_t)
            if self.id_phi_beta_required:
                beta_t = self.beta_T[t][:, self.reg_cross_unique]
                x_t = self.X_T[0, 0, self.reg_cross_unique, t]
                self.phi_T[t][:], self.beta_T[t][:, self.reg_cross_unique] = self.identify_phi_io_beta(phi_t_identified, beta_t, x_t )    
            else:
                self.phi_T[t] = phi_t_identified

    def check_regressors_seq_shape(self):
        """
        check that:
            - X_T is N x N x n_reg x T and 
            - beta_T is size_beta x n_reg x (T or 1) and 
        """
        if self.check_reg_and_coeff_pres(self.X_T, self.beta_T):
            for t in range(self.T-1):
                _, X_t = self.get_obs_t(t)
                _, _, beta_t = self.get_par_t(t)
                
                self.check_reg_and_coeff_shape(X_t, beta_t) 

    def check_regressors_cross_uniqueness(self):
        """
        return a bool tensor with one entry for each regressor. True if the regressor is uniform across links for more than a few time steps, false otherwise 
        """
        reg_cross_unique = torch.zeros(self.n_reg, dtype=bool)
        for k in range(self.n_reg):
            unique_flag_T = [(self.X_T[:,:,k,t].unique().shape[0] == 1) for t in range(self.T)]
            if all(unique_flag_T):
                reg_cross_unique[k] = True
            elif np.all(unique_flag_T) > 0.1:
                raise
        if reg_cross_unique.sum() > 1:
            raise
        return reg_cross_unique

    def run_checks(self):
       self.check_regressors_seq_shape()

    def set_all_requires_grad(self, flag):
        for p in self.parameters():
            p.requires_grad = flag

    def set_requires_grad_t(self, par_list, t, flag):
        if par_list is None:
            raise
        elif len(par_list) == 1 :
            par_list[0].requires_grad = flag
        elif len(par_list) == self.T :
            par_list[t].requires_grad = flag
        else:
            raise

    def estimate_ss_t(self, t, est_phi, est_dist_par, est_beta, tb_log_flag=False):
        """
        single snapshot Maximum logLikelihood estimate of phi_t, dist_par_un_t and beta_t
        """

        Y_t, X_t = self.get_obs_t(t)
        
        par_l_to_opt = [] 

        phi_t, dist_par_un_t, beta_t = self.get_par_t(t)

        if est_phi:
            par_l_to_opt.append(phi_t)
        if est_dist_par:
            par_l_to_opt.append(dist_par_un_t)
        if est_beta:
            par_l_to_opt.append(beta_t)

        
        def obj_fun():
            phi_t, dist_par_un_t, beta_t = self.get_par_t(t)
            return - self.loglike_t(Y_t, phi_t, X_t=X_t, beta=beta_t, dist_par_un=dist_par_un_t)

        run_name = f"{self.data_name}_SingleSnap_t_{t}"

        hparams_dict = {"est_phi" :est_phi, "est_dist_par" :est_dist_par, "est_beta" :est_beta}

        optim_torch(obj_fun, list(par_l_to_opt), max_opt_iter=self.opt_options_ss_t["max_opt_iter"], opt_n=self.opt_options_ss_t["opt_n"], lr=self.opt_options_ss_t["lr"], min_opt_iter=self.opt_options_ss_t["min_opt_iter"], run_name=run_name, tb_log_flag=tb_log_flag, hparams_dict_in=hparams_dict)


    def loglike_seq_T(self):
        loglike_T = 0
        for t in range(self.T):
            Y_t, X_t = self.get_obs_t(t)
            
            phi_t, dist_par_un_t, beta_t = self.get_par_t(t)
            
            loglike_T += self.loglike_t(Y_t, phi_t, X_t=X_t, beta=beta_t, dist_par_un=dist_par_un_t)

        return loglike_T

    def init_phi_T_from_obs(self):
        if self.phi_tv:
            for t in range(self.T):
                if self.phi_par_init == "fast_mle":
                    Y_t, _ = self.get_obs_t(t)
                    self.phi_T[t] = torch.nn.Parameter(self.start_phi_from_obs(Y_t))
                elif self.phi_par_init == "rand":
                    self.phi_T[t] = torch.rand(self.phi_T[t].shape)
                elif self.phi_par_init == "zeros":
                    self.phi_T[t] = torch.zeros(self.phi_T[t].shape)           
        else:
            raise

    def plot_phi_T(self, x=None):
        if x is None:
            x = np.array(range(self.T))
        phi_T, _, _ = self.get_seq_latent_par()
        phi_i_T = phi_T[:self.N,:]
        phi_o_T = phi_T[self.N:,:]
        fig, ax = plt.subplots(2,1)
        ax[0].plot(x, phi_i_T.T)
        ax[1].plot(x, phi_o_T.T)
        return fig, ax
    
    def plot_beta_T(self, x=None):
        if x is None:
            x = np.array(range(self.T))
        _, _, beta_T = self.get_seq_latent_par()
        n_beta = beta_T.shape[2]
        fig, ax = plt.subplots(n_beta,1)
        if n_beta == 1:
            ax.plot(x, beta_T[:,:,0].T)
            ax.plot(x, beta_T[:,:,0].T)
        else:
            for i in range(n_beta):
                ax[i].plot(x, beta_T[:,:,i].T)
        return fig, ax
        
    def estimate_ss_seq_joint(self, tb_log_flag=True):
        """
        Estimate from sequence of observations a set of parameters that can be time varying or constant. If time varying estimate a different set of parameters for each time step
        """
        
        self.run_checks()

        self.set_all_requires_grad(True)

        self.init_phi_T_from_obs()
        
        def obj_fun():
            return  - self.loglike_seq_T()
  
        self.par_dict_to_opt = {}
        self.par_dict_to_opt["phi_T"] = self.phi_T
        self.par_dict_to_opt["dist_par_un_T"] = self.dist_par_un_T
        if self.beta_T is not None:
            self.par_dict_to_opt["beta_T"] = self.beta_T

      
        run_name = f"{self.data_name}_SequenceSingleSnap"

        hparams_dict = {"phi_tv" :self.phi_tv, "dist_par_tv" :self.dist_par_tv, "beta_tv" :tens(self.beta_tv), "phi_par_init" :self.phi_par_init}

        par_l_to_opt = []
        for l in self.par_dict_to_opt.values():
            par_l_to_opt.extend(l)

        optim_torch(obj_fun, par_l_to_opt, max_opt_iter=self.opt_options_ss_seq["max_opt_iter"], opt_n=self.opt_options_ss_seq["opt_n"], lr=self.opt_options_ss_seq["lr"], min_opt_iter=self.opt_options_ss_seq["min_opt_iter"], run_name=run_name, tb_log_flag=tb_log_flag, hparams_dict_in = hparams_dict)

        self.identify_sequence()


class dirGraphs_SD(dirGraphs_sequence_ss):
    """
        Version With Score Driven parameters.
        
        init_sd_type : "unc_mean", "est_joint", "est_ss_before"
    """

    def __init__(self, Y_T, X_T=None, phi_tv=True, ovflw_lm=True, distr='gamma',  size_dist_par_un_t = 1, dist_par_tv=False, size_beta_t = 1, beta_tv=[False], rescale_SD=True, init_sd_type = "unc_mean", data_name="", 
            opt_options_sd = {"opt_n" :"ADAM", "min_opt_iter" :200, "max_opt_iter" :5000, "lr" :0.01}):

        super().__init__( Y_T = Y_T, X_T=X_T, phi_tv=phi_tv, ovflw_lm=ovflw_lm, distr=distr,  size_dist_par_un_t=size_dist_par_un_t, dist_par_tv=dist_par_tv, size_beta_t=size_beta_t, beta_tv=beta_tv, data_name=data_name)
        
        self.opt_options_sd = opt_options_sd
        self.rescale_SD = rescale_SD
        self.init_sd_type = init_sd_type
   
        self.B0 = self.re2un_B_par( torch.ones(1) * 0.98)
        self.A0 = self.re2un_A_par( torch.ones(1) * 0.001)



        if self.phi_tv:
            self.sd_stat_par_un_phi = self.define_stat_un_sd_par_dict(self.phi_T[0].shape)    

        if self.dist_par_tv:
            self.sd_stat_par_un_dist_par_un = self.define_stat_un_sd_par_dict(self.dist_par_un_T[0].shape)    
           
        if self.beta_T is not None:    
            if any(self.beta_tv):
                self.sd_stat_par_un_beta = self.define_stat_un_sd_par_dict(self.beta_T[0].shape)    
            
                if not all(self.beta_tv):
                    raise

            
    def define_stat_un_sd_par_dict(self, n_sd_par):

        sd_stat_par_dict = {"w" :nn.Parameter(torch.zeros(n_sd_par)),"B" :nn.Parameter(torch.ones(n_sd_par)*self.B0),"A" :nn.Parameter(torch.ones(n_sd_par)*self.A0)}

        if self.init_sd_type != "unc_mean":
            sd_stat_par_dict["init_val"] = nn.Parameter(torch.zeros(n_sd_par))

        return nn.ParameterDict(sd_stat_par_dict)
    
    def un2re_B_par(self, B_un):
        exp_B = torch.exp(B_un)
        return torch.div(exp_B, (1 + exp_B))

    def un2re_A_par(self, A_un):
        return  torch.exp(A_un)

    def re2un_B_par(self, B_re):
        return torch.log(torch.div(B_re, 1 - B_re))

    def re2un_A_par(self, A_re):
        return  torch.log(A_re)

    def score_t(self, t):
        pass

    def update_dynw_par(self, t_to_be_updated):
        """
        score driven update of the parameters related with the weights: phi_i phi_o
        w_i and w_o need to be vectors of size N, B and A can be scalars or Vectors
        Identify the vector before updating and after
        """

        t = t_to_be_updated-1
      
        phi_t, dist_par_un_t, beta_t = self.get_par_t(t)

        score_dict = self.score_t(t)

        if self.phi_tv:

            phi_t = self.identify_phi_io(phi_t)

            s = score_dict["phi"]
            w = self.sd_stat_par_un_phi["w"]
            B = self.un2re_B_par(self.sd_stat_par_un_phi["B"])  
            A = self.un2re_A_par(self.sd_stat_par_un_phi["A"])  
            phi_tp1 = w + torch.mul(B, phi_t) + torch.mul(A, s)

            # phi_tp1 = self.identify_phi_io(phi_tp1)

            self.phi_T[t+1] = phi_tp1

        if self.dist_par_tv:            

            s = score_dict["dist_par_un"]
            w = self.sd_stat_par_un_dist_par_un["w"]  
            B = self.un2re_B_par(self.sd_stat_par_un_dist_par_un["B"])  
            A = self.un2re_A_par(self.sd_stat_par_un_dist_par_un["A"])  

            dist_par_un_tp1 = w + torch.mul(B, dist_par_un_t) + torch.mul(A, s)

            self.dist_par_un_T[t+1] = dist_par_un_tp1

        if any(self.beta_tv):
    
            s = score_dict["beta"] 
            w = self.sd_stat_par_un_beta["w"]  
            B = self.un2re_B_par(self.sd_stat_par_un_beta["B"])  
            A = self.un2re_A_par(self.sd_stat_par_un_beta["A"])  
            
            beta_tp1 = w + torch.mul(B, beta_t) + torch.mul(A, s)

            # self.identify_phi_io_beta....
            self.beta_T[t+1] = beta_tp1

    def plot_sd_par(self):
        fig, ax = plt.subplots(3,1)
        ax[0].plot(self.sd_stat_par_un_phi["w"].detach())
        ax[0].plot(self.sd_stat_par_un_phi["B"].detach())
        ax[0].plot(self.sd_stat_par_un_phi["A"].detach())
        return fig, ax

    def get_unc_mean(self, sd_stat_par_un):
        w = sd_stat_par_un["w"]
        B = self.un2re_B_par(sd_stat_par_un["B"])  

        return w/(1-B)
        
    def roll_sd_filt(self):

        """Use the static parameters and the observations, that are attributes of the class, to fiter, and update, the dynamical parameters with  score driven dynamics.
        """

        if self.init_sd_type == "unc_mean":
            if self.phi_tv:
                self.phi_T[0] = self.get_unc_mean(self.sd_stat_par_un_phi)
            if any(self.beta_tv):
                self.beta_T[0] = self.get_unc_mean(self.sd_stat_par_un_beta)
            if self.dist_par_tv:
                self.dist_par_un_T[0] = self.get_unc_mean(self.sd_stat_par_un_dist_par_un)
        elif self.init_sd_type in ["est_joint", "est_ss_before"]:
            if self.phi_tv:
                self.phi_T[0] = self.sd_stat_par_un_phi["init_val"]
            if any(self.beta_tv):
                self.beta_T[0] = self.sd_stat_par_un_beta["init_val"]
            if self.dist_par_tv:
                self.dist_par_un_T[0] = self.sd_stat_par_un_dist_par_un["init_val"] 
        else:
            raise

        for t in range(1, self.T):
            self.update_dynw_par(t)

        self.identify_sequence()

    def append_all_par_dict_to_list(self, par_dict, par_list, keys_to_exclude=[]):
        for k, v in par_dict.items():
            if k not in keys_to_exclude:
                par_list.append(v)

    def estimate_sd(self, tb_log_flag=True):

        self.run_checks()

        self.set_all_requires_grad(True)
        
        if self.init_sd_type == "est_ss_before":
            # the inititial value of sd tv par is etimated beforehand on a single snapshot
            self.estimate_ss_t(0, True, True, True)
                
        def obj_fun():
            self.roll_sd_filt()
            return - self.loglike_seq_T()
  
        # dict define for consistency with non sd version
        self.par_dict_to_opt = self.state_dict()
        par_l_to_opt = [] 
        
        if self.phi_tv:
            if self.init_sd_type == "est_ss_before":
                par_to_exclude = ["init_val"]
            else:
                par_to_exclude = []
            self.append_all_par_dict_to_list(self.sd_stat_par_un_phi, par_l_to_opt, keys_to_exclude=par_to_exclude)
        else:
            par_l_to_opt.append(self.phi_T[0])
            self.par_dict_to_opt["phi"] = self.phi_T[0]

        if self.dist_par_tv:
            if self.init_sd_type == "est_ss_before":
                par_to_exclude = ["init_val"]
            else:
                par_to_exclude = []
            self.append_all_par_dict_to_list(self.sd_stat_par_un_dist_par_un, par_l_to_opt, keys_to_exclude=par_to_exclude)

        else:
            par_l_to_opt.append(self.dist_par_un_T[0])
            self.par_dict_to_opt["dist_par_un_T"] = self.dist_par_un_T[0]

        if any(self.beta_tv):
            if self.init_sd_type == "est_ss_before":
                par_to_exclude = ["init_val"]
            else:
                par_to_exclude = []
            self.append_all_par_dict_to_list(self.sd_stat_par_un_beta, par_l_to_opt, keys_to_exclude=par_to_exclude)
            
        elif self.beta_T is not None:
            par_l_to_opt.append(self.beta_T[0])
            self.par_dict_to_opt["beta_T"] = self.beta_T[0]


        
        run_name = f"{self.data_name}_ScoreDriven"

        hparams_dict = {"phi_tv" :self.phi_tv, "dist_par_tv" :self.dist_par_tv, "beta_tv" :tens(self.beta_tv), "init_sd_type" :self.init_sd_type}


        return optim_torch(obj_fun, list(par_l_to_opt), max_opt_iter=self.opt_options_sd["max_opt_iter"], opt_n=self.opt_options_sd["opt_n"], lr=self.opt_options_sd["lr"], min_opt_iter=self.opt_options_sd["min_opt_iter"], run_name=run_name, tb_log_flag=tb_log_flag, hparams_dict_in = hparams_dict)


class dirSpW1_sequence_ss(dirGraphs_sequence_ss):
  
    def start_phi_from_obs(self, Y_t):
        N = Y_t.shape[0]
        A_t = tens(Y_t > 0)
        S_i = Y_t.sum(dim=0)
        S_o = Y_t.sum(dim=1)
        S_sqrt = S_i.sum().sqrt()
        K_i = A_t.sum(dim=0)
        K_o = A_t.sum(dim=1)
        phi_t_0_i = torch.log(S_i/S_sqrt )#* (N/K_i) )
        phi_t_0_o = torch.log(S_o/S_sqrt )#* (N/K_o) )

        phi_t_0 = torch.cat((phi_t_0_i, phi_t_0_o))

        max_val = phi_t_0[~torch.isnan(phi_t_0)].max()
        phi_t_0[torch.isnan(phi_t_0)] = -max_val
        phi_t_0[~torch.isfinite(phi_t_0)] = - max_val
        phi_t_0_i[K_i == 0] = - max_val
        phi_t_0_o[K_o == 0] = - max_val

        return self.identify_phi_io(phi_t_0)

    def dist_from_pars(self, phi, beta, X_t, dist_par_un, A_t=None):
        """
        return a pytorch distribution. it can be matrix valued (if no A_t is given) or vector valued
        """
        N = phi.shape[0]//2
        # Restrict the distribution parameters.
        dist_par_re = self.link_dist_par(dist_par_un, N, A_t=A_t)
        if self.distr == 'gamma':
            EYcond_mat = self.cond_exp_Y(phi, beta=beta, X_t=X_t)
            if A_t is None:
                rate = torch.div(dist_par_re, EYcond_mat)
                distr_obj = torch.distributions.gamma.Gamma(dist_par_re, rate)
            else:# if A_t is given, we already took into account the dimension of dist_par above when restricting it
                rate = torch.div(dist_par_re, EYcond_mat[A_t])
                distr_obj = torch.distributions.gamma.Gamma(dist_par_re, rate)

        elif self.distr == 'lognormal':
            log_EYcond_mat = self.cond_exp_Y(phi, beta=beta, X_t=X_t, ret_log=True)
            if A_t is None:
                sigma = dist_par_re
                mu = log_EYcond_mat - (sigma ** 2) / 2
                distr_obj = torch.distributions.log_normal.LogNormal(mu, sigma)
            else:  # if A_t is given, we already took into account the dimension of dist_par above when restricting it
                sigma = dist_par_re
                mu = log_EYcond_mat[A_t] - (sigma ** 2) / 2
                distr_obj = torch.distributions.log_normal.LogNormal(mu, sigma)
        return distr_obj

    def loglike_t(self, Y_t, phi, beta, X_t, dist_par_un):
        """ The log likelihood of the zero augmented gamma network model as a function of the observations
        """
        #disregard self loops if present
        Y_t = putZeroDiag(Y_t)
        A_t = Y_t > 0
        N = A_t.shape[0]
        if dist_par_un is None:
            dist_par_un = self.dist_par_un_start_val()

        if (self.distr == 'gamma') and (self.like_type in [0, 1]):# if non torch computation of the likelihood is required
            # Restrict the distribution parameters.
            dist_par_re = self.link_dist_par(dist_par_un, N, A_t)
            if self.like_type==0:
                """ numerically stable version """
                log_EYcond_mat = self.cond_exp_Y(phi, beta=beta, X_t=X_t, ret_log=True)
                # divide the computation of the loglikelihood in 4 pieces
                tmp = (dist_par_re - 1) * torch.sum(torch.log(Y_t[A_t]))
                tmp1 = - torch.sum(A_t) * torch.lgamma(dist_par_re)
                tmp2 = - dist_par_re * torch.sum(log_EYcond_mat[A_t])
                #tmp3 = - torch.sum(torch.div(Y_t[A_t], torch.exp(log_EYcond_mat[A_t])))
                tmp3 = - torch.sum(torch.exp(torch.log(Y_t[A_t])-log_EYcond_mat[A_t] + dist_par_re.log() ))
                out = tmp + tmp1 + tmp2 + tmp3
            elif self.like_type == 1:
                EYcond_mat = self.cond_exp_Y(phi, beta=beta, X_t=X_t)
                # divide the computation of the loglikelihood in 4 pieces
                tmp = (dist_par_re - 1) * torch.sum(torch.log(Y_t[A_t]))
                tmp1 = - torch.sum(A_t) * torch.lgamma(dist_par_re)
                tmp2 = - dist_par_re * torch.sum(torch.log(EYcond_mat[A_t]))
                tmp3 = - torch.sum(torch.div(Y_t[A_t], EYcond_mat[A_t])*dist_par_re)
                out = tmp + tmp1 + tmp2 + tmp3

        else:# compute the likelihood using torch buit in functions
            distr_obj = self.dist_from_pars(phi, beta, X_t, dist_par_un, A_t=A_t)
            log_probs = distr_obj.log_prob(Y_t[A_t])
            out = torch.sum(log_probs)        # softly bound loglikelihood from below??

        #out = soft_l_bound(out, -1e20)
        return out

    def link_dist_par(self, dist_par_un, N, A_t=None):
        """
        take as input the unrestricted version of distribution parameters (the ones that we optimize)
        return their restricted versions, compatible with the different distributions
        """
        if (self.distr == 'gamma') | (self.distr == 'lognormal'):
            if dist_par_un.shape[0] == 1:
                dist_par_re = torch.exp(dist_par_un)
            elif dist_par_un.shape[0] == N:
                dist_par_re = torch.exp(dist_par_un + dist_par_un.unsqueeze(1))
                if A_t is not None:
                    dist_par_re = dist_par_re[A_t]
        return dist_par_re

    def dist_par_un_start_val(self):
        if self.distr=='gamma':
            # starting point for log(alpha) in gamma distribution
            dist_par_un0 = torch.zeros(self.size_dist_par_un_t)
        elif self.distr=='lognormal':
            # starting point for log(alpha) in gamma distribution
            dist_par_un0 = torch.zeros(self.size_dist_par_un_t)
        return dist_par_un0
    
    def check_tot_exp(self, Y_t, phi_t, X_t, beta_t):
        EYcond_mat = self.cond_exp_Y(phi_t, beta=beta_t, X_t=X_t)
        EYcond_mat = putZeroDiag(EYcond_mat)
        A_t=putZeroDiag(Y_t)>0
        return torch.sum(Y_t[A_t])/torch.sum(EYcond_mat[A_t]), torch.mean(Y_t[A_t]/EYcond_mat[A_t])

    
class dirSpW1_SD(dirSpW1_sequence_ss):

    def score_t(self, t):

        """
        given the observations and the ZA gamma parameters (i.e. the cond mean
        matrix and the dist_par_un par), return the score of the distribution wrt to, node
        specific, parameters associated with the weights
        """

        Y_t, X_t = self.get_obs_t(t)

        phi_t, dist_par_un_t, beta_t = self.get_par_t(t)

        A_t_bool = Y_t > 0
        A_t = tens(A_t_bool)
        
        score_dict = {}
        if  any(self.beta_tv) | self.dist_par_tv:
            
            # compute the score with AD using Autograd
            like_t = self.loglike_t(Y_t, phi_t, beta=beta_t, X_t=X_t, dist_par_un=dist_par_un_t)

            if any(self.beta_tv):
                score_dict["beta"] = grad(like_t, beta_t, create_graph=True)[0]
            if self.dist_par_tv:
                score_dict["dist_par_un"] = grad(like_t, dist_par_un_t, create_graph=True)[0]
            if self.rescale_SD:
                raise "Rescaling not ready for beta and dist par"

        if self.phi_tv:
            sigma_mat = self.link_dist_par(dist_par_un_t, self.N)
            log_cond_exp_Y, log_cond_exp_Y_restr, cond_exp_Y = self.cond_exp_Y(phi_t, beta=beta_t, X_t=X_t, ret_all=True)

            if self.distr == 'gamma':
                tmp = (Y_t.clone()/cond_exp_Y - A_t)*sigma_mat
                if self.rescale_SD:
                    diag_resc_mat = sigma_mat*A_t

            elif self.distr == 'lognormal':
                sigma2_mat = sigma_mat**2
                log_Y_t = torch.zeros(self.N, self.N)
                log_Y_t[A_t_bool] = Y_t[A_t_bool].log()
                tmp = (log_Y_t - log_cond_exp_Y_restr*A_t)/sigma2_mat + A_t/2
                if self.rescale_SD:
                    diag_resc_mat = (sigma_mat**2)**A_t

            if self.ovflw_lm:
                L = self.ovflw_exp_L_limit
                U = self.ovflw_exp_U_limit
                # multiply the score by the derivative of the overflow limit function
                soft_bnd_der_mat = (1 - torch.tanh(2*((log_cond_exp_Y - L)/(L - U)) + 1)**2)
                tmp = soft_bnd_der_mat * tmp
                if self.rescale_SD:
                    diag_resc_mat = diag_resc_mat * (soft_bnd_der_mat**2)

            score_phi = strIO_from_mat(tmp)

            #rescale score if required
            if self.rescale_SD:
                diag_resc = strIO_from_mat(diag_resc_mat)
                diag_resc[diag_resc==0] = 1
                score_phi = score_phi/diag_resc.sqrt()

            score_dict["phi"] = score_phi

        return score_dict





